"""Tests for evofsm_rl.model.lora — Story 4.1.

Uses a small synthetic ``nn.Module`` (a stack of fake transformer blocks
each exposing ``q_proj`` / ``k_proj`` / ``v_proj`` Linear layers) for
all tests. This exercises every branch of the peft integration —
wrapping, freezing, saving, loading, counting — in milliseconds on
CPU, without loading the full 16 GB Qwen3-VL-8B model.

Real-model tests are deferred to Story 4.4 (``scripts/run_b4_evolution.py``
smoke test), where loading the actual base model is already paid for.

Run::
    python -m pytest tests/test_lora.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from evofsm_rl.model.lora import (
    DEFAULT_RANK,
    DEFAULT_TARGET_MODULES,
    attach_lora,
    count_trainable_params,
    load_lora_checkpoint,
    save_lora_checkpoint,
)


# ─────────────────────────────────────────────────────────────────────
# Fixture: tiny synthetic transformer-like module
# ─────────────────────────────────────────────────────────────────────


class _MockLayer(nn.Module):
    """Pretend attention block with q / k / v projections."""

    def __init__(self, d: int = 32):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)

    def forward(self, x, **_kwargs):
        # Swallow any extra kwargs peft's load path may sniff with
        # (``input_ids``, ``attention_mask``, etc.). This mock only
        # cares about the first positional tensor.
        return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)


class _MockModel(nn.Module):
    """Stack of fake attention blocks; large enough that rank-8 LoRA
    adds a meaningful but still minority fraction of parameters.

    ``forward`` tolerates HF-style kwargs (``input_ids``,
    ``attention_mask``, ...) so peft probing during
    ``PeftModel.from_pretrained`` doesn't crash. The first positional
    tensor is treated as the input regardless of its kwarg name.
    """

    def __init__(self, n_layers: int = 4, d: int = 32):
        super().__init__()
        self.layers = nn.ModuleList([_MockLayer(d) for _ in range(n_layers)])

    def forward(self, x=None, *, input_ids=None, **_kwargs):
        t = x if x is not None else input_ids
        if t is None:
            raise ValueError("mock model forward needs a tensor input")
        for layer in self.layers:
            t = layer(t)
        return t


def _fresh_model() -> _MockModel:
    """Fresh synthetic model — deterministic weights so save/load tests
    can compare state dicts exactly."""
    torch.manual_seed(42)
    return _MockModel(n_layers=4, d=32)


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────


def test_attach_lora_config_valid():
    """attach_lora builds a LoraConfig with the expected parameters and
    returns a peft-wrapped model that exposes the adapter config."""
    model = _fresh_model()
    wrapped = attach_lora(
        model,
        rank=8,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0.1,
        task_type="FEATURE_EXTRACTION",
    )

    # peft-wrapped models expose .peft_config (dict: adapter_name -> config).
    assert hasattr(wrapped, "peft_config"), (
        "attach_lora did not return a peft-wrapped model"
    )
    cfg = wrapped.peft_config["default"]
    assert cfg.r == 8
    assert cfg.lora_alpha == 16
    assert abs(cfg.lora_dropout - 0.1) < 1e-9
    assert set(cfg.target_modules) == {"q_proj", "v_proj"}


def test_save_load_path_handling(tmp_path: Path):
    """save_lora_checkpoint creates the directory tree if it doesn't
    exist and writes peft's canonical adapter files."""
    model = _fresh_model()
    wrapped = attach_lora(model, rank=4, target_modules=["q_proj", "v_proj"],
                         task_type="FEATURE_EXTRACTION")

    ckpt_dir = tmp_path / "new" / "nested" / "path"
    assert not ckpt_dir.exists()

    returned = save_lora_checkpoint(wrapped, ckpt_dir)
    assert returned == ckpt_dir
    assert ckpt_dir.exists()
    # peft writes the adapter config + a weight file (the extension can
    # be .safetensors or .bin depending on peft version).
    assert (ckpt_dir / "adapter_config.json").exists()
    has_weights = any(
        (ckpt_dir / name).exists()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )
    assert has_weights, f"no adapter weights found in {ckpt_dir}"


def test_attach_lora_freezes_base_weights_and_trains_adapter():
    """Post-attach invariant: every base parameter is frozen; the only
    parameters with requires_grad=True are LoRA adapter tensors."""
    model = _fresh_model()

    # Sanity: before attach, everything is trainable.
    pre_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert pre_trainable > 0

    wrapped = attach_lora(model, rank=8, target_modules=["q_proj", "v_proj"],
                         task_type="FEATURE_EXTRACTION")

    base_names_frozen = 0
    lora_names_trainable = 0
    other_trainable = 0
    for name, param in wrapped.named_parameters():
        is_lora = ("lora_A" in name) or ("lora_B" in name)
        if is_lora:
            assert param.requires_grad, f"LoRA param {name!r} not trainable"
            lora_names_trainable += 1
        else:
            if param.requires_grad:
                other_trainable += 1
            else:
                base_names_frozen += 1

    assert lora_names_trainable > 0, "no LoRA parameters were created"
    assert base_names_frozen > 0, "base weights appear unfrozen"
    assert other_trainable == 0, (
        f"non-LoRA params are trainable after attach: expected 0, got "
        f"{other_trainable}"
    )


def test_count_trainable_params_reasonable():
    """Trainable share must be a non-trivial but bounded fraction.

    For the real 8B model at rank=16 the expected range is ~0.1–2%.
    For this tiny synthetic model with rank=8 and 4 layers × 2 targets
    the LoRA adds more proportionally — loosen the upper bound, but
    still assert the share is non-trivial and less than the whole model
    (i.e. the base is really frozen).
    """
    wrapped = attach_lora(
        _fresh_model(), rank=8, target_modules=["q_proj", "v_proj"],
        task_type="FEATURE_EXTRACTION",
    )
    counts = count_trainable_params(wrapped)
    assert counts["total"] > 0
    assert 0 < counts["trainable"] < counts["total"]
    # Lower bound: LoRA contributes at least one parameter.
    assert counts["percent"] > 0.0
    # Upper bound: the base was frozen, so we're well below 100%.
    assert counts["percent"] < 50.0


def test_count_trainable_params_all_zero_when_no_params():
    """Edge case: a model with zero parameters returns percent=0.0
    without a division-by-zero crash."""

    class _Empty(nn.Module):
        def forward(self, x):
            return x

    counts = count_trainable_params(_Empty())
    assert counts == {"trainable": 0, "total": 0, "percent": 0.0}


def test_save_load_roundtrip_preserves_outputs(tmp_path: Path):
    """Save → fresh base + load → same forward output.

    The forward output of the re-loaded model on identical input must
    match the original, bit-for-bit in fp32 (LoRA is deterministic and
    save_pretrained serializes the A/B matrices exactly).
    """
    # Build + attach + seed the LoRA weights to non-zero values so the
    # forward pass actually depends on them. (peft initializes LoRA_A
    # to small Gaussian and LoRA_B to zero by default — a vanilla
    # freshly-wrapped model would produce the same output as the base,
    # which trivially round-trips. We nudge LoRA_B off zero.)
    wrapped = attach_lora(
        _fresh_model(), rank=8, target_modules=["q_proj", "v_proj"],
        task_type="FEATURE_EXTRACTION",
    )
    with torch.no_grad():
        for name, p in wrapped.named_parameters():
            if "lora_B" in name:
                p.add_(torch.randn_like(p) * 0.01)

    # Reference output.
    wrapped.eval()
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        ref_out = wrapped(x)

    # Save the adapter.
    ckpt_dir = tmp_path / "ckpt"
    save_lora_checkpoint(wrapped, ckpt_dir)

    # Load into a fresh base.
    fresh_base = _fresh_model()
    restored = load_lora_checkpoint(fresh_base, ckpt_dir)
    restored.eval()
    with torch.no_grad():
        new_out = restored(x)

    assert torch.allclose(ref_out, new_out, atol=1e-6), (
        f"output differs after save/load: "
        f"max diff = {(ref_out - new_out).abs().max().item()}"
    )


def test_default_constants_exported():
    """The defaults are importable constants so callers don't have to
    remember magic numbers."""
    assert DEFAULT_RANK == 16
    assert DEFAULT_TARGET_MODULES == ("q_proj", "v_proj")


# ─────────────────────────────────────────────────────────────────────
# Pytest-less standalone runner (matches the project convention).
# ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import inspect
    import tempfile
    import traceback

    ns = dict(globals())
    tests = [(n, f) for n, f in ns.items()
             if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        td = tempfile.TemporaryDirectory()
        try:
            sig = inspect.signature(fn)
            kwargs = {}
            if "tmp_path" in sig.parameters:
                kwargs["tmp_path"] = Path(td.name)
            fn(**kwargs)
            passed += 1
            print(f"  ok    {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
        finally:
            td.cleanup()
    total = passed + failed
    print(f"\n{passed}/{total} passed")
    raise SystemExit(0 if failed == 0 else 1)

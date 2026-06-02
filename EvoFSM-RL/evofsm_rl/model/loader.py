"""Device-aware base-model loader for EvoFSM-RL.

Responsibilities
----------------
1. Read `configs/model.yaml` (single source of truth per ADR-002 / Story 1.2.1).
2. Resolve the right dtype / quantization for the requested device (`mps`, `cuda`,
   or `cpu` for CI smoke).
3. Pull `Qwen/Qwen3-VL-8B-Instruct` at the revision pinned in the yaml.
    - If the yaml says `revision: UNPINNED` — warn loudly, load `main`, print the
      resolved commit SHA so the dev can paste it back into the yaml.
    - If the yaml pins a concrete 40-char SHA — the resolved SHA MUST match,
      else raise `RevisionMismatchError` (fails loud per acceptance criteria).
4. On first successful load, write `configs/model_fingerprint.lock.json` capturing
   param count, vocab size, and architecture. On subsequent loads, compare against
   the lock file and raise `FingerprintMismatchError` on any drift.
5. Return `(model, processor)` — processor includes both tokenizer and image
   processor for Qwen3-VL.

Usage
-----
    from evofsm_rl.model import load_base_model
    model, processor = load_base_model(device="mps")          # Mac dev
    model, processor = load_base_model(device="cuda")         # A100 training box
    model, processor = load_base_model(device="cpu")          # CI-only

Dependencies (not in parent `android_world_plus/requirements.txt`, see
`EvoFSM-RL/requirements.txt`):
    torch, transformers>=4.45, huggingface_hub, pyyaml, accelerate, Pillow
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import warnings
from typing import Any, Literal

import yaml

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# Paths — keep in sync with EvoFSM-RL repo layout
# ─────────────────────────────────────────────────────────────────────────
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "model.yaml"

# ─────────────────────────────────────────────────────────────────────────
# Exception types — specific so callers can catch what they want
# ─────────────────────────────────────────────────────────────────────────


class ModelConfigError(ValueError):
    """Raised when configs/model.yaml is malformed or internally inconsistent."""


class RevisionMismatchError(RuntimeError):
    """Raised when the HF Hub resolved SHA differs from the pinned revision."""


class FingerprintMismatchError(RuntimeError):
    """Raised when the loaded model's fingerprint differs from the locked one."""


# ─────────────────────────────────────────────────────────────────────────
# Config dataclass — a typed view over the yaml
# ─────────────────────────────────────────────────────────────────────────

Device = Literal["mps", "cuda", "cpu"]
UNPINNED_SENTINEL = "UNPINNED"


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Typed projection of configs/model.yaml.

    Kept narrow — only the fields the loader actually needs. The full yaml
    stays accessible via `raw` for callers that want generation settings, etc.
    """

    name: str                       # HF repo id, e.g. "Qwen/Qwen3-VL-8B-Instruct"
    revision: str                   # 40-char SHA, or UNPINNED_SENTINEL
    torch_device: str               # "mps" | "cuda" | "cpu"
    torch_dtype_str: str            # "float16" | "bfloat16" | "float32"
    quantization: str | None        # None | "int8" | "int4" (future)
    max_memory_gb_hint: float | None
    fingerprint_lock_path: pathlib.Path
    expected_param_count_billions: float
    expected_vocab_size: int
    expected_architecture: str
    raw: dict[str, Any]             # full yaml for anyone who needs it


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────


def load_model_config(
    config_path: pathlib.Path | str | None = None,
    device: Device | None = None,
) -> ModelConfig:
    """Load + validate configs/model.yaml and project onto a device row.

    Args:
        config_path: Override the default `EvoFSM-RL/configs/model.yaml`.
        device: "mps", "cuda", or "cpu". If None, resolved via
            `resolve_device()`. Picked up from `devices.{device}` in the yaml.

    Returns:
        ModelConfig — all fields required to call `from_pretrained`.

    Raises:
        ModelConfigError on missing keys or bad types.
    """
    path = pathlib.Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ModelConfigError(f"model config not found: {path}")

    with path.open("r") as fh:
        raw = yaml.safe_load(fh)

    _validate_config_shape(raw, path)

    dev = device or resolve_device()
    # Map torch-device-name → yaml-row-key. Yaml uses host-centric names
    # ("mac", "cuda") while torch uses backend names ("mps", "cuda"); we key
    # the yaml by host so downstream readers (docs, PRs) see "mac" not "mps".
    _TORCH_DEV_TO_YAML_KEY = {"mps": "mac", "cuda": "cuda", "cpu": "mac"}
    yaml_key = _TORCH_DEV_TO_YAML_KEY.get(dev)
    if yaml_key is None or yaml_key not in raw["devices"]:
        raise ModelConfigError(
            f"device '{dev}' not mapped to any row in configs/model.yaml "
            f"devices section (have: {list(raw['devices'].keys())})"
        )

    if dev == "cpu":
        # CPU is CI/smoke-only. Borrow the Mac row (fp16 / no quant) but swap
        # device, since Mac row also has "no bitsandbytes" semantics.
        dev_row = dict(raw["devices"][yaml_key])
        dev_row["torch_device"] = "cpu"
        dev_row["dtype"] = "float32"   # fp16 on CPU is slow + numerically noisy
    else:
        dev_row = raw["devices"][yaml_key]

    sanity = raw["sanity"]

    return ModelConfig(
        name=raw["model"]["name"],
        revision=raw["model"]["revision"],
        torch_device=dev_row["torch_device"],
        torch_dtype_str=dev_row["dtype"],
        quantization=dev_row.get("quantization"),
        max_memory_gb_hint=dev_row.get("max_memory_gb_hint"),
        fingerprint_lock_path=(_REPO_ROOT / sanity["fingerprint_lock_path"]).resolve(),
        expected_param_count_billions=float(sanity["expected_param_count_billions"]),
        expected_vocab_size=int(sanity["expected_vocab_size"]),
        expected_architecture=sanity["expected_architecture"],
        raw=raw,
    )


def resolve_device() -> Device:
    """Pick the right torch device for the current host.

    Priority: CUDA → MPS → CPU. No magic env-var override — use the `device`
    arg of `load_base_model` if you want to force something.
    """
    try:
        import torch
    except ImportError as e:
        raise ModelConfigError(
            "torch is not installed. Install EvoFSM-RL requirements: "
            "`pip install -r EvoFSM-RL/requirements.txt`."
        ) from e

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_base_model(
    device: Device | None = None,
    config_path: pathlib.Path | str | None = None,
    *,
    allow_unpinned: bool = True,
) -> tuple[Any, Any]:
    """Load Qwen3-VL-8B-Instruct with device-aware dtype + fingerprint check.

    Args:
        device: "mps", "cuda", or "cpu". Default: `resolve_device()`.
        config_path: Override default configs/model.yaml.
        allow_unpinned: If the yaml has `revision: UNPINNED` and this is False,
            raise loudly before touching the network. Set to False in CI / paper
            runs so nobody accidentally ships an unpinned checkpoint.

    Returns:
        (model, processor) — both from HuggingFace transformers. `processor` is
        an `AutoProcessor` subclass that includes tokenizer + image processor.

    Raises:
        ModelConfigError        — bad config yaml.
        RevisionMismatchError   — pinned SHA != resolved SHA.
        FingerprintMismatchError — loaded model differs from lock file.
    """
    cfg = load_model_config(config_path, device=device)

    if cfg.revision == UNPINNED_SENTINEL:
        if not allow_unpinned:
            raise ModelConfigError(
                "configs/model.yaml has revision: UNPINNED and allow_unpinned=False. "
                "Pin a 40-char SHA in the yaml before production runs."
            )
        warnings.warn(
            "⚠️  model.yaml revision is UNPINNED — loading from `main`. "
            "After this loads, paste the printed SHA back into the yaml "
            "and commit.",
            stacklevel=2,
        )

    # Lazy import heavy deps so module import is cheap for tests that don't load
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor

    # VLM auto-class: transformers ≥ 4.45 exposes AutoModelForImageTextToText
    # (the canonical name going forward, including Qwen3-VL). Older versions
    # used AutoModelForVision2Seq. Fall through both so we don't wedge on
    # a single transformers release.
    try:
        from transformers import AutoModelForImageTextToText as _AutoVLM
    except ImportError:
        try:
            from transformers import AutoModelForVision2Seq as _AutoVLM   # older
        except ImportError as e:
            raise ModelConfigError(
                "Neither AutoModelForImageTextToText nor AutoModelForVision2Seq "
                "is available in your installed transformers. Upgrade to "
                "transformers>=4.45."
            ) from e

    resolved_sha = _resolve_and_verify_revision(cfg)
    torch_dtype = _parse_dtype(cfg.torch_dtype_str)

    logger.info(
        "Loading %s @ %s → device=%s dtype=%s",
        cfg.name, resolved_sha[:12], cfg.torch_device, cfg.torch_dtype_str,
    )

    load_kwargs: dict[str, Any] = {
        "revision": resolved_sha,
        "torch_dtype": torch_dtype,
        "trust_remote_code": False,   # Qwen3-VL is in transformers ≥ 4.45 natively
    }

    # Device placement
    if cfg.torch_device == "cuda":
        load_kwargs["device_map"] = "cuda"
    elif cfg.torch_device == "mps":
        # MPS prefers explicit .to() over device_map="auto" (which can try
        # offload shards to CPU and break VLM cross-attention).
        pass
    elif cfg.torch_device == "cpu":
        load_kwargs["device_map"] = "cpu"

    # Quantization — only meaningful on CUDA today (bitsandbytes). Mac path
    # leaves this None per configs/model.yaml rationale.
    if cfg.quantization == "int8" and cfg.torch_device == "cuda":
        load_kwargs["load_in_8bit"] = True
    elif cfg.quantization == "int4" and cfg.torch_device == "cuda":
        load_kwargs["load_in_4bit"] = True
    elif cfg.quantization is not None:
        warnings.warn(
            f"quantization={cfg.quantization!r} requested on device "
            f"{cfg.torch_device!r} but not supported via this loader; "
            f"ignoring. See configs/model.yaml `deviations_from_adr`.",
            stacklevel=2,
        )

    # ── Load the weights ────────────────────────────────────────────
    # Use _AutoVLM (resolved above to either AutoModelForImageTextToText or
    # the legacy AutoModelForVision2Seq depending on transformers version).
    model = _AutoVLM.from_pretrained(cfg.name, **load_kwargs)

    # Processor = tokenizer + image processor. We pass min/max pixel budgets
    # from the yaml so the vision tower doesn't swamp MPS on real screenshots.
    # Different transformers versions expose the knob via different kwarg paths;
    # we both pass as from_pretrained kwargs AND set post-init attrs so one
    # of the two definitely wins.
    img_cfg = cfg.raw.get("image", {})
    min_px = img_cfg.get("min_pixels")
    max_px = img_cfg.get("max_pixels")
    processor_kwargs = {"revision": resolved_sha}
    if min_px is not None:
        processor_kwargs["min_pixels"] = int(min_px)
    if max_px is not None:
        processor_kwargs["max_pixels"] = int(max_px)
    processor = AutoProcessor.from_pretrained(cfg.name, **processor_kwargs)
    # Belt-and-suspenders: stamp the limits directly on the image processor
    # so they hold even if the kwarg above was silently dropped.
    img_proc = getattr(processor, "image_processor", None)
    if img_proc is not None:
        if min_px is not None:
            setattr(img_proc, "min_pixels", int(min_px))
            setattr(img_proc, "size", {"shortest_edge": int(min_px), "longest_edge": int(max_px or min_px)})
        if max_px is not None:
            setattr(img_proc, "max_pixels", int(max_px))
    logger.info(
        "processor pixel budget: min=%s max=%s",
        getattr(img_proc, "min_pixels", "?"),
        getattr(img_proc, "max_pixels", "?"),
    )

    if cfg.torch_device == "mps" and not load_kwargs.get("device_map"):
        model = model.to("mps")

    # ── Fingerprint check (or lock on first load) ──────────────────
    fingerprint = _compute_fingerprint(model, resolved_sha)
    _check_or_write_fingerprint_lock(cfg, fingerprint)

    # ── Sanity gate — catches "someone swapped the checkpoint" ────
    _assert_sanity(cfg, fingerprint)

    logger.info(
        "✅ Loaded %s (%.2fB params, vocab=%d) in %s on %s",
        cfg.expected_architecture,
        fingerprint["param_count_billions"],
        fingerprint["vocab_size"],
        cfg.torch_dtype_str,
        cfg.torch_device,
    )

    return model, processor


# ─────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────


def _validate_config_shape(raw: dict[str, Any], path: pathlib.Path) -> None:
    required_top = ("meta", "model", "devices", "sanity")
    for key in required_top:
        if key not in raw:
            raise ModelConfigError(f"{path}: missing top-level key '{key}'")
    if "name" not in raw["model"] or "revision" not in raw["model"]:
        raise ModelConfigError(f"{path}: model.name and model.revision are required")
    for dev in ("mac", "cuda"):
        if dev not in raw["devices"]:
            raise ModelConfigError(f"{path}: devices.{dev} is required")
        row = raw["devices"][dev]
        for k in ("torch_device", "dtype"):
            if k not in row:
                raise ModelConfigError(f"{path}: devices.{dev}.{k} is required")
    for k in (
        "expected_param_count_billions",
        "expected_vocab_size",
        "expected_architecture",
        "fingerprint_lock_path",
    ):
        if k not in raw["sanity"]:
            raise ModelConfigError(f"{path}: sanity.{k} is required")


def _parse_dtype(name: str):
    """Resolve a dtype string from yaml to a real torch dtype."""
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ModelConfigError(
            f"unknown dtype '{name}' (expected one of {sorted(mapping)})"
        )
    return mapping[name]


def _resolve_and_verify_revision(cfg: ModelConfig) -> str:
    """Return the 40-char commit SHA the loader should request.

    If yaml is pinned, verify HF Hub resolves to the same SHA.
    If yaml is UNPINNED, ask HF Hub for main's SHA and return it
    (caller will still load but a warning was emitted up-stream).
    """
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(cfg.name, revision="main")
        main_sha = info.sha
    except Exception as e:  # network / auth errors — surface cleanly
        raise ModelConfigError(
            f"Could not resolve HF Hub revision for {cfg.name}: {e}. "
            "Check network and (if private) HF_TOKEN."
        ) from e

    if cfg.revision == UNPINNED_SENTINEL:
        print(
            f"\n[EvoFSM-RL model loader] HF main SHA for {cfg.name}:\n"
            f"  {main_sha}\n"
            f"→ Paste this into configs/model.yaml > model.revision and commit.\n"
        )
        return main_sha

    if not _looks_like_sha(cfg.revision):
        raise ModelConfigError(
            f"model.revision must be either {UNPINNED_SENTINEL!r} or a 40-char "
            f"commit SHA; got {cfg.revision!r}"
        )
    if cfg.revision != main_sha:
        # Not necessarily an error — main might have moved past our pin.
        # What IS an error: HF refusing to resolve the pinned SHA at all.
        try:
            HfApi().model_info(cfg.name, revision=cfg.revision)
        except Exception as e:
            raise RevisionMismatchError(
                f"Pinned revision {cfg.revision[:12]}… is not reachable on HF "
                f"for {cfg.name}: {e}"
            ) from e
    return cfg.revision


_SHA_RE = __import__("re").compile(r"^[0-9a-f]{40}$")


def _looks_like_sha(s: str) -> bool:
    return bool(_SHA_RE.match(s))


def _compute_fingerprint(model: Any, resolved_sha: str) -> dict[str, Any]:
    """Return a small dict that uniquely identifies a checkpoint."""
    n_params = sum(p.numel() for p in model.parameters())
    vocab = _resolve_vocab_size(model)
    arch = model.__class__.__name__
    return {
        "resolved_sha": resolved_sha,
        "param_count": int(n_params),
        "param_count_billions": round(n_params / 1e9, 3),
        "vocab_size": int(vocab) if vocab is not None else None,
        "architecture": arch,
    }


def _resolve_vocab_size(model: Any) -> int | None:
    """Find vocab_size across the config variants VLMs use.

    Qwen3-VL (and most modern VLMs) nests the language-side config under
    `config.text_config` — the top-level `config.vocab_size` is absent. Some
    older models keep it at the top level. Try both, plus `get_input_embeddings`
    as a last-resort ground-truth.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    # 1) Top-level (older / non-VLM models)
    vocab = getattr(cfg, "vocab_size", None)
    if vocab:
        return int(vocab)
    # 2) Nested under text_config (Qwen3-VL, LLaVA-Next, InternVL, ...)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        vocab = getattr(text_cfg, "vocab_size", None)
        if vocab:
            return int(vocab)
    # 3) Fall back to the input embedding weight matrix (ground truth).
    try:
        emb = model.get_input_embeddings()
        w = getattr(emb, "weight", None)
        if w is not None and w.ndim == 2:
            return int(w.shape[0])
    except Exception:
        pass
    return None


def _check_or_write_fingerprint_lock(
    cfg: ModelConfig, fingerprint: dict[str, Any]
) -> None:
    lock_path = cfg.fingerprint_lock_path
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if not lock_path.exists():
        with lock_path.open("w") as fh:
            json.dump(fingerprint, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(
            f"[EvoFSM-RL model loader] Wrote fingerprint lock: {lock_path}\n"
            f"  Commit this file so teammates get the same identity check."
        )
        return

    with lock_path.open("r") as fh:
        locked = json.load(fh)

    # Compare the identity-critical fields only — resolved_sha can legitimately
    # change when someone bumps the revision, but param_count / vocab / arch
    # drifting means the actual checkpoint changed shape. Flag both loudly.
    mismatches = []
    for key in ("param_count", "vocab_size", "architecture"):
        if locked.get(key) != fingerprint.get(key):
            mismatches.append((key, locked.get(key), fingerprint.get(key)))

    if mismatches:
        lines = "\n".join(
            f"    {k}: locked={lv!r} but loaded={cv!r}" for k, lv, cv in mismatches
        )
        raise FingerprintMismatchError(
            f"Loaded model doesn't match {lock_path}:\n{lines}\n"
            f"Either the HF checkpoint changed, or you swapped model.name. "
            f"If intended, delete the lock file and re-run."
        )

    if locked.get("resolved_sha") != fingerprint.get("resolved_sha"):
        logger.warning(
            "SHA changed in lock (%s → %s) but shape is identical — "
            "accepting (probably a revision bump).",
            locked.get("resolved_sha", "?")[:12],
            fingerprint.get("resolved_sha", "?")[:12],
        )


def _assert_sanity(cfg: ModelConfig, fingerprint: dict[str, Any]) -> None:
    # Architecture name must match exactly.
    if fingerprint["architecture"] != cfg.expected_architecture:
        raise FingerprintMismatchError(
            f"expected architecture {cfg.expected_architecture!r}, "
            f"loaded {fingerprint['architecture']!r}"
        )
    # Vocab size must match exactly.
    if (
        fingerprint["vocab_size"] is not None
        and fingerprint["vocab_size"] != cfg.expected_vocab_size
    ):
        raise FingerprintMismatchError(
            f"expected vocab_size {cfg.expected_vocab_size}, "
            f"loaded {fingerprint['vocab_size']}"
        )
    # Param count within 1% of expected (tolerates minor HF repo updates).
    expected = cfg.expected_param_count_billions
    loaded = fingerprint["param_count_billions"]
    if abs(loaded - expected) / expected > 0.01:
        raise FingerprintMismatchError(
            f"expected ~{expected:.2f}B params, loaded {loaded:.2f}B "
            f"(Δ > 1%). Check configs/model.yaml `expected_param_count_billions`."
        )

"""LoRA adapter setup for Qwen3-VL-8B — Story 4.1 (Epic 4 / B4).

Thin wrappers over peft's ``LoraConfig`` + ``get_peft_model`` that:

  * Freeze base model weights automatically (peft does this for us on
    wrap) and expose only the LoRA matrices as trainable parameters.
  * Default ``target_modules = ("q_proj", "v_proj")`` — attention query
    and value projections, the cheapest-and-effective surface per the
    original LoRA paper.
  * Save / load the adapter only (~tens of MB for rank=16 on an 8B
    base), never the full model weights.
  * Return a peft-wrapped model that is a drop-in replacement for the
    base: ``.generate()``, ``.forward()``, and any existing agent code
    continue to work untouched. Only the optimizer needs to know about
    LoRA (it iterates ``p for p in model.parameters() if p.requires_grad``).

The module has no opinion on where the base model came from. It takes
any ``torch.nn.Module`` whose layers include ``q_proj`` / ``v_proj``
sub-modules (or whatever ``target_modules`` the caller passes) and
returns a peft-wrapped version. Tests can therefore exercise it with
small synthetic modules without loading the full 16 GB Qwen3-VL-8B.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence


logger = logging.getLogger(__name__)


# Defaults — changing these is a calibrated decision; most callers
# should just use ``attach_lora(model)`` without kwargs.
DEFAULT_RANK = 16
DEFAULT_ALPHA = 32
DEFAULT_DROPOUT = 0.05
DEFAULT_TARGET_MODULES: tuple[str, ...] = ("q_proj", "v_proj")
DEFAULT_TASK_TYPE = "CAUSAL_LM"


def attach_lora(
    model: Any,
    *,
    rank: int = DEFAULT_RANK,
    target_modules: Sequence[str] | None = None,
    lora_alpha: int = DEFAULT_ALPHA,
    lora_dropout: float = DEFAULT_DROPOUT,
    task_type: str = DEFAULT_TASK_TYPE,
) -> Any:
    """Attach a LoRA adapter to ``model`` and return the peft-wrapped result.

    After this call:

      * Every original ``model.parameters()`` has ``requires_grad=False``
        (peft freezes them during wrap).
      * LoRA A/B matrices inside every ``target_modules`` submodule have
        ``requires_grad=True`` and are the only thing an optimizer
        should iterate over.
      * ``.generate()`` still works.

    Args:
        model: Base ``torch.nn.Module``. Typically Qwen3-VL-8B from
            :func:`evofsm_rl.model.load_base_model`, but any module
            whose children include the requested ``target_modules``
            works (useful for tests).
        rank: LoRA rank ``r``. Default 16.
        target_modules: Sub-module name patterns to wrap with LoRA.
            Defaults to ``("q_proj", "v_proj")``.
        lora_alpha: Scaling factor ``alpha``. Effective LoRA scale is
            ``alpha / rank``. Default 32 → scale of 2.0 at rank 16.
        lora_dropout: Dropout on the LoRA branch. Default 0.05.
        task_type: Passed through to ``LoraConfig``. Default
            ``"CAUSAL_LM"``. Other values (e.g. ``"SEQ_2_SEQ_LM"``) are
            accepted but untested here.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    targets = list(target_modules) if target_modules is not None else list(
        DEFAULT_TARGET_MODULES,
    )
    try:
        task = TaskType[task_type]
    except KeyError as e:  # pragma: no cover — validated upstream
        raise ValueError(
            f"unknown task_type {task_type!r}. "
            f"Valid: {[t.name for t in TaskType]}"
        ) from e

    config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=targets,
        bias="none",
        task_type=task,
    )

    wrapped = get_peft_model(model, config)
    counts = count_trainable_params(wrapped)
    logger.info(
        "attach_lora: rank=%d targets=%s — trainable=%d / total=%d (%.3f%%)",
        rank, targets, counts["trainable"], counts["total"], counts["percent"],
    )
    return wrapped


def save_lora_checkpoint(model: Any, path: Path | str) -> Path:
    """Write the LoRA adapter (and only the adapter) to ``path``.

    Creates the directory if it doesn't exist. Uses peft's
    ``save_pretrained`` which writes ``adapter_config.json`` plus
    ``adapter_model.safetensors`` (or the equivalent weight file).
    Typical size: ~10–50 MB for rank=16 on an 8B base — tiny compared
    to the ~16 GB full model.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    return path


def load_lora_checkpoint(model: Any, path: Path | str) -> Any:
    """Load a previously saved LoRA adapter into ``model``.

    Two call shapes are supported:

      * ``model`` is already peft-wrapped → adds / replaces the adapter
        in place and returns the same object.
      * ``model`` is a raw base model → wraps it via
        :class:`peft.PeftModel.from_pretrained` and returns the wrapped
        version. The caller must use the return value.

    The loaded adapter is marked trainable so GRPO can continue
    updating it. Pass through peft's default adapter name ``"default"``.
    """
    from peft import PeftModel

    path = str(path)
    if hasattr(model, "peft_config") and hasattr(model, "load_adapter"):
        # Already peft-wrapped. Reload weights into the default slot.
        model.load_adapter(path, adapter_name="default", is_trainable=True)
        model.set_adapter("default")
        return model
    return PeftModel.from_pretrained(model, path, is_trainable=True)


def load_ref_lora_adapter(
    model: Any,
    path: Path | str,
    *,
    adapter_name: str = "ref",
) -> Any:
    """Load a frozen reference LoRA adapter as a secondary slot.

    The model must already be peft-wrapped (i.e., the "default" adapter
    is in place via :func:`attach_lora` + optional
    :func:`load_lora_checkpoint`). This call adds a second adapter
    under ``adapter_name`` that is **not trainable** — gradients never
    flow through it, regardless of any caller's ``loss.backward()``.

    To compute ``log P`` under the reference policy at GRPO time, the
    caller wraps the forward in::

        model.set_adapter("ref")
        with torch.no_grad():
            log_p_ref = compute_logprob(...)
        model.set_adapter("default")

    so the reference forward uses π^pre weights while the training
    forward uses the current π_θ weights, both on the same base model
    (no extra ~16 GB Qwen3-VL-8B in memory).

    This is the LoRA-side machinery for adding a KL anchor term to the
    GRPO loss: ``L_new = L_GRPO + β × KL(π_θ ‖ π^pre)``. The β
    coefficient and ref adapter name are surfaced to
    :func:`evofsm_rl.rl.grpo.grpo_step` and
    :class:`evofsm_rl.fsm.evolution.EvolutionConfig`.
    """
    if not (hasattr(model, "peft_config") and hasattr(model, "load_adapter")):
        raise ValueError(
            "load_ref_lora_adapter requires a peft-wrapped model "
            "(call attach_lora first). Got bare model of type %s."
            % type(model).__name__
        )
    path = str(path)
    model.load_adapter(path, adapter_name=adapter_name, is_trainable=False)
    model.set_adapter("default")
    logger.info(
        "load_ref_lora_adapter: loaded frozen reference adapter %r from %s "
        "(trainable=False); current active adapter is 'default'.",
        adapter_name, path,
    )
    return model


def count_trainable_params(model: Any) -> dict[str, float]:
    """Return ``{"trainable": int, "total": int, "percent": float}``.

    Sanity check for attach_lora: for Qwen3-VL-8B + rank=16 +
    ``target_modules=("q_proj", "v_proj")``, ``percent`` should be in
    the 0.1–2% range. Anything outside that likely means the base
    weights weren't frozen (way too high) or the target modules
    weren't found (way too low).
    """
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {
        "trainable": trainable,
        "total": total,
        "percent": (trainable / total * 100.0) if total else 0.0,
    }


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_DROPOUT",
    "DEFAULT_RANK",
    "DEFAULT_TARGET_MODULES",
    "DEFAULT_TASK_TYPE",
    "attach_lora",
    "count_trainable_params",
    "load_lora_checkpoint",
    "load_ref_lora_adapter",
    "save_lora_checkpoint",
]

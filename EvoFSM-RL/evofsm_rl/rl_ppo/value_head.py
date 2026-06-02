"""Value head for PPO — a linear projection of the last-token hidden state.

The head sits on top of the **frozen base model's** last-layer hidden
state at the last token of the prompt (i.e. the same position that
would be used to predict the next token). It is *not* a LoRA-side
parameter — it has its own optimizer state and never participates in
the policy's forward pass.

Design choices:
  * Initialize the linear weights to zero. Standard practice in actor-
    critic implementations (e.g. cleanrl, TRL): keeps V(s) ≈ 0 at the
    start so it doesn't bias early PPO updates while the head is still
    miscalibrated.
  * No bias initialization tuning — leave at PyTorch default (uniform).
    Could be zeroed too but a tiny bias is harmless and matches HF's
    ``score`` head defaults for sequence-classification.
  * Save/load uses ``state_dict`` so the head can be reloaded on a
    differently-instantiated base model (the head doesn't depend on the
    base's parameter count, only on ``hidden_size``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class LinearValueHead(nn.Module):
    """Linear value head: hidden_state[:, -1, :] → scalar V(s).

    Parameters:
        hidden_size: Dimension of the base model's hidden state
            (e.g. 4096 for Qwen3-VL-8B).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.linear = nn.Linear(self.hidden_size, 1)
        # Zero-init: V(s) starts at 0 so it doesn't dominate early PPO
        # advantages while the head is still being calibrated.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project the last-token hidden state to a scalar value.

        Args:
            hidden_state: Tensor of shape ``(batch, seq_len, hidden_size)``
                from the base model's last-layer hidden state.

        Returns:
            Tensor of shape ``(batch,)`` containing V(s) for each sample.
        """
        # Last-token hidden state. (We don't try to compensate for left-
        # vs right-padding here — the caller is expected to pass either
        # an unpadded single sequence or right-padded batch where the
        # last meaningful token is the last position. For PPO training
        # we always forward one step at a time, so this is moot.)
        last = hidden_state[:, -1, :]
        # Cast to match linear weight dtype (base model emits bfloat16
        # hidden states; we keep the head in fp32 for numerical stability
        # but downcast hidden state inputs to match).
        if last.dtype != self.linear.weight.dtype:
            last = last.to(self.linear.weight.dtype)
        return self.linear(last).squeeze(-1)


def attach_value_head(model: Any, device: str | torch.device) -> LinearValueHead:
    """Construct a value head sized for ``model.config.hidden_size``.

    Args:
        model: A HuggingFace model (or peft-wrapped equivalent) whose
            ``config.hidden_size`` matches the hidden state the head
            will see. For Qwen3-VL-8B this is 4096.
        device: Target device for the head's parameters.

    Returns:
        The instantiated :class:`LinearValueHead` on ``device``.
    """
    # peft-wrapped models keep ``.config`` on the wrapped module via
    # ``__getattr__``. The base HF model exposes it directly too.
    # Multimodal models like Qwen3-VL nest hidden_size in ``text_config``
    # because the top-level config covers vision + text + audio.
    config = getattr(model, "config", None)
    hidden_size = None
    if config is not None:
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            text_cfg = getattr(config, "text_config", None)
            if text_cfg is not None:
                hidden_size = getattr(text_cfg, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(
            "attach_value_head: could not find hidden_size in "
            "model.config or model.config.text_config — "
            "cannot infer head dimension."
        )
    head = LinearValueHead(int(hidden_size))
    head.to(device)
    return head


def save_value_head(head: LinearValueHead, path: Path | str) -> Path:
    """Save the head's state_dict to ``path`` (creating parent dirs).

    The file is a regular ``torch.save`` pickle — load it with
    :func:`load_value_head`.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "hidden_size": head.hidden_size,
        },
        p,
    )
    return p


def load_value_head(head: LinearValueHead, path: Path | str) -> LinearValueHead:
    """Load ``head``'s state_dict from ``path`` in-place.

    The hidden-size in the checkpoint must match the head's
    ``hidden_size`` (we don't silently rebuild — that would mask a
    misconfiguration).
    """
    p = Path(path)
    payload = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        ckpt_hs = int(payload.get("hidden_size", head.hidden_size))
        if ckpt_hs != head.hidden_size:
            raise ValueError(
                f"load_value_head: hidden_size mismatch — checkpoint "
                f"hidden_size={ckpt_hs}, head hidden_size={head.hidden_size}"
            )
        head.load_state_dict(payload["state_dict"])
    else:
        # Backward-compat: raw state_dict file.
        head.load_state_dict(payload)
    return head


__all__ = [
    "LinearValueHead",
    "attach_value_head",
    "load_value_head",
    "save_value_head",
]

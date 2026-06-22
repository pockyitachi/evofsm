"""PPO clipped surrogate loss + value-function MSE loss.

Both functions operate on PyTorch tensors and propagate gradients to
their differentiable inputs (``new_log_probs`` for the policy term,
``predicted_values`` for the value term). They are stateless and have
no module-level state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ppo_clipped_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """PPO-clipped surrogate policy loss.

    Computes::

        ratio = exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        loss  = -mean(min(surr1, surr2))

    Args:
        new_log_probs: Log π_θ(a_t | s_t) under the **current** policy.
            Differentiable — gradients flow back to the LoRA params.
            Shape ``(N,)``.
        old_log_probs: Log π_θ(a_t | s_t) under the **behaviour** policy
            (the one that collected the trajectory). Detached —
            gradients do NOT flow back. Shape ``(N,)``.
        advantages: GAE advantages per step. Detached. Shape ``(N,)``.
        clip_eps: ε. Default 0.2 (PPO paper).

    Returns:
        Scalar tensor: the mean clipped surrogate loss (negated so that
        minimising it maximises expected advantage).
    """
    # Defensive: detach the non-grad-bearing inputs so a caller who
    # forgot can't accidentally hold the value-head graph alive.
    old = old_log_probs.detach()
    adv = advantages.detach()

    ratio = torch.exp(new_log_probs - old)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return -torch.min(surr1, surr2).mean()


def value_loss(
    predicted_values: torch.Tensor,
    target_returns: torch.Tensor,
) -> torch.Tensor:
    """MSE loss between predicted V(s_t) and GAE-estimated returns.

    Args:
        predicted_values: V_θ(s_t) under the current value head.
            Differentiable — gradients flow back to the head's
            parameters. Shape ``(N,)``.
        target_returns: Detached GAE returns (advantages + values from
            the snapshot at buffer-collection time). Shape ``(N,)``.

    Returns:
        Scalar mean-squared-error tensor.
    """
    return F.mse_loss(predicted_values, target_returns.detach())


__all__ = ["ppo_clipped_loss", "value_loss"]

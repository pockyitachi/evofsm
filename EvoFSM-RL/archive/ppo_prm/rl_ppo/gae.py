"""Generalized Advantage Estimation (GAE) for sparse-reward trajectories.

Specialised for the EvoFSM-RL setting where per-step rewards are zero
everywhere except the terminal step, which carries the trajectory-level
binary reward R ∈ {0, 1}.

References:
  * Schulman et al. 2016 — "High-Dimensional Continuous Control Using
    Generalized Advantage Estimation" (arXiv:1506.02438), eqs. (10)-(16).
"""

from __future__ import annotations


def compute_gae(
    per_step_values: list[float],
    trajectory_reward: float,
    *,
    per_step_rewards: list[float] | None = None,
    gamma: float = 0.99,
    lam: float = 1.0,
) -> tuple[list[float], list[float]]:
    """Compute GAE advantages + value-loss targets for one trajectory.

    For a trajectory of length ``T`` with values
    ``V_0, V_1, ..., V_{T-1}`` and terminal reward ``R`` (per-step
    rewards ``r_t = 0`` for ``t < T-1`` and ``r_{T-1} = R``):

      * Define ``V_T = 0`` (terminal state has no future value).
      * TD residual: ``δ_t = r_t + γ V_{t+1} - V_t``
      * GAE: ``A_t = δ_t + (γλ) A_{t+1}``, with ``A_T = 0``.

    Special cases:
      * ``λ = 1``: degenerates to pure Monte Carlo. With per-step
        ``r_t = 0`` for ``t < T-1`` and ``r_{T-1} = R`` and γ = 1:
        ``A_t = R - V_t`` for every step.
      * ``λ = 0``: pure 1-step TD. ``A_t = r_t + γ V_{t+1} - V_t``
        (high bias, low variance — not recommended for very sparse R).

    The value-loss target is ``returns = advantages + values``
    (this is the canonical actor-critic target so the value head
    learns the GAE-estimated return; with λ=1 and γ=1 it reduces to
    the Monte Carlo return ``R``, same as the trajectory reward).

    Args:
        per_step_values: V(s_t) at each step in the trajectory, as
            produced by the value head on the saved hidden states.
            Length must equal trajectory length ``T``.
        trajectory_reward: Terminal reward R (typically 0.0 or 1.0
            in this codebase).
        gamma: Discount factor γ. Default 0.99.
        lam: GAE λ. Default 1.0 (pure Monte Carlo).

    Returns:
        ``(advantages, returns)`` — both lists of length ``T``, in
        forward chronological order. Advantages are the GAE per-step
        signal used by the PPO policy loss; returns are the value-loss
        regression targets.

    Raises:
        ValueError: if ``per_step_values`` is empty.
    """
    T = len(per_step_values)
    if T == 0:
        raise ValueError("compute_gae: per_step_values must be non-empty")

    # Build per-step rewards. If caller supplied them (e.g., from a PRM),
    # use those directly. Otherwise fall back to the sparse default:
    # zero everywhere except the terminal step which carries R.
    if per_step_rewards is not None:
        if len(per_step_rewards) != T:
            raise ValueError(
                f"compute_gae: per_step_rewards length {len(per_step_rewards)} "
                f"does not match per_step_values length {T}"
            )
        rewards = [float(r) for r in per_step_rewards]
    else:
        rewards = [0.0] * (T - 1) + [float(trajectory_reward)] if T >= 1 else [float(trajectory_reward)]

    advantages = [0.0] * T
    gae = 0.0
    # Backward sweep: t = T-1, T-2, ..., 0.
    # V_{T} = 0 (terminal).
    for t in range(T - 1, -1, -1):
        r_t = rewards[t]
        v_next = 0.0 if t == T - 1 else float(per_step_values[t + 1])
        delta = r_t + gamma * v_next - float(per_step_values[t])
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    returns = [a + v for a, v in zip(advantages, per_step_values)]
    return advantages, returns


__all__ = ["compute_gae"]

"""Unit tests for evofsm_rl.rl_ppo.gae.

Covers the sparse-reward GAE math that powers the PPO trainer's
per-step credit assignment.

Run::
    python -m pytest tests/test_gae.py -v
"""

from __future__ import annotations

import pytest

from evofsm_rl.rl_ppo.gae import compute_gae


# ─────────────────────────────────────────────────────────────────────
# Monte-Carlo edge case: λ=1, γ=1 — every step gets the full return.
# ─────────────────────────────────────────────────────────────────────


def test_mc_zero_values_reward_one():
    """V=[0,0,0,0], R=1, γ=1, λ=1 → advantages all equal 1.

    Pure Monte Carlo with no value baseline collapses A_t = R - V_t = 1.
    """
    advs, rets = compute_gae([0.0, 0.0, 0.0, 0.0], 1.0, gamma=1.0, lam=1.0)
    assert advs == pytest.approx([1.0, 1.0, 1.0, 1.0])
    # returns = advantages + values
    assert rets == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_mc_constant_values_subtract_baseline():
    """V=[0.5,0.5,0.5,0.5], R=1, γ=1, λ=1 → advantages all = 0.5.

    The MC return is R=1 at every step, advantage = return - V = 0.5.
    """
    advs, rets = compute_gae([0.5, 0.5, 0.5, 0.5], 1.0, gamma=1.0, lam=1.0)
    assert advs == pytest.approx([0.5, 0.5, 0.5, 0.5])
    # returns = advantages + values = 0.5 + 0.5 = 1.0 everywhere (the
    # MC return, by construction).
    assert rets == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_mc_reward_zero():
    """R=0, V=[0,0,0,0], γ=1, λ=1 → all advantages 0."""
    advs, rets = compute_gae([0.0, 0.0, 0.0, 0.0], 0.0, gamma=1.0, lam=1.0)
    assert advs == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert rets == pytest.approx([0.0, 0.0, 0.0, 0.0])


# ─────────────────────────────────────────────────────────────────────
# Edge cases: trajectory length.
# ─────────────────────────────────────────────────────────────────────


def test_len_one_trajectory():
    """T=1, V=[0.3], R=1.0, γ=1, λ=1.

    Only step is the terminal step: δ = R - V = 0.7, A = 0.7.
    return = A + V = 1.0 (= R, as expected).
    """
    advs, rets = compute_gae([0.3], 1.0, gamma=1.0, lam=1.0)
    assert advs == pytest.approx([0.7])
    assert rets == pytest.approx([1.0])


def test_len_one_lambda_zero():
    """T=1 with λ=0 is still well-defined (lambda doesn't kick in
    because there's nothing to propagate)."""
    advs, rets = compute_gae([0.4], 1.0, gamma=0.99, lam=0.0)
    # δ_0 = 1.0 + 0.99 * 0 - 0.4 = 0.6 ; A_0 = 0.6.
    assert advs == pytest.approx([0.6])
    assert rets == pytest.approx([1.0])


def test_empty_trajectory_raises():
    with pytest.raises(ValueError):
        compute_gae([], 1.0)


# ─────────────────────────────────────────────────────────────────────
# Non-trivial lambda < 1.
# ─────────────────────────────────────────────────────────────────────


def test_lambda_zero_one_step_td():
    """λ=0 reduces to pure 1-step TD.

    With V=[0.2, 0.4, 0.6], R=1, γ=1:
      δ_2 = 1 + 0 - 0.6 = 0.4   → A_2 = 0.4
      δ_1 = 0 + 0.6 - 0.4 = 0.2 → A_1 = 0.2
      δ_0 = 0 + 0.4 - 0.2 = 0.2 → A_0 = 0.2
    """
    advs, _ = compute_gae([0.2, 0.4, 0.6], 1.0, gamma=1.0, lam=0.0)
    assert advs == pytest.approx([0.2, 0.2, 0.4])


def test_lambda_partial_propagates_back():
    """λ=0.5 propagates half the future TD residual to earlier steps.

    V=[0.0, 0.0, 0.0], R=1, γ=1, λ=0.5:
      δ_2 = 1 - 0 = 1     → A_2 = 1
      δ_1 = 0 - 0 = 0     → A_1 = 0 + 0.5*1 = 0.5
      δ_0 = 0 - 0 = 0     → A_0 = 0 + 0.5*0.5 = 0.25
    """
    advs, _ = compute_gae([0.0, 0.0, 0.0], 1.0, gamma=1.0, lam=0.5)
    assert advs == pytest.approx([0.25, 0.5, 1.0])


def test_lambda_one_propagates_full():
    """λ=1 propagates the entire residual, which with constant V
    means every step gets the full return R - V_t."""
    advs, _ = compute_gae([0.1, 0.2, 0.3], 1.0, gamma=1.0, lam=1.0)
    # With γ=1, λ=1: A_t = sum_{k=t}^{T-1} δ_k = R - V_t  (telescopes).
    assert advs == pytest.approx([0.9, 0.8, 0.7])


# ─────────────────────────────────────────────────────────────────────
# Gamma sensitivity.
# ─────────────────────────────────────────────────────────────────────


def test_gamma_discounts_future_reward():
    """V=[0,0,0], R=1, γ=0.9, λ=1.

    A_2 = 1 - 0 = 1
    A_1 = 0 + 0.9*1 - 0 = 0.9
    A_0 = 0 + 0.9*0 - 0 + 0.9*1*A_1 = 0.81

    (The λ=1 closed form with constant V=0: A_t = γ^(T-1-t) * R.)
    """
    advs, _ = compute_gae([0.0, 0.0, 0.0], 1.0, gamma=0.9, lam=1.0)
    assert advs == pytest.approx([0.81, 0.9, 1.0])


def test_returns_equal_advantages_plus_values():
    """``returns = advantages + values`` invariant must hold."""
    values = [0.1, -0.2, 0.5, 0.3]
    advs, rets = compute_gae(values, 1.0, gamma=0.95, lam=0.97)
    for a, v, r in zip(advs, values, rets):
        assert r == pytest.approx(a + v)

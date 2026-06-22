"""Lightweight unit tests for PPO loss + value head pieces.

The full PPOTrainer.step() requires a real model forward pass and a
GPU (covered at integration time via the live training scripts). Here
we just verify the surface in isolation.

Run::
    python -m pytest tests/test_ppo_components.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from evofsm_rl.rl_ppo.ppo_loss import ppo_clipped_loss, value_loss
from evofsm_rl.rl_ppo.value_head import (
    LinearValueHead,
    load_value_head,
    save_value_head,
)


# ─────────────────────────────────────────────────────────────────────
# ppo_clipped_loss
# ─────────────────────────────────────────────────────────────────────


def test_ppo_loss_zero_when_ratios_one_and_zero_advantage():
    """When new_log_probs == old_log_probs and adv == 0, loss == 0."""
    new = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    old = torch.tensor([0.0, 0.0, 0.0])
    adv = torch.tensor([0.0, 0.0, 0.0])
    loss = ppo_clipped_loss(new, old, adv)
    assert float(loss.item()) == pytest.approx(0.0)


def test_ppo_loss_negative_of_advantage_when_ratio_one():
    """ratio=1, adv > 0: loss = -mean(1*adv) = -adv.mean()."""
    new = torch.tensor([0.0, 0.0], requires_grad=True)
    old = torch.tensor([0.0, 0.0])
    adv = torch.tensor([1.0, 1.0])
    loss = ppo_clipped_loss(new, old, adv)
    assert float(loss.item()) == pytest.approx(-1.0)


def test_ppo_loss_clip_kicks_in_at_ratio_above_one_plus_eps():
    """With log(ratio) >> 0 and adv > 0, the surr2 term clips at
    (1+eps)*adv, and min picks the (smaller) clipped value → loss
    capped at -(1+eps)*adv."""
    # ratio = exp(2.0) ≈ 7.389; advantage > 0; clip eps = 0.2 → cap 1.2.
    new = torch.tensor([2.0], requires_grad=True)
    old = torch.tensor([0.0])
    adv = torch.tensor([1.0])
    loss = ppo_clipped_loss(new, old, adv, clip_eps=0.2)
    assert float(loss.item()) == pytest.approx(-1.2)


def test_ppo_loss_clip_for_negative_advantage():
    """ratio >> 1 with adv < 0: surr1 = ratio*adv is more negative,
    surr2 = (1+eps)*adv is less negative; min picks surr1 (no clip
    when ratio increase makes the policy worse on a bad action — PPO
    wants to *amplify* the negative gradient here, no cap)."""
    new = torch.tensor([2.0], requires_grad=True)
    old = torch.tensor([0.0])
    adv = torch.tensor([-1.0])
    loss = ppo_clipped_loss(new, old, adv, clip_eps=0.2)
    # ratio = exp(2) ≈ 7.389; surr1 = -7.389, surr2 = -1.2; min = -7.389;
    # loss = -(-7.389) = 7.389.
    assert float(loss.item()) == pytest.approx(7.389056, rel=1e-4)


def test_ppo_loss_has_gradient_to_new_log_probs():
    new = torch.tensor([0.1, 0.2], requires_grad=True)
    old = torch.tensor([0.0, 0.0])
    adv = torch.tensor([1.0, -1.0])
    loss = ppo_clipped_loss(new, old, adv)
    loss.backward()
    assert new.grad is not None


# ─────────────────────────────────────────────────────────────────────
# value_loss
# ─────────────────────────────────────────────────────────────────────


def test_value_loss_zero_when_perfect_prediction():
    pred = torch.tensor([0.3, 0.5, 0.7], requires_grad=True)
    tgt = torch.tensor([0.3, 0.5, 0.7])
    loss = value_loss(pred, tgt)
    assert float(loss.item()) == pytest.approx(0.0)


def test_value_loss_mse_value():
    pred = torch.tensor([0.0, 0.0], requires_grad=True)
    tgt = torch.tensor([1.0, -1.0])
    loss = value_loss(pred, tgt)
    # mean((0-1)^2, (0-(-1))^2) = 1.0
    assert float(loss.item()) == pytest.approx(1.0)


def test_value_loss_has_gradient_to_predicted():
    pred = torch.tensor([0.1], requires_grad=True)
    tgt = torch.tensor([0.0])
    loss = value_loss(pred, tgt)
    loss.backward()
    assert pred.grad is not None


# ─────────────────────────────────────────────────────────────────────
# LinearValueHead
# ─────────────────────────────────────────────────────────────────────


def test_value_head_zero_init_returns_zero():
    head = LinearValueHead(hidden_size=8)
    hidden = torch.randn(2, 5, 8)
    out = head(hidden)
    assert out.shape == (2,)
    assert torch.allclose(out, torch.zeros(2))


def test_value_head_uses_last_token():
    """After perturbing only the last token's hidden state, the output
    changes — confirms the head reads from position -1."""
    head = LinearValueHead(hidden_size=4)
    # Set linear weights to ones so the projection is the sum of the
    # input hidden (so it's easy to predict).
    with torch.no_grad():
        head.linear.weight.fill_(1.0)
        head.linear.bias.fill_(0.0)
    hidden = torch.zeros(1, 3, 4)
    hidden[0, -1, :] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = head(hidden)
    # sum = 10
    assert float(out.item()) == pytest.approx(10.0)


def test_value_head_round_trip_save_load(tmp_path: Path):
    head = LinearValueHead(hidden_size=8)
    # Make the head's state non-trivial.
    with torch.no_grad():
        head.linear.weight.copy_(torch.randn_like(head.linear.weight))
        head.linear.bias.copy_(torch.randn_like(head.linear.bias))
    target = tmp_path / "vh.pt"
    save_value_head(head, target)

    head2 = LinearValueHead(hidden_size=8)
    assert not torch.allclose(head2.linear.weight, head.linear.weight)
    load_value_head(head2, target)
    assert torch.allclose(head2.linear.weight, head.linear.weight)
    assert torch.allclose(head2.linear.bias, head.linear.bias)


def test_value_head_load_mismatched_hidden_size_raises(tmp_path: Path):
    head = LinearValueHead(hidden_size=8)
    target = tmp_path / "vh.pt"
    save_value_head(head, target)

    head2 = LinearValueHead(hidden_size=16)
    with pytest.raises(ValueError):
        load_value_head(head2, target)

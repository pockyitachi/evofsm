"""Tests for evofsm_rl.rl.grpo — Story 4.2 Part B.

These tests cover the pure-Python / pure-math pieces of the GRPO
module: reward computation, advantage computation, and replay-data
cleanup. The ``grpo_step`` training function requires a real model
forward pass with LoRA gradient flow and is exercised at integration
time in Story 4.4 (``scripts/run_b4_evolution.py`` smoke test). Here
we just sanity-check the surface in isolation.

Run::
    python -m pytest tests/test_grpo.py -v
"""

from __future__ import annotations

from pathlib import Path

import torch

from evofsm_rl.rl.grpo import (
    TrajectoryData,
    cleanup_replay_data,
    compute_advantages,
    compute_reward,
)


# ─────────────────────────────────────────────────────────────────────
# Fixture builder
# ─────────────────────────────────────────────────────────────────────


def _traj(
    *,
    fsm_variant_id: str = "v0",
    reward: float = 0.0,
    success: float = 0.0,
    n_steps: int = 1,
    task_name: str = "T",
    seed: int = 0,
    step_log_probs: list[float] | None = None,
    replay_paths: list[str] | None = None,
) -> TrajectoryData:
    return TrajectoryData(
        task_name=task_name,
        seed=seed,
        fsm_variant_id=fsm_variant_id,
        success=success,
        n_steps=n_steps,
        step_log_probs=list(step_log_probs or []),
        replay_paths=list(replay_paths or []),
        reward=reward,
    )


# ─────────────────────────────────────────────────────────────────────
# compute_reward
# ─────────────────────────────────────────────────────────────────────


def test_compute_reward_success_full_case():
    """success=1.0, 6 steps, budget=60 → 1 + 0.2*(1 - 0.1) = 1.18."""
    r = compute_reward(1.0, 6, 60)
    assert abs(r - 1.18) < 1e-9, f"got {r}"


def test_compute_reward_failure_full_case():
    """success=0.0 on the same budget → just the efficiency bonus."""
    r = compute_reward(0.0, 6, 60)
    assert abs(r - 0.18) < 1e-9, f"got {r}"


def test_compute_reward_zero_budget_no_crash():
    """A zero (or negative) step_budget is clamped rather than dividing
    by zero. The resulting efficiency ≤ 0 so the bonus degenerates."""
    assert compute_reward(1.0, 5, 0) == 1.0
    assert compute_reward(0.0, 5, -3) == 0.0


def test_compute_reward_exhausted_budget_floors_bonus():
    """When the agent hits the budget, the bonus clamps to 0."""
    assert compute_reward(1.0, 60, 60) == 1.0
    assert compute_reward(0.0, 100, 60) == 0.0


def test_compute_reward_custom_weight():
    """``w_efficiency`` is honoured."""
    r = compute_reward(1.0, 30, 60, w_efficiency=0.5)
    assert abs(r - 1.25) < 1e-9  # 1 + 0.5 * 0.5


# ─────────────────────────────────────────────────────────────────────
# compute_advantages
# ─────────────────────────────────────────────────────────────────────


def test_compute_advantages_two_groups():
    """2 variants × 2 trajs each; within-group mean subtraction."""
    trajs = [
        _traj(fsm_variant_id="A", reward=1.0),
        _traj(fsm_variant_id="A", reward=0.0),
        _traj(fsm_variant_id="B", reward=0.8),
        _traj(fsm_variant_id="B", reward=0.2),
    ]
    adv = compute_advantages(trajs)
    # Group A mean = 0.5 → advantages (+0.5, -0.5).
    # Group B mean = 0.5 → advantages (+0.3, -0.3).
    assert adv == [0.5, -0.5, 0.30000000000000004, -0.30000000000000004] or (
        abs(adv[0] - 0.5) < 1e-9
        and abs(adv[1] + 0.5) < 1e-9
        and abs(adv[2] - 0.3) < 1e-9
        and abs(adv[3] + 0.3) < 1e-9
    )


def test_compute_advantages_single_in_group():
    """A singleton group has no peer → advantage 0."""
    trajs = [
        _traj(fsm_variant_id="A", reward=1.0),
        _traj(fsm_variant_id="B", reward=0.0),
    ]
    adv = compute_advantages(trajs)
    assert adv == [0.0, 0.0]


def test_compute_advantages_all_same_reward_in_group():
    """Three trajs, identical rewards → all advantages zero."""
    trajs = [
        _traj(fsm_variant_id="A", reward=0.5),
        _traj(fsm_variant_id="A", reward=0.5),
        _traj(fsm_variant_id="A", reward=0.5),
    ]
    assert compute_advantages(trajs) == [0.0, 0.0, 0.0]


def test_compute_advantages_empty_list():
    """An empty input yields an empty output, no errors."""
    assert compute_advantages([]) == []


def test_compute_advantages_preserves_input_order():
    """Advantages are returned in the same positional order as input."""
    trajs = [
        _traj(fsm_variant_id="X", reward=2.0),  # group X [2.0, 0.0], mean 1.0
        _traj(fsm_variant_id="Y", reward=0.5),  # group Y [0.5, 1.5], mean 1.0
        _traj(fsm_variant_id="X", reward=0.0),
        _traj(fsm_variant_id="Y", reward=1.5),
    ]
    adv = compute_advantages(trajs)
    assert abs(adv[0] - 1.0) < 1e-9   # X first: 2.0 - 1.0
    assert abs(adv[1] + 0.5) < 1e-9   # Y first: 0.5 - 1.0
    assert abs(adv[2] + 1.0) < 1e-9   # X second
    assert abs(adv[3] - 0.5) < 1e-9   # Y second


def test_compute_advantages_same_fsm_different_tasks_are_separate_groups():
    """Same FSM on two different tasks → two singleton groups → both advantages 0.

    Under the old fsm_variant_id-only grouping these two trajectories would
    have shared a group and produced advantages ±0.5. The (fsm, task) tuple
    grouping per §3.2 keeps them apart, matching the original GRPO design.
    """
    trajs = [
        _traj(fsm_variant_id="A", task_name="task_X", reward=1.0),
        _traj(fsm_variant_id="A", task_name="task_Y", reward=0.0),
    ]
    assert compute_advantages(trajs) == [0.0, 0.0]


def test_compute_advantages_fsm_task_tuple_grouping():
    """4 trajs spanning 2 FSMs × 2 tasks → 4 singleton groups → all 0.

    Locks in the (fsm_variant_id, task_name) group key against future
    regressions back to FSM-only grouping.
    """
    trajs = [
        _traj(fsm_variant_id="A", task_name="task_X", reward=1.0),
        _traj(fsm_variant_id="A", task_name="task_Y", reward=0.0),
        _traj(fsm_variant_id="B", task_name="task_X", reward=0.8),
        _traj(fsm_variant_id="B", task_name="task_Y", reward=0.2),
    ]
    assert compute_advantages(trajs) == [0.0, 0.0, 0.0, 0.0]


def test_compute_advantages_within_fsm_task_group_subtracts_mean():
    """Same FSM and task with 2 rollouts → within-group advantage = r - V_i.

    This is the standard GRPO design from §3.2: V_i = mean over N rollouts
    of FSM i on task t; advantage_{i,j} = r_{i,j} - V_i.
    """
    trajs = [
        _traj(fsm_variant_id="A", task_name="task_X", reward=1.0),
        _traj(fsm_variant_id="A", task_name="task_X", reward=0.0),
        _traj(fsm_variant_id="A", task_name="task_Y", reward=0.6),
        _traj(fsm_variant_id="A", task_name="task_Y", reward=0.4),
    ]
    adv = compute_advantages(trajs)
    # (A, task_X) mean = 0.5  → advantages (+0.5, -0.5)
    # (A, task_Y) mean = 0.5  → advantages (+0.1, -0.1)
    assert abs(adv[0] - 0.5) < 1e-9
    assert abs(adv[1] + 0.5) < 1e-9
    assert abs(adv[2] - 0.1) < 1e-9
    assert abs(adv[3] + 0.1) < 1e-9


# ─────────────────────────────────────────────────────────────────────
# cleanup_replay_data
# ─────────────────────────────────────────────────────────────────────


def test_cleanup_replay_data_deletes_and_counts(tmp_path: Path):
    """Create real files, verify they disappear and the count matches."""
    # Two trajectories, 3 and 2 replay files respectively.
    paths_a = []
    for i in range(3):
        p = tmp_path / f"a_step_{i}.pt"
        torch.save({"idx": i}, p)
        paths_a.append(str(p))
    paths_b = []
    for i in range(2):
        p = tmp_path / f"b_step_{i}.pt"
        torch.save({"idx": i}, p)
        paths_b.append(str(p))

    trajs = [
        _traj(fsm_variant_id="A", replay_paths=paths_a),
        _traj(fsm_variant_id="B", replay_paths=paths_b),
    ]
    removed = cleanup_replay_data(trajs)
    assert removed == 5
    # All files gone.
    for p in paths_a + paths_b:
        assert not Path(p).exists()


def test_cleanup_replay_data_idempotent_missing_files(tmp_path: Path):
    """Cleanup that refers to already-missing files must not raise
    and must return 0 removed."""
    # Reference a path that doesn't exist.
    ghost = str(tmp_path / "never_written.pt")
    trajs = [_traj(fsm_variant_id="A", replay_paths=[ghost])]
    assert cleanup_replay_data(trajs) == 0


def test_cleanup_replay_data_empty_list_is_safe():
    assert cleanup_replay_data([]) == 0


def test_cleanup_replay_data_removes_empty_parent_dirs(tmp_path: Path):
    """Regression — ``cleanup_replay_data`` should also rmdir any
    ``episode_NNNN/`` parent that becomes empty after its ``step_*.pt``
    files are unlinked. Long runs otherwise leak hundreds of empty
    directories under ``replay/``.
    """
    ep_dir = tmp_path / "episode_0001"
    ep_dir.mkdir()
    paths = []
    for i in range(3):
        p = ep_dir / f"step_{i}.pt"
        torch.save({"idx": i}, p)
        paths.append(str(p))

    removed = cleanup_replay_data([_traj(replay_paths=paths)])
    assert removed == 3
    # Parent dir is gone.
    assert not ep_dir.exists()


def test_cleanup_replay_data_keeps_non_empty_parent_dirs(tmp_path: Path):
    """Defensive — if the parent still has content the caller didn't
    reference, we must NOT blow it away."""
    ep_dir = tmp_path / "episode_0002"
    ep_dir.mkdir()
    referenced = ep_dir / "step_0.pt"
    torch.save({"idx": 0}, referenced)
    stray = ep_dir / "unknown.bin"
    stray.write_bytes(b"do not delete me")

    cleanup_replay_data([_traj(replay_paths=[str(referenced)])])
    assert ep_dir.exists()
    assert stray.exists()


# ─────────────────────────────────────────────────────────────────────
# _compute_step_log_prob_with_grad — regression for mm_token_type_ids
# ─────────────────────────────────────────────────────────────────────


def test_compute_step_log_prob_passes_all_replay_fields():
    """Regression — the gradient-bearing forward inside GRPO must
    forward every tensor field the agent saved, including
    ``mm_token_type_ids`` and any exotic forward-compatible field.
    Earlier code hand-picked pixel_values/attention_mask/image_grid_thw
    and crashed under recent transformers Qwen3-VL.
    """
    from evofsm_rl.rl.grpo import _compute_step_log_prob_with_grad

    input_len = 4
    action_len = 2
    vocab = 8

    step_data = {
        "input_ids": torch.arange(input_len).unsqueeze(0),
        "attention_mask": torch.ones(1, input_len, dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 4, 4),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "mm_token_type_ids": torch.zeros(1, input_len, dtype=torch.long),
        "future_extra_field": torch.tensor([[0.1, 0.2]]),
        "action_token_ids": torch.tensor([5, 6]),
        "input_len": input_len,
    }

    observed: dict = {}

    class _FakeOut:
        def __init__(self, logits):
            self.logits = logits

    class _FakeModel:
        def __call__(self, **kwargs):
            observed.update(kwargs)
            seq_len = int(kwargs["input_ids"].shape[-1])
            return _FakeOut(torch.zeros(1, seq_len, vocab, requires_grad=True))

    model = _FakeModel()
    lp = _compute_step_log_prob_with_grad(model, step_data, device="cpu")
    assert lp.shape == ()

    # Forward call saw every field — explicitly including the two that
    # would have been silently dropped by the old hardcoded subset.
    for key in (
        "input_ids", "attention_mask", "pixel_values",
        "image_grid_thw", "mm_token_type_ids", "future_extra_field",
    ):
        assert key in observed, f"{key} was dropped by _compute_step_log_prob_with_grad"
    # Agent-side bookkeeping is NOT passed to the model.
    for skip in ("action_token_ids", "input_len", "step_log_prob"):
        assert skip not in observed


def test_compute_step_log_prob_extends_per_token_aux_fields():
    """Regression — every per-token auxiliary tensor (shape ends in
    input_len) must be extended to cover the generated action tokens.
    ``mm_token_type_ids`` pads with 0 (text), ``attention_mask`` with 1.
    Caught by Qwen3-VL's ``get_rope_index`` which indexes one tensor
    with another and raises IndexError on size mismatch.
    """
    from evofsm_rl.rl.grpo import _compute_step_log_prob_with_grad

    input_len = 5
    action_len = 3  # full seq = 8
    vocab = 8

    step_data = {
        "input_ids": torch.arange(input_len).unsqueeze(0),
        "attention_mask": torch.ones(1, input_len, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(1, input_len, dtype=torch.long),
        "action_token_ids": torch.tensor([6, 7, 1]),
        "input_len": input_len,
    }

    observed: dict = {}

    class _FakeOut:
        def __init__(self, logits):
            self.logits = logits

    class _FakeModel:
        def __call__(self, **kwargs):
            observed.update(kwargs)
            return _FakeOut(torch.zeros(1, kwargs["input_ids"].shape[-1], vocab, requires_grad=True))

    _compute_step_log_prob_with_grad(_FakeModel(), step_data, device="cpu")

    # input_ids was extended to full (prompt + action) length.
    assert observed["input_ids"].shape[-1] == input_len + action_len
    # Per-token aux fields were extended to the same length.
    assert observed["attention_mask"].shape[-1] == input_len + action_len
    assert observed["mm_token_type_ids"].shape[-1] == input_len + action_len
    # attention_mask appended 1s; mm_token_type_ids appended 0s.
    tail_mask = observed["attention_mask"][0, input_len:]
    tail_types = observed["mm_token_type_ids"][0, input_len:]
    assert torch.all(tail_mask == 1)
    assert torch.all(tail_types == 0)


def test_compute_step_log_prob_ignores_step_log_prob_scalar():
    """A ``step_log_prob`` float sometimes leaks into replay data (older
    files). The recomputation path must not try to ``.to(device)`` it.
    """
    from evofsm_rl.rl.grpo import _compute_step_log_prob_with_grad

    input_len = 3
    action_len = 2
    vocab = 8

    class _FakeOut:
        def __init__(self, logits):
            self.logits = logits

    class _FakeModel:
        def __call__(self, **kwargs):
            seq_len = int(kwargs["input_ids"].shape[-1])
            return _FakeOut(torch.zeros(1, seq_len, vocab, requires_grad=True))

    step_data = {
        "input_ids": torch.arange(input_len).unsqueeze(0),
        "attention_mask": torch.ones(1, input_len, dtype=torch.long),
        "action_token_ids": torch.tensor([7, 7]),
        "input_len": input_len,
        "step_log_prob": -0.42,   # scalar float — must be ignored
    }
    lp = _compute_step_log_prob_with_grad(_FakeModel(), step_data, device="cpu")
    assert lp.shape == ()


# ─────────────────────────────────────────────────────────────────────
# grpo_step early-reject behavior (min_n_active + empty-active path)
# ─────────────────────────────────────────────────────────────────────


def test_grpo_step_rejects_when_no_active_trajectories():
    """All-singleton groups → no active trajectories → optimizer untouched."""
    from evofsm_rl.rl.grpo import grpo_step

    # All trajectories in their own group (distinct task_name) ⇒ singleton
    # groups ⇒ advantage 0 for everyone ⇒ no active trajectories.
    trajs = [
        _traj(reward=1.0, task_name="task_A", replay_paths=["a"]),
        _traj(reward=0.0, task_name="task_B", replay_paths=["b"]),
    ]
    metrics = grpo_step(
        model=None, optimizer=None, trajectories=trajs, device="cpu",
    )
    assert metrics["n_active"] == 0
    assert metrics["loss"] == 0.0
    assert metrics["grad_norm"] == 0.0
    assert metrics["mean_kl"] == 0.0  # KL anchor not configured


def test_grpo_step_rejects_when_n_active_below_min_threshold():
    """n_active < min_n_active path: optimizer not stepped."""
    from evofsm_rl.rl.grpo import grpo_step

    # Two trajectories in the same group, different rewards → advantage
    # is non-zero for both → n_active = 2. With min_n_active=3, we
    # should refuse to step.
    trajs = [
        _traj(reward=1.0, task_name="T", replay_paths=["a"]),
        _traj(reward=0.0, task_name="T", replay_paths=["b"]),
    ]
    metrics = grpo_step(
        model=None,
        optimizer=None,
        trajectories=trajs,
        device="cpu",
        min_n_active=3,
    )
    assert metrics["n_active"] == 2
    assert metrics["loss"] == 0.0  # rejected, no backward
    assert metrics["grad_norm"] == 0.0


def test_kl_log_ratio_clip_caps_kl_contribution():
    """Math sanity: log_ratio clipping bounds the k3 KL value.

    With log_ratio clamped to ±10:
      max KL per step = exp(10) - 1 - 10 ≈ 22 015
    Without clipping, log_ratio=25 (observed in 2026-05-15 β=0.02 fire 2)
    yields exp(25) ≈ 7.2e10 — a >3e6× larger KL.
    """
    import math
    import torch

    # Helper that mirrors the in-grpo_step KL computation.
    def k3_kl(log_ratio_val: float, clip: float | None) -> float:
        lr = torch.tensor(log_ratio_val)
        if clip is not None and clip > 0:
            lr = lr.clamp(min=-clip, max=clip)
        ratio = torch.exp(lr)
        return float(ratio - 1.0 - lr)

    # Normal range: clip is a no-op.
    assert abs(k3_kl(0.5, clip=10.0) - k3_kl(0.5, clip=None)) < 1e-6
    assert abs(k3_kl(-1.0, clip=10.0) - k3_kl(-1.0, clip=None)) < 1e-6
    assert abs(k3_kl(5.0, clip=10.0) - k3_kl(5.0, clip=None)) < 1e-6

    # At the boundary: identity.
    assert abs(k3_kl(10.0, clip=10.0) - k3_kl(10.0, clip=None)) < 1e-3

    # Beyond clip: capped.
    kl_unclipped = k3_kl(25.0, clip=None)        # ~7.2e10
    kl_clipped = k3_kl(25.0, clip=10.0)          # ~22 015
    assert kl_unclipped > 1e10
    assert kl_clipped < 1e5
    assert kl_clipped < kl_unclipped / 1e5  # >5 orders of magnitude smaller

    # Exact value at clip boundary.
    expected = math.exp(10) - 1 - 10
    assert abs(kl_clipped - expected) < 1.0  # ~22 015


def test_grpo_step_kl_beta_without_ref_adapter_raises():
    """kl_beta > 0 but model has no 'ref' adapter → clear error."""
    from evofsm_rl.rl.grpo import grpo_step

    # Build trajs that WOULD trigger an update (n_active >= 1) so the
    # function reaches the kl check rather than early-returning.
    trajs = [
        _traj(reward=1.0, task_name="T", replay_paths=["a"]),
        _traj(reward=0.0, task_name="T", replay_paths=["b"]),
    ]

    class _NoAdapterModel:
        # No peft_config attribute → grpo_step should raise RuntimeError
        # before it tries to use the adapter.
        training = False
        def train(self, mode=True): pass
        def parameters(self):
            return iter(())

    try:
        grpo_step(
            model=_NoAdapterModel(),
            optimizer=None,
            trajectories=trajs,
            device="cpu",
            kl_beta=0.02,
            ref_adapter_name="ref",
        )
    except RuntimeError as e:
        assert "ref" in str(e).lower() or "adapter" in str(e).lower()
        return
    raise AssertionError("expected RuntimeError for missing ref adapter")


# ─────────────────────────────────────────────────────────────────────
# Standalone runner
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

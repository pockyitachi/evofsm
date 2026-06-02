"""Tests for evofsm_rl.fsm.evolution — Story 3.5.

12 tests covering the B3 evolution loop end-to-end *without* emulator,
model, or real API calls:

  Loop (1-5):
    1.  Basic run: 5 iterations, mock rollouts mixing success/failure,
        population grows, no crash.
    2.  Resume: run 3 iters, checkpoint, resume for 2 more; final state
        matches running 5 in one shot (same task sequence).
    3.  Mutation is triggered on success: reward always = 1.0 → every
        iteration calls ``mutate_fsm`` and population grows by +1.
    4.  Mutation skipped when best_reward=0 on non-periodic iterations
        (iter 1, 2), triggered on iter 3 by the periodic cadence.
    5.  MutationError is caught when skip_mutation_on_error=True; the
        loop keeps going and the population does NOT grow that iter.

  Invariants (6-7):
    6.  Directional TrueSkill: if variant A always outscores B, A's mu
        ends up > B's mu after several iterations.
    7.  iterations.jsonl has exactly one JSON line per iteration.

  Artifacts (8):
    8.  Champion FSM can be round-tripped through ``FSM.from_json`` at
        the end of the run.

  TaskSampler (9-10):
    9.  round_robin: 7 samples over 3 tasks produce the expected cycle.
    10. random: same RNG seed → same sequence.

  Config (11-12):
    11. EvolutionConfig defaults are sensible and overridable.
    12. mock rollout fixture produces valid episode dirs that
        compress_trajectory_for_reflection can read back.

Run::
    python -m pytest tests/test_evolution.py -v
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evofsm_rl.fsm import evolution as evo
from evofsm_rl.fsm.diff import DiffOp, FSMDiff
from evofsm_rl.fsm.evolution import (
    EvolutionConfig,
    RolloutResult,
    TaskSampler,
    l_c_to_seed_fsm,
    run_evolution,
    run_l_c_evolution,
)
from evofsm_rl.fsm.mutation import MutationError
from evofsm_rl.fsm.schema import (
    AbstractCategory,
    FSM,
    Layer1,
    Layer2,
    State,
)


# ─────────────────────────────────────────────────────────────────────
# Fixture helpers
# ─────────────────────────────────────────────────────────────────────


def _sample_fsm(app: str = "testapp") -> FSM:
    return FSM(
        app=app,
        layer1=Layer1(
            app=app,
            category="Productivity",
            states=[State(id="home"), State(id="main")],
        ),
        layer2=Layer2(categories=[
            AbstractCategory(
                name="ADD_ENTRY",
                precondition="list visible",
                abstract_steps=["locate primary affordance"],
            ),
        ]),
    )


def _write_mock_episode(ep_dir: Path, *, task_name: str, seed: int,
                         reward: float, n_steps: int = 3) -> Path:
    """Create a valid episode directory (meta.json + episode.jsonl)."""
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / "meta.json").write_text(json.dumps({
        "app": "testapp",
        "template": task_name,
        "seed": seed,
        "reward": reward,
        "success": reward,
        "goal": f"Test goal for {task_name}",
        "n_steps": n_steps,
    }))
    with (ep_dir / "episode.jsonl").open("w") as fh:
        for s in range(1, n_steps + 1):
            fh.write(json.dumps({
                "step": s,
                "action": {"action_type": "click", "index": s},
                "action_reason": f"step {s} reason",
                "summary": f"step {s} summary",
                "before_ui_elements_text": [f"elem_{i}" for i in range(3)],
                "status": "ok",
            }) + "\n")
    return ep_dir


def _make_rollout_fn(
    tmp_path: Path,
    reward_seq,  # callable (iter_idx) -> reward, or list of rewards
):
    """Build a mock rollout_fn that writes a real episode dir per call
    and returns a reward driven by ``reward_seq``."""
    calls = {"n": 0}

    def rollout_fn(fsm_prompt_text, task_name, seed):
        i = calls["n"]
        calls["n"] += 1
        if callable(reward_seq):
            r = float(reward_seq(i))
        else:
            r = float(reward_seq[i % len(reward_seq)])
        ep_dir = tmp_path / "episodes" / f"{task_name}_seed{seed}"
        _write_mock_episode(ep_dir, task_name=task_name, seed=seed, reward=r)
        return RolloutResult(
            task_name=task_name,
            seed=seed,
            reward=r,
            n_steps=3,
            wall_seconds=0.01,
            episode_dir=ep_dir,
        )

    rollout_fn.calls = calls  # type: ignore[attr-defined]
    return rollout_fn


def _fake_mutate_fsm(fsm, trajectories, *, task_category="",
                       model="ignored", **kwargs):
    """Deterministic stand-in for :func:`mutate_fsm`.

    Returns a child FSM that is the parent with an extra dead_end
    appended, plus a minimal FSMDiff and a constant reflection text.
    """
    child = copy.deepcopy(fsm)
    child.layer1.dead_ends.append({
        "state": "home",
        "failed_action": "fake_mutation",
        "note": f"fake child (parent_app={fsm.app}, n_trajs={len(trajectories)})",
    })
    diff = FSMDiff(
        ops=[],  # intentionally empty — we're not driving the applier here
        reflection_summary="fake mutation for tests",
        layer_tag="layer1",
    )
    return child, diff, "fake reflection text"


# ═════════════════════════════════════════════════════════════════════
# Tests
# ═════════════════════════════════════════════════════════════════════


def test_basic_loop_completes(tmp_path: Path, monkeypatch):
    """Test 1: 5 iterations of mixed-reward rollouts terminate cleanly."""
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm)
    rollout_fn = _make_rollout_fn(tmp_path, lambda i: 1.0 if i % 2 == 0 else 0.0)

    config = EvolutionConfig(n_iterations=5, m_select=1, n_rollouts=1,
                              checkpoint_every=1)
    pop = run_evolution(
        root_fsm=_sample_fsm(),
        task_templates=["TaskA", "TaskB"],
        rollout_fn=rollout_fn,
        config=config,
        output_dir=tmp_path,
        app_category="Productivity",
    )
    # At least the root survives. Population grew because at least some
    # iterations succeeded + every third forced mutation.
    assert pop.size >= 1
    assert (tmp_path / "population.json").exists()
    assert (tmp_path / "iterations.jsonl").exists()


def test_resume_matches_single_run(tmp_path: Path, monkeypatch):
    """Test 2: split 5 iters as 3 + 2 with resume; end state matches 5-in-one.

    We compare the sequence of ``task_name`` values in iterations.jsonl
    and the final population size. Because the task sampler is seeded
    from ``config.seed_base`` and re-advanced past completed iterations
    on resume, both runs should draw the same tasks in the same order.
    """
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm)

    def build_run(output: Path):
        rollout_fn = _make_rollout_fn(output, [1.0] * 1000)  # always succeed
        config = EvolutionConfig(n_iterations=5, m_select=1, n_rollouts=1,
                                  checkpoint_every=1)
        return rollout_fn, config

    # Run A: full 5 iters in one shot.
    dir_a = tmp_path / "run_a"
    dir_a.mkdir()
    fn_a, cfg = build_run(dir_a)
    run_evolution(
        root_fsm=_sample_fsm(),
        task_templates=["TaskA", "TaskB", "TaskC"],
        rollout_fn=fn_a,
        config=cfg,
        output_dir=dir_a,
        app_category="Productivity",
    )

    # Run B: 3 iters, then resume for 2 more.
    dir_b = tmp_path / "run_b"
    dir_b.mkdir()
    fn_b1 = _make_rollout_fn(dir_b, [1.0] * 1000)
    run_evolution(
        root_fsm=_sample_fsm(),
        task_templates=["TaskA", "TaskB", "TaskC"],
        rollout_fn=fn_b1,
        config=EvolutionConfig(n_iterations=3, m_select=1, n_rollouts=1),
        output_dir=dir_b,
        app_category="Productivity",
    )
    fn_b2 = _make_rollout_fn(dir_b, [1.0] * 1000)
    run_evolution(
        root_fsm=_sample_fsm(),
        task_templates=["TaskA", "TaskB", "TaskC"],
        rollout_fn=fn_b2,
        config=EvolutionConfig(n_iterations=5, m_select=1, n_rollouts=1),
        output_dir=dir_b,
        app_category="Productivity",
        resume=True,
    )

    def _task_seq(d: Path) -> list[str]:
        return [
            json.loads(line)["task_name"]
            for line in (d / "iterations.jsonl").open()
        ]

    assert _task_seq(dir_a) == _task_seq(dir_b)
    pop_a = json.loads((dir_a / "population.json").read_text())
    pop_b = json.loads((dir_b / "population.json").read_text())
    assert len(pop_a["variants"]) == len(pop_b["variants"])


def test_mutation_triggered_on_success(tmp_path: Path, monkeypatch):
    """Test 3: reward always 1.0 → every iteration mutates → pop grows by N."""
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm)
    rollout_fn = _make_rollout_fn(tmp_path, [1.0] * 100)
    config = EvolutionConfig(n_iterations=4, m_select=1, n_rollouts=1)
    pop = run_evolution(
        root_fsm=_sample_fsm(),
        task_templates=["TaskA"],
        rollout_fn=rollout_fn,
        config=config,
        output_dir=tmp_path,
        app_category="Productivity",
    )
    # Root + 1 child per iteration.
    assert pop.size == 1 + 4
    lines = list((tmp_path / "iterations.jsonl").open())
    assert len(lines) == 4
    for line in lines:
        row = json.loads(line)
        assert row["mutation_success"] is True
        assert row["child_id"] is not None


def test_mutation_skipped_on_failure_except_periodic(
    tmp_path: Path, monkeypatch,
):
    """Test 4: all-failure rollouts skip mutation on iters 1, 2 but fire
    on iter 3 by periodic cadence (mutation_every_n_iters=3)."""
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm)
    rollout_fn = _make_rollout_fn(tmp_path, [0.0] * 100)
    config = EvolutionConfig(
        n_iterations=3, m_select=1, n_rollouts=1,
        mutation_every_n_iters=3,
    )
    pop = run_evolution(
        root_fsm=_sample_fsm(),
        task_templates=["TaskA"],
        rollout_fn=rollout_fn,
        config=config,
        output_dir=tmp_path,
        app_category="Productivity",
    )
    # Only iter 3 mutates.
    assert pop.size == 2
    lines = [json.loads(l) for l in (tmp_path / "iterations.jsonl").open()]
    assert lines[0]["mutation_success"] is False
    assert lines[0]["child_id"] is None
    assert lines[1]["mutation_success"] is False
    assert lines[1]["child_id"] is None
    assert lines[2]["mutation_success"] is True
    assert lines[2]["child_id"] is not None


def test_mutation_error_is_caught_when_skip_flag_set(
    tmp_path: Path, monkeypatch,
):
    """Test 5: mutate_fsm raising MutationError is logged, loop continues."""
    def broken_mutate(*_args, **_kwargs):
        raise MutationError("simulated parse failure")

    monkeypatch.setattr(evo, "mutate_fsm", broken_mutate)
    rollout_fn = _make_rollout_fn(tmp_path, [1.0] * 100)  # always try to mutate
    config = EvolutionConfig(
        n_iterations=3, m_select=1, n_rollouts=1,
        skip_mutation_on_error=True,
    )
    pop = run_evolution(
        root_fsm=_sample_fsm(),
        task_templates=["TaskA"],
        rollout_fn=rollout_fn,
        config=config,
        output_dir=tmp_path,
        app_category="Productivity",
    )
    # Loop finished; no children grew because every mutation attempt failed.
    assert pop.size == 1
    lines = [json.loads(l) for l in (tmp_path / "iterations.jsonl").open()]
    assert all(l["mutation_success"] is False for l in lines)
    assert all(l["error"] and "MutationError" in l["error"] for l in lines)


def test_trueskill_updates_are_directional(tmp_path: Path, monkeypatch):
    """Test 6: after 8 rounds where variant A beats B every time, A's mu > B's mu.

    We force M=2 so both variants compete each round. A's rollouts return
    1.0, B's return 0.0. Since the population is seeded with only root,
    we prime it with one extra child up front (the "B" variant) via our
    fake mutate, then drive rollouts to reward A (= whichever happens to
    be the champion at the time) and penalize B.
    """
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm)

    # Build population externally with 2 variants and a distinguishing
    # state id on the "B" variant so the rollout can tell them apart
    # from the prompt text alone.
    from evofsm_rl.fsm.population import Population
    pop = Population(_sample_fsm(), window_size=5, delta_sigma=0.5)
    b_fsm = _sample_fsm()
    b_fsm.layer1.states.append(State(id="b_variant_marker"))
    child = pop.add_child("gen0_root", b_fsm, iteration=0,
                           metadata={"label": "B"})
    # Checkpoint it so run_evolution picks it up with resume=True.
    (tmp_path / "population.json").write_text(
        json.dumps(pop.to_json(), indent=2),
    )

    def directional_rollout(fsm_prompt_text, task_name, seed):
        """Reward the root FSM (= no marker), penalize the B variant
        (= marker present). ``FSM.to_prompt_text`` uppercases state ids
        in its STATES block, so match case-insensitively."""
        is_b = "b_variant_marker" in fsm_prompt_text.lower()
        r = 0.0 if is_b else 1.0
        ep_dir = tmp_path / "episodes" / f"{task_name}_seed{seed}_{int(r)}_{is_b}"
        _write_mock_episode(ep_dir, task_name=task_name, seed=seed, reward=r)
        return RolloutResult(
            task_name=task_name, seed=seed, reward=r,
            n_steps=3, wall_seconds=0.01, episode_dir=ep_dir,
        )

    # Run 8 iterations; both variants selected each round (m=2).
    config = EvolutionConfig(
        n_iterations=8, m_select=2, n_rollouts=1,
        mutation_every_n_iters=99,  # disable periodic mutation
        checkpoint_every=1,
    )
    final_pop = run_evolution(
        root_fsm=_sample_fsm(),
        task_templates=["TaskA"],
        rollout_fn=directional_rollout,
        config=config,
        output_dir=tmp_path,
        app_category="Productivity",
        resume=True,   # pick up our pre-built population
    )
    root = final_pop.get("gen0_root")
    loser = final_pop.get(child.id)
    assert root.rating.mu > loser.rating.mu, (
        f"root mu ({root.rating.mu:.2f}) should exceed loser mu "
        f"({loser.rating.mu:.2f}) after 8 directional rounds"
    )


def test_iterations_jsonl_has_one_line_per_iteration(
    tmp_path: Path, monkeypatch,
):
    """Test 7: 5 iterations → exactly 5 JSON-parsable lines."""
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm)
    rollout_fn = _make_rollout_fn(tmp_path, [1.0] * 100)
    run_evolution(
        root_fsm=_sample_fsm(),
        task_templates=["TaskA"],
        rollout_fn=rollout_fn,
        config=EvolutionConfig(n_iterations=5, m_select=1, n_rollouts=1),
        output_dir=tmp_path,
        app_category="Productivity",
    )
    lines = list((tmp_path / "iterations.jsonl").open())
    assert len(lines) == 5
    for ln in lines:
        row = json.loads(ln)
        for key in ("iteration", "task_name", "seed", "selected_ids",
                     "rewards", "best_id", "best_reward",
                     "mutation_success", "wall_seconds"):
            assert key in row


def test_champion_fsm_round_trips_through_schema(
    tmp_path: Path, monkeypatch,
):
    """Test 8: the champion's FSM serializes and re-loads cleanly."""
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm)
    rollout_fn = _make_rollout_fn(tmp_path, [1.0] * 100)
    pop = run_evolution(
        root_fsm=_sample_fsm(),
        task_templates=["TaskA"],
        rollout_fn=rollout_fn,
        config=EvolutionConfig(n_iterations=3, m_select=1, n_rollouts=1),
        output_dir=tmp_path,
        app_category="Productivity",
    )
    champion_json = pop.champion.fsm.to_json()
    # Serialize to a string and back, then into an FSM via from_json.
    rehydrated = FSM.from_json(json.loads(json.dumps(champion_json)))
    assert rehydrated.to_json() == champion_json


def test_task_sampler_round_robin():
    """Test 9: round_robin cycles through the task list."""
    s = TaskSampler(["a", "b", "c"], mode="round_robin", rng_seed=0)
    seq = [s.sample() for _ in range(7)]
    assert seq == ["a", "b", "c", "a", "b", "c", "a"]


def test_task_sampler_random_is_deterministic_by_seed():
    """Test 10: same rng_seed → same sample sequence."""
    s1 = TaskSampler(["a", "b", "c", "d"], mode="random", rng_seed=7)
    s2 = TaskSampler(["a", "b", "c", "d"], mode="random", rng_seed=7)
    seq1 = [s1.sample() for _ in range(10)]
    seq2 = [s2.sample() for _ in range(10)]
    assert seq1 == seq2


def test_evolution_config_defaults_and_overrides():
    """Test 11: all fields default sensibly; individual fields can be overridden."""
    default = EvolutionConfig()
    assert default.n_iterations > 0
    assert default.m_select >= 1
    assert default.n_rollouts >= 1
    assert default.window_size >= 1
    assert default.task_sample_mode in ("random", "round_robin")
    assert default.mutation_model  # non-empty
    assert default.mutation_every_n_iters >= 1
    # mutation_enabled defaults to True so existing B3 / B4-Tier-B
    # callers see no behavior change after the 2026-05-13 addition.
    assert default.mutation_enabled is True

    # Override one field; others stay default.
    custom = EvolutionConfig(n_iterations=50, mutation_every_n_iters=5)
    assert custom.n_iterations == 50
    assert custom.mutation_every_n_iters == 5
    assert custom.m_select == default.m_select
    assert custom.mutation_enabled is True

    # Explicit opt-out for the Tier-C / no-L_C path.
    disabled = EvolutionConfig(mutation_enabled=False)
    assert disabled.mutation_enabled is False


def test_mock_rollout_produces_readable_episode_dirs(tmp_path: Path):
    """Test 12: the fixture's rollout_fn writes dirs that
    ``compress_trajectory_for_reflection`` can read back."""
    from evofsm_rl.fsm.mutation import compress_trajectory_for_reflection
    fn = _make_rollout_fn(tmp_path, [1.0, 0.0])
    r1 = fn("SomeTask", "SomeTask", 42)
    r2 = fn("SomeTask", "SomeTask", 43)
    traj1 = compress_trajectory_for_reflection(r1.episode_dir)
    traj2 = compress_trajectory_for_reflection(r2.episode_dir)
    assert traj1.reward == 1.0
    assert traj2.reward == 0.0
    assert traj1.n_steps > 0
    # Sanity: step fields populated.
    assert traj1.steps[0].action.get("action_type") == "click"
    assert traj1.steps[0].ui_elements  # non-empty


# ═════════════════════════════════════════════════════════════════════
# L_C evolution (Story 3.5 — B3 canonical path)
# ═════════════════════════════════════════════════════════════════════


def _sample_l_c() -> Layer2:
    """Minimal Layer-2 with two categories so mutations have room to bite."""
    return Layer2(categories=[
        AbstractCategory(
            name="ADD_ENTRY",
            precondition="list visible",
            abstract_steps=["locate primary affordance", "confirm"],
            failure_modes=["confirming too early"],
            verification_checklist=["new entry visible"],
        ),
        AbstractCategory(
            name="QUERY_INFO",
            precondition="list visible",
            abstract_steps=["scroll full list", "read attribute"],
            failure_modes=["answering from truncated cell"],
            verification_checklist=["answer matches detail view"],
        ),
    ])


def _fake_mutate_fsm_l2(fsm, trajectories, *, task_category="",
                          layer2_only=False, model="ignored", **kwargs):
    """Mutation stand-in for L_C tests.

    Produces a child whose Layer 2 has one extra failure_mode appended
    to the first category, leaving Layer 1 untouched. Records the
    ``layer2_only`` kwarg on the function object so tests can assert it.
    """
    _fake_mutate_fsm_l2.last_kwargs = {
        "task_category": task_category,
        "layer2_only": layer2_only,
        "model": model,
    }
    child = copy.deepcopy(fsm)
    if child.layer2.categories:
        child.layer2.categories[0].failure_modes.append(
            "[mock] newly discovered failure mode",
        )
    diff = FSMDiff(
        ops=[DiffOp(
            layer="layer2", target="category", op="modify",
            key=child.layer2.categories[0].name if child.layer2.categories else "X",
            value={"failure_modes": list(
                child.layer2.categories[0].failure_modes,
            ) if child.layer2.categories else []},
        )],
        reflection_summary="mock L2-only mutation",
        layer_tag="layer2",
    )
    return child, diff, "mock reflection text"


def test_l_c_to_seed_fsm_produces_empty_layer1():
    """``l_c_to_seed_fsm`` wraps a Layer 2 into an FSM whose Layer 1 is
    deliberately empty."""
    l_c = _sample_l_c()
    fsm = l_c_to_seed_fsm(l_c, app_name="pro_expense", category="Finance")
    assert fsm.app == "pro_expense"
    assert fsm.layer1.app == "pro_expense"
    assert fsm.layer1.category == "Finance"
    assert fsm.layer1.states == []
    assert fsm.layer1.transitions == []
    assert fsm.layer1.strategies == []
    assert fsm.layer1.dead_ends == []
    # Layer 2 is the passed-in L_C (by reference is fine).
    assert fsm.layer2 is l_c
    # Metadata records the provenance so later consumers can tell this
    # wasn't synthesized from trajectories.
    assert fsm.metadata.get("source") == "l_c_seed"


def test_run_l_c_evolution_basic_completes(tmp_path: Path, monkeypatch):
    """L_C evolution runs 5 iterations with mocked rollouts + mutation."""
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm_l2)
    rollout_fn = _make_rollout_fn(tmp_path, lambda i: 1.0 if i % 2 == 0 else 0.0)
    config = EvolutionConfig(n_iterations=5, m_select=1, n_rollouts=1,
                              checkpoint_every=1)

    pop = run_l_c_evolution(
        initial_l_c=_sample_l_c(),
        app_name="pro_expense",
        app_category="Finance",
        task_templates=["ExpenseAdd", "ExpenseDelete"],
        rollout_fn=rollout_fn,
        config=config,
        output_dir=tmp_path,
    )
    # Loop completed + at least one child landed (some iterations got
    # reward=1.0 so mutation should have triggered at least once).
    assert pop.size >= 1
    assert (tmp_path / "population.json").exists()
    assert (tmp_path / "iterations.jsonl").exists()


def test_run_l_c_evolution_saves_versioned_snapshots(
    tmp_path: Path, monkeypatch,
):
    """Every successful mutation produces an ``l_c_v{N}_iter{I}.json``
    file, and ``l_c_champion.json`` stays current."""
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm_l2)
    # Always succeed so mutation fires every iter.
    rollout_fn = _make_rollout_fn(tmp_path, [1.0] * 100)
    config = EvolutionConfig(n_iterations=4, m_select=1, n_rollouts=1)

    run_l_c_evolution(
        initial_l_c=_sample_l_c(),
        app_name="pro_expense",
        app_category="Finance",
        task_templates=["TaskA"],
        rollout_fn=rollout_fn,
        config=config,
        output_dir=tmp_path,
    )

    # Initial snapshot always present and distinct from champion.
    initial_path = tmp_path / "l_c_v0_initial.json"
    assert initial_path.exists()
    # One versioned snapshot per successful mutation (= 4 iterations,
    # all reward=1.0).
    versioned = sorted(tmp_path.glob("l_c_v*_iter*.json"))
    assert len(versioned) == 4
    # Filenames follow the v{N}_iter{I} convention.
    for p in versioned:
        assert "_iter" in p.name and p.name.startswith("l_c_v")
    # Champion file exists and is valid FSM JSON.
    champ_path = tmp_path / "l_c_champion.json"
    assert champ_path.exists()
    champ_fsm = FSM.from_json(json.loads(champ_path.read_text()))
    # Champion has evolved beyond the initial L_C (the fake mutation
    # appends a failure_mode to ADD_ENTRY each round).
    champ_add = next(c for c in champ_fsm.layer2.categories
                      if c.name == "ADD_ENTRY")
    assert any(
        "newly discovered failure mode" in fm for fm in champ_add.failure_modes
    )


def test_run_l_c_evolution_initial_snapshot_preserved(
    tmp_path: Path, monkeypatch,
):
    """``l_c_v0_initial.json`` is written once and never overwritten,
    even after successful mutations change the champion."""
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm_l2)
    rollout_fn = _make_rollout_fn(tmp_path, [1.0] * 100)

    initial_l_c = _sample_l_c()
    # Snapshot the initial content *before* evolution for comparison.
    initial_json_str = json.dumps(initial_l_c.to_json(), sort_keys=True)

    run_l_c_evolution(
        initial_l_c=initial_l_c,
        app_name="pro_expense",
        app_category="Finance",
        task_templates=["TaskA"],
        rollout_fn=rollout_fn,
        config=EvolutionConfig(n_iterations=3, m_select=1, n_rollouts=1),
        output_dir=tmp_path,
    )

    initial_path = tmp_path / "l_c_v0_initial.json"
    saved = json.loads(initial_path.read_text())
    # The file wraps an FSM; its layer2 is what we compare.
    saved_l2 = saved["layer2"]
    assert json.dumps(saved_l2, sort_keys=True) == initial_json_str


def test_run_l_c_evolution_calls_mutate_fsm_with_layer2_only(
    tmp_path: Path, monkeypatch,
):
    """Confirm mutate_fsm is invoked with ``layer2_only=True``."""
    seen_kwargs: list[dict] = []

    def spying_mutate(fsm, trajectories, *,
                       task_category="", layer2_only=False,
                       model="ignored", **kwargs):
        seen_kwargs.append({
            "task_category": task_category,
            "layer2_only": layer2_only,
        })
        return _fake_mutate_fsm_l2(
            fsm, trajectories,
            task_category=task_category,
            layer2_only=layer2_only,
            model=model,
        )

    monkeypatch.setattr(evo, "mutate_fsm", spying_mutate)
    rollout_fn = _make_rollout_fn(tmp_path, [1.0] * 100)

    run_l_c_evolution(
        initial_l_c=_sample_l_c(),
        app_name="pro_expense",
        app_category="Finance",
        task_templates=["TaskA"],
        rollout_fn=rollout_fn,
        config=EvolutionConfig(n_iterations=2, m_select=1, n_rollouts=1),
        output_dir=tmp_path,
    )
    # Each iteration with reward=1.0 triggers exactly one mutate_fsm call.
    assert len(seen_kwargs) == 2
    for kw in seen_kwargs:
        assert kw["layer2_only"] is True
        assert kw["task_category"] == "Finance"


def test_run_l_c_evolution_requires_non_empty_category(tmp_path: Path):
    """Empty ``app_category`` is rejected — Tier-C apps have no L_C to
    evolve, and the caller is expected to skip them at a higher level."""
    try:
        run_l_c_evolution(
            initial_l_c=_sample_l_c(),
            app_name="whatever",
            app_category="",
            task_templates=["TaskA"],
            rollout_fn=_make_rollout_fn(tmp_path, [0.0]),
            config=EvolutionConfig(n_iterations=1),
            output_dir=tmp_path,
        )
    except ValueError as e:
        assert "app_category" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on empty app_category")


def test_run_l_c_evolution_injects_layer2_only_prompt(
    tmp_path: Path, monkeypatch,
):
    """The ``rollout_fn`` should receive a prompt containing only the
    LAYER 2 block — no LAYER 1 APP_SPECIFIC banner (matches B2 shape)."""
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm_l2)

    captured: list[str] = []

    def capturing_rollout(fsm_prompt_text, task_name, seed):
        captured.append(fsm_prompt_text)
        ep_dir = tmp_path / "episodes" / f"{task_name}_seed{seed}"
        _write_mock_episode(ep_dir, task_name=task_name, seed=seed,
                              reward=1.0)
        return RolloutResult(
            task_name=task_name, seed=seed, reward=1.0,
            n_steps=3, wall_seconds=0.01, episode_dir=ep_dir,
        )

    run_l_c_evolution(
        initial_l_c=_sample_l_c(),
        app_name="pro_expense",
        app_category="Finance",
        task_templates=["TaskA"],
        rollout_fn=capturing_rollout,
        config=EvolutionConfig(n_iterations=2, m_select=1, n_rollouts=1),
        output_dir=tmp_path,
    )

    assert captured, "rollout_fn should have been called"
    for prompt in captured:
        assert "LAYER 2: GENERIC" in prompt
        # Layer-2-only injection suppresses the LAYER 1 banner.
        assert "LAYER 1: APP_SPECIFIC" not in prompt
        # The L_C category tag is included so the agent knows context.
        assert "L_C CATEGORY: Finance" in prompt


# ═════════════════════════════════════════════════════════════════════
# B4: LoRA + GRPO integration (Story 4.3)
# ═════════════════════════════════════════════════════════════════════
#
# These tests exercise the ``enable_lora=True`` path in ``run_evolution``
# end-to-end without loading peft or a real Qwen3-VL model. They mock
# ``grpo_step``, ``save_lora_checkpoint``, and ``cleanup_replay_data`` at
# the evolution module's namespace so the real functions (which touch
# torch tensors + peft + filesystem) never run. A lightweight ``_FakeModel``
# exposes a single ``torch.nn.Parameter`` so the AdamW optimizer
# constructor has something legal to iterate; ``count_trainable_params``
# sees the parameter and reports >0 trainable.


class _FakeModel:
    """Minimal stand-in for a peft-wrapped Qwen3-VL.

    Exposes one ``torch.nn.Parameter(requires_grad=True)`` through
    ``.parameters()`` so ``torch.optim.AdamW`` + ``count_trainable_params``
    are happy. No forward pass; the mocked ``grpo_step`` never calls one.
    """

    def __init__(self, *, trainable: bool = True):
        import torch
        self._p = torch.nn.Parameter(torch.zeros(1), requires_grad=trainable)

    def parameters(self):
        return iter([self._p])


def _fake_trajectory(variant_id: str = "v0", reward: float = 1.0):
    """Minimal TrajectoryData-like payload that rollouts return in B4."""
    from evofsm_rl.rl.grpo import TrajectoryData
    return TrajectoryData(
        task_name="TaskA",
        seed=0,
        fsm_variant_id=variant_id,
        success=float(reward > 0),
        n_steps=3,
        step_log_probs=[-1.0, -0.5, -0.2],
        replay_paths=[],
        reward=float(reward),
    )


def _make_b4_rollout_fn(tmp_path: Path, reward_seq):
    """Like ``_make_rollout_fn`` but also attaches a ``trajectory_data``
    object to every :class:`RolloutResult` so B4 wiring exercises.
    """
    calls = {"n": 0}

    def rollout_fn(fsm_prompt_text, task_name, seed):
        i = calls["n"]
        calls["n"] += 1
        if callable(reward_seq):
            r = float(reward_seq(i))
        else:
            r = float(reward_seq[i % len(reward_seq)])
        ep_dir = tmp_path / "episodes" / f"{task_name}_seed{seed}"
        _write_mock_episode(ep_dir, task_name=task_name, seed=seed, reward=r)
        return RolloutResult(
            task_name=task_name,
            seed=seed,
            reward=r,
            n_steps=3,
            wall_seconds=0.01,
            episode_dir=ep_dir,
            trajectory_data=_fake_trajectory(reward=r),
        )

    rollout_fn.calls = calls  # type: ignore[attr-defined]
    return rollout_fn


def test_b4_new_fields_default_off():
    """Test B4-1: ``EvolutionConfig`` defaults preserve B3 behavior.

    Every new field must have a default that keeps ``enable_lora=False``
    and leaves the B3 code path unchanged.
    """
    cfg = EvolutionConfig()
    assert cfg.enable_lora is False
    assert cfg.lora_rank == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    assert cfg.lora_target_modules == ("q_proj", "v_proj")
    assert cfg.lora_lr == 1e-5
    assert cfg.lora_update_every_k == 8
    assert cfg.lora_max_grad_norm == 1.0
    assert cfg.lora_checkpoint_every_k == 4
    assert cfg.lora_device == "cuda"


def test_b4_rollout_result_trajectory_data_default_none():
    """Test B4-2: ``RolloutResult.trajectory_data`` defaults to None so
    existing callers keep working without modification."""
    r = RolloutResult(
        task_name="T", seed=0, reward=1.0, n_steps=1, wall_seconds=0.01,
    )
    assert r.trajectory_data is None


def test_b4_enable_lora_false_preserves_b3_behavior(
    tmp_path: Path, monkeypatch,
):
    """Test B4-3: With ``enable_lora=False`` (default), no LoRA code runs
    even when ``model`` happens to be supplied — grpo_step never called,
    no lora_checkpoints/ dir created, population evolves as in B3.
    """
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm)
    grpo_calls: list = []
    ckpt_calls: list = []
    monkeypatch.setattr(
        evo, "grpo_step",
        lambda *a, **kw: (grpo_calls.append((a, kw)) or {
            "loss": 0.0, "grad_norm": 0.0, "mean_advantage": 0.0,
            "advantage_std": 0.0, "advantage_abs_max": 0.0,
            "mean_reward": 0.0, "n_trajectories": 0, "n_active": 0,
        }),
    )
    monkeypatch.setattr(
        evo, "save_lora_checkpoint",
        lambda m, p: (ckpt_calls.append(p) or Path(p)),
    )

    rollout_fn = _make_b4_rollout_fn(tmp_path, [1.0] * 100)
    run_evolution(
        root_fsm=_sample_fsm(),
        task_templates=["TaskA"],
        rollout_fn=rollout_fn,
        config=EvolutionConfig(n_iterations=3, m_select=1, n_rollouts=1),
        output_dir=tmp_path,
        app_category="Productivity",
    )
    assert grpo_calls == []
    assert ckpt_calls == []
    assert not (tmp_path / "lora_checkpoints").exists()


def test_b4_enable_lora_requires_model(tmp_path: Path):
    """Test B4-4: ``enable_lora=True`` without ``model=`` raises ValueError
    before any iteration runs."""
    rollout_fn = _make_b4_rollout_fn(tmp_path, [1.0] * 100)
    with pytest.raises(ValueError, match="enable_lora=True requires"):
        run_evolution(
            root_fsm=_sample_fsm(),
            task_templates=["TaskA"],
            rollout_fn=rollout_fn,
            config=EvolutionConfig(
                n_iterations=1, enable_lora=True,
            ),
            output_dir=tmp_path,
            app_category="Productivity",
            model=None,
        )


def test_b4_enable_lora_requires_trainable_params(tmp_path: Path):
    """Test B4-5: ``enable_lora=True`` with a model whose parameters all
    have ``requires_grad=False`` raises ValueError."""
    rollout_fn = _make_b4_rollout_fn(tmp_path, [1.0] * 100)
    with pytest.raises(ValueError, match="no trainable parameters"):
        run_evolution(
            root_fsm=_sample_fsm(),
            task_templates=["TaskA"],
            rollout_fn=rollout_fn,
            config=EvolutionConfig(
                n_iterations=1, enable_lora=True,
            ),
            output_dir=tmp_path,
            app_category="Productivity",
            model=_FakeModel(trainable=False),
        )


def test_b4_grpo_step_called_every_k_iterations(
    tmp_path: Path, monkeypatch,
):
    """Test B4-6: With ``lora_update_every_k=2`` and 1 traj/iteration,
    ``grpo_step`` is called on iter 2 and iter 4 (not 1 or 3)."""
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm)
    grpo_invocations: list = []

    def fake_grpo(model, optimizer, trajs, *, device, max_grad_norm, **_kwargs):
        grpo_invocations.append({
            "n_traj": len(trajs),
            "device": device,
            "max_grad_norm": max_grad_norm,
        })
        return {
            "loss": 0.5, "grad_norm": 0.1, "mean_advantage": 0.0,
            "advantage_std": 0.05, "advantage_abs_max": 0.1,
            "mean_reward": 1.0, "n_trajectories": len(trajs), "n_active": 0,
        }

    cleanup_invocations: list = []
    monkeypatch.setattr(evo, "grpo_step", fake_grpo)
    monkeypatch.setattr(
        evo, "cleanup_replay_data",
        lambda trajs: (cleanup_invocations.append(len(trajs)) or 0),
    )
    monkeypatch.setattr(
        evo, "save_lora_checkpoint",
        lambda m, p: Path(p),  # no-op — cadence checked in separate test
    )

    rollout_fn = _make_b4_rollout_fn(tmp_path, [1.0] * 100)
    cfg = EvolutionConfig(
        n_iterations=4, m_select=1, n_rollouts=1,
        enable_lora=True,
        lora_update_every_k=2,
        lora_checkpoint_every_k=999,  # disable periodic ckpt for this test
        lora_device="cpu",
    )
    run_evolution(
        root_fsm=_sample_fsm(),
        task_templates=["TaskA"],
        rollout_fn=rollout_fn,
        config=cfg,
        output_dir=tmp_path,
        app_category="Productivity",
        model=_FakeModel(),
    )
    # 4 iterations × 1 traj each = 4 trajs buffered. Threshold=2, so
    # grpo fires after iter 2 (flushes 2) and after iter 4 (flushes 2).
    # Final-drain block runs only if buffer non-empty — it's already flushed.
    assert len(grpo_invocations) == 2
    assert all(inv["n_traj"] == 2 for inv in grpo_invocations)
    assert all(inv["device"] == "cpu" for inv in grpo_invocations)
    assert cleanup_invocations == [2, 2]


def test_b4_cleanup_replay_data_called_after_grpo(
    tmp_path: Path, monkeypatch,
):
    """Test B4-7: Every GRPO step is paired with a ``cleanup_replay_data``
    call, in that order, on the same buffer."""
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm)
    events: list[str] = []

    def fake_grpo(model, optimizer, trajs, *, device, max_grad_norm, **_kwargs):
        events.append(f"grpo[{len(trajs)}]")
        return {
            "loss": 0.0, "grad_norm": 0.0, "mean_advantage": 0.0,
            "advantage_std": 0.0, "advantage_abs_max": 0.0,
            "mean_reward": 0.0,
            "n_trajectories": len(trajs), "n_active": 0,
        }

    def fake_cleanup(trajs):
        events.append(f"cleanup[{len(trajs)}]")
        return 0

    monkeypatch.setattr(evo, "grpo_step", fake_grpo)
    monkeypatch.setattr(evo, "cleanup_replay_data", fake_cleanup)
    monkeypatch.setattr(evo, "save_lora_checkpoint", lambda m, p: Path(p))

    rollout_fn = _make_b4_rollout_fn(tmp_path, [1.0] * 100)
    cfg = EvolutionConfig(
        n_iterations=4, m_select=1, n_rollouts=1,
        enable_lora=True,
        lora_update_every_k=2,
        lora_checkpoint_every_k=999,
        lora_device="cpu",
    )
    run_evolution(
        root_fsm=_sample_fsm(),
        task_templates=["TaskA"],
        rollout_fn=rollout_fn,
        config=cfg,
        output_dir=tmp_path,
        app_category="Productivity",
        model=_FakeModel(),
    )
    # grpo must precede cleanup every time; two pairs in total.
    assert events == ["grpo[2]", "cleanup[2]", "grpo[2]", "cleanup[2]"]


def test_b4_grpo_not_called_when_no_trajectory_data(
    tmp_path: Path, monkeypatch,
):
    """Test B4-8: Rollouts that don't populate ``trajectory_data``
    (legacy rollout_fn or infra crash) leave the buffer empty — grpo is
    never invoked even with ``enable_lora=True``.
    """
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm)
    grpo_calls: list = []
    monkeypatch.setattr(
        evo, "grpo_step",
        lambda *a, **kw: (grpo_calls.append(1) or {
            "loss": 0.0, "grad_norm": 0.0, "mean_advantage": 0.0,
            "advantage_std": 0.0, "advantage_abs_max": 0.0,
            "mean_reward": 0.0, "n_trajectories": 0, "n_active": 0,
        }),
    )
    monkeypatch.setattr(evo, "cleanup_replay_data", lambda t: 0)
    monkeypatch.setattr(evo, "save_lora_checkpoint", lambda m, p: Path(p))

    # Legacy rollout_fn — no trajectory_data attached.
    legacy_rollout_fn = _make_rollout_fn(tmp_path, [1.0] * 100)
    run_evolution(
        root_fsm=_sample_fsm(),
        task_templates=["TaskA"],
        rollout_fn=legacy_rollout_fn,
        config=EvolutionConfig(
            n_iterations=4, m_select=1, n_rollouts=1,
            enable_lora=True, lora_update_every_k=2,
            lora_checkpoint_every_k=999, lora_device="cpu",
        ),
        output_dir=tmp_path,
        app_category="Productivity",
        model=_FakeModel(),
    )
    assert grpo_calls == []


def test_b4_lora_checkpoint_every_k_iterations(
    tmp_path: Path, monkeypatch,
):
    """Test B4-9: With ``lora_checkpoint_every_k=2`` and 5 iterations,
    periodic ``save_lora_checkpoint`` fires on iters 2, 4 → two
    ``iter_####`` dirs, plus the final ``final/``.
    """
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm)
    monkeypatch.setattr(
        evo, "grpo_step",
        lambda *a, **kw: {
            "loss": 0.0, "grad_norm": 0.0, "mean_advantage": 0.0,
            "advantage_std": 0.0, "advantage_abs_max": 0.0,
            "mean_reward": 0.0, "n_trajectories": 0, "n_active": 0,
        },
    )
    monkeypatch.setattr(evo, "cleanup_replay_data", lambda t: 0)

    ckpt_paths: list[Path] = []

    def fake_save(model, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_model.bin").write_text("stub")
        ckpt_paths.append(p)
        return p

    monkeypatch.setattr(evo, "save_lora_checkpoint", fake_save)

    rollout_fn = _make_b4_rollout_fn(tmp_path, [1.0] * 100)
    run_evolution(
        root_fsm=_sample_fsm(),
        task_templates=["TaskA"],
        rollout_fn=rollout_fn,
        config=EvolutionConfig(
            n_iterations=5, m_select=1, n_rollouts=1,
            enable_lora=True,
            lora_update_every_k=999,  # disable GRPO cadence here
            lora_checkpoint_every_k=2,
            lora_device="cpu",
        ),
        output_dir=tmp_path,
        app_category="Productivity",
        model=_FakeModel(),
    )
    # 2 periodic (iter 2, 4) + 1 final = 3 total.
    assert len(ckpt_paths) == 3
    names = [p.name for p in ckpt_paths]
    assert names[0] == "iter_0002"
    assert names[1] == "iter_0004"
    assert names[2] == "final"
    # All live under lora_checkpoints/.
    for p in ckpt_paths:
        assert p.parent.name == "lora_checkpoints"


def test_b4_final_lora_checkpoint_always_written(
    tmp_path: Path, monkeypatch,
):
    """Test B4-10: Even when ``lora_checkpoint_every_k`` is huge (no
    periodic ckpts), ``final/`` is always saved after the loop exits.
    """
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm)
    monkeypatch.setattr(
        evo, "grpo_step",
        lambda *a, **kw: {
            "loss": 0.0, "grad_norm": 0.0, "mean_advantage": 0.0,
            "advantage_std": 0.0, "advantage_abs_max": 0.0,
            "mean_reward": 0.0, "n_trajectories": 0, "n_active": 0,
        },
    )
    monkeypatch.setattr(evo, "cleanup_replay_data", lambda t: 0)

    ckpt_paths: list[Path] = []
    monkeypatch.setattr(
        evo, "save_lora_checkpoint",
        lambda m, p: (ckpt_paths.append(Path(p)) or Path(p)),
    )

    rollout_fn = _make_b4_rollout_fn(tmp_path, [1.0] * 100)
    run_evolution(
        root_fsm=_sample_fsm(),
        task_templates=["TaskA"],
        rollout_fn=rollout_fn,
        config=EvolutionConfig(
            n_iterations=3, m_select=1, n_rollouts=1,
            enable_lora=True,
            lora_update_every_k=999,
            lora_checkpoint_every_k=999,
            lora_device="cpu",
        ),
        output_dir=tmp_path,
        app_category="Productivity",
        model=_FakeModel(),
    )
    # Only the post-loop final/ checkpoint.
    assert len(ckpt_paths) == 1
    assert ckpt_paths[0].name == "final"


def test_b4_final_drain_flushes_leftover_buffer(
    tmp_path: Path, monkeypatch,
):
    """Test B4-11: Buffered trajectories that never hit the
    ``lora_update_every_k`` threshold during the loop are still flushed
    by the post-loop drain so no GPU memory / disk is leaked.
    """
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm)
    events: list = []

    def fake_grpo(model, optimizer, trajs, *, device, max_grad_norm, **_kwargs):
        events.append(("grpo", len(trajs)))
        return {
            "loss": 0.0, "grad_norm": 0.0, "mean_advantage": 0.0,
            "advantage_std": 0.0, "advantage_abs_max": 0.0,
            "mean_reward": 0.0,
            "n_trajectories": len(trajs), "n_active": 0,
        }

    monkeypatch.setattr(evo, "grpo_step", fake_grpo)
    monkeypatch.setattr(
        evo, "cleanup_replay_data",
        lambda t: (events.append(("cleanup", len(t))) or 0),
    )
    monkeypatch.setattr(evo, "save_lora_checkpoint", lambda m, p: Path(p))

    rollout_fn = _make_b4_rollout_fn(tmp_path, [1.0] * 100)
    # 3 iterations × 1 traj = 3 trajs buffered, threshold = 100 (never
    # hit during the loop). Final drain must flush all 3.
    run_evolution(
        root_fsm=_sample_fsm(),
        task_templates=["TaskA"],
        rollout_fn=rollout_fn,
        config=EvolutionConfig(
            n_iterations=3, m_select=1, n_rollouts=1,
            enable_lora=True,
            lora_update_every_k=100,
            lora_checkpoint_every_k=999,
            lora_device="cpu",
        ),
        output_dir=tmp_path,
        app_category="Productivity",
        model=_FakeModel(),
    )
    # Exactly one final-drain grpo + cleanup over all 3 trajs.
    assert events == [("grpo", 3), ("cleanup", 3)]


def test_b4_run_l_c_evolution_forwards_model(
    tmp_path: Path, monkeypatch,
):
    """Test B4-12: ``run_l_c_evolution`` threads ``model=`` through to
    ``run_evolution`` so B4 works from the L_C entry point too.
    """
    monkeypatch.setattr(evo, "mutate_fsm", _fake_mutate_fsm_l2)
    captured: dict = {}

    def fake_grpo(model, optimizer, trajs, *, device, max_grad_norm, **_kwargs):
        # The model passed to grpo_step must be the same object we handed
        # to run_l_c_evolution — proof the kwarg threaded all the way down.
        captured["model"] = model
        return {
            "loss": 0.0, "grad_norm": 0.0, "mean_advantage": 0.0,
            "advantage_std": 0.0, "advantage_abs_max": 0.0,
            "mean_reward": 0.0,
            "n_trajectories": len(trajs), "n_active": 0,
        }

    monkeypatch.setattr(evo, "grpo_step", fake_grpo)
    monkeypatch.setattr(evo, "cleanup_replay_data", lambda t: 0)
    monkeypatch.setattr(evo, "save_lora_checkpoint", lambda m, p: Path(p))

    fake_model = _FakeModel()
    rollout_fn = _make_b4_rollout_fn(tmp_path, [1.0] * 100)
    run_l_c_evolution(
        initial_l_c=_sample_l_c(),
        app_name="pro_expense",
        app_category="Finance",
        task_templates=["TaskA"],
        rollout_fn=rollout_fn,
        config=EvolutionConfig(
            n_iterations=2, m_select=1, n_rollouts=1,
            enable_lora=True,
            lora_update_every_k=2,
            lora_checkpoint_every_k=999,
            lora_device="cpu",
        ),
        output_dir=tmp_path,
        model=fake_model,
    )
    assert captured.get("model") is fake_model


# ─────────────────────────────────────────────────────────────────────
# Standalone runner (pytest-optional)
# ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import inspect
    import tempfile
    import traceback

    class _MP:
        def __init__(self):
            self._undo: list = []

        def setattr(self, target, value=None):
            import importlib
            if isinstance(target, str):
                mod_name, _, attr = target.rpartition(".")
                mod = importlib.import_module(mod_name)
                old = getattr(mod, attr, None)
                setattr(mod, attr, value)
                self._undo.append((mod, attr, old))
            else:
                mod = target
                old = getattr(mod, value, None)
                # monkeypatch module attr 3-arg form unused in this suite
                raise NotImplementedError

        def undo(self):
            while self._undo:
                mod, attr, old = self._undo.pop()
                setattr(mod, attr, old)

    ns = dict(globals())
    tests = [(name, fn) for name, fn in ns.items()
             if name.startswith("test_") and callable(fn)]
    passed = failed = 0
    for name, fn in tests:
        mp = _MP()
        td = tempfile.TemporaryDirectory()
        try:
            sig = inspect.signature(fn)
            kwargs = {}
            if "monkeypatch" in sig.parameters:
                class _MPShim:
                    def __init__(self, mp): self.mp = mp
                    def setattr(self, mod_or_target, attr_or_value, value=None):
                        if value is None and isinstance(mod_or_target, str):
                            self.mp.setattr(mod_or_target, attr_or_value)
                        else:
                            old = getattr(mod_or_target, attr_or_value, None)
                            setattr(mod_or_target, attr_or_value, value)
                            self.mp._undo.append(
                                (mod_or_target, attr_or_value, old),
                            )
                kwargs["monkeypatch"] = _MPShim(mp)
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
            mp.undo()
            td.cleanup()
    print(f"\n{passed}/{passed + failed} passed")
    raise SystemExit(0 if failed == 0 else 1)

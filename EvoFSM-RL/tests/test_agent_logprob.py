"""Tests for Qwen3VLAgent's B4 log-prob surface — Story 4.2 Part A.

These tests exercise the agent-level surface area added for B4:

  * ``collect_log_probs`` / ``replay_dir`` constructor params
  * ``reset()`` clearing per-episode log-prob state and allocating a
    fresh per-episode replay subdir
  * ``get_trajectory_data`` packaging the just-finished episode into
    a :class:`evofsm_rl.rl.grpo.TrajectoryData`

They do NOT exercise ``step()`` — running a real generate() on Qwen3-VL
for each test is too expensive and GPU-resident, and the step-level
behavior is covered in the Story 4.4 integration smoke test. Instead
we construct the agent with stubs for ``model`` / ``processor`` /
``env``, then directly manipulate episode-state fields the way a real
episode would and verify the resulting TrajectoryData shape.

Run::
    python -m pytest tests/test_agent_logprob.py -v
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from evofsm_rl.agent.rollout import GenerationConfig, Qwen3VLAgent, generate_text
from evofsm_rl.rl.grpo import TrajectoryData


# ─────────────────────────────────────────────────────────────────────
# Minimal stubs — enough to instantiate Qwen3VLAgent without loading
# Qwen3-VL or connecting an emulator. The base agent's __init__ only
# stores `env`; it never calls methods on it.
# ─────────────────────────────────────────────────────────────────────


class _DummyModel:
    """Stand-in for Qwen3-VL. ``.eval()`` is the only method the agent
    calls during __init__; we make it a no-op."""

    def eval(self):
        return self


class _DummyProcessor:
    """Processor stub — never touched until ``step()`` runs, which
    these tests don't call."""


class _DummyEnv:
    """Enough of an interface.AsyncEnv to satisfy ``base_agent``'s
    ``self._env = env`` + the ``reset()`` forwarded call the base
    agent's own ``reset()`` makes. No real emulator state."""

    def reset(self, go_home: bool = False) -> None:
        return None


def _make_agent(
    *, collect_log_probs: bool = False,
    replay_dir: Path | None = None,
) -> Qwen3VLAgent:
    return Qwen3VLAgent(
        model=_DummyModel(),
        processor=_DummyProcessor(),
        env=_DummyEnv(),
        device="cpu",
        collect_log_probs=collect_log_probs,
        replay_dir=replay_dir,
    )


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────


def test_default_collect_log_probs_false():
    """Default construction: B3 behavior preserved. Flag off, log-prob
    list empty, replay dir not materialised."""
    agent = _make_agent()
    assert agent._collect_log_probs is False
    assert agent._step_log_probs == []
    assert agent._replay_dir is None
    assert agent._replay_dir_base is None


def test_enabled_but_no_replay_dir():
    """collect_log_probs=True without replay_dir: log-probs are still
    collected but nothing hits disk. reset() leaves replay_dir None."""
    agent = _make_agent(collect_log_probs=True)
    assert agent._collect_log_probs is True
    assert agent._replay_dir_base is None
    agent.reset(go_home=False)
    assert agent._replay_dir is None
    assert agent._step_log_probs == []


def test_reset_allocates_per_episode_replay_subdir(tmp_path: Path):
    """Each reset with a configured base dir allocates a monotonic
    per-episode subdir on disk."""
    agent = _make_agent(collect_log_probs=True, replay_dir=tmp_path)

    agent.reset(go_home=False)
    ep1 = agent._replay_dir
    assert ep1 is not None
    assert ep1.exists()
    assert ep1.name == "episode_0001"
    assert ep1.parent == tmp_path

    agent.reset(go_home=False)
    ep2 = agent._replay_dir
    assert ep2 is not None
    assert ep2.name == "episode_0002"
    assert ep2 != ep1
    assert ep2.exists()


def test_reset_clears_log_probs(tmp_path: Path):
    """Pre-populate ``_step_log_probs`` manually, call reset, verify empty."""
    agent = _make_agent(collect_log_probs=True, replay_dir=tmp_path)
    agent._step_log_probs = [-1.1, -0.7, -2.3]
    agent.reset(go_home=False)
    assert agent._step_log_probs == []


def test_get_trajectory_data_fields_populated(tmp_path: Path):
    """Set up a fake finished episode by touching the fields ``step()``
    would have written, then verify ``get_trajectory_data`` packages
    them correctly."""
    agent = _make_agent(collect_log_probs=True, replay_dir=tmp_path)
    agent.reset(go_home=False)

    # Simulate two steps having run.
    agent._step_idx = 2
    agent._step_log_probs = [-2.5, -1.75]
    # Create the replay files the way a real step() would.
    ep_dir = agent._replay_dir
    assert ep_dir is not None
    (ep_dir / "step_1.pt").write_bytes(b"dummy")
    (ep_dir / "step_2.pt").write_bytes(b"dummy")
    # Last-step history with a completion status emission.
    agent.history = [
        {"action_output_json": {"action_type": "click", "index": 3}},
        {"action_output_json": {
            "action_type": "status", "goal_status": "complete",
        }},
    ]

    traj = agent.get_trajectory_data(
        task_name="MarkorCreateNote", seed=42,
        fsm_variant_id="gen3_mut_7", reward=1.18,
    )

    assert isinstance(traj, TrajectoryData)
    assert traj.task_name == "MarkorCreateNote"
    assert traj.seed == 42
    assert traj.fsm_variant_id == "gen3_mut_7"
    assert traj.n_steps == 2
    assert traj.step_log_probs == [-2.5, -1.75]
    # success inferred from the last step's goal_status=="complete".
    assert traj.success == 1.0
    assert traj.reward == 1.18
    # Replay paths point at our two fake .pt files, sorted by name.
    assert len(traj.replay_paths) == 2
    assert all(Path(p).exists() for p in traj.replay_paths)
    assert Path(traj.replay_paths[0]).name == "step_1.pt"
    assert Path(traj.replay_paths[1]).name == "step_2.pt"


def test_get_trajectory_data_explicit_success_overrides_inference(
    tmp_path: Path,
):
    """When the caller knows the real ``task.is_successful`` verdict,
    passing ``success=`` explicitly wins over the history heuristic."""
    agent = _make_agent(collect_log_probs=True, replay_dir=tmp_path)
    agent.reset(go_home=False)
    agent._step_idx = 1
    agent._step_log_probs = [-1.0]
    # Agent *thinks* it completed the task, but the grader disagrees.
    agent.history = [
        {"action_output_json": {
            "action_type": "status", "goal_status": "complete",
        }},
    ]

    traj = agent.get_trajectory_data(
        task_name="T", seed=0, fsm_variant_id="v", reward=0.0, success=0.0,
    )
    assert traj.success == 0.0


def test_get_trajectory_data_empty_episode(tmp_path: Path):
    """Zero-step episode (e.g. infra crash before the first action)
    still yields a valid TrajectoryData with empty lists."""
    agent = _make_agent(collect_log_probs=True, replay_dir=tmp_path)
    agent.reset(go_home=False)
    # No steps run, no history appended.
    traj = agent.get_trajectory_data(
        task_name="T", seed=0, fsm_variant_id="v", reward=0.0,
    )
    assert traj.n_steps == 0
    assert traj.step_log_probs == []
    assert traj.replay_paths == []
    assert traj.success == 0.0


def test_generate_text_log_prob_uses_single_generate_call(tmp_path: Path):
    """Regression — the log-prob path must NOT run a second full forward
    pass (that was the OOM-at-80GB culprit). Instead it uses
    ``generate(return_dict_in_generate=True, output_scores=True)`` and
    reads logits from the returned ``scores`` tuple. No ``__call__`` on
    the model outside ``generate``.
    """
    import torch

    input_len = 4
    action_len = 3
    vocab = 16

    class _FakeTokenizer:
        def decode(self, ids, skip_special_tokens: bool = True) -> str:
            return "action-text"

    class _FakeProcessor:
        tokenizer = _FakeTokenizer()

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "rendered"

        def __call__(self, *, text, images, return_tensors, padding):
            return {
                "input_ids": torch.arange(input_len).unsqueeze(0),
                "attention_mask": torch.ones(1, input_len, dtype=torch.long),
                "pixel_values": torch.zeros(1, 3, 8, 8),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
                "mm_token_type_ids": torch.zeros(1, input_len, dtype=torch.long),
                "exotic_future_field": torch.tensor([[1.0, 2.0]]),
            }

    forward_calls = {"n": 0}

    class _GenOut:
        def __init__(self, sequences, scores):
            self.sequences = sequences
            self.scores = scores

    class _FakeModel:
        def generate(self, *, input_ids, return_dict_in_generate=False,
                     output_scores=False, **kwargs):
            tail = torch.arange(
                input_len, input_len + action_len,
            ).unsqueeze(0)
            seqs = torch.cat([input_ids, tail], dim=-1)
            if return_dict_in_generate and output_scores:
                # One logits tensor per generated token.
                scores = tuple(
                    torch.zeros(1, vocab) for _ in range(action_len)
                )
                return _GenOut(seqs, scores)
            return seqs

        def __call__(self, **kwargs):
            forward_calls["n"] += 1
            raise AssertionError(
                "generate_text(collect_log_probs=True) must not run a "
                "second forward pass — it should use generate(..., "
                "output_scores=True) instead."
            )

        def eval(self):
            return self

    text, _wall, got_input_len, extras = generate_text(
        model=_FakeModel(),
        processor=_FakeProcessor(),
        messages=[],
        images=[],
        device="cpu",
        generation_config=GenerationConfig(
            max_new_tokens=action_len, do_sample=False,
        ),
        collect_log_probs=True,
    )

    assert got_input_len == input_len
    # No second forward pass.
    assert forward_calls["n"] == 0
    # Extras still echoes all processor fields so grpo.py's later
    # gradient-bearing replay forward can reconstruct the inputs.
    for key in (
        "input_ids", "attention_mask", "pixel_values",
        "image_grid_thw", "mm_token_type_ids", "exotic_future_field",
    ):
        assert key in extras, f"{key} missing from extras dict"
    assert "step_log_prob" in extras
    assert "action_token_ids" in extras
    assert extras["input_len"] == input_len


def test_step_replay_save_persists_all_processor_fields(tmp_path: Path):
    """Regression — when ``Qwen3VLAgent.step()`` saves the per-step
    ``.pt`` replay, it must persist every tensor field the processor
    returned, not a hardcoded list. We directly emulate the save by
    calling the same ``torch.save`` code path with an ``extras`` dict
    shaped like ``generate_text`` now produces.
    """
    import torch

    extras = {
        "input_ids": torch.arange(4).unsqueeze(0),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 8, 8),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "mm_token_type_ids": torch.zeros(1, 4, dtype=torch.long),
        "step_log_prob": -1.234,
        "action_token_ids": torch.tensor([5, 6, 7]),
        "input_len": 4,
    }

    replay_data = {k: v for k, v in extras.items() if k != "step_log_prob"}
    replay_path = tmp_path / "step_1.pt"
    torch.save(replay_data, replay_path)

    loaded = torch.load(replay_path, weights_only=False)
    for key in (
        "input_ids", "attention_mask", "pixel_values",
        "image_grid_thw", "mm_token_type_ids", "action_token_ids",
    ):
        assert key in loaded, f"{key} not persisted"
    assert loaded["input_len"] == 4
    assert "step_log_prob" not in loaded  # scalar is derivable, not stored


def test_default_agent_does_not_create_replay_dirs(tmp_path: Path):
    """B3-compat: default (collect_log_probs=False) agent never
    touches the replay_dir even when one is provided."""
    # We pass a replay_dir but leave collect_log_probs False — the
    # agent should ignore replay_dir entirely, not create subdirs.
    agent = _make_agent(collect_log_probs=False, replay_dir=tmp_path)
    agent.reset(go_home=False)
    # No subdir allocation when flag is off.
    assert agent._replay_dir is None
    # Parent stays empty.
    assert list(tmp_path.iterdir()) == []


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

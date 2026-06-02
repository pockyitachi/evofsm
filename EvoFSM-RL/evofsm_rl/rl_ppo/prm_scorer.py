"""PRM (Process Reward Model) scorer used at PPO rollout time.

Loads a trained PRM head (Linear(hidden_size → 1) on the frozen base
model's last-token hidden state) and produces a per-step scalar reward
score from a finished trajectory's saved data on disk.

The PRM was trained (see ``scripts/run_prm_train.py``) on Sonnet step
labels using the same input format Sonnet saw at labeling time:
    GOAL + BEFORE screenshot/UI + ACTION/REASON + AFTER screenshot/UI + agent SUMMARY → score.

At PPO rollout time:
    1. Agent runs its trajectory, writing ``episode.jsonl`` + per-step
       ``step_*_before.png`` / ``step_*_after.png``.
    2. After the rollout, :func:`score_trajectory` reads those files and
       runs PRM forward per step to produce ``per_step_rewards``.
    3. The rewards are attached to :class:`TrajectoryData` and used by
       :class:`PPOTrainer` via :func:`compute_gae(per_step_rewards=...)`.

This module does NOT do any policy forward / training — it just runs
the frozen base + frozen PRM head once per step to produce a scalar.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Match what run_prm_train.py used at training time
PRM_PROMPT_TEMPLATE = """Goal: {goal}

BEFORE action:
[image 1 above]
UI elements: {before_ui}

Action taken: {action}
Reasoning: {action_reason}

AFTER action:
[image 2 above]
UI elements: {after_ui}
Reflection: {summary}

Evaluate this step's progress toward the goal."""


def load_prm_head(path: Path | str, model: Any, device: str | Any):
    """Load a trained PRM head's state_dict and return a frozen module.

    Args:
        path: Path to ``.pt`` file (state_dict torch.save).
        model: Loaded base model (needed to infer hidden_size).
        device: Target device.

    Returns:
        :class:`LinearValueHead` instance with weights loaded and grads off.
    """
    import torch
    from evofsm_rl.rl_ppo.value_head import attach_value_head

    head = attach_value_head(model, device)
    state = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    head.load_state_dict(state)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def score_trajectory(
    episode_dir: Path | str,
    *,
    prm_head: Any,
    model: Any,
    processor: Any,
    device: str,
    ui_trunc: int = 2000,
) -> list[float] | None:
    """Score every step in a saved trajectory using the PRM.

    Args:
        episode_dir: Directory containing ``episode.jsonl`` + screenshot PNGs
            written by the agent's :meth:`save_episode`.
        prm_head: PRM head from :func:`load_prm_head`.
        model, processor: Base model + processor (frozen for PRM eval).
        device: Device string.
        ui_trunc: Truncate before/after UI text to this many chars (matches
            Sonnet labeling truncation).

    Returns:
        A list of per-step scalar rewards (length = number of steps), each
        in roughly [0, 1] (the PRM was trained to predict Sonnet's
        {0, 0.25, 0.5, 0.75, 1.0} scores via MSE; output is unbounded but
        typically clamped to [0, 1] downstream). Returns None if scoring
        failed for the whole trajectory (e.g., missing episode.jsonl).
    """
    import torch
    from PIL import Image

    ep_path = Path(episode_dir)
    jsonl_p = ep_path / "episode.jsonl"
    if not jsonl_p.exists():
        logger.warning("score_trajectory: missing %s", jsonl_p)
        return None

    with jsonl_p.open() as f:
        steps = [json.loads(line) for line in f if line.strip()]
    if not steps:
        return None

    scores: list[float] = []
    for step_idx, step in enumerate(steps):
        # Required fields
        req = (
            "goal", "action", "action_reason", "summary",
            "before_ui_elements_text", "after_ui_elements_text",
            "before_screenshot_path", "after_screenshot_path",
        )
        if not all(step.get(k) is not None for k in req):
            scores.append(0.5)  # neutral fallback if step data incomplete
            continue
        bp = ep_path / Path(step["before_screenshot_path"]).name
        ap = ep_path / Path(step["after_screenshot_path"]).name
        if not bp.exists() or not ap.exists():
            scores.append(0.5)
            continue

        try:
            before_pil = Image.open(bp).convert("RGB")
            after_pil = Image.open(ap).convert("RGB")
            prompt_text = PRM_PROMPT_TEMPLATE.format(
                goal=step["goal"],
                before_ui=step["before_ui_elements_text"][:ui_trunc],
                after_ui=step["after_ui_elements_text"][:ui_trunc],
                action=json.dumps(step["action"]),
                action_reason=step["action_reason"],
                summary=step["summary"],
            )
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": before_pil},
                    {"type": "image", "image": after_pil},
                    {"type": "text", "text": prompt_text},
                ],
            }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = processor(
                text=[text],
                images=[before_pil, after_pil],
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            v = prm_head(hidden)
            score = float(v.detach().item())
            # Clamp to [0, 1] — PRM trained to predict Sonnet scores in this range
            score = max(0.0, min(1.0, score))
            scores.append(score)
            del before_pil, after_pil, outputs, hidden, v, inputs
            if torch.cuda.is_available() and isinstance(device, str) and device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            logger.warning("PRM forward failed at step %d: %s", step_idx, e)
            scores.append(0.5)

    return scores


__all__ = ["load_prm_head", "score_trajectory", "PRM_PROMPT_TEMPLATE"]

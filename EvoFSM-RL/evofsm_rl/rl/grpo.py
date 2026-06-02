"""GRPO (Group Relative Policy Optimization) for LoRA weight updates — Story 4.2.

This module implements the weight-side half of B4:

  * :class:`TrajectoryData` — one rollout packaged for GRPO, including
    the per-step replay paths produced by
    :class:`evofsm_rl.agent.rollout.Qwen3VLAgent` with
    ``collect_log_probs=True``.
  * :func:`compute_reward` — the scalar reward used as the fitness
    signal: task success plus an efficiency bonus.
  * :func:`compute_advantages` — GRPO's within-group mean subtraction:
    each trajectory's advantage is its reward minus the mean reward of
    trajectories that share the same ``(fsm_variant_id, task_name)``
    tuple, per the original §3.2 design.
  * :func:`grpo_step` — one AdamW weight update on the LoRA parameters,
    computed from a fresh forward pass on the replay data so gradients
    flow through the LoRA matrices.
  * :func:`cleanup_replay_data` — delete the replay ``*.pt`` files
    after the GRPO step has consumed them.

The whole pipeline is intentionally file-based: rollout writes per-step
tensors to disk in the agent's ``replay_dir``, GRPO reads them back at
update time, and the files are deleted once the update has used them.
This keeps the in-memory footprint of the evolution loop small even
across long runs, at the cost of ~10 MB of disk per step (Qwen3-VL's
pixel_values are the dominant contributor).

No emulator or live agent is needed at GRPO time — only the
peft-wrapped model, an AdamW optimizer over the LoRA parameters, and
the trajectory objects.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


logger = logging.getLogger(__name__)


# Special sentinel for ``ref_adapter_name`` in :func:`grpo_step`. When the
# caller passes this value, the KL anchor uses the **base model** (LoRA
# adapter disabled) as the reference policy, instead of a named LoRA
# adapter. Used by Phase 1 pretraining: there's no pre-existing π^pre to
# anchor against, so we anchor the freshly-attached LoRA back to base
# policy (the LoRA's starting point at iter 0 is BA=0 by init, ≡ base).
BASE_REF_SENTINEL = "_BASE_"


# ─────────────────────────────────────────────────────────────────────
# TrajectoryData
# ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class TrajectoryData:
    """One agent trajectory with everything GRPO needs.

    Attributes:
        task_name: AndroidWorld template id.
        seed: Task seed the episode ran with.
        fsm_variant_id: ID of the L_C variant that produced the rollout.
            Trajectories sharing the same ``(fsm_variant_id, task_name)``
            tuple form one "group" for within-group advantage computation.
        success: Final success signal (0.0 / 0.5 / 1.0).
        n_steps: Number of agent steps consumed.
        step_log_probs: Per-step ``log P(action | state)`` under the
            policy that rolled out the trajectory. Detached floats,
            only used for logging / debugging; the GRPO update itself
            recomputes these with gradient via ``replay_paths``.
        replay_paths: Absolute paths to the per-step ``*.pt`` files the
            agent wrote during rollout. GRPO loads these back and does
            a fresh forward pass so the LoRA parameters get a gradient.
        reward: Scalar reward — typically
            :func:`compute_reward(success, n_steps, step_budget)`.
    """

    task_name: str
    seed: int
    fsm_variant_id: str
    success: float
    n_steps: int
    step_log_probs: list[float]
    replay_paths: list[str]
    reward: float
    # PRM-style per-step rewards (B6/PPO+PRM). Length must equal len(replay_paths)
    # if provided. None = use sparse trajectory-level R only (default GRPO/PPO).
    per_step_rewards: list[float] | None = None


# ─────────────────────────────────────────────────────────────────────
# Reward + advantage
# ─────────────────────────────────────────────────────────────────────


def compute_reward(
    success: float,
    n_steps: int,
    step_budget: int,
    *,
    w_efficiency: float = 0.2,
) -> float:
    """``R = success + w_efficiency * max(0, 1 - n_steps / step_budget)``.

    The efficiency bonus caps at ``w_efficiency`` (when ``n_steps == 0``)
    and is zero once the agent exhausts or exceeds the budget.

    Args:
        success: Typically 0.0 / 0.5 / 1.0 from the task grader.
        n_steps: Steps consumed during the rollout.
        step_budget: Per-template step budget. If ``<= 0`` we clamp to
            1 to avoid division-by-zero; the bonus then degenerates to
            ``max(0, 1 - n_steps)`` which is ≤ 0 for any real episode,
            effectively disabling the bonus.
        w_efficiency: Weight on the efficiency bonus. Default ``0.2``
            (per §3.X of the algorithm design — small enough that
            success dominates).
    """
    budget = max(int(step_budget), 1)
    efficiency = max(0.0, 1.0 - (n_steps / budget))
    return float(success) + float(w_efficiency) * efficiency


def compute_advantages(
    trajectories: list[TrajectoryData],
) -> list[float]:
    """Within-group advantage = reward minus group-mean reward.

    Groups are defined by the ``(fsm_variant_id, task_name)`` tuple,
    matching the original §3.2 design where each iteration runs M
    FSMs × N rollouts on a single sampled task and the baseline
    V_i = mean of FSM i's N rollouts on that task. A group of size
    1 has no peer to compare against, so those trajectories receive
    advantage 0 (they contribute nothing to the GRPO update).

    Note: an earlier implementation grouped by ``fsm_variant_id``
    alone, which produced an across-task baseline that diverges
    from the design. Pre-2026-05-13 ``grpo_metrics.jsonl`` numbers
    are not directly comparable to post-fix runs.

    Returns advantages in the same order as ``trajectories``.
    """
    if not trajectories:
        return []

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, traj in enumerate(trajectories):
        groups[(traj.fsm_variant_id, traj.task_name)].append(i)

    advantages = [0.0] * len(trajectories)
    for indices in groups.values():
        if len(indices) <= 1:
            # No within-group comparison possible; advantage stays 0.0.
            continue
        rewards = [trajectories[i].reward for i in indices]
        mean_r = sum(rewards) / len(rewards)
        for i in indices:
            advantages[i] = float(trajectories[i].reward - mean_r)
    return advantages


# ─────────────────────────────────────────────────────────────────────
# Replay helpers (internal)
# ─────────────────────────────────────────────────────────────────────


def _load_replay_step(path: str, device: str) -> dict[str, Any]:
    """Load one step's replay ``.pt`` to a target device.

    Tensors move with ``.to(device)``; non-tensor fields (like
    ``input_len``) pass through unchanged.
    """
    raw = torch.load(path, map_location="cpu", weights_only=True)
    moved: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device) if v.numel() > 0 else v
        else:
            moved[k] = v
    return moved


def _compute_step_log_prob_with_grad(
    model: Any, step_data: dict[str, Any], device: str,
) -> torch.Tensor:
    """Forward pass on one step's replay, returning ``log P(action | state)``
    as a scalar tensor whose gradient flows into the LoRA parameters.

    Reconstructs the full (prompt + action) sequence the agent saw at
    rollout time, runs the model under its current (possibly-updated)
    LoRA weights, and sums per-token log-probs for the action positions.

    Every tensor field in ``step_data`` except the agent-only bookkeeping
    (``action_token_ids``, ``input_len``, ``step_log_prob``) is forwarded
    to the model so the kwargs contract stays consistent with whatever
    the processor emitted (``mm_token_type_ids``, ``image_grid_thw``,
    etc.). Hard-coding a subset here would re-introduce the bug where
    transformers versions that add new required fields silently break.
    """
    action_ids: torch.Tensor = step_data["action_token_ids"]
    input_len: int = int(step_data["input_len"])

    action_ids_1d = action_ids.flatten()
    action_len = int(action_ids_1d.shape[-1])

    input_ids: torch.Tensor = step_data["input_ids"]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.shape[-1] >= input_len + action_len:
        full_ids = input_ids
    else:
        full_ids = torch.cat(
            [input_ids[:, :input_len], action_ids_1d.unsqueeze(0)], dim=-1,
        )

    # Build forward kwargs from every tensor field the agent saved.
    # Skip the fields that are agent-side bookkeeping — everything else
    # (pixel_values, attention_mask, image_grid_thw, mm_token_type_ids,
    # and any future additions) passes through as the processor emitted.
    #
    # Per-token fields (shape ends in ``input_len``) need to grow to
    # match the full (prompt + action) sequence length. attention_mask
    # pads with 1s; everything else pads with 0s. 0 is correct for
    # ``mm_token_type_ids`` (0 = text, 1 = image placeholder) and a
    # safe default for future flag-style per-token fields.
    skip_keys = {"action_token_ids", "input_len", "step_log_prob", "input_ids"}
    full_len = int(full_ids.shape[-1])
    fwd_kwargs: dict[str, Any] = {"input_ids": full_ids.to(device)}
    for key, val in step_data.items():
        if key in skip_keys:
            continue
        if not isinstance(val, torch.Tensor) or val.numel() == 0:
            continue
        if val.dim() >= 2 and val.shape[-1] == input_len and full_len > input_len:
            pad_val = 1 if key == "attention_mask" else 0
            pad_shape = list(val.shape)
            pad_shape[-1] = full_len - input_len
            pad = torch.full(
                pad_shape, pad_val, dtype=val.dtype, device=val.device,
            )
            val = torch.cat([val, pad], dim=-1)
        fwd_kwargs[key] = val.to(device)

    outputs = model(**fwd_kwargs)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Log-prob of the action tokens. Autoregressive shift: logits[t]
    # predicts token[t+1], so the action-token slice starts at input_len-1.
    action_logits = logits[0, input_len - 1 : input_len - 1 + action_len, :]
    log_probs = torch.log_softmax(action_logits, dim=-1)
    token_log_probs = log_probs.gather(
        1, action_ids_1d.to(device).unsqueeze(1),
    ).squeeze(1)
    return token_log_probs.sum()


# ─────────────────────────────────────────────────────────────────────
# Public: GRPO update + replay cleanup
# ─────────────────────────────────────────────────────────────────────


def grpo_step(
    model: Any,
    optimizer: Any,
    trajectories: list[TrajectoryData],
    *,
    device: str = "cuda",
    max_grad_norm: float = 1.0,
    min_n_active: int = 1,
    kl_beta: float = 0.0,
    ref_adapter_name: str = "ref",
    kl_log_ratio_clip: float = 10.0,
) -> dict[str, float]:
    """One GRPO weight update using buffered trajectories.

    Pipeline:
      1. Compute within-group advantages. Zero-advantage trajectories
         (singletons or all-tied groups) are dropped.
      2. If fewer than ``min_n_active`` trajectories remain, skip the
         update entirely. ``min_n_active=1`` is the legacy behavior
         (only ``n_active == 0`` is rejected); set to 3+ to refuse
         outlier-dominated fires where one positive trajectory would
         carry the whole gradient.
      3. For each active trajectory, for each step, load the replay
         ``.pt`` from disk and run a forward pass under the current
         LoRA weights. Accumulate a scalar loss:
         ``loss_j = -advantage_j * sum_t log P(a_t | s_t)``.
      4. If ``kl_beta > 0``, also forward under the reference adapter
         (frozen π^pre, loaded via
         :func:`evofsm_rl.model.lora.load_ref_lora_adapter` under the
         name ``ref_adapter_name``) and add a KL-anchor term to each
         step's loss::

             step_loss = pg_loss + kl_beta * (log π_θ - log π^pre)

         The gradient of this added term is ``β × ∂ log π_θ / ∂θ``
         (the ref logprob is detached), which pulls π_θ back toward
         π^pre whenever it drifts. This is the standard RLHF/PPO
         pattern (OpenAI InstructGPT, Anthropic Claude RL) and is the
         load-bearing fix for Phase 3's "premature-termination"
         mode-collapse where π_θ learned to fire ``status: complete``
         at step 1 (see ``docs/results/`` once written).
      5. Average over active trajectories, backprop, clip LoRA
         gradients, step the optimizer.

    The ``optimizer`` is expected to iterate LoRA parameters only
    (``[p for p in model.parameters() if p.requires_grad]``). The base
    model weights stay frozen by construction. The reference adapter
    is loaded with ``is_trainable=False`` so its parameters do not
    receive gradients regardless of caller state.

    Returns a metrics dict for logging. When ``kl_beta > 0``, the
    dict additionally includes ``mean_kl`` (token-summed log-ratio
    averaged over active steps), useful for sanity-checking that the
    anchor is actually active.
    """
    advantages = compute_advantages(trajectories)
    active: list[tuple[TrajectoryData, float]] = [
        (traj, adv)
        for traj, adv in zip(trajectories, advantages)
        if abs(adv) > 1e-8 and traj.replay_paths
    ]

    mean_reward = (
        sum(t.reward for t in trajectories) / len(trajectories)
        if trajectories else 0.0
    )

    if len(active) < max(1, min_n_active):
        logger.info(
            "grpo_step: only %d active trajectories (need >= %d); "
            "optimizer not stepped.",
            len(active), max(1, min_n_active),
        )
        return {
            "loss": 0.0,
            "grad_norm": 0.0,
            "mean_advantage": 0.0,
            "advantage_std": 0.0,
            "advantage_abs_max": 0.0,
            "mean_reward": float(mean_reward),
            "n_trajectories": len(trajectories),
            "n_active": len(active),
            "mean_kl": 0.0,
            "skipped_min_n_active": int(len(active) < max(1, min_n_active) and len(active) > 0),
        }

    # Sanity-check the ref adapter is loaded when caller requested KL.
    # Done BEFORE touching the optimizer so a misconfigured call doesn't
    # leave any state mutated. peft stores adapter names in
    # ``model.peft_config`` (dict). The special sentinel
    # ``BASE_REF_SENTINEL`` bypasses this check: KL anchors to the base
    # policy via ``disable_adapter_layers()`` instead of a named adapter.
    kl_active = kl_beta > 0.0
    kl_ref_is_base = (ref_adapter_name == BASE_REF_SENTINEL)
    if kl_active and not kl_ref_is_base:
        cfg = getattr(model, "peft_config", None)
        if cfg is None or ref_adapter_name not in cfg:
            raise RuntimeError(
                f"grpo_step: kl_beta={kl_beta} > 0 but adapter "
                f"{ref_adapter_name!r} not found on model. Load it via "
                f"evofsm_rl.model.lora.load_ref_lora_adapter() first, "
                f"or pass ref_adapter_name=BASE_REF_SENTINEL to anchor "
                f"to the base policy."
            )

    optimizer.zero_grad()
    # Per-step backward. The math-equivalent batched formulation is
    #
    #   L = (1 / N_active) * sum_j (1/T_j) * sum_t [-adv_j * log P(a_t^j)]
    #
    # (T_j = number of replay steps in trajectory j). Per-trajectory
    # normalization by T_j keeps the loss magnitude bounded as episode
    # length grows. Without it, a 30-step trajectory produces ~30× the
    # gradient of a 1-step trajectory, and on long-horizon apps the
    # gradient norm hits 50–270 (observed in B4 v2 sweep:
    # traces/b4_evolution_v2/*/grpo_metrics.jsonl). With the default
    # max_grad_norm=1.0 clip those fires get their effective LR cut by
    # 50–270×, making updates erratic and largely random in magnitude.
    # Normalizing per T_j brings the typical grad_norm into O(1), so
    # the clip becomes a guard against true outliers rather than the
    # main signal-attenuation mechanism.
    #
    # By linearity the loss decomposes into independent per-step terms
    #
    #   L = sum_j sum_t (-adv_j / (N_active * T_j)) * log P(a_t^j),
    #
    # so we can call .backward() on each step's scalar loss and
    # accumulate gradients on the LoRA parameters exactly the same
    # way a single backward on the summed tensor would — but without
    # holding 60+ step-worth of activations simultaneously. With
    # 7K-token contexts on Qwen3-VL-8B the batched graph peaks at 70+
    # GB of activations on an 80 GB A100 and OOMs; per-step backward
    # keeps peak to ~one step's worth (~5-8 GB). Between steps we call
    # ``empty_cache`` so the fragmentation from allocator expansion
    # doesn't pile up either.
    #
    # HF gradient checkpointing is gated by ``self.training`` — it's a
    # no-op when the model is in eval mode. The B4 agent sets eval at
    # init time (dropout off during rollout), so we need to flip the
    # model into train mode for the backward pass and restore afterward.
    # Without this the ``gradient_checkpointing_enable(...)`` call from
    # run_b4_evolution.py silently has no effect and peak activation
    # memory stays at ~80 GB on long sequences.
    was_training = model.training
    model.train()
    total_loss_value = 0.0
    total_kl_value = 0.0
    n_steps_total = 0
    scale = 1.0 / len(active)

    try:
        for traj, adv in active:
            n_steps_traj = len(traj.replay_paths)
            if n_steps_traj == 0:
                continue
            traj_scale = scale / n_steps_traj
            for replay_path in traj.replay_paths:
                step_data = _load_replay_step(replay_path, device)

                # Forward 1: current LoRA (gradient flows here).
                step_lp = _compute_step_log_prob_with_grad(model, step_data, device)

                # Forward 2: reference policy (no_grad, no gradient).
                # Two modes:
                #   - kl_ref_is_base: disable the LoRA adapter entirely
                #     so the forward is computed under base policy. Used
                #     by Phase 1 pretraining (no prior π^pre exists).
                #   - otherwise: switch to the named ref LoRA adapter
                #     and forward under that. Used by Phase 3 (anchor
                #     to π^pre).
                # Both modes run on the SAME base model — no extra
                # ~16 GB Qwen3-VL-8B copy in memory.
                kl_term_value = 0.0
                if kl_active:
                    if kl_ref_is_base:
                        model.disable_adapter_layers()
                        try:
                            with torch.no_grad():
                                step_lp_ref = _compute_step_log_prob_with_grad(
                                    model, step_data, device,
                                )
                        finally:
                            model.enable_adapter_layers()
                    else:
                        model.set_adapter(ref_adapter_name)
                        with torch.no_grad():
                            step_lp_ref = _compute_step_log_prob_with_grad(
                                model, step_data, device,
                            )
                        model.set_adapter("default")
                    # Schulman k3 KL estimator (PPO-style):
                    #   kl = exp(log_ratio) - 1 - log_ratio,  log_ratio = log π_θ - log π_ref
                    # Properties (vs the simpler k1 estimator
                    # `log π_θ - log π_ref` which we used in 2026-05-15
                    # smoke and which had broken-direction gradient):
                    #   * Always ≥ 0 (true to KL's non-negativity)
                    #   * Gradient = (ratio - 1) × ∂log π_θ/∂θ, which
                    #     pulls log π_θ DOWN when curr > ref and pulls
                    #     it UP when curr < ref — i.e., always toward
                    #     π_ref. k1 instead has gradient ∂log π_θ/∂θ
                    #     unconditionally, which is "always decrease
                    #     log π_θ" (entropy maximization), wrong sign.
                    # Reference: Schulman 2020,
                    # "Approximating KL Divergence" (arxiv blog).
                    #
                    # log_ratio CLIPPING (added 2026-05-15 after smoke
                    # found β=0.02 fire 2 had log_ratio drift into the
                    # ~25 range, causing exp(25) ≈ 7e10 and a 16-billion
                    # KL value that drove grad_norm to 2e11). Clamping
                    # log_ratio to [-clip, +clip] before exp() bounds
                    # KL ≤ exp(clip) - 1 - clip. With clip=10:
                    #   max KL contribution per step ≈ 22 015
                    #   max kl_loss contribution ≈ β × 22015 (bounded)
                    # The clamp's gradient is zero outside [-clip, +clip],
                    # so extreme-outlier steps no longer get any KL
                    # pull-back (they also no longer destroy training).
                    # The remaining 99%+ of steps with |log_ratio| < clip
                    # are unaffected — clamp is identity in that range.
                    log_ratio = step_lp - step_lp_ref.detach()
                    if kl_log_ratio_clip > 0:
                        log_ratio = torch.clamp(
                            log_ratio,
                            min=-kl_log_ratio_clip,
                            max=kl_log_ratio_clip,
                        )
                    ratio = torch.exp(log_ratio)
                    kl_term = ratio - 1.0 - log_ratio  # tensor, ≥ 0
                    kl_loss = kl_beta * traj_scale * kl_term
                    pg_loss = -float(adv) * traj_scale * step_lp
                    step_loss = pg_loss + kl_loss
                    kl_term_value = float(kl_term.detach().item())
                else:
                    step_loss = -float(adv) * traj_scale * step_lp

                step_loss.backward()
                total_loss_value += float(step_loss.detach().item())
                total_kl_value += kl_term_value
                n_steps_total += 1
                # Drop references so autograd can free the step's graph.
                del step_data, step_lp, step_loss
                if torch.cuda.is_available() and device.startswith("cuda"):
                    torch.cuda.empty_cache()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm).item()
        )
        optimizer.step()
        optimizer.zero_grad()
    finally:
        # Always restore the caller's training state — rollouts downstream
        # expect eval mode (no dropout noise during action-selection).
        model.train(was_training)
        # And restore the default adapter in case anything in the loop
        # raised before set_adapter("default") was called. Skip for
        # base-anchor mode (no adapter switching happens there).
        if kl_active and not kl_ref_is_base:
            try:
                model.set_adapter("default")
            except Exception:  # pragma: no cover — defensive
                logger.exception("grpo_step: failed to restore default adapter")
        if kl_active and kl_ref_is_base:
            try:
                model.enable_adapter_layers()
            except Exception:  # pragma: no cover — defensive
                logger.exception("grpo_step: failed to re-enable adapter layers")

    adv_values = [a for _, a in active]
    mean_advantage = sum(adv_values) / len(adv_values)
    # advantage_std and advantage_abs_max are the load-bearing signal-
    # strength metrics; mean_advantage is constructively ~0 (GRPO
    # subtracts the group mean from rewards, so within any group the
    # advantages sum to 0 — and across groups they average to ~0 too).
    # Kept in the dict for back-compat with existing grpo_metrics.jsonl
    # consumers, but don't read it as a learning-signal indicator.
    variance = sum((a - mean_advantage) ** 2 for a in adv_values) / len(adv_values)
    adv_std = variance ** 0.5
    adv_abs_max = max(abs(a) for a in adv_values)
    return {
        "loss": total_loss_value,
        "grad_norm": grad_norm,
        "mean_advantage": float(mean_advantage),
        "advantage_std": float(adv_std),
        "advantage_abs_max": float(adv_abs_max),
        "mean_reward": float(mean_reward),
        "n_trajectories": len(trajectories),
        "n_active": len(active),
        "mean_kl": (
            float(total_kl_value / n_steps_total) if (kl_active and n_steps_total > 0) else 0.0
        ),
        "skipped_min_n_active": 0,
    }


def cleanup_replay_data(trajectories: list[TrajectoryData]) -> int:
    """Delete every ``replay_path`` referenced by the given trajectories.

    Idempotent. Missing files are skipped silently. Returns the number
    of files actually removed (useful as a log line, e.g.
    ``"freed 42 replay files (=%d MB)"``).

    After the files are unlinked, any ``episode_NNNN/`` parent directory
    that becomes empty is also removed — otherwise a long run leaks
    hundreds of empty dirs under ``replay/``. ``rmdir`` only succeeds on
    empty directories, so an episode whose files weren't fully consumed
    (rare — implies a bug in the caller) is left alone for inspection.
    """
    count = 0
    dirs_to_check: set[Path] = set()
    for traj in trajectories:
        for p in traj.replay_paths:
            path = Path(p)
            dirs_to_check.add(path.parent)
            if path.exists():
                try:
                    path.unlink()
                    count += 1
                except OSError as e:
                    logger.warning(
                        "cleanup_replay_data: could not remove %s: %s", p, e,
                    )
    for d in dirs_to_check:
        if not d.exists():
            continue
        try:
            if not any(d.iterdir()):
                d.rmdir()
        except OSError as e:
            logger.warning(
                "cleanup_replay_data: could not rmdir %s: %s", d, e,
            )
    return count


__all__ = [
    "BASE_REF_SENTINEL",
    "TrajectoryData",
    "cleanup_replay_data",
    "compute_advantages",
    "compute_reward",
    "grpo_step",
]

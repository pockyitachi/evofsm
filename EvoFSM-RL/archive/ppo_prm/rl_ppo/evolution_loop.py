"""PPO version of the L_C evolution loop.

Parallel to :func:`evofsm_rl.fsm.evolution.run_l_c_evolution` (which
uses GRPO). Same outer loop — select / rollout / mutate / log — but
calls a :class:`evofsm_rl.rl_ppo.trainer.PPOTrainer` instead of
``grpo_step`` when the trajectory buffer fills up.

We deliberately duplicate the loop body rather than refactoring
``run_evolution`` to take a pluggable trainer. The user's design
decision (CLAUDE.md PPO ticket) is that PPO must sit alongside GRPO
without touching the existing module — touching ``fsm/evolution.py``
to add a strategy hook would violate that.

What we reuse from ``fsm/evolution.py``:
  * :class:`EvolutionConfig`, :class:`IterationResult`,
    :class:`TaskSampler`, :class:`Population` setup, mutation logic,
    L_C-snapshot bookkeeping — all unchanged.

What we replace:
  * GRPO optimizer construction + ``grpo_step`` + replay cleanup →
    :class:`PPOTrainer` add_trajectory / step / clear_buffer.
  * GRPO metrics JSONL → ``ppo_metrics.jsonl``.
  * Per-iteration log line — switched to surface PPO-side metrics.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

# Re-use everything that's not GRPO-specific from the base loop.
from evofsm_rl.fsm.evolution import (
    EvolutionConfig,
    IterationResult,
    TaskSampler,
    l_c_to_seed_fsm,
)
from evofsm_rl.fsm.mutation import (
    CompressedTrajectory,
    MutationError,
    compress_trajectory_for_reflection,
    mutate_fsm,
)
from evofsm_rl.fsm.population import Population
from evofsm_rl.fsm.schema import FSM, Layer2
from evofsm_rl.model.lora import count_trainable_params, save_lora_checkpoint
from evofsm_rl.rl_ppo.trainer import PPOTrainer
from evofsm_rl.rl_ppo.value_head import (
    LinearValueHead,
    attach_value_head,
    save_value_head,
)


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# PPO-specific config extension
# ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class PPOEvolutionConfig:
    """PPO-side hyperparameters layered on top of :class:`EvolutionConfig`.

    The base :class:`EvolutionConfig` carries the FSM-evolution knobs
    (M, N, mutation cadence, etc.) and the LoRA knobs. PPO adds:
      * value head learning rate (LoRA lr is in the base config)
      * value head path (loaded into the head before training)
      * PPO clip ε
      * GAE λ + γ + optional λ schedule
      * value loss coefficient
    """

    value_lr: float = 1e-4
    clip_eps: float = 0.2
    value_loss_coef: float = 0.5
    gae_gamma: float = 0.99
    gae_lambda: float = 1.0
    gae_lambda_after_iter: int | None = None
    gae_lambda_after_value: float = 0.95
    value_head_path: Path | None = None
    value_head_save_dir: Path | None = None  # written every K iters + at end


# ─────────────────────────────────────────────────────────────────────
# Loop body
# ─────────────────────────────────────────────────────────────────────


def run_evolution_ppo(
    root_fsm: FSM,
    task_templates: list[str],
    rollout_fn: Any,            # evofsm_rl.fsm.evolution.RolloutFn
    config: EvolutionConfig,
    ppo_config: PPOEvolutionConfig,
    *,
    output_dir: Path,
    app_category: str = "",
    resume: bool = True,
    layer2_only: bool = False,
    l_c_snapshot: bool = False,
    model: Any = None,
) -> Population:
    """PPO version of :func:`evofsm_rl.fsm.evolution.run_evolution`.

    See that function's docstring for the FSM-evolution semantics.
    Differences:
      * ``config.enable_lora`` must be True (PPO updates the LoRA).
      * ``model`` must be a peft-wrapped model (same as GRPO path).
      * A separate value head is constructed and trained alongside.
      * GRPO replaced by :class:`PPOTrainer`.
      * Logs metrics to ``ppo_metrics.jsonl`` (not ``grpo_metrics.jsonl``).
    """
    if not task_templates:
        raise ValueError("run_evolution_ppo: task_templates must be non-empty")
    if not config.enable_lora:
        raise ValueError(
            "run_evolution_ppo: config.enable_lora must be True (PPO is "
            "the weight-update path)."
        )
    if model is None:
        raise ValueError(
            "run_evolution_ppo: model is required (PPO needs a peft-"
            "wrapped policy to backward through)."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / "population.json"
    log_path = output_dir / "iterations.jsonl"
    ppo_log_path = output_dir / "ppo_metrics.jsonl"

    # ── Resume or fresh start ────────────────────────────────────
    if resume and checkpoint_path.exists():
        logger.info("Resuming evolution from checkpoint: %s", checkpoint_path)
        population = Population.from_json(json.loads(checkpoint_path.read_text()))
        completed = sum(1 for _ in log_path.open()) if log_path.exists() else 0
        start_iter = completed + 1
    else:
        population = Population(
            root_fsm=root_fsm,
            window_size=config.window_size,
            selection_lambda=config.selection_lambda,
            selection_temperature=config.selection_temperature,
            delta_sigma=config.delta_sigma,
        )
        start_iter = 1

    task_sampler = TaskSampler(
        task_templates,
        mode=config.task_sample_mode,
        rng_seed=config.seed_base,
    )
    for _ in range(start_iter - 1):
        task_sampler.sample()

    select_rng = random.Random(config.seed_base)
    for _ in range((start_iter - 1) * max(1, config.m_select)):
        select_rng.random()

    snapshot_counter = 0
    if l_c_snapshot:
        existing = list(output_dir.glob("l_c_v*_iter*.json"))
        for p in existing:
            try:
                n = int(p.name.split("_v", 1)[1].split("_iter", 1)[0])
                snapshot_counter = max(snapshot_counter, n)
            except (IndexError, ValueError):
                continue

    logger.info(
        "PPO evolution config: app=%s, category=%r, tasks=%d, "
        "iterations=%d (start=%d), M=%d, N=%d, layer2_only=%s",
        root_fsm.app, app_category, len(task_templates),
        config.n_iterations, start_iter,
        config.m_select, config.n_rollouts, layer2_only,
    )

    # ── PPO setup ────────────────────────────────────────────────
    value_head = attach_value_head(model, config.lora_device)
    trainer = PPOTrainer(
        model=model,
        value_head=value_head,
        lora_lr=config.lora_lr,
        value_lr=ppo_config.value_lr,
        value_head_path=ppo_config.value_head_path,
        clip_eps=ppo_config.clip_eps,
        value_loss_coef=ppo_config.value_loss_coef,
        gae_gamma=ppo_config.gae_gamma,
        gae_lambda=ppo_config.gae_lambda,
        gae_lambda_after_iter=ppo_config.gae_lambda_after_iter,
        gae_lambda_after_value=ppo_config.gae_lambda_after_value,
        max_grad_norm=config.lora_max_grad_norm,
        min_n_active=config.lora_min_n_active,
        device=config.lora_device,
        log_path=ppo_log_path,
    )

    counts = count_trainable_params(model)
    logger.info(
        "PPO enabled: LoRA trainable=%d / total=%d (%.3f%%), "
        "lora_lr=%g value_lr=%g clip_eps=%g lambda=%g lambda_switch=%s",
        counts["trainable"], counts["total"], counts["percent"],
        config.lora_lr, ppo_config.value_lr,
        ppo_config.clip_eps, ppo_config.gae_lambda,
        ppo_config.gae_lambda_after_iter,
    )
    lora_ckpt_dir = output_dir / "lora_checkpoints"
    lora_ckpt_dir.mkdir(exist_ok=True)
    value_head_dir = ppo_config.value_head_save_dir or (output_dir / "value_head_checkpoints")
    value_head_dir.mkdir(exist_ok=True, parents=True)

    # ── Main loop ────────────────────────────────────────────────
    for iteration in range(start_iter, config.n_iterations + 1):
        t0 = time.monotonic()
        task_name = task_sampler.sample()
        seed = config.seed_base + iteration
        logger.info(
            "=== Iteration %d/%d: task=%s seed=%d ===",
            iteration, config.n_iterations, task_name, seed,
        )

        m = min(config.m_select, len(population.window))
        selected = population.select(m, rng=select_rng)
        selected_ids = [v.id for v in selected]
        logger.info("Selected variants: %s", selected_ids)

        rewards: dict[str, float] = {}
        episode_dirs: dict[str, list[Path]] = {}
        for variant in selected:
            if layer2_only:
                fsm_text = variant.fsm.layer2.to_prompt_text(
                    category=variant.fsm.layer1.category,
                )
            else:
                fsm_text = variant.fsm.to_prompt_text()
            per_variant: list[float] = []
            dirs: list[Path] = []
            for j in range(config.n_rollouts):
                rollout_seed = seed * 100 + j
                result = rollout_fn(
                    fsm_prompt_text=fsm_text,
                    task_name=task_name,
                    seed=rollout_seed,
                )
                per_variant.append(result.reward)
                if result.episode_dir is not None:
                    dirs.append(Path(result.episode_dir))
                if result.trajectory_data is not None:
                    trainer.add_trajectory(result.trajectory_data)
            avg = sum(per_variant) / max(1, len(per_variant))
            rewards[variant.id] = avg
            episode_dirs[variant.id] = dirs
            logger.info(
                "  %s: avg reward=%.2f over %d rollout(s)",
                variant.id, avg, len(per_variant),
            )

        population.update_ratings(
            variant_ids=selected_ids,
            rewards=[rewards[v] for v in selected_ids],
        )

        best_id = max(selected_ids, key=lambda v: rewards[v])
        best_reward = rewards[best_id]
        best_variant = population.get(best_id)

        # ── Mutation (same as B3/B4 paths) ──
        mutation_success = False
        child_id: str | None = None
        error_msg: str | None = None

        force_periodic = (iteration % config.mutation_every_n_iters == 0)
        should_mutate = config.mutation_enabled and ((best_reward > 0) or force_periodic)
        if should_mutate:
            best_dirs = episode_dirs.get(best_id, [])
            trajectories: list[CompressedTrajectory] = []
            for ep_dir in best_dirs:
                try:
                    trajectories.append(
                        compress_trajectory_for_reflection(ep_dir)
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to compress trajectory at %s: %s", ep_dir, e,
                    )

            if not trajectories:
                error_msg = "no trajectory dirs available for reflection"
                logger.info(
                    "Skipping mutation at iter %d: %s", iteration, error_msg,
                )
            else:
                try:
                    new_fsm, diff, _reflection = mutate_fsm(
                        best_variant.fsm,
                        trajectories,
                        task_category=app_category,
                        layer2_only=layer2_only,
                        bootstrap=config.bootstrap_fsm,
                        model=config.mutation_model,
                    )
                    child = population.add_child(
                        parent_id=best_id,
                        child_fsm=new_fsm,
                        iteration=iteration,
                        metadata={
                            "reflection_summary": diff.reflection_summary,
                            "ops_count": len(diff.ops),
                            "layer_tag": diff.layer_tag,
                            "task_name": task_name,
                            "parent_best_reward": best_reward,
                        },
                    )
                    child_id = child.id
                    mutation_success = True
                    logger.info(
                        "Mutation OK: %s -> %s (%d ops, layer_tag=%s)",
                        best_id, child_id, len(diff.ops), diff.layer_tag,
                    )

                    if l_c_snapshot:
                        snapshot_counter += 1
                        snap_path = (
                            output_dir
                            / f"l_c_v{snapshot_counter}_iter{iteration}.json"
                        )
                        snap_path.write_text(
                            json.dumps(child.fsm.to_json(), indent=2)
                        )
                        champ_path = output_dir / "l_c_champion.json"
                        champ_path.write_text(
                            json.dumps(population.champion.fsm.to_json(), indent=2)
                        )
                except MutationError as e:
                    error_msg = f"MutationError: {e}"
                    logger.warning(
                        "MutationError at iter %d: %s", iteration, e,
                    )
                    if not config.skip_mutation_on_error:
                        raise
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}"
                    logger.warning(
                        "Unexpected mutation error at iter %d: %s",
                        iteration, e,
                    )
                    if not config.skip_mutation_on_error:
                        raise

        # ── Log iteration ──
        champion = population.champion
        iter_result = IterationResult(
            iteration=iteration,
            task_name=task_name,
            seed=seed,
            selected_ids=selected_ids,
            rewards=rewards,
            best_id=best_id,
            best_reward=best_reward,
            mutation_success=mutation_success,
            child_id=child_id,
            wall_seconds=time.monotonic() - t0,
            error=error_msg,
            champion_id=champion.id,
            champion_mu=float(champion.rating.mu),
            champion_sigma=float(champion.rating.sigma),
        )
        with log_path.open("a") as fh:
            fh.write(json.dumps(dataclasses.asdict(iter_result)) + "\n")

        if iteration % config.checkpoint_every == 0:
            checkpoint_path.write_text(
                json.dumps(population.to_json(), indent=2)
            )

        # ── PPO update when buffer fills ──
        if trainer.buffer_size() >= config.lora_update_every_k:
            metrics = trainer.step(iteration=iteration)
            logger.info(
                "PPO step @ iter %d: policy_loss=%.4f value_loss=%.4f "
                "grad_norm=%.3f adv_std=%.3f adv_max=%.3f "
                "mean_reward=%.3f lambda=%.2f n_traj=%d n_active=%d",
                iteration, metrics["policy_loss"], metrics["value_loss"],
                metrics["grad_norm"], metrics["advantage_std"],
                metrics["advantage_max"], metrics["mean_reward"],
                metrics["gae_lambda"],
                metrics["n_trajectories"], metrics["n_active"],
            )
            trainer.clear_buffer(cleanup_replay=True)

        # ── Periodic LoRA + value-head adapter checkpoint ──
        if iteration % config.lora_checkpoint_every_k == 0:
            ckpt = lora_ckpt_dir / f"iter_{iteration:04d}"
            save_lora_checkpoint(model, ckpt)
            logger.info("Saved LoRA adapter to %s", ckpt)
            vh_ckpt = value_head_dir / f"iter_{iteration:04d}.pt"
            save_value_head(value_head, vh_ckpt)
            logger.info("Saved value head to %s", vh_ckpt)

        # ── Memory hygiene ──
        import gc
        import torch as _torch_mem
        gc.collect()
        if _torch_mem.cuda.is_available():
            _torch_mem.cuda.empty_cache()

        logger.info(
            "Iter %d done: best=%s (R=%.2f) champion=%s (mu=%.2f) "
            "pop=%d mutation=%s %.1fs",
            iteration, best_id, best_reward,
            population.champion.id, population.champion.rating.mu,
            population.size,
            "ok" if mutation_success else ("skip" if not should_mutate else "err"),
            time.monotonic() - t0,
        )

    # Final population checkpoint.
    checkpoint_path.write_text(json.dumps(population.to_json(), indent=2))

    # Drain remaining buffer + final adapter / value-head checkpoint.
    if trainer.buffer_size() > 0:
        metrics = trainer.step(iteration=config.n_iterations)
        logger.info(
            "PPO step (final drain): policy_loss=%.4f value_loss=%.4f "
            "grad_norm=%.3f n_traj=%d n_active=%d",
            metrics["policy_loss"], metrics["value_loss"],
            metrics["grad_norm"],
            metrics["n_trajectories"], metrics["n_active"],
        )
        trainer.clear_buffer(cleanup_replay=True)
    final_ckpt = lora_ckpt_dir / "final"
    save_lora_checkpoint(model, final_ckpt)
    logger.info("Saved final LoRA adapter to %s", final_ckpt)
    final_vh = value_head_dir / "final.pt"
    save_value_head(value_head, final_vh)
    logger.info("Saved final value head to %s", final_vh)

    logger.info(
        "PPO evolution complete. Champion: %s (mu=%.2f, sigma=%.2f, pop=%d)",
        population.champion.id,
        population.champion.rating.mu,
        population.champion.rating.sigma,
        population.size,
    )
    return population


def run_l_c_evolution_ppo(
    initial_l_c: Layer2,
    app_name: str,
    app_category: str,
    task_templates: list[str],
    rollout_fn: Any,
    config: EvolutionConfig,
    ppo_config: PPOEvolutionConfig,
    *,
    output_dir: Path,
    resume: bool = True,
    model: Any = None,
) -> Population:
    """PPO version of :func:`evofsm_rl.fsm.evolution.run_l_c_evolution`.

    Same semantics: evolves L_C (Layer 2) on one app's T_adapt, wraps
    the loop into :func:`run_evolution_ppo` with ``layer2_only=True``
    and ``l_c_snapshot=True``.
    """
    if not app_category:
        raise ValueError(
            "run_l_c_evolution_ppo: app_category must be non-empty."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_fsm = l_c_to_seed_fsm(initial_l_c, app_name, app_category)

    initial_path = output_dir / "l_c_v0_initial.json"
    if not initial_path.exists():
        initial_path.write_text(json.dumps(seed_fsm.to_json(), indent=2))
        logger.info("Wrote initial L_C snapshot: %s", initial_path)

    return run_evolution_ppo(
        root_fsm=seed_fsm,
        task_templates=task_templates,
        rollout_fn=rollout_fn,
        config=config,
        ppo_config=ppo_config,
        output_dir=output_dir,
        app_category=app_category,
        resume=resume,
        layer2_only=True,
        l_c_snapshot=True,
        model=model,
    )


__all__ = [
    "PPOEvolutionConfig",
    "run_evolution_ppo",
    "run_l_c_evolution_ppo",
]

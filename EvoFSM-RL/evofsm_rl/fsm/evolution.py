"""Evolution loop orchestrator for B3 (Self-Evolving FSM, frozen weights).

Implements the outer loop of ``plan/algorithm_design.md`` §3.2 minus the
RL weight update (that's B4 / Epic 4). Each iteration:

  1. Sample one task template from the app's task set.
  2. Select M FSM variants from the population window.
  3. Run N rollouts per variant on that task (all variants see the same
     emulator seed so comparisons are paired).
  4. Feed rewards into the population's TrueSkill ratings.
  5. If the best variant this round scored > 0 (or every third
     iteration, to force exploration from failure), compress its recent
     trajectories and call :func:`mutate_fsm` → add child to population.
  6. Checkpoint the population to disk so the run is resumable.

The loop is **decoupled from hardware**: it takes a ``rollout_fn``
callable via the :class:`RolloutFn` protocol, which is whatever the
caller wants — a real emulator+agent closure (for production runs), a
mock that hits pre-recorded traces (for tests), a dry-run stub that
flips a random coin (for CLI smoke testing). The loop itself has no
imports of the agent, model loader, or emulator harness.

No side effects beyond ``output_dir``: a JSON population checkpoint
(``population.json``) overwritten after every ``checkpoint_every``
iterations, an append-only JSONL of per-iteration results
(``iterations.jsonl``), and whatever episode traces the rollout function
chooses to write under ``output_dir / "episodes"``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Protocol

from evofsm_rl.fsm.mutation import (
    CompressedTrajectory,
    MutationError,
    compress_trajectory_for_reflection,
    mutate_fsm,
)
from evofsm_rl.fsm.population import Population
from evofsm_rl.fsm.schema import FSM, Layer1, Layer2
from evofsm_rl.model.lora import count_trainable_params, save_lora_checkpoint
from evofsm_rl.rl.grpo import cleanup_replay_data, grpo_step


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Rollout interface
# ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class RolloutResult:
    """Outcome of one agent episode.

    Attributes:
        task_name: Template id (e.g. ``"MarkorCreateNote"``).
        seed: Task seed passed through to the emulator.
        reward: Final scalar reward, typically 0.0 or 1.0 from the task's
            built-in rule-based checker.
        n_steps: Agent steps consumed (success or timeout).
        wall_seconds: Episode wall-clock time.
        episode_dir: Directory the caller persisted the trace into, if
            any. The evolution loop feeds this to
            :func:`compress_trajectory_for_reflection` when mutating.
            Leave ``None`` for rollouts that don't persist (e.g.
            dry-run, smoke tests).
        error: Non-null iff the rollout crashed (infra error, OOM, etc.).
            A failed episode can still have ``reward=0.0`` without an
            ``error`` — the distinction is "agent tried and failed" vs.
            "we couldn't even run the agent".
    """
    task_name: str
    seed: int
    reward: float
    n_steps: int
    wall_seconds: float
    episode_dir: Path | None = None
    error: str | None = None
    trajectory_data: Any | None = None


class RolloutFn(Protocol):
    """The callable signature expected by :func:`run_evolution`.

    Implementations own emulator/model state; the loop only knows how
    to pass the FSM prompt text, a task name, and a seed.
    """

    def __call__(
        self,
        fsm_prompt_text: str,
        task_name: str,
        seed: int,
    ) -> RolloutResult:
        ...


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class EvolutionConfig:
    """Hyperparameters for the B3 evolution loop.

    Defaults encode the values from §3.2 / §6.3 / §9. Only
    ``n_iterations`` typically varies across runs.
    """

    n_iterations: int = 30
    m_select: int = 2
    n_rollouts: int = 1
    task_sample_mode: str = "random"          # "random" | "round_robin"
    mutation_model: str = "claude-opus-4-7"
    checkpoint_every: int = 1
    seed_base: int = 100
    skip_mutation_on_error: bool = True
    mutation_every_n_iters: int = 3           # forced mutation cadence when best_reward=0
    mutation_enabled: bool = True             # when False, both signal-driven and periodic
                                              # mutation are skipped entirely; used for the
                                              # Tier-C B4 path where no L_C anchor exists
                                              # and bootstrap is deliberately out of scope.

    # Population hyperparams (pass-through to the Population constructor).
    window_size: int = 15
    selection_lambda: float = 1.0
    selection_temperature: float = 1.0
    delta_sigma: float = 1.5

    # B4 / Epic 4 — LoRA + GRPO. All guarded by ``enable_lora=False`` so
    # the default config is byte-identical to B3. The runner (Story 4.4)
    # is responsible for calling ``attach_lora`` with these hyperparams
    # before handing the wrapped model to :func:`run_evolution`.
    enable_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    lora_lr: float = 1e-5
    lora_update_every_k: int = 8
    lora_max_grad_norm: float = 1.0
    lora_checkpoint_every_k: int = 4
    lora_device: str = "cuda"
    # GRPO stability knobs (added 2026-05-15 to fix Phase 3 mode-collapse).
    # ``lora_min_n_active`` rejects fires below threshold; ``lora_kl_beta``
    # adds a KL anchor to a frozen reference adapter (set
    # ``lora_ref_adapter_name`` accordingly and load it on the model
    # before calling :func:`run_evolution`).
    lora_min_n_active: int = 1
    lora_kl_beta: float = 0.0
    lora_ref_adapter_name: str = "ref"
    lora_kl_log_ratio_clip: float = 10.0
    # Tier-C FSM bootstrap (CLAUDE.md 2026-05-13 option b). When True,
    # Claude Opus mutations are framed as "cold-start synthesize" instead
    # of "incremental diff", so they can populate an initially-empty L_C
    # from target-app trajectories alone. Used for Tier-C apps that have
    # no source-pool category match. Requires ``mutation_enabled=True``
    # and an empty initial L_C in the population.
    bootstrap_fsm: bool = False


# ─────────────────────────────────────────────────────────────────────
# Task sampler
# ─────────────────────────────────────────────────────────────────────


class TaskSampler:
    """Samples one task template per iteration.

    Modes:
      * ``"random"``: i.i.d. uniform-with-replacement, seeded for
        determinism across resumes.
      * ``"round_robin"``: cycle through the task list in declared order;
        wrap around on overflow.
    """

    def __init__(
        self,
        tasks: list[str],
        mode: str = "random",
        rng_seed: int = 42,
    ):
        if not tasks:
            raise ValueError("TaskSampler: tasks list must be non-empty")
        if mode not in ("random", "round_robin"):
            raise ValueError(
                f"TaskSampler: mode must be 'random' or 'round_robin', got {mode!r}"
            )
        self._tasks = list(tasks)
        self._mode = mode
        self._rng = random.Random(rng_seed)
        self._idx = 0

    def sample(self) -> str:
        if self._mode == "round_robin":
            task = self._tasks[self._idx % len(self._tasks)]
            self._idx += 1
            return task
        return self._rng.choice(self._tasks)


# ─────────────────────────────────────────────────────────────────────
# Per-iteration log record
# ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class IterationResult:
    """One row in ``iterations.jsonl``.

    Serializes cleanly via :func:`dataclasses.asdict` since all fields
    are primitives or JSON-friendly containers (no ``Path`` escapes).
    """

    iteration: int
    task_name: str
    seed: int
    selected_ids: list[str]
    rewards: dict[str, float]
    best_id: str
    best_reward: float
    mutation_success: bool
    child_id: str | None
    wall_seconds: float
    error: str | None = None
    # Champion snapshot at end of iteration — used by the convergence
    # plot utility (scripts/plot_convergence.py) and any downstream
    # offline analysis. Defaulted so older JSONL rows remain readable.
    champion_id: str | None = None
    champion_mu: float | None = None
    champion_sigma: float | None = None


# ─────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────


def l_c_to_seed_fsm(
    l_c: Layer2, app_name: str, category: str,
) -> FSM:
    """Wrap an ``L_C`` (Layer 2) as a minimal FSM for the population.

    The resulting FSM's Layer 1 is structurally valid but empty
    (app + category set; no states / transitions / strategies /
    dead_ends). Layer 2 is the passed-in ``L_C`` by reference — callers
    who plan to keep mutating the original should pass a ``deepcopy``.

    Used by :func:`run_l_c_evolution` to seed a population whose
    variants carry nothing except per-iteration LAYER-2 refinements.
    """
    return FSM(
        app=app_name,
        layer1=Layer1(
            app=app_name,
            category=category,
            states=[],
            transitions=[],
            strategies=[],
            dead_ends=[],
        ),
        layer2=l_c,
        metadata={"source": "l_c_seed", "category": category},
    )


def run_evolution(
    root_fsm: FSM,
    task_templates: list[str],
    rollout_fn: RolloutFn,
    config: EvolutionConfig,
    *,
    output_dir: Path,
    app_category: str = "",
    resume: bool = True,
    layer2_only: bool = False,
    l_c_snapshot: bool = False,
    model: Any = None,
) -> Population:
    """Run the B3 evolution loop for one target app.

    Args:
        root_fsm: Starting FSM (typically from
            ``artifacts/static_fsms/{app}.json``; can also be a
            category-seeded shell).
        task_templates: Task template names this app supports.
            Non-empty. Used by the :class:`TaskSampler` each iteration.
        rollout_fn: Callable conforming to :class:`RolloutFn`. The
            evolution loop never touches an emulator or model directly.
        config: Hyperparameters (:class:`EvolutionConfig`).
        output_dir: Writable directory. Populated with:

            * ``population.json`` — latest population state, overwritten
              every ``checkpoint_every`` iterations.
            * ``iterations.jsonl`` — one line per iteration, append-only.
            * ``episodes/`` — subdirectory passed through to
              ``rollout_fn`` for per-episode traces (its contents are
              the rollout function's concern).
        app_category: Play Store category string for layered mutation
            reflection. Empty string triggers LAYER-1-only mutation
            (correct for Tier-C apps whose category has no matching L_C).
        resume: If ``True`` and ``{output_dir}/population.json`` exists,
            pick up from that checkpoint and advance the task sampler
            past the already-completed iterations. If ``False`` (or no
            checkpoint), start from a fresh single-variant population.
        layer2_only: When True, inject only ``variant.fsm.layer2.to_prompt_text``
            into the rollout prompt (matches B2 injection shape) and pass
            ``layer2_only=True`` through to :func:`mutate_fsm` so
            mutations are constrained to LAYER-2 category edits. Used by
            :func:`run_l_c_evolution`; typically leave False for the
            full-FSM evolution path.
        l_c_snapshot: When True, after every successful mutation write
            an ``l_c_v{N}_iter{I}.json`` snapshot of the new child under
            ``output_dir`` and refresh ``l_c_champion.json`` to the
            current champion's FSM. Also used by
            :func:`run_l_c_evolution`.
        model: Required only when ``config.enable_lora=True``. A peft-
            wrapped base model (already attached via
            :func:`evofsm_rl.model.lora.attach_lora` by the caller). The
            loop creates an :class:`torch.optim.AdamW` over the model's
            trainable parameters and runs a GRPO update every
            ``config.lora_update_every_k`` iterations, then checkpoints
            the adapter every ``config.lora_checkpoint_every_k``
            iterations under ``{output_dir}/lora_checkpoints/``. When
            ``enable_lora=False`` (default, B3 path), ``model`` is
            ignored.

    Returns:
        The final :class:`Population` (the champion is whatever reached
        the top of ``population.window`` by the last iteration).

    Raises:
        :class:`MutationError` is caught per-iteration and logged;
        whether it propagates is controlled by
        ``config.skip_mutation_on_error``. Rollout exceptions are not
        caught — a broken emulator should abort the run.
    """
    if not task_templates:
        raise ValueError("run_evolution: task_templates must be non-empty")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / "population.json"
    log_path = output_dir / "iterations.jsonl"

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
    # Advance sampler so resumed runs don't repeat the already-sampled
    # sequence.
    for _ in range(start_iter - 1):
        task_sampler.sample()

    # Separate RNG for population.select — seeded the same way for
    # reproducibility but advanced past resumed iterations.
    select_rng = random.Random(config.seed_base)
    # Population.select uses rng.random() once per variant picked, so
    # advancing by (start_iter - 1) * m_select is the right draw count.
    # When the window has fewer variants than m_select, actual draws are
    # smaller — this is a mild approximation of the pre-resume RNG state
    # but keeps determinism good enough for research reproducibility.
    for _ in range((start_iter - 1) * max(1, config.m_select)):
        select_rng.random()

    # Track the highest snapshot version already on disk so a resumed
    # run keeps monotonically numbering new L_C snapshots.
    snapshot_counter = 0
    if l_c_snapshot:
        existing = list(output_dir.glob("l_c_v*_iter*.json"))
        for p in existing:
            # Filename format: l_c_v{N}_iter{I}.json
            try:
                n = int(p.name.split("_v", 1)[1].split("_iter", 1)[0])
                snapshot_counter = max(snapshot_counter, n)
            except (IndexError, ValueError):
                continue

    logger.info(
        "Evolution config: app=%s, category=%r, tasks=%d, "
        "iterations=%d (start=%d), M=%d, N=%d, layer2_only=%s, l_c_snapshot=%s",
        root_fsm.app, app_category, len(task_templates),
        config.n_iterations, start_iter,
        config.m_select, config.n_rollouts,
        layer2_only, l_c_snapshot,
    )

    # ── B4: LoRA + GRPO setup (only when enable_lora=True) ──────
    # The optimizer, trajectory buffer, and checkpoint directory live
    # for the lifetime of this call. On resume the optimizer state is
    # reset — acceptable because GRPO's update is a single step and the
    # LoRA adapter itself is saved to disk every K iterations.
    optimizer: Any = None
    trajectory_buffer: list[Any] = []
    lora_ckpt_dir: Path | None = None
    grpo_log_path = output_dir / "grpo_metrics.jsonl"
    if config.enable_lora:
        if model is None:
            raise ValueError(
                "run_evolution: enable_lora=True requires a peft-wrapped "
                "model via the model= kwarg. Call attach_lora(base_model) "
                "in the runner and pass the result."
            )
        import torch

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError(
                "run_evolution: enable_lora=True but the provided model has "
                "no trainable parameters. Ensure attach_lora() was called "
                "before handing the model to run_evolution."
            )
        optimizer = torch.optim.AdamW(trainable_params, lr=config.lora_lr)
        counts = count_trainable_params(model)
        logger.info(
            "B4 enabled: LoRA trainable=%d / total=%d (%.3f%%), "
            "lr=%g, update_every_k=%d, checkpoint_every_k=%d",
            counts["trainable"], counts["total"], counts["percent"],
            config.lora_lr, config.lora_update_every_k,
            config.lora_checkpoint_every_k,
        )
        lora_ckpt_dir = output_dir / "lora_checkpoints"
        lora_ckpt_dir.mkdir(exist_ok=True)

    for iteration in range(start_iter, config.n_iterations + 1):
        t0 = time.monotonic()
        task_name = task_sampler.sample()
        seed = config.seed_base + iteration
        logger.info(
            "=== Iteration %d/%d: task=%s seed=%d ===",
            iteration, config.n_iterations, task_name, seed,
        )

        # ── Select ──
        m = min(config.m_select, len(population.window))
        selected = population.select(m, rng=select_rng)
        selected_ids = [v.id for v in selected]
        logger.info("Selected variants: %s", selected_ids)

        # ── Rollout ──
        rewards: dict[str, float] = {}
        episode_dirs: dict[str, list[Path]] = {}
        for variant in selected:
            # In L_C-evolution mode, inject only the Layer-2 block (byte-
            # compatible with B2's L_C injection). Otherwise inject the
            # full two-layer FSM text.
            if layer2_only:
                fsm_text = variant.fsm.layer2.to_prompt_text(
                    category=variant.fsm.layer1.category,
                )
            else:
                fsm_text = variant.fsm.to_prompt_text()
            per_variant: list[float] = []
            dirs: list[Path] = []
            for j in range(config.n_rollouts):
                # Paired seeds: every variant sees the same rollout_seed so
                # the only source of variance is the FSM content.
                rollout_seed = seed * 100 + j
                result = rollout_fn(
                    fsm_prompt_text=fsm_text,
                    task_name=task_name,
                    seed=rollout_seed,
                )
                per_variant.append(result.reward)
                if result.episode_dir is not None:
                    dirs.append(Path(result.episode_dir))
                # B4: accumulate log-prob-bearing trajectories for GRPO.
                if config.enable_lora and result.trajectory_data is not None:
                    trajectory_buffer.append(result.trajectory_data)
            avg = sum(per_variant) / max(1, len(per_variant))
            rewards[variant.id] = avg
            episode_dirs[variant.id] = dirs
            logger.info(
                "  %s: avg reward=%.2f over %d rollout(s)",
                variant.id, avg, len(per_variant),
            )

        # ── Update ratings ──
        population.update_ratings(
            variant_ids=selected_ids,
            rewards=[rewards[v] for v in selected_ids],
        )

        # ── Identify best ──
        best_id = max(selected_ids, key=lambda v: rewards[v])
        best_reward = rewards[best_id]
        best_variant = population.get(best_id)

        # ── Mutation ──
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

                    # L_C-evolution snapshots: one file per new child,
                    # plus the always-current champion pointer.
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

        # ── Checkpoint ──
        if iteration % config.checkpoint_every == 0:
            checkpoint_path.write_text(
                json.dumps(population.to_json(), indent=2)
            )

        # ── B4: GRPO update when buffer fills ──
        if (
            config.enable_lora
            and len(trajectory_buffer) >= config.lora_update_every_k
        ):
            metrics = grpo_step(
                model,
                optimizer,
                trajectory_buffer,
                device=config.lora_device,
                max_grad_norm=config.lora_max_grad_norm,
                min_n_active=config.lora_min_n_active,
                kl_beta=config.lora_kl_beta,
                ref_adapter_name=config.lora_ref_adapter_name,
                kl_log_ratio_clip=config.lora_kl_log_ratio_clip,
            )
            logger.info(
                "GRPO step @ iter %d: loss=%.4f grad_norm=%.3f "
                "adv_std=%.3f adv_max=%.3f mean_reward=%.3f mean_kl=%.4f "
                "n_traj=%d n_active=%d",
                iteration, metrics["loss"], metrics["grad_norm"],
                metrics["advantage_std"], metrics["advantage_abs_max"],
                metrics["mean_reward"], metrics.get("mean_kl", 0.0),
                metrics["n_trajectories"], metrics["n_active"],
            )
            with grpo_log_path.open("a") as fh:
                fh.write(json.dumps({
                    "iteration": iteration,
                    "trigger": "buffer_full",
                    **metrics,
                }) + "\n")
            cleanup_replay_data(trajectory_buffer)
            trajectory_buffer.clear()

        # ── B4: periodic LoRA adapter checkpoint ──
        if (
            config.enable_lora
            and lora_ckpt_dir is not None
            and iteration % config.lora_checkpoint_every_k == 0
        ):
            ckpt = lora_ckpt_dir / f"iter_{iteration:04d}"
            save_lora_checkpoint(model, ckpt)
            logger.info("Saved LoRA adapter to %s", ckpt)

        # ── B4: end-of-iteration memory hygiene ──
        # Even with per-step backward + per-GRPO empty_cache, PyTorch's
        # allocator fragments across iterations on long sequences; a
        # short-horizon app can reach iter 12 before cumulative
        # fragmentation pushes it over 80 GB (observed 2026-04-23 on
        # system_settings). A full empty_cache + gc.collect between
        # iterations releases unreferenced cached blocks so the next
        # iteration starts from a consistent allocator state.
        if config.enable_lora:
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

    # Final checkpoint (covers the case where checkpoint_every > 1 and
    # the last iteration wasn't on the cadence).
    checkpoint_path.write_text(json.dumps(population.to_json(), indent=2))

    # ── B4: drain any remaining buffered trajectories + final LoRA
    # adapter checkpoint. Done regardless of ``lora_checkpoint_every_k``
    # cadence so the final adapter is always on disk at ``final/``.
    if config.enable_lora:
        if trajectory_buffer:
            metrics = grpo_step(
                model,
                optimizer,
                trajectory_buffer,
                device=config.lora_device,
                max_grad_norm=config.lora_max_grad_norm,
                min_n_active=config.lora_min_n_active,
                kl_beta=config.lora_kl_beta,
                ref_adapter_name=config.lora_ref_adapter_name,
                kl_log_ratio_clip=config.lora_kl_log_ratio_clip,
            )
            logger.info(
                "GRPO step (final drain): loss=%.4f grad_norm=%.3f "
                "n_traj=%d n_active=%d",
                metrics["loss"], metrics["grad_norm"],
                metrics["n_trajectories"], metrics["n_active"],
            )
            with grpo_log_path.open("a") as fh:
                fh.write(json.dumps({
                    "iteration": config.n_iterations,
                    "trigger": "final_drain",
                    **metrics,
                }) + "\n")
            cleanup_replay_data(trajectory_buffer)
            trajectory_buffer.clear()
        if lora_ckpt_dir is not None:
            final_ckpt = lora_ckpt_dir / "final"
            save_lora_checkpoint(model, final_ckpt)
            logger.info("Saved final LoRA adapter to %s", final_ckpt)

    logger.info(
        "Evolution complete. Champion: %s (mu=%.2f, sigma=%.2f, pop=%d)",
        population.champion.id,
        population.champion.rating.mu,
        population.champion.rating.sigma,
        population.size,
    )
    return population


def run_l_c_evolution(
    initial_l_c: Layer2,
    app_name: str,
    app_category: str,
    task_templates: list[str],
    rollout_fn: RolloutFn,
    config: EvolutionConfig,
    *,
    output_dir: Path,
    resume: bool = True,
    model: Any = None,
) -> Population:
    """Evolve a category-level ``L_C`` (Layer 2) on one app's T_adapt.

    This is the canonical B3 entry point. Unlike :func:`run_evolution`,
    which evolves a full two-layer FSM, this function evolves only the
    LAYER-2 (category-level abstract knowledge). Each population variant
    is an :class:`FSM` wrapper whose Layer 1 is deliberately empty; the
    substance of every mutation lives in ``layer2``.

    Args:
        initial_l_c: Starting ``L_C``, typically loaded via
            :func:`evofsm_rl.fsm.aggregator.load_L_C`.
        app_name: Held-out app whose T_adapt this evolution runs on.
            Surfaced to the mutation prompt as context (the reflection
            reminds the model that edits must generalize beyond the
            single app).
        app_category: Play Store category — determines which ``L_C``
            the run edits. Must be non-empty.
        task_templates: T_adapt task template names. **Never** pass
            T_eval here — T_eval is reserved for frozen evaluation
            after evolution finishes.
        rollout_fn: Same protocol as :func:`run_evolution`. The injected
            prompt text is the Layer-2 block only, matching B2's L_C
            injection shape.
        config: Hyperparameters. Mutation model / window_size / etc.
            all apply.
        output_dir: Writable directory. Additional files beyond
            :func:`run_evolution`'s outputs:

              * ``l_c_v0_initial.json`` — pristine copy of the starting
                L_C (written once; never overwritten, even on resume).
              * ``l_c_v{N}_iter{I}.json`` — snapshot of each new child
                as it's produced. ``N`` is a monotonic mutation counter,
                ``I`` is the iteration that produced it.
              * ``l_c_champion.json`` — the current champion's FSM,
                refreshed every time a new child lands.
        resume: Same semantics as :func:`run_evolution`.

    Returns:
        The final :class:`Population`. The champion's ``layer2`` is the
        evolved L_C to hand off to T_eval.

    Raises:
        ValueError: if ``app_category`` is empty — L_C evolution only
            makes sense when the agent has a target category to refine.
    """
    if not app_category:
        raise ValueError(
            "run_l_c_evolution: app_category must be non-empty. "
            "Tier-C apps (no matching source-pool category) have no L_C "
            "to evolve — skip them at the caller."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_fsm = l_c_to_seed_fsm(initial_l_c, app_name, app_category)

    # Pristine snapshot of the starting L_C. Written only if it doesn't
    # already exist; on resume we must not clobber the original.
    initial_path = output_dir / "l_c_v0_initial.json"
    if not initial_path.exists():
        initial_path.write_text(json.dumps(seed_fsm.to_json(), indent=2))
        logger.info("Wrote initial L_C snapshot: %s", initial_path)

    return run_evolution(
        root_fsm=seed_fsm,
        task_templates=task_templates,
        rollout_fn=rollout_fn,
        config=config,
        output_dir=output_dir,
        app_category=app_category,
        resume=resume,
        layer2_only=True,
        l_c_snapshot=True,
        model=model,
    )


__all__ = [
    "EvolutionConfig",
    "IterationResult",
    "RolloutFn",
    "RolloutResult",
    "TaskSampler",
    "l_c_to_seed_fsm",
    "run_evolution",
    "run_l_c_evolution",
]

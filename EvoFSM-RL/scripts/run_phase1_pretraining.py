#!/usr/bin/env python3
"""Phase 1 of EvoFSM-RL: shared-LoRA GRPO pretraining over a source pool.

This is the *minimal* Phase 1 (per memory note ``project-phase1-minimal-scope``):

  * Shared LoRA trained jointly across multiple source apps
  * Static FSMs (loaded from ``artifacts/static_fsms/``) — no mutation,
    no L_C aggregation, no cross-app crossover
  * Single produced artifact: ``π^pre_θ`` (the shared LoRA adapter),
    saved periodically under ``lora_checkpoints/iter_NNNN/`` and
    finally under ``lora_checkpoints/final/``

Algorithm (per ``plan/algorithm_design.md`` §8.2, minimal-scope form)::

    loop iter:
        a ← sample_uniform(--apps)
        t ← sample(templates_of[a])
        F ← static_fsms[a]                 # frozen, per app
        for j in 1..N:                      # N≥2 required by F5 grouping
            τ_j = rollout(π_θ, system_prompt=F, app=a, task=t, seed=…)
        buffer.extend(trajectories)
        if buffer.size ≥ K:
            grpo_step(model, buffer)        # uses F5 (FSM, task) grouping
            buffer.clear()
        save LoRA every --checkpoint-every iters

GRPO group key is ``(fsm_variant_id, task_name)`` where
``fsm_variant_id = "static_{app}"``. With ``N≥2`` rollouts on the
same (app, task) per iter we get a group of size N inside one
``(static_{app}, task_name)`` bucket, so within-group advantage is
non-zero by construction.

Pilot example (4 apps × ~50 iter each, ~200 iter total)::

    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      python EvoFSM-RL/scripts/run_phase1_pretraining.py \\
        --apps markor bluecoins calculator contacts \\
        --n-iterations 200 \\
        --n-rollouts 2 \\
        --buffer-size 10 \\
        --lora-rank 16 --lora-lr 3e-4 \\
        --checkpoint-every 50 \\
        --console-port 5710 --grpc-port 8710 \\
        --adb-path $ANDROID_HOME/platform-tools/adb \\
        --output-dir EvoFSM-RL/traces/phase1_pilot_v01/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("run_phase1_pretraining")


# ─────────────────────────────────────────────────────────────────────
# Source-pool resolution: per-app template list from splits.yaml
# ─────────────────────────────────────────────────────────────────────


def _load_source_pool_apps(project_root: Path) -> dict[str, dict[str, Any]]:
    """Return {app_name: {category, templates}} aggregating apps across
    the pool selectors needed for "shared TTA" — by default the source
    pool only. When the env var ``EVOFSM_PHASE1_INCLUDE_TARGETS=1`` is
    set, also include each Tier-B/Tier-C target app's **T_adapt** subset
    (template-disjoint from T_eval). This lets one ``run_phase1_*``
    invocation train a single shared LoRA on source pool + target
    adaptation templates, which is the "Paradigm B / shared-TTA"
    experiment paired with the per-app B4 ``run_b4_evolution.py`` runs.

    Reads ``configs/splits.yaml``.
    """
    import os
    import yaml

    splits_path = project_root / "configs" / "splits.yaml"
    if not splits_path.exists():
        raise FileNotFoundError(f"No splits.yaml at {splits_path}")
    splits = yaml.safe_load(splits_path.read_text())
    out: dict[str, dict[str, Any]] = {}

    only_targets = os.environ.get("EVOFSM_PHASE1_ONLY_TARGETS") == "1"

    # Pool 1: source_pool (default, all templates per app are training).
    # Skipped entirely when EVOFSM_PHASE1_ONLY_TARGETS=1 — that flag means
    # we want the shared-TTA experiment that only sees target T_adapt.
    if not only_targets:
        source = splits.get("source_pool", {})
        for app, info in source.items():
            out[app] = {
                "category": info.get("category"),
                "templates": list(info.get("templates", [])),
            }

    # Pool 2 (opt-in): Tier-B + Tier-C target apps, using only T_adapt
    # (NOT T_eval — never let T_eval into training).
    if only_targets or os.environ.get("EVOFSM_PHASE1_INCLUDE_TARGETS") == "1":
        for pool_key in ("tier_B_held_out", "tier_C_held_out"):
            tier = splits.get(pool_key, {})
            for app, info in tier.items():
                ta = list(info.get("T_adapt", []))
                if not ta:
                    continue
                # Prefix-avoiding name collision unlikely; use raw app names.
                if app in out:
                    # Source pool already has this app — skip (shouldn't happen
                    # since source/tier_B/tier_C are disjoint by design).
                    continue
                out[app] = {
                    "category": info.get("category"),
                    "templates": ta,
                }

    if not out:
        raise ValueError("splits.yaml produced no apps for Phase-1 training")
    return out


def _load_static_fsm_prompt(static_fsms_dir: Path, app: str) -> str:
    """Read the human-readable serialized static FSM for ``app``."""
    txt_path = static_fsms_dir / f"{app}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(
            f"No static FSM text at {txt_path}. "
            f"Build it with scripts/build_all_fsms.py first."
        )
    return txt_path.read_text()


# ─────────────────────────────────────────────────────────────────────
# Rollout fn builder (model + LoRA + emulator + agent)
# ─────────────────────────────────────────────────────────────────────
#
# Mirrors ``_build_b4_rollout_fn`` in ``run_b4_evolution.py`` but:
#   * No L_C / mutation machinery (Phase 1 is static-FSM)
#   * ``app_name`` is a per-call argument, not closure-captured, because
#     Phase 1 cycles through multiple apps in one process
#   * ``fsm_variant_id`` is set to ``f"static_{app_name}"`` so the
#     F5 grouping by ``(fsm_variant_id, task_name)`` puts one group per
#     (app, task) tuple, matching §3.2 design.


def _build_pretrain_rollout_fn(
    *,
    episodes_dir: Path,
    replay_dir: Path,
    console_port: int,
    grpc_port: int,
    adb_path: str | None,
    device: str | None,
    max_steps_multiplier: float,
    emulator_setup: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: tuple[str, ...],
    step_budget: int,
    use_dense_reward: bool = False,
):
    """Construct the Phase-1 rollout closure.

    Returns ``(rollout_fn, env, model)``. The caller owns lifecycle of
    ``env`` (call ``env.close()`` at the end) and ``model`` (the same
    PEFT-wrapped object that GRPO will backward through).
    """
    from evofsm_rl.agent.rollout import GenerationConfig, Qwen3VLAgent
    from evofsm_rl.env import harness
    from evofsm_rl.model import (
        load_base_model,
        load_model_config,
        resolve_device,
    )
    from evofsm_rl.model.lora import attach_lora, count_trainable_params
    from evofsm_rl.rl.grpo import compute_reward

    resolved_device = device or resolve_device()
    logger.info("Loading Qwen3-VL-8B on device=%s", resolved_device)
    t0 = time.monotonic()
    model, processor = load_base_model(device=resolved_device)
    logger.info("Model loaded in %.1fs", time.monotonic() - t0)

    cfg = load_model_config(device=resolved_device)
    gen_cfg = GenerationConfig.from_yaml(cfg.raw.get("generation", {}))

    logger.info(
        "Attaching LoRA: rank=%d alpha=%d dropout=%.3f targets=%s",
        lora_rank, lora_alpha, lora_dropout, lora_target_modules,
    )
    model = attach_lora(
        model,
        rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=tuple(lora_target_modules),
    )
    counts = count_trainable_params(model)
    logger.info(
        "LoRA attached: trainable=%d / total=%d (%.3f%%)",
        counts["trainable"], counts["total"], counts["percent"],
    )

    # Optional: warm-start LoRA from a pretrained checkpoint (e.g. v3-B
    # for the "shared TTA" variant). Triggered by env var to avoid CLI
    # churn — matches the EVOFSM_PHASE1_* family.
    import os as _os
    _init_lora_from = _os.environ.get("EVOFSM_PHASE1_INIT_LORA_FROM")
    if _init_lora_from:
        from evofsm_rl.model.lora import load_lora_checkpoint
        logger.info("Loading pretrained LoRA from %s", _init_lora_from)
        load_lora_checkpoint(model, _init_lora_from)
        logger.info("Loaded pretrained LoRA — Phase 1 continues from this state.")

    # Gradient checkpointing — same rationale as run_b4_evolution: cuts
    # peak activation memory ~50% in exchange for ~20% step time, needed
    # to keep long-horizon rollouts off the OOM cliff on 80 GB cards.
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        logger.info("Gradient checkpointing enabled (use_reentrant=False)")
    except Exception as e:
        logger.warning(
            "Failed to enable gradient checkpointing (%s) — continuing "
            "without it; may OOM on long-horizon tasks.", e,
        )

    connect_kwargs: dict[str, Any] = {
        "console_port": console_port,
        "grpc_port": grpc_port,
        "emulator_setup": emulator_setup,
    }
    if adb_path:
        connect_kwargs["adb_path"] = adb_path
    logger.info("Connecting emulator %s", connect_kwargs)
    env = harness.connect(**connect_kwargs)

    agent = Qwen3VLAgent(
        model=model,
        processor=processor,
        env=env,
        device=resolved_device,
        generation_config=gen_cfg,
        collect_log_probs=True,
        replay_dir=replay_dir,
    )

    episodes_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)

    def rollout(
        *,
        fsm_prompt_text: str,
        app_name: str,
        task_name: str,
        seed: int,
    ):
        """Run one rollout, persist the episode, return TrajectoryData.

        Returns a dict ``{trajectory_data, success, n_steps, wall_seconds,
        episode_dir, error}`` instead of a ``RolloutResult`` because we
        do not need the evolution-loop integration shape.
        """
        agent.set_l_c_prompt_text(fsm_prompt_text)
        variant_id = f"static_{app_name}"

        t_start = time.monotonic()
        try:
            result = harness.run_template(
                template_name=task_name,
                seed=seed,
                env=env,
                agent=agent,
                max_steps_multiplier=max_steps_multiplier,
                use_dense_reward=use_dense_reward,
            )
        except Exception as e:
            logger.exception("Rollout crashed on %s seed=%d", task_name, seed)
            return {
                "trajectory_data": None,
                "success": 0.0,
                "n_steps": 0,
                "wall_seconds": time.monotonic() - t_start,
                "episode_dir": None,
                "error": f"{type(e).__name__}: {e}",
            }

        ep_dir: Path | None = None
        if result.error is None and result.n_steps > 0:
            try:
                ep_dir = agent.save_episode(
                    episodes_dir,
                    success=result.success,
                    template=task_name,
                    seed=seed,
                    app=app_name,
                    tier="source",
                )
                if ep_dir is not None:
                    ep_dir = Path(ep_dir)
            except Exception:
                logger.exception(
                    "save_episode failed for %s seed=%d (continuing)",
                    task_name, seed,
                )
                ep_dir = None

        reward_scalar = compute_reward(
            success=float(result.success),
            n_steps=int(result.n_steps),
            step_budget=step_budget,
        )
        traj_data: Any = None
        if result.error is None and result.n_steps > 0:
            try:
                traj_data = agent.get_trajectory_data(
                    task_name=task_name,
                    seed=seed,
                    fsm_variant_id=variant_id,
                    reward=reward_scalar,
                    success=float(result.success),
                )
            except Exception:
                logger.exception(
                    "get_trajectory_data failed for %s seed=%d (continuing)",
                    task_name, seed,
                )
                traj_data = None

        return {
            "trajectory_data": traj_data,
            "success": float(result.success),
            "n_steps": int(result.n_steps),
            "wall_seconds": float(result.wall_seconds),
            "episode_dir": ep_dir,
            "error": result.error,
        }

    return rollout, env, model


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Phase 1 pretraining: shared LoRA + GRPO over a source-pool "
            "subset, static FSMs, no mutation."
        ),
    )
    p.add_argument(
        "--apps", nargs="+", required=True,
        help="Source-pool apps to train on. Each must be present in "
             "configs/splits.yaml > source_pool and have a static FSM "
             "under artifacts/static_fsms/{app}.txt. Pilot: 4 apps. "
             "Full: 12 apps.",
    )
    p.add_argument("--n-iterations", type=int, default=200)
    p.add_argument(
        "--n-rollouts", type=int, default=2,
        help="N — rollouts per (app, task) per iter. Must be >= 2 for "
             "the F5 (fsm_variant_id, task_name) grouping to produce "
             "non-zero advantages.",
    )
    p.add_argument(
        "--buffer-size", type=int, default=10,
        help="K — fire grpo_step once the trajectory buffer reaches "
             "this many entries.",
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=50,
        help="Save LoRA adapter every N iterations + at the end.",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="All artifacts (episodes, replay, grpo_metrics.jsonl, "
             "iterations.jsonl, lora_checkpoints/) go here.",
    )

    # Emulator / model knobs (same as run_b4_evolution defaults).
    p.add_argument("--console-port", type=int, default=5710)
    p.add_argument("--grpc-port", type=int, default=8710)
    p.add_argument("--adb-path", type=str, default=None)
    p.add_argument("--device", choices=("mps", "cuda", "cpu"), default=None)
    p.add_argument("--emulator-setup", action="store_true")
    p.add_argument("--max-steps-multiplier", type=float, default=10.0)
    p.add_argument(
        "--step-budget", type=int, default=60,
        help="Per-template step budget for the efficiency-bonus reward.",
    )

    # LoRA / GRPO.
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules", type=str, default="q_proj,v_proj",
        help="Comma-separated peft target_modules names.",
    )
    p.add_argument("--lora-lr", type=float, default=3e-4)
    p.add_argument("--lora-max-grad-norm", type=float, default=1.0)

    # KL anchor + clip + reject — same fixes as Phase 3 v2 (2026-05-15).
    # Phase 1 v2 (without these) had Tier-B 11.1% / Tier-C 0% T_eval
    # (via B4-revert v2), worse than Phase 1 pilot, evidence that
    # 600-iter unanchored training drifts off the manifold.
    p.add_argument(
        "--kl-beta", type=float, default=0.0,
        help="KL anchor strength. β=0.05 recommended for v3 (matches "
             "the value validated in Phase 3 v2 smoke + sweep). β=0 "
             "disables the anchor (legacy Phase 1 v1/v2 behavior).",
    )
    p.add_argument(
        "--kl-log-ratio-clip", type=float, default=10.0,
        help="Numerical safety clip on |log_ratio| before exp(). "
             "Caps KL contribution per step at exp(clip)-1-clip. "
             "Default 10 → max ~22 015 (vs unclipped ~7e10 observed "
             "in 2026-05-15 β=0.02 smoke). Set 0 to disable.",
    )
    p.add_argument(
        "--min-n-active", type=int, default=1,
        help="Reject GRPO fires whose post-advantage active-trajectory "
             "count is below this threshold. Default 1 = legacy. Set "
             "to 3+ for Phase 1 v3 to refuse outlier-dominated fires.",
    )
    p.add_argument(
        "--anchor-to-base", action="store_true",
        help="Anchor the KL term to the BASE policy (LoRA disabled) "
             "rather than a pre-trained reference LoRA. Recommended for "
             "Phase 1 v3 since no prior π^pre exists — the LoRA starts "
             "from BA=0 ≡ base, so anchoring to base keeps it from "
             "drifting away as training accumulates updates.",
    )

    # Misc.
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for app/task sampling. Rollout seeds are derived "
             "from --seed-base and the iteration index, independent of "
             "this.",
    )
    p.add_argument(
        "--seed-base", type=int, default=100,
        help="Base rollout seed; per-rollout seed is "
             "seed_base + iter*N*100 + j*100 (paired across N).",
    )
    p.add_argument(
        "--static-fsms-dir", type=Path, default=None,
        help="Defaults to EvoFSM-RL/artifacts/static_fsms.",
    )
    p.add_argument(
        "--use-dense-reward", action="store_true",
        help="Opt-in: training rollouts call task.get_dense_reward(env) "
             "instead of is_successful(env). Multi-row tasks return "
             "partial credit ∈ [0, 1]; other tasks fall back to binary. "
             "Default OFF — see CLAUDE.md 'Dense reward design rule'.",
    )
    p.add_argument("-v", "--verbose", action="store_true")

    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    if args.n_rollouts < 2:
        logger.error(
            "--n-rollouts=%d is < 2. Under F5 grouping every group "
            "would have size 1 and contribute zero advantage. Phase 1 "
            "requires N >= 2.",
            args.n_rollouts,
        )
        return 2

    project_root = Path(__file__).resolve().parents[1]
    static_fsms_dir = args.static_fsms_dir or (
        project_root / "artifacts" / "static_fsms"
    )

    # 1. Resolve source pool + per-app templates from splits.yaml.
    try:
        source_pool = _load_source_pool_apps(project_root)
    except Exception:
        logger.exception("Failed to load source_pool from splits.yaml")
        return 2

    apps: list[str] = list(args.apps)
    for app in apps:
        if app not in source_pool:
            logger.error(
                "App %r is not in splits.yaml > source_pool. "
                "Available: %s",
                app, sorted(source_pool.keys()),
            )
            return 2
        if not source_pool[app]["templates"]:
            logger.error("App %r has no templates in splits.yaml.", app)
            return 2

    # 2. Preload static FSM prompts (fail fast if any missing).
    # Three modes:
    #   * default: load artifacts/static_fsms/<app>.txt (Phase 1 v3-B path)
    #   * EVOFSM_PHASE1_NO_STATIC_FSM=1: empty prompt (no FSM)
    #   * EVOFSM_PHASE1_USE_CATEGORY_LC=1: shared-TTA fair-comparison mode
    #     - Tier-B apps: load source-pool category L_C, render to text
    #     - Tier-C apps: load b4_k4_unified evolved L_C (per app, frozen)
    import json as _json
    import os as _os
    skip_static_fsm = _os.environ.get("EVOFSM_PHASE1_NO_STATIC_FSM") == "1"
    use_category_lc = _os.environ.get("EVOFSM_PHASE1_USE_CATEGORY_LC") == "1"
    static_fsm_prompts: dict[str, str] = {}

    if use_category_lc:
        # Resolve every app's L_C via category lookup for Tier-B, b4_k4 evolved
        # L_C for Tier-C. Both rendered via Layer2.to_prompt_text(category).
        from evofsm_rl.fsm.aggregator import category_to_slug, load_L_C
        from evofsm_rl.fsm.schema import Layer2
        import yaml as _yaml
        splits_path = project_root / "configs" / "splits.yaml"
        splits_data = _yaml.safe_load(splits_path.read_text())
        tier_b = splits_data.get("tier_B_held_out", {})
        tier_c = splits_data.get("tier_C_held_out", {})
        lc_dir = project_root / "artifacts" / "L_C"
        b4_unified_dir = project_root / "traces" / "b4_k4_unified"
        for app in apps:
            if app in tier_b:
                cat = tier_b[app].get("category")
                lc_path = lc_dir / f"{category_to_slug(cat)}.json"
                if not lc_path.exists():
                    logger.error("Tier-B %s: no L_C at %s", app, lc_path)
                    return 2
                _, layer2 = load_L_C(lc_path)
                static_fsm_prompts[app] = layer2.to_prompt_text(category=cat)
                logger.info(
                    "Loaded source-pool L_C for Tier-B %s (category=%s, %d cats)",
                    app, cat, len(layer2.categories),
                )
            elif app in tier_c:
                cat = tier_c[app].get("category")
                fsm_path = b4_unified_dir / app / "l_c_champion.json"
                if not fsm_path.exists():
                    logger.warning(
                        "Tier-C %s: no champion L_C at %s; using empty",
                        app, fsm_path,
                    )
                    static_fsm_prompts[app] = ""
                    continue
                with fsm_path.open() as f:
                    fsm_json = _json.load(f)
                layer2 = Layer2.from_json(fsm_json.get("layer2", {"categories": []}))
                static_fsm_prompts[app] = layer2.to_prompt_text(category=cat)
                logger.info(
                    "Loaded bootstrapped L_C for Tier-C %s (category=%s, %d cats)",
                    app, cat, len(layer2.categories),
                )
            else:
                logger.warning("App %s in neither tier_B nor tier_C; empty L_C", app)
                static_fsm_prompts[app] = ""
    else:
        for app in apps:
            if skip_static_fsm:
                static_fsm_prompts[app] = ""
                continue
            try:
                static_fsm_prompts[app] = _load_static_fsm_prompt(
                    static_fsms_dir, app,
                )
            except Exception:
                logger.exception("Failed to load static FSM for %s", app)
                return 2

    logger.info(
        "Phase 1 config: apps=%s n_iterations=%d N=%d K_buffer=%d "
        "checkpoint_every=%d lora_rank=%d lr=%g",
        apps, args.n_iterations, args.n_rollouts, args.buffer_size,
        args.checkpoint_every, args.lora_rank, args.lora_lr,
    )

    # 3. Set up output dirs + log files.
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_dir / "episodes"
    replay_dir = output_dir / "replay"
    lora_ckpt_dir = output_dir / "lora_checkpoints"
    lora_ckpt_dir.mkdir(exist_ok=True)
    grpo_log_path = output_dir / "grpo_metrics.jsonl"
    iter_log_path = output_dir / "iterations.jsonl"

    # 4. Record run config for reproducibility.
    (output_dir / "phase1_config.json").write_text(json.dumps({
        "apps": apps,
        "n_iterations": args.n_iterations,
        "n_rollouts": args.n_rollouts,
        "buffer_size": args.buffer_size,
        "checkpoint_every": args.checkpoint_every,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "lora_lr": args.lora_lr,
        "lora_max_grad_norm": args.lora_max_grad_norm,
        "seed": args.seed,
        "seed_base": args.seed_base,
        "step_budget": args.step_budget,
        "max_steps_multiplier": args.max_steps_multiplier,
    }, indent=2))

    # 5. Build rollout fn (loads model, attaches LoRA, connects emulator).
    target_modules = tuple(
        m.strip() for m in args.lora_target_modules.split(",") if m.strip()
    )
    rollout_fn, env, model = _build_pretrain_rollout_fn(
        episodes_dir=episodes_dir,
        replay_dir=replay_dir,
        console_port=args.console_port,
        grpc_port=args.grpc_port,
        adb_path=args.adb_path,
        device=args.device,
        max_steps_multiplier=args.max_steps_multiplier,
        emulator_setup=args.emulator_setup,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=target_modules,
        step_budget=args.step_budget,
        use_dense_reward=args.use_dense_reward,
    )

    # 6. Build AdamW optimizer over LoRA params.
    import torch

    from evofsm_rl.model.lora import save_lora_checkpoint
    from evofsm_rl.rl.grpo import cleanup_replay_data, grpo_step

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        logger.error(
            "Model has no trainable parameters after attach_lora — "
            "Phase 1 cannot proceed."
        )
        return 2
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lora_lr)

    # Resolve KL anchor reference. For Phase 1 v3, --anchor-to-base is
    # the recommended path: the LoRA starts from BA=0 ≡ base policy, so
    # anchoring to base keeps it from drifting off the manifold. No
    # ref-adapter loading needed — grpo_step handles the BASE sentinel
    # by calling disable_adapter_layers() during the ref forward.
    from evofsm_rl.rl.grpo import BASE_REF_SENTINEL

    if args.kl_beta > 0:
        if args.anchor_to_base:
            ref_adapter_name = BASE_REF_SENTINEL
            logger.info(
                "KL anchor mode: anchored to BASE policy (LoRA disabled "
                "during ref forward). β=%g  log_ratio_clip=%g  "
                "min_n_active=%d",
                args.kl_beta, args.kl_log_ratio_clip, args.min_n_active,
            )
        else:
            logger.error(
                "--kl-beta > 0 requires --anchor-to-base for Phase 1 "
                "(no prior π^pre exists). Pass --anchor-to-base or "
                "set --kl-beta 0 to disable the anchor.",
            )
            return 2
    else:
        ref_adapter_name = "ref"  # unused when kl_beta=0
        logger.info(
            "KL anchor disabled (--kl-beta=0). Legacy Phase 1 v1/v2 "
            "behavior. Recommend --kl-beta 0.05 --anchor-to-base "
            "--kl-log-ratio-clip 10 --min-n-active 3 for v3.",
        )
    device_resolved = args.device or "cuda"

    rng = random.Random(args.seed)
    trajectory_buffer: list[Any] = []
    t_run = time.monotonic()
    n_fires = 0

    # 7. Main loop.
    try:
        for iteration in range(1, args.n_iterations + 1):
            iter_t0 = time.monotonic()

            # ── sample app & task ──
            app = rng.choice(apps)
            template = rng.choice(source_pool[app]["templates"])
            fsm_prompt = static_fsm_prompts[app]

            logger.info(
                "=== Iteration %d/%d: app=%s task=%s ===",
                iteration, args.n_iterations, app, template,
            )

            # ── N rollouts on the same (app, task) ──
            per_iter_rewards: list[float] = []
            per_iter_traj_count = 0
            iter_error: str | None = None
            for j in range(args.n_rollouts):
                rollout_seed = args.seed_base + iteration * 100 + j
                result = rollout_fn(
                    fsm_prompt_text=fsm_prompt,
                    app_name=app,
                    task_name=template,
                    seed=rollout_seed,
                )
                per_iter_rewards.append(result["success"])
                if result["error"] and iter_error is None:
                    iter_error = result["error"]
                if result["trajectory_data"] is not None:
                    trajectory_buffer.append(result["trajectory_data"])
                    per_iter_traj_count += 1

            avg_success = sum(per_iter_rewards) / max(1, len(per_iter_rewards))
            logger.info(
                "  %s/%s: avg success=%.2f over %d rollout(s), "
                "buffered_trajs=%d/%d, total_in_buffer=%d",
                app, template, avg_success, args.n_rollouts,
                per_iter_traj_count, args.n_rollouts, len(trajectory_buffer),
            )

            # ── log this iter ──
            with iter_log_path.open("a") as fh:
                fh.write(json.dumps({
                    "iteration": iteration,
                    "app": app,
                    "task_name": template,
                    "n_rollouts": args.n_rollouts,
                    "successes": per_iter_rewards,
                    "avg_success": avg_success,
                    "buffered_trajs": per_iter_traj_count,
                    "total_in_buffer": len(trajectory_buffer),
                    "wall_seconds": time.monotonic() - iter_t0,
                    "error": iter_error,
                }) + "\n")

            # ── GRPO fire when buffer fills ──
            if len(trajectory_buffer) >= args.buffer_size:
                metrics = grpo_step(
                    model,
                    optimizer,
                    trajectory_buffer,
                    device=device_resolved,
                    max_grad_norm=args.lora_max_grad_norm,
                    min_n_active=args.min_n_active,
                    kl_beta=args.kl_beta,
                    ref_adapter_name=ref_adapter_name,
                    kl_log_ratio_clip=args.kl_log_ratio_clip,
                )
                n_fires += 1
                logger.info(
                    "GRPO step #%d @ iter %d: loss=%.4f grad_norm=%.3f "
                    "adv_std=%.3f adv_max=%.3f mean_reward=%.3f mean_kl=%.4f "
                    "n_traj=%d n_active=%d",
                    n_fires, iteration, metrics["loss"], metrics["grad_norm"],
                    metrics["advantage_std"], metrics["advantage_abs_max"],
                    metrics["mean_reward"], metrics.get("mean_kl", 0.0),
                    metrics["n_trajectories"], metrics["n_active"],
                )
                with grpo_log_path.open("a") as fh:
                    fh.write(json.dumps({
                        "iteration": iteration,
                        "fire_index": n_fires,
                        "trigger": "buffer_full",
                        **metrics,
                    }) + "\n")
                cleanup_replay_data(trajectory_buffer)
                trajectory_buffer.clear()

            # ── checkpoint LoRA periodically ──
            if iteration % args.checkpoint_every == 0:
                ckpt = lora_ckpt_dir / f"iter_{iteration:04d}"
                save_lora_checkpoint(model, ckpt)
                logger.info("Saved LoRA adapter to %s", ckpt)

            # ── end-of-iter memory hygiene (matches run_b4_evolution) ──
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 8. Final-drain remaining buffered trajectories.
        if trajectory_buffer:
            metrics = grpo_step(
                model,
                optimizer,
                trajectory_buffer,
                device=device_resolved,
                max_grad_norm=args.lora_max_grad_norm,
                min_n_active=args.min_n_active,
                kl_beta=args.kl_beta,
                ref_adapter_name=ref_adapter_name,
                kl_log_ratio_clip=args.kl_log_ratio_clip,
            )
            n_fires += 1
            logger.info(
                "GRPO final drain: loss=%.4f grad_norm=%.3f "
                "n_traj=%d n_active=%d",
                metrics["loss"], metrics["grad_norm"],
                metrics["n_trajectories"], metrics["n_active"],
            )
            with grpo_log_path.open("a") as fh:
                fh.write(json.dumps({
                    "iteration": args.n_iterations,
                    "fire_index": n_fires,
                    "trigger": "final_drain",
                    **metrics,
                }) + "\n")
            cleanup_replay_data(trajectory_buffer)
            trajectory_buffer.clear()

        # 9. Always save final LoRA, regardless of cadence.
        final_ckpt = lora_ckpt_dir / "final"
        save_lora_checkpoint(model, final_ckpt)
        logger.info("Saved final LoRA adapter to %s", final_ckpt)

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                logger.warning("env.close() failed", exc_info=True)

    wall_min = (time.monotonic() - t_run) / 60
    print()
    print("=" * 64)
    print(f"Phase 1 pilot done")
    print(f"Apps:            {apps}")
    print(f"Iterations:      {args.n_iterations}")
    print(f"GRPO fires:      {n_fires}")
    print(f"LoRA ckpts ->    {lora_ckpt_dir}")
    print(f"  final ->       {final_ckpt}")
    print(f"GRPO metrics ->  {grpo_log_path}")
    print(f"Iter log ->      {iter_log_path}")
    print(f"Wall time:       {wall_min:.1f} min")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())

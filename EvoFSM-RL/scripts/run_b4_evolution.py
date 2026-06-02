#!/usr/bin/env python3
"""Run B4 L_C evolution + LoRA fine-tuning for one testing-set app.

B4 is B3 with two extra moving parts bolted on top:

  * The policy's LoRA adapters are attached before the first rollout
    and stay attached for the whole run.
  * The agent records per-step log-probs + replay tensors; the
    evolution loop buffers them and every ``--lora-update-every``
    iterations runs a GRPO weight update on the buffer, then deletes
    the replay files.

Everything else - L_C mutation cadence, population selection,
resume-from-checkpoint - is unchanged from B3.

See ``scripts/run_b3_evolution.py`` for the B3 variant. The two scripts
deliberately share their dry-run and plotting helpers but keep their
rollout factories separate because the real B4 rollout needs the agent
to package a :class:`TrajectoryData` for GRPO.

Examples::

    # Standard B4 run (live model + emulator + Claude mutation):
    export ANDROID_HOME=/shared/linqiang/evofsm_project/android-sdk
    export CUDA_VISIBLE_DEVICES=2
    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      python EvoFSM-RL/scripts/run_b4_evolution.py \\
        --app pro_expense \\
        --n-iterations 20 \\
        --lora-rank 16 --lora-lr 1e-4 \\
        --lora-update-every 5 \\
        --output-dir EvoFSM-RL/traces/b4_evolution/pro_expense \\
        --console-port 5710 --grpc-port 8710 \\
        --adb-path $ANDROID_HOME/platform-tools/adb

    # Path-A sanity (LoRA attached but never updated -> should match B3):
    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      python EvoFSM-RL/scripts/run_b4_evolution.py \\
        --app pro_expense --n-iterations 20 --disable-grpo ...

    # Dry-run (no emulator, no model load, mock rollouts):
    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      python EvoFSM-RL/scripts/run_b4_evolution.py \\
        --app pro_expense --dry-run --n-iterations 10 \\
        --output-dir EvoFSM-RL/traces/b4_dry_test

B4 is restricted to Tier-B apps (categories with a matching source-pool
L_C). Tier-C apps cannot produce meaningful advantages because their
variants never share a category with any other trajectory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Re-use everything we can from the B3 runner. Same dir so we import by
# inserting scripts/ into sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_b3_evolution import (  # noqa: E402
    _build_mock_rollout_fn,
    _resolve_app,
)


logger = logging.getLogger("run_b4_evolution")


# ─────────────────────────────────────────────────────────────────────
# Real rollout with log-prob collection + TrajectoryData packaging.
# ─────────────────────────────────────────────────────────────────────
#
# Diverges from B3's ``_build_real_rollout_fn`` in three ways:
#   1. Calls ``attach_lora`` before building the agent.
#   2. Builds the agent with ``collect_log_probs=True`` + ``replay_dir``.
#   3. Returns ``RolloutResult.trajectory_data`` populated via
#      ``agent.get_trajectory_data(...)`` so the evolution loop can
#      forward it to GRPO.
# A lot of the body mirrors B3 on purpose — factoring further would
# make the flow harder to read. Kept as a standalone function.


def _build_b4_rollout_fn(
    *,
    app_name: str,
    pool_label: str,
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
    init_lora_from: Path | None = None,
    ref_lora_from: Path | None = None,
    ref_adapter_name: str = "ref",
    use_dense_reward: bool = False,
):
    """Construct a B4 rollout closure.

    Returns ``(rollout_fn, env, model)``. The caller owns lifecycle of
    ``env`` (``env.close()``) and ``model`` (it is the object that
    ``run_evolution`` will receive via ``model=`` and that GRPO will
    call ``.backward()`` through, so must stay alive across the whole
    run).

    If ``init_lora_from`` is provided, the freshly attached LoRA
    adapter is overwritten with weights from that directory (the
    Phase-1 ``π^pre_θ`` artifact) before any training starts. The
    adapter remains trainable, so Phase 3 / B4 GRPO continues updating
    from the pretrained state.
    """
    from evofsm_rl.agent.rollout import GenerationConfig, Qwen3VLAgent
    from evofsm_rl.env import harness
    from evofsm_rl.model import (
        load_base_model, load_model_config, resolve_device,
    )
    from evofsm_rl.model.lora import (
        attach_lora,
        count_trainable_params,
        load_lora_checkpoint,
        load_ref_lora_adapter,
    )
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

    if init_lora_from is not None:
        init_path = Path(init_lora_from)
        if not init_path.exists():
            raise FileNotFoundError(
                f"--init-lora-from path does not exist: {init_path}"
            )
        logger.info(
            "Loading pretrained LoRA from %s (Phase 1 π^pre_θ)", init_path,
        )
        model = load_lora_checkpoint(model, init_path)
        counts = count_trainable_params(model)
        logger.info(
            "LoRA loaded from checkpoint: trainable=%d / total=%d (%.3f%%)",
            counts["trainable"], counts["total"], counts["percent"],
        )

    if ref_lora_from is not None:
        ref_path = Path(ref_lora_from)
        if not ref_path.exists():
            raise FileNotFoundError(
                f"--ref-lora-from path does not exist: {ref_path}"
            )
        logger.info(
            "Loading reference LoRA from %s as adapter %r (for KL anchor)",
            ref_path, ref_adapter_name,
        )
        model = load_ref_lora_adapter(
            model, ref_path, adapter_name=ref_adapter_name,
        )
        counts = count_trainable_params(model)
        logger.info(
            "Ref adapter attached (frozen). Trainable params unchanged: "
            "%d / %d (%.3f%%)",
            counts["trainable"], counts["total"], counts["percent"],
        )

    # Gradient checkpointing — recompute forward activations during
    # backward instead of storing them. Trades ~20% step time for a
    # ~50% cut in peak activation memory. Required for B4: without it
    # the GRPO gradient-bearing forward on long-horizon tasks
    # (Expense*Multi*, Simple*Calendar*, Retro*Playlist*) pushes an
    # 80 GB H100 over the limit and the run OOMs mid-sweep (empirically
    # confirmed on 2026-04-23 first-pass sweep: 4/6 apps crashed).
    # ``use_reentrant=False`` is the modern PyTorch recommendation;
    # reentrant checkpointing has quirks around PEFT-wrapped models.
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        # PEFT-wrapped HF models need input embeddings to require grad
        # for gradient-checkpointing to propagate; peft exposes a
        # helper for this.
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

    # Monotonic counter of rollouts within the run. Paired with
    # ``fsm_variant_id`` in the TrajectoryData so GRPO can group properly.
    # The evolution loop itself does not tell us which variant we are
    # running, so we reconstruct it from the injected prompt text via a
    # stable hash — enough to distinguish "same variant" vs "different
    # variant" within a single GRPO window.
    import hashlib

    def _variant_id_from_prompt(prompt_text: str) -> str:
        h = hashlib.md5(prompt_text.encode("utf-8")).hexdigest()[:10]
        return f"v_{h}"

    def real(fsm_prompt_text: str, task_name: str, seed: int):
        from evofsm_rl.fsm.evolution import RolloutResult

        agent.set_l_c_prompt_text(fsm_prompt_text)
        variant_id = _variant_id_from_prompt(fsm_prompt_text)

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
            return RolloutResult(
                task_name=task_name,
                seed=seed,
                reward=0.0,
                n_steps=0,
                wall_seconds=time.monotonic() - t_start,
                episode_dir=None,
                error=f"{type(e).__name__}: {e}",
                trajectory_data=None,
            )

        ep_dir: Path | None = None
        if result.error is None and result.n_steps > 0:
            try:
                ep_dir = agent.save_episode(
                    episodes_dir,
                    success=result.success,
                    template=task_name,
                    seed=seed,
                    app=app_name,
                    tier=pool_label,
                )
                if ep_dir is not None:
                    ep_dir = Path(ep_dir)
            except Exception:
                logger.exception(
                    "save_episode failed for %s seed=%d (continuing)",
                    task_name, seed,
                )
                ep_dir = None

        # Package the trajectory for GRPO. The reward uses the same
        # formula described in the algorithm design so GRPO's advantage
        # computation stays consistent with the fitness signal.
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

        return RolloutResult(
            task_name=task_name,
            seed=seed,
            reward=float(result.success),
            n_steps=int(result.n_steps),
            wall_seconds=float(result.wall_seconds),
            episode_dir=ep_dir,
            error=result.error,
            trajectory_data=traj_data,
        )

    # Stash processor on the model object so callers (e.g., PPO+PRM)
    # can reuse it without re-loading. The agent owns the same processor
    # via its self._processor attribute.
    try:
        model.processor = processor  # noqa: SLF001
    except Exception:
        pass
    return real, env, model


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run B4 evolution (L_C + LoRA GRPO) for one target app.",
    )
    # App + run-level args (mostly same as B3).
    p.add_argument("--app", required=True,
                   help="Target app. Must be a Tier-B app (its category "
                        "has a source-pool L_C). Tier-C is rejected.")
    p.add_argument("--n-iterations", "--iterations", dest="n_iterations",
                   type=int, default=20)
    p.add_argument("--m-select", type=int, default=2)
    p.add_argument("--n-rollouts", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Defaults to EvoFSM-RL/traces/b4_evolution/{app}.")
    p.add_argument("--l-c-dir", type=Path, default=None,
                   help="Directory holding per-category L_C files. "
                        "Defaults to EvoFSM-RL/artifacts/L_C.")

    # Emulator / model.
    p.add_argument("--console-port", type=int, default=5710)
    p.add_argument("--grpc-port", type=int, default=8710)
    p.add_argument("--adb-path", type=str, default=None)
    p.add_argument("--device", choices=("mps", "cuda", "cpu"), default=None)
    p.add_argument("--emulator-setup", action="store_true")
    p.add_argument("--max-steps-multiplier", type=float, default=10.0)
    p.add_argument("--step-budget", type=int, default=60,
                   help="Per-template step budget for the reward's "
                        "efficiency bonus. Default 60.")

    # Evolution (same as B3).
    p.add_argument("--seed-base", type=int, default=100)
    p.add_argument("--window-size", type=int, default=15)
    p.add_argument("--mutation-model", default="claude-opus-4-7")
    p.add_argument("--task-sample-mode", choices=("random", "round_robin"),
                   default="random")
    p.add_argument("--mutation-every-n-iters", type=int, default=3)
    p.add_argument("--no-resume", action="store_true")

    # LoRA / GRPO (new).
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", type=str,
                   default="q_proj,v_proj",
                   help="Comma-separated peft target_modules names.")
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--lora-update-every", type=int, default=5,
                   help="Call grpo_step every N iterations.")
    p.add_argument("--lora-checkpoint-every", type=int, default=5,
                   help="Save LoRA adapter every N iterations.")
    p.add_argument("--lora-max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--init-lora-from", type=Path, default=None,
        help="Path to a Phase-1 LoRA checkpoint directory (e.g. "
             "traces/phase1_pilot_v01/lora_checkpoints/final). Loads "
             "the pretrained π^pre_θ adapter before training begins, "
             "so this run continues from it instead of random init. "
             "When omitted, B4 starts from fresh random LoRA "
             "(legacy v2 behavior — not the paper's B4).",
    )
    p.add_argument(
        "--ref-lora-from", type=Path, default=None,
        help="Path to a frozen LoRA checkpoint to load as the KL-anchor "
             "reference policy (π^pre). Required when --kl-beta > 0. "
             "Typically the same directory as --init-lora-from (the "
             "Phase 1 pretrained adapter), so we anchor π_θ to its "
             "starting point.",
    )
    p.add_argument(
        "--kl-beta", type=float, default=0.0,
        help="Strength of the KL anchor term in the GRPO loss: "
             "L_new = L_GRPO + β × E_step[log π_θ - log π_ref]. "
             "β=0 disables (legacy). β=0.01–0.05 is the standard RLHF "
             "range and the load-bearing fix for Phase 3's "
             "premature-termination mode-collapse (see CLAUDE.md). "
             "Requires --ref-lora-from when > 0.",
    )
    p.add_argument(
        "--min-n-active", type=int, default=1,
        help="Reject GRPO fires whose post-advantage active-trajectory "
             "count is below this threshold (default 1 = legacy: only "
             "n_active=0 is rejected). Set to 3+ to refuse "
             "outlier-dominated fires.",
    )
    p.add_argument(
        "--kl-log-ratio-clip", type=float, default=10.0,
        help="Numerical safety clip on |log π_θ - log π^pre| before "
             "exp() in the k3 KL estimator. Default 10.0 caps KL "
             "contribution per step at exp(10) ≈ 22 015. Set to 0 "
             "to disable clipping (legacy; observed exp blow-up to "
             "~7e10 in 2026-05-15 β=0.02 smoke fire 2). 99%+ of "
             "training steps have |log_ratio| < 5 so this clip is a "
             "no-op in the common case.",
    )
    p.add_argument("--disable-grpo", action="store_true",
                   help="Attach LoRA but skip every grpo_step call "
                        "(Path-A sanity check: should match B3).")
    p.add_argument(
        "--enable-bootstrap-fsm", action="store_true",
        help="Tier-C FSM bootstrap mode (CLAUDE.md 2026-05-13 option b). "
             "Like --allow-no-l-c (starts with empty L_C stub), but KEEPS "
             "FSM mutation ENABLED. Claude Opus mutation is framed as "
             "'cold-start synthesize from target-app trajectories' instead "
             "of the standard 'incremental diff'. Use this for Tier-C apps "
             "to test whether the system can discover category abstractions "
             "online without a source-pool prior. Mutually compatible with "
             "--allow-no-l-c (bootstrap subsumes it).",
    )
    p.add_argument("--allow-no-l-c", action="store_true",
                   help="Allow running on a target app whose Play "
                        "category has no source-pool L_C match (Tier-C "
                        "B4 path, option (a) in CLAUDE.md). When set, "
                        "the initial L_C is an empty Layer2 stub and "
                        "FSM mutation is disabled — only LoRA updates "
                        "happen during the 20-iter loop. Without this "
                        "flag, Tier-C apps are rejected (preserves the "
                        "original Tier-B-only safety check).")

    p.add_argument("--dry-run", action="store_true",
                   help="Mock rollouts, no emulator, no model load. LoRA "
                        "is NOT attached in dry-run (no model to attach "
                        "it to). Useful for smoke-testing loop wiring.")
    p.add_argument(
        "--use-dense-reward", action="store_true",
        help="Opt-in: training rollouts call task.get_dense_reward(env) "
             "instead of is_successful(env). Multi-row tasks "
             "(AddMultipleRows / DeleteMultipleRows subclasses) return "
             "partial credit ∈ [0, 1]; other tasks fall back to binary. "
             "Default OFF — see CLAUDE.md 'Dense reward design rule'. "
             "T_eval (run_b4_teval.py) is unaffected.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # Resolve app. Tier-B is the canonical B4 path (has source-pool L_C).
    # Tier-C is allowed when ``--allow-no-l-c`` is set: B4 runs with an
    # empty initial L_C and mutation disabled (option (a) — pure LoRA TTA
    # on the target's T_adapt, no in-context evolution). See CLAUDE.md
    # 2026-05-13 "Tier-C B4 design decision" for rationale.
    resolved = _resolve_app(args.app)
    if resolved is None:
        logger.error(
            "App %r is not registered in configs/splits.yaml (source_pool / "
            "tier_B_held_out / tier_C_held_out).", args.app,
        )
        return 2
    task_templates, app_category, pool_label = resolved
    # Bootstrap subsumes --allow-no-l-c (both permit an empty initial L_C;
    # bootstrap additionally keeps mutation enabled).
    allow_no_l_c_effective = args.allow_no_l_c or args.enable_bootstrap_fsm
    if pool_label != "tier_B" and not allow_no_l_c_effective:
        logger.error(
            "B4 default path requires a Tier-B app. Got %s in pool=%s. "
            "For Tier-C (no source-pool L_C match), rerun with "
            "--allow-no-l-c (option a: pure-LoRA TTA, mutation OFF) or "
            "--enable-bootstrap-fsm (option b: bootstrap LAYER 2 from "
            "target trajectories, mutation ON).",
            args.app, pool_label,
        )
        return 2
    if not task_templates:
        logger.error(
            "App %r resolved to empty T_adapt. Populate splits.yaml first.",
            args.app,
        )
        return 2

    project_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or (
        project_root / "traces" / "b4_evolution" / args.app
    )
    l_c_dir = args.l_c_dir or (project_root / "artifacts" / "L_C")

    # Load initial L_C (or build an empty stub for Tier-C / no-match).
    from evofsm_rl.fsm.aggregator import category_to_slug, load_L_C
    from evofsm_rl.fsm.schema import Layer2

    l_c_path = l_c_dir / f"{category_to_slug(app_category)}.json"
    l_c_missing = not l_c_path.exists()
    if l_c_missing:
        if not allow_no_l_c_effective:
            logger.error(
                "No L_C file at %s for app=%s (category=%s). Build the L_C "
                "for this category first (scripts/build_L_C.py), or rerun "
                "with --allow-no-l-c (option a) or --enable-bootstrap-fsm "
                "(option b) to handle this Tier-C target.",
                l_c_path, args.app, app_category,
            )
            return 2
        # Tier-C path: empty L_C means no in-context category prior. The
        # population starts with a single root variant whose LAYER 2 has
        # zero abstract categories. Mutation behavior depends on which
        # flag is set:
        #   --allow-no-l-c            → mutation DISABLED (option a)
        #   --enable-bootstrap-fsm    → mutation ENABLED with bootstrap
        #                               prompts (option b)
        initial_l_c = Layer2(categories=[])
        mode_label = (
            "FSM bootstrap mode (option b: mutation ON)"
            if args.enable_bootstrap_fsm
            else "pure-LoRA TTA (option a: mutation OFF)"
        )
        logger.info(
            "App=%s (%s) category=%s  tasks=%d  initial_l_c=<EMPTY>  %s",
            args.app, pool_label, app_category, len(task_templates),
            mode_label,
        )
    else:
        _l_c_category, initial_l_c = load_L_C(l_c_path)
        logger.info(
            "App=%s (%s) category=%s  tasks=%d  initial_l_c=%s (n_cats=%d)",
            args.app, pool_label, app_category, len(task_templates),
            l_c_path, len(initial_l_c.categories),
        )

    # Mutation will run (and thus need the API key) unless we are in the
    # option-(a) path: empty L_C AND --allow-no-l-c without --enable-bootstrap-fsm.
    mutation_will_run = (not l_c_missing) or args.enable_bootstrap_fsm
    if mutation_will_run and not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error(
            "ANTHROPIC_API_KEY is not set — mutation requires it. "
            "(Mutation runs when L_C exists, or when --enable-bootstrap-fsm "
            "is set on a Tier-C target.)",
        )
        return 2
    if not mutation_will_run:
        logger.info(
            "ANTHROPIC_API_KEY not required for this run: option-(a) "
            "path (empty L_C + mutation disabled).",
        )

    episodes_dir = output_dir / "episodes"
    replay_dir = output_dir / "replay"

    # Build rollout fn (+ env + model if real).
    env: Any = None
    model: Any = None
    if args.dry_run:
        rollout_fn = _build_mock_rollout_fn(
            seed_base=args.seed_base,
            episodes_dir=episodes_dir,
            app_name=args.app,
        )
    else:
        target_modules = tuple(
            m.strip() for m in args.lora_target_modules.split(",") if m.strip()
        )
        # Sanity-check: --kl-beta > 0 requires --ref-lora-from.
        if args.kl_beta > 0 and args.ref_lora_from is None:
            logger.error(
                "--kl-beta=%.4f > 0 but --ref-lora-from is unset. "
                "Either provide a Phase-1 LoRA path to anchor against, "
                "or set --kl-beta=0 to disable the KL anchor.",
                args.kl_beta,
            )
            return 2
        rollout_fn, env, model = _build_b4_rollout_fn(
            app_name=args.app,
            pool_label=pool_label,
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
            init_lora_from=args.init_lora_from,
            ref_lora_from=args.ref_lora_from,
            use_dense_reward=args.use_dense_reward,
        )

    # GRPO is only enabled on real runs and only when not suppressed.
    grpo_enabled = (not args.dry_run) and (not args.disable_grpo)

    from evofsm_rl.fsm.evolution import EvolutionConfig, run_l_c_evolution

    # When L_C is missing:
    #   - option (a) (--allow-no-l-c only)            → mutation OFF
    #   - option (b) (--enable-bootstrap-fsm)         → mutation ON,
    #                                                    bootstrap=True
    # The bootstrap mutation path uses a different reflection + diff
    # prompt that frames the empty L_C as something to populate from
    # scratch (see evofsm_rl/fsm/mutation.py:_build_bootstrap_*).
    # When L_C is missing (Tier-C path with --allow-no-l-c) we disable
    # mutation entirely: there is no source-pool category content to
    # diff against, and bootstrapping LAYER 2 from sparse target-app
    # trajectories is reserved for option (b) — a separate next-step
    # experiment per CLAUDE.md 2026-05-13.
    # mutation_enabled: ON when we have an L_C to evolve (Tier-B path)
    # OR when we explicitly want to bootstrap from empty (option b).
    mutation_enabled = (not l_c_missing) or args.enable_bootstrap_fsm

    config = EvolutionConfig(
        n_iterations=args.n_iterations,
        m_select=args.m_select,
        n_rollouts=args.n_rollouts,
        task_sample_mode=args.task_sample_mode,
        mutation_model=args.mutation_model,
        seed_base=args.seed_base,
        window_size=args.window_size,
        mutation_every_n_iters=args.mutation_every_n_iters,
        mutation_enabled=mutation_enabled,
        # B4-specific.
        enable_lora=grpo_enabled,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(
            m.strip()
            for m in args.lora_target_modules.split(",") if m.strip()
        ),
        lora_lr=args.lora_lr,
        lora_update_every_k=args.lora_update_every,
        lora_max_grad_norm=args.lora_max_grad_norm,
        lora_checkpoint_every_k=args.lora_checkpoint_every,
        lora_device=(args.device or "cuda"),
        lora_min_n_active=args.min_n_active,
        lora_kl_beta=args.kl_beta,
        lora_ref_adapter_name="ref",
        lora_kl_log_ratio_clip=args.kl_log_ratio_clip,
        bootstrap_fsm=args.enable_bootstrap_fsm,
    )
    logger.info(
        "Config: enable_lora=%s (disable-grpo=%s, dry-run=%s) "
        "update_every=%d checkpoint_every=%d lr=%g "
        "min_n_active=%d kl_beta=%g ref_lora=%s",
        grpo_enabled, args.disable_grpo, args.dry_run,
        args.lora_update_every, args.lora_checkpoint_every, args.lora_lr,
        args.min_n_active, args.kl_beta,
        str(args.ref_lora_from) if args.ref_lora_from else "(none)",
    )

    t_run = time.monotonic()
    try:
        population = run_l_c_evolution(
            initial_l_c=initial_l_c,
            app_name=args.app,
            app_category=app_category,
            task_templates=task_templates,
            rollout_fn=rollout_fn,
            config=config,
            output_dir=output_dir,
            resume=not args.no_resume,
            model=model if grpo_enabled else None,
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                logger.warning("env.close() failed", exc_info=True)

    wall_min = (time.monotonic() - t_run) / 60
    print()
    print("=" * 64)
    print(f"App:             {args.app}  ({pool_label}, {app_category})")
    print(f"Iterations:      {config.n_iterations}")
    print(f"GRPO enabled:    {grpo_enabled}")
    print(f"Population size: {population.size}")
    print(f"Champion:        {population.champion.id}")
    print(f"Champion mu:     {population.champion.rating.mu:.2f}")
    print(f"Champion sigma:  {population.champion.rating.sigma:.2f}")
    print(f"Output dir:      {output_dir}")
    print(f"Wall time:       {wall_min:.1f} min")
    print("=" * 64)

    champion_path = output_dir / "l_c_champion.json"
    champion_path.write_text(
        json.dumps(population.champion.fsm.to_json(), indent=2)
    )
    print(f"L_C champion -> {champion_path}")
    print(f"Initial L_C  -> {output_dir / 'l_c_v0_initial.json'}")
    if grpo_enabled:
        print(f"LoRA ckpts   -> {output_dir / 'lora_checkpoints'}")

    # Convergence plot (best-effort; never fails the run).
    try:
        from plot_convergence import plot_convergence

        plot_path = plot_convergence(
            log_dir=output_dir,
            app_name=args.app,
            n_iterations=config.n_iterations,
            has_grpo=grpo_enabled,
        )
        if plot_path is not None:
            print(f"Convergence  -> {plot_path}")
    except Exception:
        logger.warning("plot_convergence failed", exc_info=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

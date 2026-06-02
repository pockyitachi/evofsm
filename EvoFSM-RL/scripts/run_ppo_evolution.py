#!/usr/bin/env python3
"""Run B4-style L_C evolution + PPO LoRA fine-tuning for one app.

This is the PPO twin of ``scripts/run_b4_evolution.py``. It shares the
exact same FSM evolution outer loop, rollout factory, and CLI surface,
but swaps the GRPO weight-update path for PPO + GAE with a frozen-base
value head.

Key differences vs. run_b4_evolution.py:
  * Uses :func:`evofsm_rl.rl_ppo.run_l_c_evolution_ppo` (lives in the
    rl_ppo package, not fsm.evolution).
  * Requires ``--value-head-from`` (a pretrained value head from
    ``run_ppo_value_pretrain.py``). The head is loaded into the
    PPOTrainer at construction time and continues to update during
    PPO training.
  * Adds ``--gae-lambda`` (default 1.0) and
    ``--gae-lambda-switch-after-iter`` (default None) for the
    Monte-Carlo / adaptive schedule.
  * Adds ``--clip-eps`` (default 0.2) and ``--value-lr`` (default 1e-4,
    separate from --lora-lr).
  * Writes ``ppo_metrics.jsonl`` instead of ``grpo_metrics.jsonl``.

The mutation / population / emulator / agent / replay-collection
machinery is byte-identical to B4. We just call a different trainer
when the trajectory buffer fills up.

Example::

    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      python EvoFSM-RL/scripts/run_ppo_evolution.py \\
        --app pro_expense \\
        --n-iterations 20 \\
        --lora-rank 16 --lora-lr 3e-4 --value-lr 1e-4 \\
        --lora-update-every 5 \\
        --gae-lambda 1.0 \\
        --value-head-from EvoFSM-RL/artifacts/value_head_v01.pt \\
        --output-dir EvoFSM-RL/traces/ppo_evolution/pro_expense \\
        --console-port 5710 --grpc-port 8710 \\
        --adb-path $ANDROID_HOME/platform-tools/adb

    # Adaptive lambda ablation: λ=1 first 20 iter, then λ=0.95.
    ... --gae-lambda 1.0 --gae-lambda-switch-after-iter 20 ...
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

# Re-use both B3 helpers and B4's rollout factory + app resolver.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_b3_evolution import (  # noqa: E402
    _build_mock_rollout_fn,
    _resolve_app,
)
from run_b4_evolution import _build_b4_rollout_fn  # noqa: E402


logger = logging.getLogger("run_ppo_evolution")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Run PPO L_C evolution (LoRA + PPO + GAE with a frozen-base "
            "value head) for one target app. PPO replaces GRPO; "
            "everything else mirrors run_b4_evolution.py."
        ),
    )

    # ── App + run-level args (same as B4) ─────────────────────────
    p.add_argument("--app", required=True,
                   help="Target app. Must be a Tier-B app unless --allow-no-l-c "
                        "or --enable-bootstrap-fsm is set.")
    p.add_argument("--n-iterations", "--iterations", dest="n_iterations",
                   type=int, default=20)
    p.add_argument("--m-select", type=int, default=2)
    p.add_argument("--n-rollouts", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Defaults to EvoFSM-RL/traces/ppo_evolution/{app}.")
    p.add_argument("--l-c-dir", type=Path, default=None,
                   help="Directory holding per-category L_C files. "
                        "Defaults to EvoFSM-RL/artifacts/L_C.")

    # ── Emulator / model ─────────────────────────────────────────
    p.add_argument("--console-port", type=int, default=5710)
    p.add_argument("--grpc-port", type=int, default=8710)
    p.add_argument("--adb-path", type=str, default=None)
    p.add_argument("--device", choices=("mps", "cuda", "cpu"), default=None)
    p.add_argument("--emulator-setup", action="store_true")
    p.add_argument("--max-steps-multiplier", type=float, default=10.0)
    p.add_argument("--step-budget", type=int, default=60)

    # ── Evolution (same as B3/B4) ─────────────────────────────────
    p.add_argument("--seed-base", type=int, default=100)
    p.add_argument("--window-size", type=int, default=15)
    p.add_argument("--mutation-model", default="claude-opus-4-7")
    p.add_argument("--task-sample-mode", choices=("random", "round_robin"),
                   default="random")
    p.add_argument("--mutation-every-n-iters", type=int, default=3)
    p.add_argument("--no-resume", action="store_true")

    # ── LoRA (same as B4) ─────────────────────────────────────────
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", type=str,
                   default="q_proj,v_proj")
    p.add_argument("--lora-lr", type=float, default=3e-4,
                   help="AdamW lr for the LoRA params. Separate from "
                        "--value-lr (PPO has two optimizers).")
    p.add_argument("--lora-update-every", type=int, default=5,
                   help="Call PPOTrainer.step() every N iterations "
                        "(buffer-full trigger).")
    p.add_argument("--lora-checkpoint-every", type=int, default=5)
    p.add_argument("--lora-max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--init-lora-from", type=Path, default=None,
        help="Optional Phase-1 LoRA checkpoint dir to warm-start from. "
             "When omitted, PPO starts from random LoRA init.",
    )
    p.add_argument(
        "--min-n-active", type=int, default=1,
        help="Reject PPO fires below this active-trajectory count.",
    )

    # ── PPO-specific ──────────────────────────────────────────────
    p.add_argument(
        "--value-head-from", type=Path, required=True,
        help="Path to the pretrained value head .pt file (from "
             "scripts/run_ppo_value_pretrain.py). The head is loaded "
             "into the PPOTrainer at startup and continues to update "
             "during PPO training.",
    )
    p.add_argument(
        "--value-lr", type=float, default=1e-4,
        help="AdamW lr for the value head. Separate from --lora-lr.",
    )
    p.add_argument(
        "--clip-eps", type=float, default=0.2,
        help="PPO clipping epsilon. Default 0.2 (PPO paper).",
    )
    p.add_argument(
        "--value-loss-coef", type=float, default=0.5,
        help="Weight on the value MSE loss term in the combined "
             "policy + value loss. Default 0.5.",
    )
    p.add_argument(
        "--gae-gamma", type=float, default=0.99,
        help="GAE discount γ. Default 0.99.",
    )
    p.add_argument(
        "--gae-lambda", type=float, default=1.0,
        help="GAE λ. Default 1.0 (pure Monte Carlo — primary path).",
    )
    p.add_argument(
        "--gae-lambda-switch-after-iter", type=int, default=None,
        help="If set, switch GAE λ from --gae-lambda to "
             "--gae-lambda-after-value after iteration N. Default None "
             "(no schedule). Set to 20 for the documented adaptive "
             "ablation (λ=1.0 first 20 iter, then λ=0.95).",
    )
    p.add_argument(
        "--gae-lambda-after-value", type=float, default=0.95,
        help="λ value to switch to after --gae-lambda-switch-after-iter. "
             "Default 0.95 (standard PPO-style GAE).",
    )

    # ── Tier-C / bootstrap (same as B4) ───────────────────────────
    p.add_argument(
        "--enable-bootstrap-fsm", action="store_true",
        help="Tier-C bootstrap mode: empty initial L_C + mutation ON "
             "with bootstrap-prompt framing.",
    )
    p.add_argument(
        "--allow-no-l-c", action="store_true",
        help="Tier-C no-L_C mode: empty initial L_C + mutation OFF.",
    )

    # ── Dense reward (same as B4) ────────────────────────────────
    p.add_argument(
        "--use-dense-reward", action="store_true",
        help="Opt-in dense reward for training rollouts.",
    )

    # ── PRM (B6 — Process Reward Model) ──────────────────────────
    p.add_argument(
        "--prm-path", type=Path, default=None,
        help="Path to a trained PRM head .pt file (from "
             "scripts/run_prm_train.py). When set, every step of every "
             "rollout is scored by the PRM and the dense per-step rewards "
             "are passed to GAE instead of the sparse trajectory-level R. "
             "This is the B6 configuration. Default None (B5 / pure PPO).",
    )
    p.add_argument(
        "--prm-binary-safety-net", action="store_true",
        help="Add trajectory binary R to the PRM score at the terminal "
             "step (Math-Shepherd-style mix). Helps guard against PRM "
             "reward hacking. No-op when --prm-path is not set.",
    )

    p.add_argument("--dry-run", action="store_true",
                   help="Mock rollouts, no emulator, no model load. "
                        "PPO is NOT exercised in dry-run.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # ── Resolve app ──────────────────────────────────────────────
    resolved = _resolve_app(args.app)
    if resolved is None:
        logger.error(
            "App %r is not registered in configs/splits.yaml.", args.app,
        )
        return 2
    task_templates, app_category, pool_label = resolved
    allow_no_l_c_effective = args.allow_no_l_c or args.enable_bootstrap_fsm
    if pool_label != "tier_B" and not allow_no_l_c_effective:
        logger.error(
            "PPO default path requires a Tier-B app. Got %s in pool=%s. "
            "Pass --allow-no-l-c or --enable-bootstrap-fsm for Tier-C.",
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
        project_root / "traces" / "ppo_evolution" / args.app
    )
    l_c_dir = args.l_c_dir or (project_root / "artifacts" / "L_C")

    from evofsm_rl.fsm.aggregator import category_to_slug, load_L_C
    from evofsm_rl.fsm.schema import Layer2

    l_c_path = l_c_dir / f"{category_to_slug(app_category)}.json"
    l_c_missing = not l_c_path.exists()
    if l_c_missing:
        if not allow_no_l_c_effective:
            logger.error(
                "No L_C file at %s for app=%s (category=%s).",
                l_c_path, args.app, app_category,
            )
            return 2
        initial_l_c = Layer2(categories=[])
        mode_label = (
            "FSM bootstrap (option b)"
            if args.enable_bootstrap_fsm
            else "pure-LoRA TTA (option a)"
        )
        logger.info(
            "App=%s (%s) category=%s tasks=%d initial_l_c=<EMPTY> %s",
            args.app, pool_label, app_category, len(task_templates),
            mode_label,
        )
    else:
        _l_c_category, initial_l_c = load_L_C(l_c_path)
        logger.info(
            "App=%s (%s) category=%s tasks=%d initial_l_c=%s (n_cats=%d)",
            args.app, pool_label, app_category, len(task_templates),
            l_c_path, len(initial_l_c.categories),
        )

    # ── Sanity: --value-head-from must exist ─────────────────────
    if not args.value_head_from.exists():
        logger.error(
            "--value-head-from path does not exist: %s. Run "
            "scripts/run_ppo_value_pretrain.py first.",
            args.value_head_from,
        )
        return 2

    # ── Mutation API key check (same as B4) ──────────────────────
    mutation_will_run = (not l_c_missing) or args.enable_bootstrap_fsm
    if mutation_will_run and not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error(
            "ANTHROPIC_API_KEY is not set — mutation requires it.",
        )
        return 2

    episodes_dir = output_dir / "episodes"
    replay_dir = output_dir / "replay"

    # ── Build rollout fn (+ env + model if real) ─────────────────
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
        # PPO does not use a KL anchor / ref adapter (the clipped
        # surrogate already constrains the trust region). We pass
        # kl_beta=0 / ref_lora_from=None to the B4 rollout factory.
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
            ref_lora_from=None,
            use_dense_reward=args.use_dense_reward,
        )

        # ── If --prm-path given, wrap rollout_fn with PRM scoring ──
        # After each completed rollout we score every step via the PRM
        # and attach per_step_rewards to the trajectory. PPOTrainer will
        # use those dense per-step rewards in GAE instead of sparse R.
        if args.prm_path is not None:
            import dataclasses
            from evofsm_rl.rl_ppo.prm_scorer import load_prm_head, score_trajectory

            if not args.prm_path.exists():
                logger.error("--prm-path file not found: %s", args.prm_path)
                return 2
            logger.info("Loading PRM head from %s ...", args.prm_path)
            prm_head = load_prm_head(args.prm_path, model, args.device or "cuda")
            processor = getattr(model, "processor", None)
            if processor is None:
                logger.error(
                    "PRM scoring requires model.processor — "
                    "_build_b4_rollout_fn was expected to stash it."
                )
                return 2
            logger.info(
                "PRM scoring enabled. binary_safety_net=%s",
                args.prm_binary_safety_net,
            )

            _original_rollout_fn = rollout_fn
            _prm_device = args.device or "cuda"

            def _rollout_with_prm(fsm_prompt_text, task_name, seed):
                result = _original_rollout_fn(fsm_prompt_text, task_name, seed)
                if result.trajectory_data is None or result.episode_dir is None:
                    return result
                try:
                    per_step_rewards = score_trajectory(
                        result.episode_dir,
                        prm_head=prm_head,
                        model=model,
                        processor=processor,
                        device=_prm_device,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "PRM scoring failed for %s seed=%d — falling back to sparse R",
                        task_name, seed,
                    )
                    return result
                if per_step_rewards is None:
                    return result
                # If lengths mismatch, fall back rather than passing invalid data.
                if len(per_step_rewards) != len(result.trajectory_data.replay_paths):
                    logger.warning(
                        "PRM produced %d scores for %d replay_paths on %s seed=%d; "
                        "falling back to sparse R",
                        len(per_step_rewards),
                        len(result.trajectory_data.replay_paths),
                        task_name, seed,
                    )
                    return result
                # Optional: add binary R as safety net to the terminal step.
                if args.prm_binary_safety_net:
                    per_step_rewards = list(per_step_rewards)
                    per_step_rewards[-1] = per_step_rewards[-1] + float(result.trajectory_data.reward)
                new_traj = dataclasses.replace(
                    result.trajectory_data,
                    per_step_rewards=per_step_rewards,
                )
                return dataclasses.replace(result, trajectory_data=new_traj)

            rollout_fn = _rollout_with_prm

    if args.dry_run:
        logger.error(
            "PPO is not exercised in --dry-run (no model loaded). Either "
            "drop --dry-run or use scripts/run_b3_evolution.py for a "
            "trainer-free smoke test.",
        )
        return 2

    from evofsm_rl.fsm.evolution import EvolutionConfig
    from evofsm_rl.rl_ppo import PPOEvolutionConfig, run_l_c_evolution_ppo

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
        enable_lora=True,
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
        lora_kl_beta=0.0,                       # PPO has no KL anchor.
        lora_ref_adapter_name="ref",
        lora_kl_log_ratio_clip=10.0,
        bootstrap_fsm=args.enable_bootstrap_fsm,
    )

    ppo_config = PPOEvolutionConfig(
        value_lr=args.value_lr,
        clip_eps=args.clip_eps,
        value_loss_coef=args.value_loss_coef,
        gae_gamma=args.gae_gamma,
        gae_lambda=args.gae_lambda,
        gae_lambda_after_iter=args.gae_lambda_switch_after_iter,
        gae_lambda_after_value=args.gae_lambda_after_value,
        value_head_path=args.value_head_from,
    )

    logger.info(
        "PPO config: lora_lr=%g value_lr=%g clip_eps=%g lambda=%g "
        "lambda_switch=%s gamma=%g value_coef=%g update_every=%d",
        args.lora_lr, args.value_lr, args.clip_eps,
        args.gae_lambda, args.gae_lambda_switch_after_iter,
        args.gae_gamma, args.value_loss_coef,
        args.lora_update_every,
    )

    t_run = time.monotonic()
    try:
        population = run_l_c_evolution_ppo(
            initial_l_c=initial_l_c,
            app_name=args.app,
            app_category=app_category,
            task_templates=task_templates,
            rollout_fn=rollout_fn,
            config=config,
            ppo_config=ppo_config,
            output_dir=output_dir,
            resume=not args.no_resume,
            model=model,
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
    print(f"Trainer:         PPO + GAE (λ={args.gae_lambda}"
          + (f" → {args.gae_lambda_after_value} after iter {args.gae_lambda_switch_after_iter}"
             if args.gae_lambda_switch_after_iter else "")
          + ")")
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
    print(f"LoRA ckpts   -> {output_dir / 'lora_checkpoints'}")
    print(f"PPO metrics  -> {output_dir / 'ppo_metrics.jsonl'}")
    print(f"Value head   -> {output_dir / 'value_head_checkpoints'}")

    # Convergence plot — uses the same util as B3/B4; the per-iter log
    # schema is identical.
    try:
        from plot_convergence import plot_convergence

        plot_path = plot_convergence(
            log_dir=output_dir,
            app_name=args.app,
            n_iterations=config.n_iterations,
            has_grpo=True,
        )
        if plot_path is not None:
            print(f"Convergence  -> {plot_path}")
    except Exception:
        logger.warning("plot_convergence failed", exc_info=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

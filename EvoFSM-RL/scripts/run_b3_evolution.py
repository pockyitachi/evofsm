#!/usr/bin/env python3
"""Run B3 L_C evolution for one testing-set app.

B3 starts from the same static ``L_C`` (category-level abstract action
library) as B2, then evolves it on the held-out app's **T_adapt** task
set. Only LAYER 2 is ever updated; the per-app LAYER 1 stays empty.
T_eval is never touched during evolution — it is reserved for the
post-evolution frozen evaluation.

Glues :func:`evofsm_rl.fsm.evolution.run_l_c_evolution` to the real
Qwen3-VL agent + AWAvd2 emulator. Mirrors ``scripts/run_b2_eval.py``'s
model-load and env-connect pattern.

Examples::

    # Live run (tmux window with ANTHROPIC_API_KEY set, model + emulator):
    export ANDROID_HOME=/shared/linqiang/evofsm_project/android-sdk
    export CUDA_VISIBLE_DEVICES=2
    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      python EvoFSM-RL/scripts/run_b3_evolution.py \\
        --app pro_expense \\
        --iterations 20 \\
        --output-dir EvoFSM-RL/traces/b3_evolution/pro_expense \\
        --console-port 5710 --grpc-port 8710 \\
        --adb-path $ANDROID_HOME/platform-tools/adb

    # Dry-run (no model, no emulator, mock rollouts):
    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      python EvoFSM-RL/scripts/run_b3_evolution.py \\
        --app pro_expense --iterations 5 --dry-run

A run is automatically resumed if ``{output-dir}/population.json`` exists
(unless ``--no-resume`` is passed).

Tier-C apps have no matching source-pool ``L_C`` and therefore cannot
run B3 — the script exits cleanly with a message in that case.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random as _random
import sys
import time
from pathlib import Path
from typing import Any


logger = logging.getLogger("run_b3_evolution")


# ─────────────────────────────────────────────────────────────────────
# App resolution (source_pool ∪ tier_B ∪ tier_C via splits.py loaders)
# ─────────────────────────────────────────────────────────────────────


def _resolve_app(app_name: str) -> tuple[list[str], str, str] | None:
    """Resolve an app name to (task_templates, category, pool_label).

    Preference order:
      * Held-out apps (Tier-B / Tier-C) — use T_adapt (the adaptation
        set). T_eval is reserved for frozen post-evolution evaluation.
      * Source-pool apps — use the full templates list. B3 evolution on
        a source app is a sanity-check / ablation, not the main use case.

    Returns ``None`` when the app is unknown.
    """
    from evofsm_rl.splits import (
        get_source_pool, get_tier_B_apps, get_tier_C_apps,
    )

    for loader, label in (
        (get_tier_B_apps, "tier_B"),
        (get_tier_C_apps, "tier_C"),
    ):
        pool = loader()
        if app_name in pool:
            info = pool[app_name]
            return list(info.T_adapt), info.category, label

    source = get_source_pool()
    if app_name in source:
        info = source[app_name]
        return list(info.templates), info.category, "source"

    return None


# ─────────────────────────────────────────────────────────────────────
# Mock rollout (for --dry-run smoke testing the loop)
# ─────────────────────────────────────────────────────────────────────


def _build_mock_rollout_fn(
    seed_base: int,
    episodes_dir: Path,
    app_name: str,
):
    """Mock rollout_fn that returns ~30% success and writes throwaway
    episode directories with the minimum schema the mutation pipeline
    expects. Used only by ``--dry-run`` for loop sanity checks."""
    rng = _random.Random(seed_base)

    def mock(fsm_prompt_text: str, task_name: str, seed: int):
        from evofsm_rl.fsm.evolution import RolloutResult

        reward = 1.0 if rng.random() < 0.3 else 0.0
        n_steps = rng.randint(3, 15)

        ep_dir = episodes_dir / f"{task_name}_seed{seed}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "meta.json").write_text(json.dumps({
            "app": app_name,
            "template": task_name,
            "seed": seed,
            "reward": reward,
            "success": reward,
            "goal": f"[dry-run] {task_name}",
            "n_steps": n_steps,
        }))
        with (ep_dir / "episode.jsonl").open("w") as fh:
            for step in range(1, n_steps + 1):
                fh.write(json.dumps({
                    "step": step,
                    "action": {"action_type": "click", "index": step},
                    "action_reason": f"[mock] step {step}",
                    "summary": f"[mock] did step {step}",
                    "before_ui_elements_text": [f"ui_elem_{k}" for k in range(3)],
                    "status": "ok",
                }) + "\n")
        return RolloutResult(
            task_name=task_name,
            seed=seed,
            reward=reward,
            n_steps=n_steps,
            wall_seconds=0.01,
            episode_dir=ep_dir,
        )

    return mock


# ─────────────────────────────────────────────────────────────────────
# Real rollout (Qwen3-VL agent + AWAvd2 emulator)
# ─────────────────────────────────────────────────────────────────────


def _build_real_rollout_fn(
    *,
    app_name: str,
    pool_label: str,
    episodes_dir: Path,
    console_port: int,
    grpc_port: int,
    adb_path: str | None,
    device: str | None,
    max_steps_multiplier: float,
    emulator_setup: bool,
):
    """Construct a closure that runs one real episode per call.

    Loads the model + connects to the emulator once up front and
    returns both the rollout function and the env handle (caller
    is responsible for ``env.close()``).
    """
    from evofsm_rl.agent.rollout import GenerationConfig, Qwen3VLAgent
    from evofsm_rl.env import harness
    from evofsm_rl.model import (
        load_base_model, load_model_config, resolve_device,
    )

    resolved_device = device or resolve_device()
    logger.info("Loading Qwen3-VL-8B on device=%s", resolved_device)
    t0 = time.monotonic()
    model, processor = load_base_model(device=resolved_device)
    logger.info("Model loaded in %.1fs", time.monotonic() - t0)

    cfg = load_model_config(device=resolved_device)
    gen_cfg = GenerationConfig.from_yaml(cfg.raw.get("generation", {}))

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
    )

    episodes_dir.mkdir(parents=True, exist_ok=True)

    def real(fsm_prompt_text: str, task_name: str, seed: int):
        """Run one episode with the provided FSM prompt injected."""
        from evofsm_rl.fsm.evolution import RolloutResult

        # Reuse the B2 injection channel: the agent splices whatever
        # text we give it right after PROMPT_PREFIX. In B2 this was L_C
        # only; in B3 we pass the full FSM prompt (Layer 1 + Layer 2).
        agent.set_l_c_prompt_text(fsm_prompt_text)

        t_start = time.monotonic()
        try:
            result = harness.run_template(
                template_name=task_name,
                seed=seed,
                env=env,
                agent=agent,
                max_steps_multiplier=max_steps_multiplier,
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
            )

        # Persist the trace only if the episode actually ran.
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

        return RolloutResult(
            task_name=task_name,
            seed=seed,
            reward=float(result.success),
            n_steps=int(result.n_steps),
            wall_seconds=float(result.wall_seconds),
            episode_dir=ep_dir,
            error=result.error,
        )

    return real, env


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run B3 evolution loop for one target app.",
    )
    p.add_argument("--app", required=True,
                   help="Target app name (key in configs/splits.yaml). "
                        "Must belong to a category that has an L_C in "
                        "artifacts/L_C/ — Tier-C apps are not supported.")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--m-select", type=int, default=2)
    p.add_argument("--n-rollouts", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Defaults to EvoFSM-RL/traces/b3_evolution/{app}.")
    p.add_argument("--l-c-dir", type=Path, default=None,
                   help="Directory holding the per-category L_C files. "
                        "Defaults to EvoFSM-RL/artifacts/L_C.")
    p.add_argument("--splits-yaml", type=Path, default=None,
                   help="Path to splits.yaml. Defaults to "
                        "EvoFSM-RL/configs/splits.yaml.")
    p.add_argument("--console-port", type=int, default=5710)
    p.add_argument("--grpc-port", type=int, default=8710)
    p.add_argument("--adb-path", type=str, default=None)
    p.add_argument("--device", choices=("mps", "cuda", "cpu"), default=None)
    p.add_argument("--emulator-setup", action="store_true")
    p.add_argument("--max-steps-multiplier", type=float, default=10.0)
    p.add_argument("--dry-run", action="store_true",
                   help="Use mock rollouts — no model, no emulator, "
                        "no ANTHROPIC_API_KEY requirement beyond mutation.")
    p.add_argument("--seed-base", type=int, default=100)
    p.add_argument("--window-size", type=int, default=15)
    p.add_argument("--mutation-model", default="claude-opus-4-7")
    p.add_argument("--task-sample-mode", choices=("random", "round_robin"),
                   default="random")
    p.add_argument("--mutation-every-n-iters", type=int, default=3)
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore existing checkpoint and start fresh.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # Resolve app
    resolved = _resolve_app(args.app)
    if resolved is None:
        logger.error(
            "App %r is not registered in configs/splits.yaml (source_pool / "
            "tier_B_held_out / tier_C_held_out).", args.app,
        )
        return 2
    task_templates, app_category, pool_label = resolved
    if not task_templates:
        logger.error(
            "App %r resolved to an empty task list (pool=%s). T_adapt "
            "may not be populated for this app yet.",
            args.app, pool_label,
        )
        return 2

    # Resolve paths
    project_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or (
        project_root / "traces" / "b3_evolution" / args.app
    )
    l_c_dir = args.l_c_dir or (project_root / "artifacts" / "L_C")
    splits_yaml = args.splits_yaml or (
        project_root / "configs" / "splits.yaml"
    )

    # Load the matching L_C for this app's category.
    from evofsm_rl.fsm.aggregator import category_to_slug, load_L_C

    l_c_path = l_c_dir / f"{category_to_slug(app_category)}.json"
    if not l_c_path.exists():
        logger.error(
            "No L_C file at %s for app=%s (category=%s). This is the "
            "expected situation for Tier-C apps whose category is not in "
            "the source pool — B3 cannot run on those apps. Build an L_C "
            "for this category first, or pick a different app.",
            l_c_path, args.app, app_category,
        )
        return 2

    _l_c_category, initial_l_c = load_L_C(l_c_path)
    logger.info(
        "App=%s (%s) category=%s  tasks=%d  initial_l_c=%s (n_cats=%d)",
        args.app, pool_label, app_category, len(task_templates),
        l_c_path, len(initial_l_c.categories),
    )

    # Check ANTHROPIC_API_KEY up front (mutation will need it regardless
    # of dry-run, since dry-run still mutates the FSM between iterations).
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error(
            "ANTHROPIC_API_KEY is not set — mutation requires it. "
            "Export it in the tmux session before invoking this script.",
        )
        return 2

    # Build rollout function
    episodes_dir = output_dir / "episodes"
    env = None
    if args.dry_run:
        rollout_fn = _build_mock_rollout_fn(
            seed_base=args.seed_base,
            episodes_dir=episodes_dir,
            app_name=args.app,
        )
    else:
        rollout_fn, env = _build_real_rollout_fn(
            app_name=args.app,
            pool_label=pool_label,
            episodes_dir=episodes_dir,
            console_port=args.console_port,
            grpc_port=args.grpc_port,
            adb_path=args.adb_path,
            device=args.device,
            max_steps_multiplier=args.max_steps_multiplier,
            emulator_setup=args.emulator_setup,
        )

    # Config
    from evofsm_rl.fsm.evolution import EvolutionConfig, run_l_c_evolution

    config = EvolutionConfig(
        n_iterations=args.iterations,
        m_select=args.m_select,
        n_rollouts=args.n_rollouts,
        task_sample_mode=args.task_sample_mode,
        mutation_model=args.mutation_model,
        seed_base=args.seed_base,
        window_size=args.window_size,
        mutation_every_n_iters=args.mutation_every_n_iters,
    )

    # Run
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
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                logger.warning("env.close() failed", exc_info=True)

    # Report
    wall_min = (time.monotonic() - t_run) / 60
    print()
    print("=" * 64)
    print(f"App:             {args.app}  ({pool_label}, {app_category})")
    print(f"Iterations:      {config.n_iterations}")
    print(f"Population size: {population.size}")
    print(f"Champion:        {population.champion.id}")
    print(f"Champion mu:     {population.champion.rating.mu:.2f}")
    print(f"Champion sigma:  {population.champion.rating.sigma:.2f}")
    print(f"Output dir:      {output_dir}")
    print(f"Wall time:       {wall_min:.1f} min")
    print("=" * 64)

    # run_l_c_evolution writes l_c_champion.json on every mutation; make
    # sure the final state is reflected there even if the last iteration
    # was a skip.
    champion_path = output_dir / "l_c_champion.json"
    champion_path.write_text(
        json.dumps(population.champion.fsm.to_json(), indent=2)
    )
    print(f"L_C champion -> {champion_path}")
    print(f"Initial L_C  -> {output_dir / 'l_c_v0_initial.json'}")

    # Convergence plot (best-effort — a failure here never fails the run).
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from plot_convergence import plot_convergence

        plot_path = plot_convergence(
            log_dir=output_dir,
            app_name=args.app,
            n_iterations=config.n_iterations,
            has_grpo=False,
        )
        if plot_path is not None:
            print(f"Convergence  -> {plot_path}")
    except Exception:
        logger.warning("plot_convergence failed", exc_info=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

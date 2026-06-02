#!/usr/bin/env python3
"""Evaluate B4 (joint LoRA + evolved L_C) on $T_{\\text{eval}}$ (held-out tasks).

For each Tier-B and Tier-C target app, runs every $T_{\\text{eval}}$ template
across K seeds with:

  * The frozen LoRA adapter from
    ``traces/b4_phase3_v01/{app}/lora_checkpoints/final``
    (produced by ``run_b4_evolution.py`` after the 20-iter Phase 3 sweep).
  * The frozen L_C champion from
    ``traces/b4_phase3_v01/{app}/l_c_champion.json``
    (also written by the Phase 3 sweep). For Tier-C apps this is the
    empty-LAYER 2 stub used during the no-L_C Phase 3 run; the resulting
    prompt is byte-identical to B1 on Tier-C.

This is the test-time half of the EvoFSM-RL framework: both axes are
frozen, the agent rolls out one episode per (template, seed), and we
score success with AndroidWorld's rule-based grader. The resulting
numbers are the paper's headline B4 row, comparable with the B1/B2/B3
arms reported by ``run_b3_teval.py`` (which uses the same seed list
``[40, 41, 42]`` by default).

Resumes from CSV: rows already present are not re-run, so an
interrupted sweep can be re-launched cheaply.

Usage::

    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      python EvoFSM-RL/scripts/run_b4_teval.py \\
        --console-port 5710 --grpc-port 8710 \\
        --adb-path $ANDROID_HOME/platform-tools/adb \\
        --seeds 40 41 42 \\
        --phase3-dir EvoFSM-RL/traces/b4_phase3_v01 \\
        --output-dir EvoFSM-RL/traces/b4_teval

Optional flags::

    --apps simple_calendar_pro system_settings    # subset of apps
    --include-tier-c                              # include Tier-C apps
    --device cuda|mps|cpu                         # override device autodetect
    --summary-only                                # skip running, print summary
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("run_b4_teval")


# ─────────────────────────────────────────────────────────────────────
# L_C text rendering (shared with run_b3_teval.py via duplicate)
# ─────────────────────────────────────────────────────────────────────


def _render_l_c_from_fsm_json(path: Path) -> str | None:
    """Render the LAYER-2 block of an L_C champion JSON to prompt text.

    Returns ``None`` if the FSM has an empty LAYER 2 (the Tier-C
    `--allow-no-l-c` case during Phase 3). The agent then runs with no
    L_C injection, identically to the B1 prompt.
    """
    from evofsm_rl.fsm.aggregator import load_L_C
    from evofsm_rl.fsm.schema import FSM, Layer2

    # Champion files written by run_b4_evolution.py are FSM JSON dicts
    # with both layer1 and layer2 keys (we wrote `population.champion.fsm.to_json()`).
    data = json.loads(path.read_text())
    if "layer2" in data:
        layer2 = Layer2.from_json(data["layer2"])
    else:
        # Fallback: treat as a raw Layer2 dict.
        layer2 = Layer2.from_json(data)
    category = data.get("layer1", {}).get("category", "")
    if not layer2.categories:
        return None
    return layer2.to_prompt_text(category=category)


# ─────────────────────────────────────────────────────────────────────
# CSV I/O
# ─────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "baseline",  # always "b4_joint" in this script
    "app",
    "tier",
    "category",
    "template",
    "seed",
    "success",
    "n_steps",
    "wall_seconds",
    "error",
    "lora_path",
    "champion_path",
]


def _init_csv(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()


def _append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _load_completed(path: Path) -> set[tuple[str, str, str, int]]:
    """Return the (baseline, app, template, seed) tuples already in CSV."""
    if not path.exists():
        return set()
    done: set[tuple[str, str, str, int]] = set()
    with path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                done.add((row["baseline"], row["app"], row["template"], int(row["seed"])))
            except (KeyError, ValueError):
                continue
    return done


# ─────────────────────────────────────────────────────────────────────
# Plan
# ─────────────────────────────────────────────────────────────────────


def _build_plan(
    apps: list[str],
    seeds: list[int],
    phase3_dir: Path,
) -> list[dict[str, Any]]:
    """Enumerate every (app, template, seed) episode to evaluate.

    Resolves per-app `lora_checkpoints/final` and `l_c_champion.json`
    paths from the Phase 3 sweep output dir, and fails fast if any are
    missing.
    """
    from evofsm_rl.splits import get_tier_B_apps, get_tier_C_apps

    tb = get_tier_B_apps()
    tc = get_tier_C_apps()

    plan: list[dict[str, Any]] = []
    for app in apps:
        if app in tb:
            info = tb[app]
            tier = "tier_B"
        elif app in tc:
            info = tc[app]
            tier = "tier_C"
        else:
            raise ValueError(
                f"app {app!r} is not in Tier-B or Tier-C. "
                f"Tier-B: {sorted(tb.keys())}. "
                f"Tier-C: {sorted(tc.keys())}."
            )

        templates = list(info.T_eval)
        if not templates:
            logger.warning("app %s has empty T_eval; skipping", app)
            continue

        app_dir = phase3_dir / app
        lora_path = app_dir / "lora_checkpoints" / "final"
        champ_path = app_dir / "l_c_champion.json"
        if not lora_path.exists():
            raise FileNotFoundError(
                f"B4 final LoRA missing for {app}: {lora_path}. "
                f"Run Phase 3 sweep for this app first."
            )
        if not champ_path.exists():
            raise FileNotFoundError(
                f"B4 L_C champion missing for {app}: {champ_path}."
            )

        for template in templates:
            for seed in seeds:
                plan.append({
                    "app": app,
                    "tier": tier,
                    "category": info.category,
                    "template": template,
                    "seed": seed,
                    "lora_path": str(lora_path),
                    "champion_path": str(champ_path),
                })
    return plan


# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────


def _print_summary(csv_path: Path, apps: list[str]) -> None:
    if not csv_path.exists():
        logger.warning("No results CSV at %s yet.", csv_path)
        return
    rows = []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            rows.append(row)

    per_app: dict[str, dict[str, float]] = {}
    per_tier_succ: dict[str, list[float]] = {"tier_B": [], "tier_C": []}
    for row in rows:
        try:
            s = float(row["success"])
        except ValueError:
            continue
        app = row["app"]
        tier = row.get("tier", "tier_B")
        per_app.setdefault(app, {"sum": 0.0, "n": 0})
        per_app[app]["sum"] += s
        per_app[app]["n"] += 1
        if tier in per_tier_succ:
            per_tier_succ[tier].append(s)

    print()
    print("=" * 64)
    print("B4 T_eval results (success rate per app)")
    print("=" * 64)
    print(f"{'App':<25}{'Tier':<10}{'n':>5}{'SR':>10}")
    print("-" * 64)
    for app in sorted(per_app.keys()):
        n = per_app[app]["n"]
        sr = per_app[app]["sum"] / n * 100 if n else 0.0
        tier = next((r["tier"] for r in rows if r["app"] == app), "")
        print(f"{app:<25}{tier:<10}{n:>5}{sr:>9.1f}%")
    print("-" * 64)
    for tier, vals in per_tier_succ.items():
        if vals:
            sr = sum(vals) / len(vals) * 100
            print(f"{tier} overall: n={len(vals)}, SR={sr:.1f}%")
    print("=" * 64)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--apps", nargs="*", default=None,
                   help="Subset of apps to evaluate. Default: all Tier-B "
                        "(+ Tier-C when --include-tier-c is passed).")
    p.add_argument("--include-tier-c", action="store_true",
                   help="Also evaluate every Tier-C app with non-empty T_eval.")
    p.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42],
                   help="Task seeds. Default: 40 41 42 (matches B3 T_eval).")
    p.add_argument("--phase3-dir", type=Path,
                   default=Path("EvoFSM-RL/traces/b4_phase3_v01"),
                   help="Directory containing per-app B4 Phase 3 outputs "
                        "(lora_checkpoints/final + l_c_champion.json).")
    p.add_argument("--output-dir", type=Path,
                   default=Path("EvoFSM-RL/traces/b4_teval"))
    p.add_argument("--console-port", type=int, default=5710)
    p.add_argument("--grpc-port", type=int, default=8710)
    p.add_argument("--adb-path", type=str, default=None)
    p.add_argument("--device", choices=("mps", "cuda", "cpu"), default=None)
    p.add_argument("--max-steps-multiplier", type=float, default=10.0)
    p.add_argument("--emulator-setup", action="store_true")
    p.add_argument("--lora-rank", type=int, default=16,
                   help="Rank of the LoRA slot we attach before loading each "
                        "app's adapter. Must match Phase 3 sweep value (16).")
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", type=str,
                   default="q_proj,v_proj",
                   help="Must match the modules used during Phase 3.")
    p.add_argument("--fixed-lora-path", type=str, default=None,
                   help="If set, load this single LoRA adapter once at startup "
                        "and DO NOT hot-swap per app. L_C champions are still "
                        "hot-swapped from --phase3-dir/{app}/l_c_champion.json. "
                        "Use this to evaluate 'pi^pre + B4 L_C' (the (A) "
                        "diagnostic ablation): isolates damage from the Phase 3 "
                        "LoRA training by reverting LoRA to the Phase 1 "
                        "pretrained checkpoint while keeping the B4-evolved "
                        "FSM L_C.")
    p.add_argument("--baseline-tag", type=str, default="b4_joint",
                   help="Tag written to the 'baseline' column of results.csv. "
                        "Use a distinct tag (e.g. 'b4_revert_lora') when "
                        "running --fixed-lora-path so the rows are separable.")
    p.add_argument("--summary-only", action="store_true",
                   help="Skip running; print summary from existing CSV.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    from evofsm_rl.splits import get_tier_B_apps, get_tier_C_apps

    tb = list(get_tier_B_apps().keys())
    tc_with_eval = [
        app for app, info in get_tier_C_apps().items() if info.T_eval
    ]
    if args.apps:
        apps = args.apps
    elif args.include_tier_c:
        apps = tb + tc_with_eval
    else:
        apps = tb
    project_root = Path(__file__).resolve().parents[1]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(exist_ok=True)

    if args.summary_only:
        _print_summary(csv_path, apps)
        return 0

    plan = _build_plan(apps=apps, seeds=args.seeds, phase3_dir=args.phase3_dir)
    if not plan:
        logger.error("Empty plan — nothing to do.")
        return 2

    _init_csv(csv_path)
    completed = _load_completed(csv_path)
    to_run = [
        ep for ep in plan
        if (args.baseline_tag, ep["app"], ep["template"], ep["seed"]) not in completed
    ]
    logger.info(
        "Plan: %d episodes total, %d already done, %d to run.",
        len(plan), len(plan) - len(to_run), len(to_run),
    )
    if not to_run:
        logger.info("All episodes already completed — printing summary.")
        _print_summary(csv_path, apps)
        return 0

    # Load model + attach a fresh LoRA slot once. We then call
    # ``load_lora_checkpoint`` per app to hot-swap weights into the
    # default adapter rather than re-attaching from scratch.
    from evofsm_rl.agent.rollout import GenerationConfig, Qwen3VLAgent
    from evofsm_rl.env import harness
    from evofsm_rl.model import (
        load_base_model,
        load_model_config,
        resolve_device,
    )
    from evofsm_rl.model.lora import (
        attach_lora,
        count_trainable_params,
        load_lora_checkpoint,
    )

    device = args.device or resolve_device()
    logger.info("Loading Qwen3-VL on device=%s", device)
    t_load = time.monotonic()
    model, processor = load_base_model(device=device)
    logger.info("Model loaded in %.1fs", time.monotonic() - t_load)

    cfg = load_model_config(device=device)
    gen_cfg = GenerationConfig.from_yaml(cfg.raw.get("generation", {}))

    target_modules = tuple(
        m.strip() for m in args.lora_target_modules.split(",") if m.strip()
    )
    logger.info("Attaching LoRA slot rank=%d targets=%s", args.lora_rank, target_modules)
    model = attach_lora(
        model,
        rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    counts = count_trainable_params(model)
    logger.info(
        "LoRA slot attached: trainable=%d / total=%d (%.3f%%)",
        counts["trainable"], counts["total"], counts["percent"],
    )

    if args.fixed_lora_path is not None:
        logger.info(
            "Fixed LoRA mode: loading %s once; per-app LoRA hot-swap DISABLED.",
            args.fixed_lora_path,
        )
        model = load_lora_checkpoint(model, args.fixed_lora_path)

    connect_kwargs: dict[str, Any] = {
        "console_port": args.console_port,
        "grpc_port": args.grpc_port,
        "emulator_setup": args.emulator_setup,
    }
    if args.adb_path:
        connect_kwargs["adb_path"] = args.adb_path
    logger.info("Connecting emulator %s", connect_kwargs)
    env = harness.connect(**connect_kwargs)

    # Eval-mode agent: no log-prob collection, no replay tensors. T_eval
    # never trains, so we skip the grad-bearing forward setup.
    agent = Qwen3VLAgent(
        model=model, processor=processor, env=env,
        device=device, generation_config=gen_cfg,
    )

    # Sweep. We hot-swap LoRA + L_C text when crossing app boundaries.
    current_app: str | None = None
    overall_t0 = time.monotonic()
    try:
        for i, ep in enumerate(to_run, start=1):
            app = ep["app"]
            template = ep["template"]
            seed = ep["seed"]
            lora_path = ep["lora_path"]
            champ_path = ep["champion_path"]

            if app != current_app:
                if args.fixed_lora_path is None:
                    logger.info(
                        "→ switching to app %s: loading LoRA %s + champion %s",
                        app, lora_path, champ_path,
                    )
                    model = load_lora_checkpoint(model, lora_path)
                    agent.model = model  # noqa: SLF001 (agent stores a ref)
                else:
                    logger.info(
                        "→ switching to app %s: keeping fixed LoRA, loading champion %s",
                        app, champ_path,
                    )
                l_c_text = _render_l_c_from_fsm_json(Path(champ_path))
                agent.set_l_c_prompt_text(l_c_text)
                current_app = app

            t0 = time.monotonic()
            logger.info(
                "[%d/%d] %s | %s | seed=%d",
                i, len(to_run), app, template, seed,
            )
            try:
                result = harness.run_template(
                    template_name=template,
                    seed=seed,
                    env=env,
                    agent=agent,
                    max_steps_multiplier=args.max_steps_multiplier,
                )
                success = float(result.success)
                n_steps = int(result.n_steps)
                wall = float(result.wall_seconds)
                err_msg = result.error

                if err_msg is None and n_steps > 0:
                    try:
                        agent.save_episode(
                            episodes_dir,
                            success=success,
                            template=template,
                            seed=seed,
                            app=app,
                            tier=ep["tier"],
                        )
                    except Exception:
                        logger.exception(
                            "save_episode failed for %s seed=%d", template, seed,
                        )
            except Exception as e:
                logger.exception(
                    "Rollout crashed: %s seed=%d", template, seed,
                )
                success = 0.0
                n_steps = 0
                wall = time.monotonic() - t0
                err_msg = f"{type(e).__name__}: {e}"

            row = {
                "baseline": args.baseline_tag,
                "app": app,
                "tier": ep["tier"],
                "category": ep["category"],
                "template": template,
                "seed": seed,
                "success": success,
                "n_steps": n_steps,
                "wall_seconds": f"{wall:.1f}",
                "error": err_msg or "",
                "lora_path": (
                    args.fixed_lora_path if args.fixed_lora_path is not None
                    else lora_path
                ),
                "champion_path": champ_path,
            }
            _append_row(csv_path, row)
            logger.info(
                "  → success=%.2f n_steps=%d wall=%.1fs err=%s",
                success, n_steps, wall, err_msg or "-",
            )
    finally:
        try:
            env.close()
        except Exception:
            logger.warning("env.close() failed", exc_info=True)

    overall_wall = (time.monotonic() - overall_t0) / 60
    logger.info("Sweep done in %.1f min.", overall_wall)
    _print_summary(csv_path, apps)
    return 0


if __name__ == "__main__":
    sys.exit(main())

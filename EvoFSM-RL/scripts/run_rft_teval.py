#!/usr/bin/env python3
"""Evaluate the RFT-trained LoRA on $T_{eval}$ (Tier-B + Tier-C, 35 templates).

Thin variant of ``run_b4_teval.py`` for the Rejection Fine-Tuning baseline:

  * Loads a single fixed LoRA adapter (no per-app hot-swap).
  * Disables L_C injection on **all** apps — RFT does not use any
    workflow knowledge, the prompt is byte-identical to B1.
  * Same 35-template T_eval list as B1/B2/B3/B4, same K=3 seeds.

Usage::

    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      CUDA_VISIBLE_DEVICES=5 EVOFSM_ADB_SERVER_PORT=5050 \\
      python EvoFSM-RL/scripts/run_rft_teval.py \\
        --init-lora-from EvoFSM-RL/traces/rft_v01/lora_checkpoints/final \\
        --console-port 5720 --grpc-port 8720 \\
        --adb-path $ANDROID_HOME/platform-tools/adb \\
        --output-dir EvoFSM-RL/traces/rft_v01_teval
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

logger = logging.getLogger("run_rft_teval")


CSV_FIELDS = [
    "baseline", "app", "tier", "category", "template", "seed",
    "success", "n_steps", "wall_seconds", "error", "lora_path",
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


def _build_plan(apps: list[str], seeds: list[int]) -> list[dict[str, Any]]:
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
            raise ValueError(f"Unknown app {app!r}")
        templates = list(info.T_eval)
        if not templates:
            logger.warning("Skipping %s (empty T_eval)", app)
            continue
        for template in templates:
            for seed in seeds:
                plan.append({
                    "app": app,
                    "tier": tier,
                    "category": info.category,
                    "template": template,
                    "seed": seed,
                })
    return plan


def _print_summary(csv_path: Path, baseline_tag: str) -> None:
    if not csv_path.exists():
        logger.warning("No CSV at %s", csv_path)
        return
    rows = []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("baseline") != baseline_tag:
                continue
            rows.append(row)

    per_app: dict[str, dict[str, float]] = {}
    per_tier: dict[str, list[float]] = {"tier_B": [], "tier_C": []}
    for row in rows:
        try:
            s = float(row["success"])
        except ValueError:
            continue
        app = row["app"]
        tier = row.get("tier", "tier_B")
        per_app.setdefault(app, {"sum": 0.0, "n": 0, "tier": tier})
        per_app[app]["sum"] += s
        per_app[app]["n"] += 1
        if tier in per_tier:
            per_tier[tier].append(s)

    print()
    print("=" * 64)
    print(f"RFT T_eval results (baseline={baseline_tag})")
    print("=" * 64)
    print(f"{'App':<25}{'Tier':<10}{'n':>5}{'SR':>10}")
    print("-" * 64)
    for app in sorted(per_app.keys()):
        d = per_app[app]
        sr = d["sum"] / d["n"] * 100 if d["n"] else 0.0
        print(f"{app:<25}{d['tier']:<10}{d['n']:>5}{sr:>9.1f}%")
    print("-" * 64)
    for tier, vals in per_tier.items():
        if vals:
            sr = sum(vals) / len(vals) * 100
            print(f"{tier} overall: n={len(vals)}, SR={sr:.1f}%")
    all_vals = per_tier["tier_B"] + per_tier["tier_C"]
    if all_vals:
        print(f"OVERALL:           n={len(all_vals)}, SR={sum(all_vals)/len(all_vals)*100:.1f}%")
    print("=" * 64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--init-lora-from", type=str, required=True,
                   help="Path to the RFT-trained LoRA adapter directory.")
    p.add_argument("--apps", nargs="*", default=None)
    p.add_argument("--include-tier-c", action="store_true", default=True,
                   help="Include Tier-C apps (default ON — matches B1/B2/B3 T_eval).")
    p.add_argument("--exclude-tier-c", action="store_true",
                   help="If set, only Tier-B apps.")
    p.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42])
    p.add_argument("--output-dir", type=Path,
                   default=Path("EvoFSM-RL/traces/rft_v01_teval"))
    p.add_argument("--console-port", type=int, default=5720)
    p.add_argument("--grpc-port", type=int, default=8720)
    p.add_argument("--adb-path", type=str, default=None)
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    p.add_argument("--max-steps-multiplier", type=float, default=10.0)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", type=str, default="q_proj,v_proj")
    p.add_argument("--baseline-tag", type=str, default="rft")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    from evofsm_rl.splits import get_tier_B_apps, get_tier_C_apps
    tb = list(get_tier_B_apps().keys())
    tc_with_eval = [app for app, info in get_tier_C_apps().items() if info.T_eval]
    if args.apps:
        apps = args.apps
    elif args.exclude_tier_c:
        apps = tb
    else:
        apps = tb + tc_with_eval

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(exist_ok=True)

    if args.summary_only:
        _print_summary(csv_path, args.baseline_tag)
        return 0

    plan = _build_plan(apps=apps, seeds=args.seeds)
    if not plan:
        logger.error("Empty plan.")
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
        _print_summary(csv_path, args.baseline_tag)
        return 0

    # ── Load model + attach LoRA + load RFT adapter ────────────────
    from evofsm_rl.agent.rollout import GenerationConfig, Qwen3VLAgent
    from evofsm_rl.env import harness
    from evofsm_rl.model import load_base_model, load_model_config, resolve_device
    from evofsm_rl.model.lora import attach_lora, count_trainable_params, load_lora_checkpoint

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
    model = attach_lora(
        model, rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=target_modules,
    )
    counts = count_trainable_params(model)
    logger.info(
        "LoRA slot attached: trainable=%d / total=%d (%.3f%%)",
        counts["trainable"], counts["total"], counts["percent"],
    )

    lora_path = args.init_lora_from
    if not Path(lora_path).exists():
        logger.error("LoRA path not found: %s", lora_path)
        return 2
    logger.info("Loading RFT LoRA from %s", lora_path)
    model = load_lora_checkpoint(model, lora_path)
    model.eval()

    connect_kwargs: dict[str, Any] = {
        "console_port": args.console_port,
        "grpc_port": args.grpc_port,
        "emulator_setup": False,
    }
    if args.adb_path:
        connect_kwargs["adb_path"] = args.adb_path
    logger.info("Connecting emulator %s", connect_kwargs)
    env = harness.connect(**connect_kwargs)

    # Agent with no L_C injection — byte-identical prompt to B1.
    agent = Qwen3VLAgent(
        model=model, processor=processor, env=env,
        device=device, generation_config=gen_cfg,
        l_c_prompt_text=None,
    )

    overall_t0 = time.monotonic()
    try:
        for i, ep in enumerate(to_run, start=1):
            app = ep["app"]
            template = ep["template"]
            seed = ep["seed"]

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
                "lora_path": lora_path,
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
    _print_summary(csv_path, args.baseline_tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())

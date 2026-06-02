#!/usr/bin/env python3
"""Evaluate B3 evolved L_C vs B2 static L_C on T_eval (held-out tasks).

For each Tier-B app, runs every T_eval template across K seeds under two
injection conditions:

  * ``b2_static``  — the original per-category L_C from
    ``artifacts/L_C/{category_slug}.json`` (same file B2 shipped with).
  * ``b3_evolved`` — the evolved champion L_C from
    ``traces/b3_evolution/{app}/l_c_champion.json`` (written by
    ``run_b3_evolution.py`` on every successful mutation).

Injection mechanics match B2 / B3 evolution's rollout: load the FSM,
render its LAYER-2 block as prompt text (via ``Layer2.to_prompt_text``),
and hand that to ``agent.set_l_c_prompt_text``. The agent splices the
text in after ``PROMPT_PREFIX``; nothing about the action/summary call
graph changes between conditions. The **only** thing that differs
across B2/B3 rows is the L_C content.

Appends to a single CSV (``results.csv``) after each episode so a crash
loses at most one in-progress row. On re-run the script detects rows
already present and skips those (simple resume).

Usage::

    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      python EvoFSM-RL/scripts/run_b3_teval.py \\
        --console-port 5710 --grpc-port 8710 \\
        --adb-path $ANDROID_HOME/platform-tools/adb \\
        --seeds 40 41 42 \\
        --output-dir EvoFSM-RL/traces/b3_teval

Optional flags:

  --apps pro_expense system_settings    # subset of apps (default depends on tier flag)
  --include-tier-c                      # also evaluate Tier-C T_eval (no L_C injected)
  --skip-b2                             # only run the b3_evolved arm
  --skip-b3                             # only run the b2_static arm
  --device cuda|mps|cpu                 # override device autodetect

Tier-C handling
---------------
Tier-C apps have no matching source-pool L_C (by definition of Tier-C),
so B3 cannot evolve an L_C for them. When ``--include-tier-c`` is set,
those apps are added to the plan with ``l_c_text=None`` in BOTH the
``b2_static`` and ``b3_evolved`` arms — the agent runs with a plain
B1-style prompt (no workflow knowledge injected) for every Tier-C
episode. The two arms are still labeled separately in the CSV so the
summary table can report the expected ``+0.0pp`` null result and keep
the format symmetric with the Tier-B numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


logger = logging.getLogger("run_b3_teval")


CSV_FIELDS = [
    "baseline", "app", "category", "template", "seed",
    "success", "n_steps", "wall_seconds", "error",
]


# ─────────────────────────────────────────────────────────────────────
# L_C loading
# ─────────────────────────────────────────────────────────────────────


def _render_l_c_from_fsm_json(path: Path) -> str:
    """Load an FSM JSON file and render its Layer-2 block as prompt text.

    The champion file is a full FSM wrapper (empty Layer 1 + evolved
    Layer 2). We deliberately render just the Layer-2 block so the
    injection shape matches B2's and B3 evolution's own rollout path —
    what the agent sees at action-selection time is a LAYER-2 block
    only, not a mostly-empty LAYER-1 banner.
    """
    from evofsm_rl.fsm.schema import FSM

    fsm = FSM.from_json(json.loads(path.read_text()))
    return fsm.layer2.to_prompt_text(category=fsm.layer1.category)


def _render_l_c_from_l_c_file(category: str, l_c_dir: Path) -> str:
    """Resolve the per-category static L_C and render as prompt text."""
    from evofsm_rl.fsm.aggregator import category_to_slug, load_L_C

    path = l_c_dir / f"{category_to_slug(category)}.json"
    if not path.exists():
        raise FileNotFoundError(f"Static L_C not found at {path}")
    cat, layer2 = load_L_C(path)
    return layer2.to_prompt_text(category=cat)


# ─────────────────────────────────────────────────────────────────────
# CSV helpers (append-safe, header-on-create, resume-on-restart)
# ─────────────────────────────────────────────────────────────────────


def _init_csv(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()


def _append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=CSV_FIELDS).writerow(row)


def _load_completed(path: Path) -> set[tuple[str, str, str, int]]:
    """Return the set of (baseline, app, template, seed) already in CSV."""
    if not path.exists():
        return set()
    done: set[tuple[str, str, str, int]] = set()
    with path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                done.add((
                    row["baseline"], row["app"], row["template"],
                    int(row["seed"]),
                ))
            except (KeyError, ValueError):
                continue
    return done


# ─────────────────────────────────────────────────────────────────────
# Plan build (all episodes up-front; makes the main loop dumb)
# ─────────────────────────────────────────────────────────────────────


def _build_plan(
    apps: list[str],
    seeds: list[int],
    skip_b2: bool,
    skip_b3: bool,
    project_root: Path,
) -> list[dict[str, Any]]:
    """Enumerate every (baseline, app, template, seed) episode to run.

    Accepts both Tier-B and Tier-C apps:
      * Tier-B: both arms inject a (possibly different) L_C text. Fails
        fast with a clear error if the B3 champion or static L_C file
        is missing for any requested app.
      * Tier-C: both arms run with ``l_c_text=None`` (no L_C injected —
        matches B1 prompt shape). The two arms should produce
        statistically equivalent results (a known null-result control
        for the summary).
    """
    from evofsm_rl.splits import get_tier_B_apps, get_tier_C_apps

    tb = get_tier_B_apps()
    tc = get_tier_C_apps()
    l_c_dir = project_root / "artifacts" / "L_C"
    b3_dir = project_root / "traces" / "b3_evolution"

    plan: list[dict[str, Any]] = []
    for app in apps:
        if app in tb:
            info = tb[app]
            tier = "tier_B"
            is_tier_c = False
        elif app in tc:
            info = tc[app]
            tier = "tier_C"
            is_tier_c = True
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
        category = info.category

        if is_tier_c:
            # Both arms run identically (no L_C) — Tier-C apps have no
            # source-pool category mapping, so there's no L_C to
            # evolve and no L_C to inject.
            b2_text: str | None = None
            b3_text: str | None = None
        else:
            # Pre-resolve injection text per baseline so a missing file
            # fails fast rather than 50 episodes into the run.
            b2_text = None
            b3_text = None
            if not skip_b2:
                b2_text = _render_l_c_from_l_c_file(category, l_c_dir)
            if not skip_b3:
                champ_path = b3_dir / app / "l_c_champion.json"
                if not champ_path.exists():
                    raise FileNotFoundError(
                        f"B3 champion missing for {app}: {champ_path}. "
                        f"Run B3 evolution for this app first or pass --skip-b3."
                    )
                b3_text = _render_l_c_from_fsm_json(champ_path)

        # b3_evolved first so any early-run crash has B3 data (the
        # novel condition); b2_static is re-run from scratch but is
        # a known quantity.
        arms = []
        if not skip_b3:
            arms.append(("b3_evolved", b3_text))
        if not skip_b2:
            arms.append(("b2_static", b2_text))
        for baseline, text in arms:
            # For Tier-B, text=None means the baseline was skipped.
            # For Tier-C, text=None is the intended value.
            if not is_tier_c and text is None:
                continue
            for template in templates:
                for seed in seeds:
                    plan.append({
                        "baseline": baseline,
                        "app": app,
                        "category": category,
                        "tier": tier,
                        "template": template,
                        "seed": seed,
                        "l_c_text": text,  # None for Tier-C by design
                    })
    return plan


# ─────────────────────────────────────────────────────────────────────
# Summary rendering
# ─────────────────────────────────────────────────────────────────────


def _read_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open() as fh:
        return list(csv.DictReader(fh))


def _pct(d: dict[str, float]) -> str:
    if not d["n"]:
        return "     —"
    return f"{d['succ']/d['n']*100:5.1f}% ({d['succ']:.1f}/{int(d['n'])})"


def _delta_str(b2: dict[str, float], b3: dict[str, float]) -> str:
    if not (b2["n"] and b3["n"]):
        return "     —"
    delta = (b3["succ"] / b3["n"]) - (b2["succ"] / b2["n"])
    return f"{'+' if delta >= 0 else ''}{delta*100:5.1f}pp"


def _tier_of(app: str) -> str:
    """Resolve app → tier_B / tier_C (source / unknown fall back).

    Cheap lookup at summary time; avoids having to persist tier to CSV.
    """
    from evofsm_rl.splits import get_tier_B_apps, get_tier_C_apps
    if app in get_tier_B_apps():
        return "tier_B"
    if app in get_tier_C_apps():
        return "tier_C"
    return "other"


def _print_summary(csv_path: Path, apps: list[str]) -> None:
    rows = _read_rows(csv_path)
    if not rows:
        print("\n(no results yet)")
        return

    # Aggregate per (baseline, app)
    agg: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"n": 0, "succ": 0.0}
    )
    for r in rows:
        key = (r["baseline"], r["app"])
        agg[key]["n"] += 1
        agg[key]["succ"] += float(r["success"])

    # Partition apps by tier for the detail section.
    tier_b_apps = [a for a in apps if _tier_of(a) == "tier_B"]
    tier_c_apps = [a for a in apps if _tier_of(a) == "tier_C"]

    def _subtotals(app_list: list[str]) -> tuple[dict, dict]:
        b2 = {"n": 0, "succ": 0.0}
        b3 = {"n": 0, "succ": 0.0}
        for app in app_list:
            x2 = agg.get(("b2_static", app), {"n": 0, "succ": 0.0})
            x3 = agg.get(("b3_evolved", app), {"n": 0, "succ": 0.0})
            b2["n"] += x2["n"]; b2["succ"] += x2["succ"]
            b3["n"] += x3["n"]; b3["succ"] += x3["succ"]
        return b2, b3

    def _print_section(title: str, app_list: list[str]) -> None:
        if not app_list:
            return
        print()
        print(f"## {title}")
        print(f"  {'App':22}  {'B2 SR':>14}  {'B3 SR':>14}  {'Δ':>9}")
        print(f"  {'-' * 22}  {'-' * 14}  {'-' * 14}  {'-' * 9}")
        for app in app_list:
            b2 = agg.get(("b2_static", app), {"n": 0, "succ": 0.0})
            b3 = agg.get(("b3_evolved", app), {"n": 0, "succ": 0.0})
            print(f"  {app:22}  {_pct(b2):>14}  {_pct(b3):>14}  "
                  f"{_delta_str(b2, b3):>9}")

    print()
    print("=" * 78)
    print("B2 (static L_C) vs B3 (evolved L_C) on T_eval")
    print("=" * 78)

    _print_section("Tier-B apps (L_C injected; B3 uses evolved champion)",
                     tier_b_apps)
    _print_section("Tier-C apps (no L_C for either arm — null control)",
                     tier_c_apps)

    # Tier roll-up + overall — user-requested compact format at the bottom.
    tb_b2, tb_b3 = _subtotals(tier_b_apps)
    tc_b2, tc_b3 = _subtotals(tier_c_apps)
    all_b2 = {"n": tb_b2["n"] + tc_b2["n"], "succ": tb_b2["succ"] + tc_b2["succ"]}
    all_b3 = {"n": tb_b3["n"] + tc_b3["n"], "succ": tb_b3["succ"] + tc_b3["succ"]}

    print()
    print("## Tier roll-up")
    print(f"  {'':10}  {'B2 SR':>14}  {'B3 SR':>14}  {'Δ':>9}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*9}")
    if tier_b_apps:
        print(f"  {'Tier-B':10}  {_pct(tb_b2):>14}  {_pct(tb_b3):>14}  "
              f"{_delta_str(tb_b2, tb_b3):>9}")
    if tier_c_apps:
        extra = "  (expected ≈ 0)" if tier_c_apps else ""
        print(f"  {'Tier-C':10}  {_pct(tc_b2):>14}  {_pct(tc_b3):>14}  "
              f"{_delta_str(tc_b2, tc_b3):>9}{extra}")
    print(f"  {'Overall':10}  {_pct(all_b2):>14}  {_pct(all_b3):>14}  "
          f"{_delta_str(all_b2, all_b3):>9}")
    print()
    print(f"CSV: {csv_path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--apps", nargs="*", default=None,
                   help="Subset of apps to evaluate. Default: all Tier-B "
                        "(+ Tier-C when --include-tier-c is passed).")
    p.add_argument("--include-tier-c", action="store_true",
                   help="Also evaluate every Tier-C app with non-empty T_eval. "
                        "Tier-C has no matching L_C — both B2 and B3 arms run "
                        "without any L_C injection (B1-shaped prompt). Δ ≈ 0 "
                        "is expected and serves as a null control.")
    p.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42],
                   help="Task seeds to run. Default: 40 41 42.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("EvoFSM-RL/traces/b3_teval"))
    p.add_argument("--console-port", type=int, default=5710)
    p.add_argument("--grpc-port", type=int, default=8710)
    p.add_argument("--adb-path", type=str, default=None)
    p.add_argument("--device", choices=("mps", "cuda", "cpu"), default=None)
    p.add_argument("--max-steps-multiplier", type=float, default=10.0)
    p.add_argument("--emulator-setup", action="store_true")
    p.add_argument("--skip-b2", action="store_true",
                   help="Only run the B3 evolved-L_C arm.")
    p.add_argument("--skip-b3", action="store_true",
                   help="Only run the B2 static-L_C arm.")
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

    # Build the plan up front so errors (missing champion, bad app) fail fast.
    plan = _build_plan(
        apps=apps, seeds=args.seeds,
        skip_b2=args.skip_b2, skip_b3=args.skip_b3,
        project_root=project_root,
    )
    if not plan:
        logger.error("Empty plan — nothing to do.")
        return 2

    _init_csv(csv_path)
    completed = _load_completed(csv_path)
    to_run = [
        ep for ep in plan
        if (ep["baseline"], ep["app"], ep["template"], ep["seed"])
        not in completed
    ]
    logger.info(
        "Plan: %d episodes total, %d already done, %d to run.",
        len(plan), len(plan) - len(to_run), len(to_run),
    )
    for baseline in ("b3_evolved", "b2_static"):
        n = sum(1 for ep in to_run if ep["baseline"] == baseline)
        if n:
            logger.info("  %s: %d episodes", baseline, n)
    if not to_run:
        logger.info("All episodes already completed — printing summary.")
        _print_summary(csv_path, apps)
        return 0

    # Load model + connect emulator once.
    from evofsm_rl.agent.rollout import GenerationConfig, Qwen3VLAgent
    from evofsm_rl.env import harness
    from evofsm_rl.model import (
        load_base_model, load_model_config, resolve_device,
    )

    device = args.device or resolve_device()
    logger.info("Loading Qwen3-VL on device=%s", device)
    t_load = time.monotonic()
    model, processor = load_base_model(device=device)
    logger.info("Model loaded in %.1fs", time.monotonic() - t_load)

    cfg = load_model_config(device=device)
    gen_cfg = GenerationConfig.from_yaml(cfg.raw.get("generation", {}))

    connect_kwargs: dict[str, Any] = {
        "console_port": args.console_port,
        "grpc_port": args.grpc_port,
        "emulator_setup": args.emulator_setup,
    }
    if args.adb_path:
        connect_kwargs["adb_path"] = args.adb_path
    logger.info("Connecting emulator %s", connect_kwargs)
    env = harness.connect(**connect_kwargs)

    agent = Qwen3VLAgent(
        model=model, processor=processor, env=env,
        device=device, generation_config=gen_cfg,
    )

    # Sweep
    overall_t0 = time.monotonic()
    try:
        for i, ep in enumerate(to_run, start=1):
            baseline = ep["baseline"]
            app = ep["app"]
            template = ep["template"]
            seed = ep["seed"]
            category = ep["category"]
            l_c_text = ep["l_c_text"]
            tag = "B3" if baseline == "b3_evolved" else "B2"

            agent.set_l_c_prompt_text(l_c_text)

            t0 = time.monotonic()
            logger.info(
                "[%d/%d] Running: %s | %s | %s | seed=%d",
                i, len(to_run), baseline, app, template, seed,
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

                # Persist trace under episodes/{baseline}/.
                if err_msg is None and n_steps > 0:
                    try:
                        arm_dir = episodes_dir / baseline
                        arm_dir.mkdir(exist_ok=True)
                        agent.save_episode(
                            arm_dir,
                            success=success,
                            template=template,
                            seed=seed,
                            app=app,
                            tier="tier_B",
                        )
                    except Exception:
                        logger.exception(
                            "save_episode failed for %s/%s seed=%d",
                            baseline, template, seed,
                        )
            except Exception as e:
                logger.exception(
                    "Episode crashed: %s %s seed=%d", app, template, seed,
                )
                success = 0.0
                n_steps = 0
                wall = time.monotonic() - t0
                err_msg = f"{type(e).__name__}: {e}"

            logger.info(
                "[%s] %s | %s | seed=%d | success=%.1f | %d steps | %.1fs",
                tag, app, template, seed, success, n_steps, wall,
            )

            _append_row(csv_path, {
                "baseline": baseline,
                "app": app,
                "category": category,
                "template": template,
                "seed": seed,
                "success": success,
                "n_steps": n_steps,
                "wall_seconds": f"{wall:.2f}",
                "error": err_msg or "",
            })
    finally:
        try:
            env.close()
        except Exception:
            logger.warning("env.close() failed", exc_info=True)

    wall_min = (time.monotonic() - overall_t0) / 60
    logger.info("Sweep wall time: %.1f min", wall_min)

    _print_summary(csv_path, apps)
    return 0


if __name__ == "__main__":
    sys.exit(main())

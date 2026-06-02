#!/usr/bin/env python3
"""Story B2 — Static-FSM baseline on T_eval (held-out apps).

Runs Qwen3-VL-M3A on every T_eval template from ``configs/splits.yaml``,
injecting the per-category L_C (LAYER-2) into the agent's action prompt
for Tier-B apps (category is in source pool) and falling back to the B1
zero-shot prompt for Tier-C apps (category absent from source pool).
The Tier-B vs Tier-C gap quantifies the transfer value of L_C.

Usage::

    export CUDA_VISIBLE_DEVICES=2
    export ANDROID_HOME=/shared/linqiang/evofsm_project/android-sdk
    export TMPDIR=/shared/linqiang/evofsm_project/tmp
    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      python EvoFSM-RL/scripts/run_b2_eval.py \\
        --splits-yaml EvoFSM-RL/configs/splits.yaml \\
        --l-c-dir EvoFSM-RL/artifacts/L_C \\
        --output-dir EvoFSM-RL/traces/b2_teval_v01 \\
        --tier both \\
        --seeds 30 \\
        --console-port 5710 --grpc-port 8710 \\
        --adb-path $ANDROID_HOME/platform-tools/adb

Outputs:
  * ``{output_dir}/{Template}_seed{N}/``  — Story-2.0 per-episode trace
    (meta.json + episode.jsonl + step PNGs), matching B1's on-disk shape
    so ``traces/m3a_teval_v01/`` and this directory are directly diff-able.
  * ``{output_dir}/summary_seed{N}.jsonl`` — one JSON line per template,
    schema: ``{seed, template, app, tier, category, l_c_injected,
    success, n_steps, wall_s, parse_failures, self_reported, error}``.

This script only drives the eval. It does NOT train anything, does NOT
mutate the L_C files, and does NOT modify the FSMs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


logger = logging.getLogger("b2_eval")


DEFAULT_SEED = 30
DEFAULT_B1_DIR = Path("EvoFSM-RL/traces/m3a_teval_v01")


# ─────────────────────────────────────────────────────────────────────────
# Row / aggregation helpers
# ─────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class B2Row:
    seed: int
    template: str
    app: str
    tier: str                # "tier_B" | "tier_C"
    category: str | None     # None when app is unknown to splits.yaml
    l_c_injected: bool
    success: float
    n_steps: int
    wall_s: float
    parse_failures: int
    alias_hits: int
    clamp_hits: int
    self_reported: int
    error: str | None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _parse_seeds(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )


def _aggregate(rows: list[B2Row]) -> dict[str, Any]:
    if not rows:
        return {}
    ran = [r for r in rows if r.error is None]
    return {
        "n_total": len(rows),
        "n_ran": len(ran),
        "n_errored": len(rows) - len(ran),
        "success_rate_over_all": sum(r.success for r in ran) / len(rows) if rows else 0.0,
        "success_rate_over_ran": sum(r.success for r in ran) / len(ran) if ran else 0.0,
        "mean_steps": sum(r.n_steps for r in ran) / len(ran) if ran else 0.0,
        "total_wall_s": sum(r.wall_s for r in rows),
        "l_c_injected_count": sum(1 for r in rows if r.l_c_injected),
    }


def _per_tier(rows: list[B2Row]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for tier_key in ("tier_B", "tier_C"):
        group = [r for r in rows if r.tier == tier_key]
        ran = [r for r in group if r.error is None]
        out[tier_key] = {
            "n": len(group),
            "n_ran": len(ran),
            "success_rate": (sum(r.success for r in ran) / len(group)) if group else 0.0,
            "mean_steps": (sum(r.n_steps for r in ran) / len(ran)) if ran else 0.0,
            "l_c_injected_count": sum(1 for r in group if r.l_c_injected),
        }
    return out


def _per_app(rows: list[B2Row]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    apps = sorted({r.app for r in rows})
    for app in apps:
        group = [r for r in rows if r.app == app]
        ran = [r for r in group if r.error is None]
        out[app] = {
            "tier": group[0].tier,
            "l_c_injected": group[0].l_c_injected,
            "n": len(group),
            "n_ran": len(ran),
            "success_rate": (sum(r.success for r in ran) / len(group)) if group else 0.0,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────
# B1 comparison (optional)
# ─────────────────────────────────────────────────────────────────────────


def _try_read_b1_seed_summary(b1_dir: Path, seed: int) -> dict | None:
    """Return B1's per-seed summary dict if it exists; else None."""
    for candidate in (f"seed{seed}.json", f"summary_seed{seed}.json"):
        p = b1_dir / candidate
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
    return None


def _b1_vs_b2_table(b1_summary: dict | None, b2_rows: list[B2Row]) -> str:
    """Render a side-by-side Tier-B / Tier-C / Overall SR table."""
    if not b1_summary:
        return "(B1 summary not found — skip comparison)"

    # B1's summary_10task schema uses `per_tier` dict keyed by free-form
    # tier strings. B1's T_eval baseline uses the same split keys ("tier_B",
    # "tier_C") so we can directly look up. Fall back gracefully.
    b1_per_tier = b1_summary.get("per_tier") or {}
    b1_agg = b1_summary.get("aggregate") or {}

    b2_per_tier = _per_tier(b2_rows)
    b2_agg = _aggregate(b2_rows)

    def _fmt(x: float | None) -> str:
        return f"{x:.1%}" if isinstance(x, (int, float)) else "n/a"

    def _b1_tier_sr(k: str) -> float | None:
        row = b1_per_tier.get(k)
        return row.get("success_rate") if isinstance(row, dict) else None

    lines = [
        "| Split    | B1 SR  | B2 SR  | Δ       |",
        "|----------|-------:|-------:|--------:|",
    ]
    for k, label in (("tier_B", "Tier-B"), ("tier_C", "Tier-C")):
        b1_sr = _b1_tier_sr(k)
        b2_sr = b2_per_tier[k]["success_rate"] if b2_per_tier[k]["n"] else None
        delta = (b2_sr - b1_sr) if (b1_sr is not None and b2_sr is not None) else None
        delta_fmt = f"{delta:+.1%}" if delta is not None else "n/a"
        lines.append(f"| {label}   | {_fmt(b1_sr):>6} | {_fmt(b2_sr):>6} | {delta_fmt:>7} |")
    b1_overall = b1_agg.get("success_rate_over_all")
    b2_overall = b2_agg.get("success_rate_over_all")
    delta = (b2_overall - b1_overall) if (b1_overall is not None and b2_overall is not None) else None
    lines.append(
        f"| Overall  | {_fmt(b1_overall):>6} | {_fmt(b2_overall):>6} | "
        f"{(f'{delta:+.1%}' if delta is not None else 'n/a'):>7} |"
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Template plan build — who to run, with which L_C
# ─────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Plan:
    """One row of the B2 run plan — template + resolved L_C state."""
    template: str
    app: str
    tier: str                 # "tier_B" | "tier_C"
    category: str | None
    l_c_prompt_text: str | None


def _build_plan(
    splits_yaml_path: Path,
    l_c_dir: Path,
    tier_filter: str,
) -> list[Plan]:
    """Expand T_eval into a flat plan list, resolving L_C per app.

    ``tier_filter`` is one of "B", "C", "both". The plan preserves the
    natural splits.yaml order (apps alphabetical by YAML key, templates
    in their declared order), which keeps the run deterministic.
    """
    from evofsm_rl.fsm import resolve_l_c_for_app
    from evofsm_rl.splits import get_tier_B_apps, get_tier_C_apps

    want_b = tier_filter in ("B", "both")
    want_c = tier_filter in ("C", "both")

    plan: list[Plan] = []

    if want_b:
        for app, split in get_tier_B_apps().items():
            text = resolve_l_c_for_app(app, splits_yaml_path, l_c_dir)
            for template in split.T_eval:
                plan.append(Plan(
                    template=template,
                    app=app,
                    tier="tier_B",
                    category=split.category,
                    l_c_prompt_text=text,
                ))

    if want_c:
        for app, split in get_tier_C_apps().items():
            # Tier-C: resolve_l_c_for_app returns None (by design).
            text = resolve_l_c_for_app(app, splits_yaml_path, l_c_dir)
            for template in split.T_eval:
                plan.append(Plan(
                    template=template,
                    app=app,
                    tier="tier_C",
                    category=split.category,
                    l_c_prompt_text=text,
                ))

    return plan


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--splits-yaml", type=Path,
                   default=Path("EvoFSM-RL/configs/splits.yaml"))
    p.add_argument("--l-c-dir", type=Path,
                   default=Path("EvoFSM-RL/artifacts/L_C"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("EvoFSM-RL/traces/b2_teval_v01"),
                   help="Where per-episode trajectory dirs AND summary "
                        "JSONLs are written. Sibling of traces/m3a_teval_v01.")
    p.add_argument("--tier", choices=("B", "C", "both"), default="both")
    p.add_argument("--no-l-c", action="store_true",
                   help="Force-disable L_C injection for every template. "
                        "Use this flag to run the B1 baseline with the SAME "
                        "runner / output schema as B2, so the two can be diff-ed "
                        "cleanly. Tier-B templates that would normally inject "
                        "are left without L_C.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", type=_parse_seeds, default=None,
                   help="Comma-separated seeds. Default: [30].")
    p.add_argument("--device", choices=("mps", "cuda", "cpu"), default=None)
    p.add_argument("--max-steps-multiplier", type=float, default=10.0)
    p.add_argument("--console-port", type=int, default=5554)
    p.add_argument("--grpc-port", type=int, default=8554)
    p.add_argument("--adb-path", type=str, default=None)
    p.add_argument("--emulator-setup", action="store_true")
    p.add_argument("--b1-dir", type=Path, default=DEFAULT_B1_DIR,
                   help="Path to B1 T_eval trace dir for side-by-side "
                        "comparison. Skipped if summary JSONs are absent.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    _setup_logging(args.verbose)

    # Resolve seeds
    if args.seeds is None and args.seed is None:
        args.seeds = [DEFAULT_SEED]
    elif args.seeds is None:
        args.seeds = [args.seed]
    elif args.seed is not None:
        logger.warning("Both --seed and --seeds given; using --seeds.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build plan
    plan = _build_plan(args.splits_yaml, args.l_c_dir, args.tier)
    if not plan:
        logger.error("Empty plan for tier=%s — nothing to do.", args.tier)
        return 2

    # B1-mode override: blank every plan row's l_c_prompt_text. Keeps the
    # runner / schema identical so the two sweeps diff cleanly.
    if args.no_l_c:
        logger.info("--no-l-c: forcing l_c_prompt_text=None on all %d templates "
                    "(B1 baseline via the B2 runner).", len(plan))
        plan = [dataclasses.replace(pl, l_c_prompt_text=None) for pl in plan]

    apps_seen = sorted({pl.app for pl in plan})
    n_lc = sum(1 for pl in plan if pl.l_c_prompt_text is not None)
    logger.info(
        "Plan: %d templates across %d app(s) %s; %d with L_C injection, %d without.",
        len(plan), len(apps_seen), apps_seen, n_lc, len(plan) - n_lc,
    )

    # Late imports — keep --help fast.
    from evofsm_rl.agent.rollout import GenerationConfig, Qwen3VLAgent
    from evofsm_rl.env import harness
    from evofsm_rl.model import load_base_model, load_model_config, resolve_device

    resolved_device = args.device or resolve_device()
    logger.info("Loading Qwen3-VL-8B on device=%s", resolved_device)
    t_load = time.monotonic()
    model, processor = load_base_model(device=resolved_device)
    logger.info("Model loaded in %.1fs", time.monotonic() - t_load)

    cfg = load_model_config(device=resolved_device)
    gen_cfg = GenerationConfig.from_yaml(cfg.raw.get("generation", {}))

    connect_kwargs: dict[str, Any] = {
        "console_port": args.console_port,
        "grpc_port": args.grpc_port,
        "emulator_setup": args.emulator_setup,
    }
    if args.adb_path:
        connect_kwargs["adb_path"] = args.adb_path
    env = harness.connect(**connect_kwargs)

    agent = Qwen3VLAgent(
        model=model,
        processor=processor,
        env=env,
        device=resolved_device,
        generation_config=gen_cfg,
    )

    overall_t0 = time.monotonic()
    try:
        for seed_i, seed in enumerate(args.seeds, start=1):
            logger.info("#" * 72)
            logger.info("# B2 seed %d (%d/%d)  —  %d templates",
                        seed, seed_i, len(args.seeds), len(plan))
            logger.info("#" * 72)

            rows: list[B2Row] = []
            last_app: str | None = None
            for i, pl in enumerate(plan, start=1):
                # Swap L_C only when the app changes (idempotent otherwise).
                if pl.app != last_app:
                    agent.set_l_c_prompt_text(pl.l_c_prompt_text)
                    logger.info(
                        "[app switch] %s (tier=%s, category=%s, l_c_injected=%s)",
                        pl.app, pl.tier, pl.category, pl.l_c_prompt_text is not None,
                    )
                    last_app = pl.app

                # Resume if a prior trajectory exists — same pattern as
                # baseline_10task.py.
                meta_path = args.output_dir / f"{pl.template}_seed{seed}" / "meta.json"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                        row = B2Row(
                            seed=seed,
                            template=pl.template,
                            app=pl.app,
                            tier=pl.tier,
                            category=pl.category,
                            l_c_injected=bool(meta.get("l_c_injected", pl.l_c_prompt_text is not None)),
                            success=float(meta.get("success", 0.0)),
                            n_steps=int(meta.get("n_steps", 0)),
                            wall_s=float(meta.get("wall_s_total", 0.0)),
                            parse_failures=int(meta.get("parse_failures", 0)),
                            alias_hits=int(meta.get("alias_hits", 0)),
                            clamp_hits=int(meta.get("clamp_hits", 0)),
                            self_reported=int(meta.get("self_reported", 0)),
                            error=None,
                        )
                        rows.append(row)
                        logger.info(
                            "[seed=%d %d/%d] %s SKIPPED (resumed: success=%.2f)",
                            seed, i, len(plan), pl.template, row.success,
                        )
                        continue
                    except Exception:
                        logger.warning("corrupt meta at %s — will re-run", meta_path)

                logger.info("=" * 72)
                logger.info(
                    "[seed=%d %d/%d] Running %s  app=%s tier=%s l_c=%s",
                    seed, i, len(plan), pl.template, pl.app, pl.tier,
                    pl.l_c_prompt_text is not None,
                )

                t_task = time.monotonic()
                try:
                    result = harness.run_template(
                        template_name=pl.template,
                        seed=seed,
                        env=env,
                        agent=agent,
                        max_steps_multiplier=args.max_steps_multiplier,
                    )
                    row = B2Row(
                        seed=seed,
                        template=pl.template,
                        app=pl.app,
                        tier=pl.tier,
                        category=pl.category,
                        l_c_injected=pl.l_c_prompt_text is not None,
                        success=result.success,
                        n_steps=result.n_steps,
                        wall_s=result.wall_seconds,
                        parse_failures=agent._parse_failures,
                        alias_hits=agent._alias_hits,
                        clamp_hits=agent._clamp_hits,
                        self_reported=agent._self_reported,
                        error=result.error,
                    )
                    if result.error is None and result.n_steps > 0:
                        try:
                            agent.save_episode(
                                args.output_dir,
                                success=result.success,
                                template=pl.template,
                                seed=seed,
                                app=pl.app,
                                tier=pl.tier,
                            )
                            # Annotate meta.json with B2-specific fields.
                            mp = args.output_dir / f"{pl.template}_seed{seed}" / "meta.json"
                            if mp.exists():
                                m = json.loads(mp.read_text())
                                m["l_c_injected"] = pl.l_c_prompt_text is not None
                                m["category"] = pl.category
                                mp.write_text(json.dumps(m, indent=2))
                        except Exception:
                            logger.exception(
                                "save_episode failed for %s (sweep continues)",
                                pl.template,
                            )
                except Exception as e:
                    logger.exception("Unhandled error on %s", pl.template)
                    row = B2Row(
                        seed=seed,
                        template=pl.template,
                        app=pl.app,
                        tier=pl.tier,
                        category=pl.category,
                        l_c_injected=pl.l_c_prompt_text is not None,
                        success=0.0,
                        n_steps=0,
                        wall_s=time.monotonic() - t_task,
                        parse_failures=0,
                        alias_hits=0,
                        clamp_hits=0,
                        self_reported=0,
                        error=f"{type(e).__name__}: {e}",
                    )

                rows.append(row)
                logger.info(
                    "[seed=%d %d/%d] %s done: success=%.2f steps=%d wall=%.1fs err=%s",
                    seed, i, len(plan), pl.template, row.success, row.n_steps,
                    row.wall_s, row.error,
                )

            # ── End-of-seed: summary JSONL (one line per template) ──
            summary_path = args.output_dir / f"summary_seed{seed}.jsonl"
            with summary_path.open("w") as fh:
                for r in rows:
                    fh.write(json.dumps(r.as_dict()) + "\n")

            # ── End-of-seed: aggregates + B1 comparison ──
            agg = _aggregate(rows)
            per_tier = _per_tier(rows)
            per_app = _per_app(rows)

            print()
            print(f"# B2 seed {seed} — {len(plan)} templates "
                  f"({agg['l_c_injected_count']} with L_C, "
                  f"{len(plan) - agg['l_c_injected_count']} without)")
            print()
            print("## Aggregate")
            print(f"- success_rate (all):  {agg['success_rate_over_all']:.1%}")
            print(f"- success_rate (ran):  {agg['success_rate_over_ran']:.1%} "
                  f"({agg['n_ran']}/{agg['n_total']} ran)")
            print(f"- mean_steps (ran):    {agg['mean_steps']:.1f}")
            print()
            print("## Per-tier")
            for tk, stats in per_tier.items():
                print(f"- {tk}: n={stats['n']:2d}  SR={stats['success_rate']:.1%}  "
                      f"l_c_injected={stats['l_c_injected_count']}")
            print()
            print("## Per-app")
            for app, stats in per_app.items():
                marker = "L_C" if stats["l_c_injected"] else "   "
                print(f"  {marker}  {app:22s} ({stats['tier']})  "
                      f"n={stats['n']:2d}  SR={stats['success_rate']:.1%}")
            print()
            print("## B1 vs B2")
            b1 = _try_read_b1_seed_summary(args.b1_dir, seed)
            print(_b1_vs_b2_table(b1, rows))
            print()
            print(f"Summary JSONL: {summary_path}")

    finally:
        try:
            env.close()
        except Exception:
            logger.warning("env.close() failed", exc_info=True)

    print()
    print(f"## All seeds complete ({len(args.seeds)} seeds × {len(plan)} templates)")
    print(f"   Total wall: {(time.monotonic() - overall_t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())

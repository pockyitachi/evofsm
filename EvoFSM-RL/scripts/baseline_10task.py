#!/usr/bin/env python3
"""Story 1.4.5 — 10-task Qwen3-VL-8B baseline sweep.

Loads the model + connects the emulator once, then runs a small
difficulty-tiered task list at seed=30 and reports per-task success,
step count, parse-failure rate, alias hit count, and wall time.

The goal is a baseline number comparable to AndroidLab 2025's ~28% on
the AndroidWorld subset — not perfect coverage. Tasks were picked to
span stock-Android / AW-installed / multi-step interactive.

Usage:
    cd /shared/linqiang/evofsm_project && source .venv/bin/activate
    export ANDROID_HOME=/shared/linqiang/evofsm_project/android-sdk
    export TMPDIR=/shared/linqiang/evofsm_project/tmp
    export CUDA_VISIBLE_DEVICES=2
    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      python EvoFSM-RL/scripts/baseline_10task.py \\
      --console-port 5700 --grpc-port 8700 \\
      --adb-path $ANDROID_HOME/platform-tools/adb \\
      --output-dir EvoFSM-RL/traces/baseline_10task

    # First run on a fresh AVD also needs:
    ... --emulator-setup

The report is written as both stdout markdown AND
``<output-dir>/summary.json`` for later diffing.
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

logger = logging.getLogger("baseline_10task")


# Difficulty-tiered 10-task list. Order is random so a crash early doesn't
# bias the easy/hard distribution — pre-tag each with its intended tier.
DEFAULT_TASKS: list[tuple[str, str]] = [
    # (template_name, tier)
    ("SystemBrightnessMin",        "easy"),      # stock Settings
    ("SystemWifiTurnOff",          "easy"),      # stock Settings
    ("SystemCopyToClipboard",      "easy"),      # clipper app
    ("SimpleCalendarAddOneEvent",  "medium"),    # Simple Calendar Pro
    ("ContactsAddContact",         "medium"),    # stock Contacts
    ("ClockTimerEntry",            "medium"),    # stock Clock
    ("MarkorCreateNote",           "medium"),    # Markor
    ("MarkorAddNoteHeader",        "medium"),    # Markor
    ("SystemBluetoothTurnOn",      "hard"),      # multi-step Settings
    ("ClockStopWatchRunning",      "hard"),      # multi-step Clock
]

DEFAULT_SEED = 30


def _parse_seeds(s: str) -> list[int]:
    """Parse '30,31,32' or '30' into a list of ints. Used for --seeds."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


@dataclasses.dataclass
class TaskRow:
    template: str
    tier: str
    success: float
    n_steps: int
    wall_s: float
    parse_failures: int
    alias_hits: int
    clamp_hits: int
    self_reported: int      # agent emitted status:complete/infeasible (max 1 per episode); independent of `success`
    error: str | None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )


def _format_markdown_table(rows: list[TaskRow]) -> str:
    header = (
        "| Task | Tier | Success | Steps | Wall (s) | Parse fails | Alias hits | Self-reported | Error |\n"
        "|------|------|:------:|:-----:|---------:|:-----------:|:----------:|:------------:|-------|"
    )
    lines = [header]
    for r in rows:
        err = (r.error or "").replace("|", "\\|")
        if len(err) > 60:
            err = err[:57] + "..."
        lines.append(
            f"| {r.template} | {r.tier} | "
            f"{r.success:.2f} | {r.n_steps} | {r.wall_s:.1f} | "
            f"{r.parse_failures} | {r.alias_hits} | {r.self_reported} | {err} |"
        )
    return "\n".join(lines)


def _aggregate(rows: list[TaskRow]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}
    ran = [r for r in rows if r.error is None]
    successes = [r.success for r in ran]
    return {
        "n_total": n,
        "n_ran": len(ran),
        "n_errored": n - len(ran),
        "success_rate_over_all": sum(successes) / n if n else 0.0,
        "success_rate_over_ran": (sum(successes) / len(ran)) if ran else 0.0,
        "mean_steps": sum(r.n_steps for r in ran) / len(ran) if ran else 0.0,
        "total_parse_failures": sum(r.parse_failures for r in ran),
        "total_alias_hits": sum(r.alias_hits for r in ran),
        "total_self_reported": sum(r.self_reported for r in ran),
        "total_steps_ran": sum(r.n_steps for r in ran),
        "total_wall_s": sum(r.wall_s for r in ran),
    }


def _per_tier(rows: list[TaskRow]) -> dict[str, dict[str, float]]:
    tiers: dict[str, list[TaskRow]] = {}
    for r in rows:
        tiers.setdefault(r.tier, []).append(r)
    out: dict[str, dict[str, float]] = {}
    for tier, group in tiers.items():
        ran = [r for r in group if r.error is None]
        out[tier] = {
            "n": len(group),
            "success_rate": (sum(r.success for r in ran) / len(group))
            if group else 0.0,
            "mean_steps": (sum(r.n_steps for r in ran) / len(ran))
            if ran else 0.0,
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=None,
                   help="Single seed (back-compat). Use --seeds for K>1 sweeps.")
    p.add_argument("--seeds", type=_parse_seeds, default=None,
                   help="Comma-separated list of seeds, e.g. '30,31,32,33,34'. "
                        "Wraps the sweep in an outer loop over seeds. "
                        "Default: [30] (or [--seed] if given).")
    p.add_argument("--device", choices=("mps", "cuda", "cpu"), default=None)
    p.add_argument("--max-steps-multiplier", type=float, default=10.0)
    p.add_argument("--console-port", type=int, default=5554)
    p.add_argument("--grpc-port", type=int, default=8554)
    p.add_argument("--adb-path", type=str, default=None)
    p.add_argument("--emulator-setup", action="store_true",
                   help="Install the 16 vanilla AW apks on first connect.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write summary_seed{N}.json. "
                        "Alias for --summary-dir (back-compat).")
    p.add_argument("--summary-dir", type=Path, default=None,
                   help="Where to write seed{N}.json per-seed summaries. "
                        "Defaults to --output-dir if unset.")
    p.add_argument("--tasks", type=str, nargs="*", default=None,
                   help="Override task list (format: 'Template:tier'). "
                        "Default is the canonical 10.")
    p.add_argument("--tasks-file", type=Path, default=None,
                   help="File with one 'Template:tier' per line (# for comments). "
                        "Overrides --tasks when given.")
    p.add_argument("--trajectory-dir", type=Path, default=None,
                   help="If set, persist per-step trajectory to "
                        "{dir}/{template}_seed{N}/ "
                        "(meta.json + episode.jsonl + step_*_before/after.png). "
                        "Story 2.0 — Epic 2 prep. No-op when unset.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    _setup_logging(args.verbose)

    # Resolve --seeds vs --seed (back-compat)
    if args.seeds is None and args.seed is None:
        args.seeds = [DEFAULT_SEED]
    elif args.seeds is None:
        args.seeds = [args.seed]
    elif args.seed is not None:
        logger.warning("Both --seed and --seeds given; using --seeds, ignoring --seed.")

    # Resolve --summary-dir vs --output-dir (alias)
    if args.summary_dir is None:
        if args.output_dir is None:
            args.summary_dir = Path("EvoFSM-RL/traces/baseline_10task")
        else:
            args.summary_dir = args.output_dir
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    def _parse_spec(spec: str) -> tuple[str, str]:
        if ":" in spec:
            name, tier = spec.split(":", 1)
            return name.strip(), tier.strip()
        return spec.strip(), "unspecified"

    if args.tasks_file:
        tasks: list[tuple[str, str]] = []
        for line in args.tasks_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tasks.append(_parse_spec(line))
    elif args.tasks:
        tasks = [_parse_spec(s) for s in args.tasks]
    else:
        tasks = DEFAULT_TASKS

    # Late imports so --help is fast.
    from evofsm_rl.agent.rollout import GenerationConfig, Qwen3VLAgent
    from evofsm_rl.env import harness
    from evofsm_rl.model import load_base_model, load_model_config, resolve_device

    # 1. Load model once.
    resolved_device = args.device or resolve_device()
    logger.info("Loading Qwen3-VL-8B on device=%s", resolved_device)
    t_load = time.monotonic()
    model, processor = load_base_model(device=resolved_device)
    logger.info("Model loaded in %.1fs", time.monotonic() - t_load)

    cfg = load_model_config(device=resolved_device)
    gen_cfg = GenerationConfig.from_yaml(cfg.raw.get("generation", {}))
    logger.info("Generation config: %s", gen_cfg)

    # 2. Connect env once.
    connect_kwargs: dict[str, Any] = {
        "console_port": args.console_port,
        "grpc_port": args.grpc_port,
        "emulator_setup": args.emulator_setup,
    }
    if args.adb_path:
        connect_kwargs["adb_path"] = args.adb_path
    logger.info("Connecting emulator (%s)", connect_kwargs)
    env = harness.connect(**connect_kwargs)

    # 3. Single reusable agent — metrics reset each episode via agent.reset().
    agent = Qwen3VLAgent(
        model=model,
        processor=processor,
        env=env,
        device=resolved_device,
        generation_config=gen_cfg,
    )

    # 3b. (optional) Trajectory-persistence prep — Story 2.0.
    # template -> app lookup so meta.json can carry the app key for downstream
    # FSM synthesis. Source of truth is configs/task_categories.csv.
    template_to_app: dict[str, str] = {}
    if args.trajectory_dir is not None:
        import csv
        cats_csv = Path(__file__).resolve().parents[1] / "configs" / "task_categories.csv"
        if cats_csv.exists():
            with cats_csv.open() as fh:
                for row in csv.DictReader(fh):
                    template_to_app[row["task_name"]] = row["app"]
            logger.info("Loaded %d template→app mappings for trajectory metadata.",
                        len(template_to_app))
        else:
            logger.warning("configs/task_categories.csv not found at %s; "
                           "trajectory meta.json will record app='unknown'.", cats_csv)
        args.trajectory_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume helper ────────────────────────────────────────────
    def _try_resume_row(template: str, tier: str, seed: int) -> TaskRow | None:
        """Reconstruct a TaskRow from a previously-persisted trajectory's
        meta.json. Returns None if no prior run found OR meta is corrupted
        (in which case we'll re-run). Only valid when --trajectory-dir is set
        (otherwise we have no on-disk per-task evidence)."""
        if args.trajectory_dir is None:
            return None
        meta_path = args.trajectory_dir / f"{template}_seed{seed}" / "meta.json"
        if not meta_path.exists():
            return None
        try:
            with meta_path.open() as fh:
                m = json.load(fh)
            return TaskRow(
                template=template,
                tier=tier,
                success=float(m.get("success", 0.0)),
                n_steps=int(m.get("n_steps", 0)),
                wall_s=float(m.get("wall_s_total", 0.0)),
                parse_failures=int(m.get("parse_failures", 0)),
                alias_hits=int(m.get("alias_hits", 0)),
                clamp_hits=int(m.get("clamp_hits", 0)),
                self_reported=int(m.get("self_reported", 0)),
                error=None,
            )
        except Exception:
            logger.warning("Corrupt meta.json at %s; will re-run.", meta_path)
            return None

    # 4. Sweep — outer loop over seeds, inner loop over tasks.
    n_seeds = len(args.seeds)
    overall_t0 = time.monotonic()
    try:
        for seed_i, seed in enumerate(args.seeds, start=1):
            logger.info("#" * 72)
            logger.info("# Seed %d (%d/%d)  —  %d tasks", seed, seed_i, n_seeds, len(tasks))
            logger.info("#" * 72)

            rows: list[TaskRow] = []
            n_resumed = 0
            for i, (template, tier) in enumerate(tasks, start=1):
                # Resume: skip the LLM run if a prior trajectory exists on disk.
                resumed = _try_resume_row(template, tier, seed)
                if resumed is not None:
                    rows.append(resumed)
                    n_resumed += 1
                    logger.info("[seed=%d %d/%d] %s SKIPPED (resumed: success=%.2f steps=%d)",
                                seed, i, len(tasks), template, resumed.success, resumed.n_steps)
                    continue

                logger.info("=" * 72)
                logger.info("[seed=%d %d/%d] Running %s (tier=%s)",
                            seed, i, len(tasks), template, tier)

                t_task = time.monotonic()
                try:
                    result = harness.run_template(
                        template_name=template,
                        seed=seed,
                        env=env,
                        agent=agent,
                        max_steps_multiplier=args.max_steps_multiplier,
                    )
                    row = TaskRow(
                        template=template,
                        tier=tier,
                        success=result.success,
                        n_steps=result.n_steps,
                        wall_s=result.wall_seconds,
                        parse_failures=agent._parse_failures,  # noqa: SLF001
                        alias_hits=agent._alias_hits,          # noqa: SLF001
                        clamp_hits=agent._clamp_hits,          # noqa: SLF001
                        self_reported=agent._self_reported,    # noqa: SLF001
                        error=result.error,
                    )
                    # Story 2.0 — persist trajectory before the next episode's
                    # agent.reset() wipes self.history. Skip when init errored
                    # (n_steps=0) since history would belong to the prior task.
                    if (
                        args.trajectory_dir is not None
                        and result.error is None
                        and result.n_steps > 0
                    ):
                        try:
                            agent.save_episode(
                                args.trajectory_dir,
                                success=result.success,
                                template=template,
                                seed=seed,
                                app=template_to_app.get(template, "unknown"),
                                tier=tier,
                            )
                        except Exception:
                            logger.exception(
                                "save_episode failed for %s (sweep continues)", template,
                            )
                except Exception as e:
                    logger.exception("Unhandled error on %s", template)
                    row = TaskRow(
                        template=template,
                        tier=tier,
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
                    "[seed=%d %d/%d] %s done: success=%.2f steps=%d wall=%.1fs "
                    "parse_fails=%d alias_hits=%d self_reported=%d err=%s",
                    seed, i, len(tasks), template, row.success, row.n_steps,
                    row.wall_s, row.parse_failures, row.alias_hits,
                    row.self_reported, row.error,
                )

            # ── End-of-seed report + JSON dump ────────────────────────
            agg = _aggregate(rows)
            per_tier = _per_tier(rows)
            print()
            print(f"# Seed {seed} ({seed_i}/{n_seeds}) — {len(tasks)} tasks "
                  f"({n_resumed} resumed from disk)")
            print()
            print(_format_markdown_table(rows))
            print()
            print("## Aggregate (this seed)")
            print(f"- Tasks attempted:       {agg['n_total']}")
            print(f"- Tasks ran (no infra err): {agg['n_ran']}")
            print(f"- Success rate (all):    {agg['success_rate_over_all']:.1%}")
            if agg["n_ran"] > 0:
                print(f"- Success rate (ran):    {agg['success_rate_over_ran']:.1%}")
                print(f"- Mean steps (ran):      {agg['mean_steps']:.1f}")
            print(f"- Total parse failures:  {agg['total_parse_failures']}")
            print(f"- Total alias hits:      {agg['total_alias_hits']}")
            print(f"- Self-reported done:    {agg['total_self_reported']} / {agg['n_ran']}")
            print(f"- Total wall time:       {agg['total_wall_s']:.1f}s "
                  f"(skip-resumed tasks contribute their original wall_s_total)")

            summary = {
                "seed": seed,
                "device": resolved_device,
                "model_name": cfg.name,
                "model_revision": cfg.revision,
                "n_resumed": n_resumed,
                "tasks": [r.as_dict() for r in rows],
                "aggregate": agg,
                "per_tier": per_tier,
            }
            out_path = args.summary_dir / f"seed{seed}.json"
            with out_path.open("w") as fh:
                json.dump(summary, fh, indent=2, default=str)
            print(f"\nSummary JSON: {out_path}")

    finally:
        try:
            env.close()
        except Exception:
            logger.warning("env.close() failed", exc_info=True)

    print()
    print(f"## All seeds complete ({n_seeds} seeds × {len(tasks)} tasks "
          f"= {n_seeds * len(tasks)} episodes)")
    print(f"   Total wall (this process): {(time.monotonic() - overall_t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())

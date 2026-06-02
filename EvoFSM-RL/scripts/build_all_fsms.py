#!/usr/bin/env python3
"""Story 2.2.4 — batch FSM synthesis over the 12 source-pool apps.

For each source-pool app:

    1. compress_trajectories(app, trajectory_dir) → compressed text
    2. Print rough token estimate (len // 4)
    3. build_fsm(app, category, compressed_text) → FSM   [skipped on --dry-run]
    4. lint_layer2(fsm) → print pass/fail + violations
    5. Save artifacts/static_fsms/{app}.json + {app}.txt
    6. sleep 2s between apps (be polite to the API)

Per-app errors are caught and logged; the sweep continues. A markdown
summary table is printed at the end.

Usage:
    cd /shared/linqiang/evofsm_project && source .venv/bin/activate
    PYTHONPATH=android_world_plus:EvoFSM-RL python EvoFSM-RL/scripts/build_all_fsms.py \\
        --trajectory-dir EvoFSM-RL/traces/source_pool_trajectories \\
        --output-dir EvoFSM-RL/artifacts/static_fsms \\
        --splits-yaml EvoFSM-RL/configs/splits.yaml \\
        --categories-csv EvoFSM-RL/configs/task_categories.csv \\
        --dry-run        # ← compress + token-estimate only; no API
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("build_all_fsms")


@dataclasses.dataclass
class AppRow:
    """One row of the final summary table."""
    app: str
    category: str
    episodes: int = 0
    est_tokens: int = 0
    n_states: int | str = "—"
    n_transitions: int | str = "—"
    n_l2_categories: int | str = "—"
    lint: str = "—"           # "PASS" | "FAIL (N)" | "—" when skipped
    status: str = "—"          # "OK" | "DRY-RUN" | "FAILED: <kind>"
    error: str | None = None


# ─────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────


def _load_source_pool_apps(splits_yaml: Path) -> list[str]:
    """Return alphabetical list of the 12 source-pool app keys."""
    import yaml
    y = yaml.safe_load(splits_yaml.read_text())
    return sorted(y["source_pool"].keys())


def _load_app_categories(csv_path: Path, apps: list[str]) -> dict[str, str]:
    """Build {app: play_category} from configs/task_categories.csv.

    Multiple rows per app share the same play_category; we take the first.
    Raises if any of the requested apps has no row in the CSV.
    """
    out: dict[str, str] = {}
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            app = row["app"]
            if app in apps and app not in out:
                out[app] = row["play_category"]
    missing = [a for a in apps if a not in out]
    if missing:
        raise RuntimeError(
            f"No play_category found in {csv_path} for apps: {missing}"
        )
    return out


# ─────────────────────────────────────────────────────────────────────────
# Per-app pipeline
# ─────────────────────────────────────────────────────────────────────────


def _process_one_app(
    app: str,
    category: str,
    trajectory_dir: Path,
    output_dir: Path,
    *,
    dry_run: bool,
) -> AppRow:
    """Run the 6-step pipeline for one app. Never raises; returns an AppRow
    with status set on failure."""
    from evofsm_rl.fsm import build_fsm, compress_trajectories, lint_layer2

    row = AppRow(app=app, category=category)

    # ── 1. compress ──
    try:
        logger.info("=" * 72)
        logger.info("[%s] compressing trajectories...", app)
        compressed = compress_trajectories(app, trajectory_dir)
    except Exception as e:
        logger.exception("[%s] compress_trajectories failed", app)
        row.status = f"FAILED: compress ({type(e).__name__})"
        row.error = str(e)
        return row

    row.episodes = compressed.count("=== Episode: ")
    row.est_tokens = len(compressed) // 4
    logger.info(
        "[%s] %d episodes, %d chars, ~%d tokens (cap %d)",
        app, row.episodes, len(compressed), row.est_tokens, 160_000,
    )

    if row.episodes == 0:
        logger.warning("[%s] no matching episodes — skipping", app)
        row.status = "FAILED: no episodes"
        return row

    if dry_run:
        row.status = "DRY-RUN"
        return row

    # ── 3. build_fsm (LLM call) ──
    try:
        logger.info("[%s] calling LLM to synthesize FSM...", app)
        t0 = time.monotonic()
        fsm = build_fsm(app_name=app, category=category, compressed_text=compressed)
        logger.info("[%s] FSM synthesized in %.1fs", app, time.monotonic() - t0)
    except Exception as e:
        logger.exception("[%s] build_fsm failed", app)
        row.status = f"FAILED: build ({type(e).__name__})"
        row.error = str(e)
        return row

    row.n_states = len(fsm.layer1.states)
    row.n_transitions = len(fsm.layer1.transitions)
    row.n_l2_categories = len(fsm.layer2.categories)

    # ── 4. lint ──
    passed, violations = lint_layer2(fsm)
    if passed:
        row.lint = "PASS"
        logger.info("[%s] lint PASS", app)
    else:
        row.lint = f"FAIL ({len(violations)})"
        logger.warning("[%s] lint FAIL — %d violations:", app, len(violations))
        for v in violations:
            logger.warning("    %s", v)

    # ── 5. save artifacts ──
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{app}.json"
    txt_path = output_dir / f"{app}.txt"
    try:
        with json_path.open("w") as fh:
            json.dump(fsm.to_json(), fh, indent=2, ensure_ascii=False)
        txt_path.write_text(fsm.to_prompt_text())
        logger.info("[%s] wrote %s and %s", app, json_path, txt_path)
    except Exception as e:
        logger.exception("[%s] artifact write failed", app)
        row.status = f"FAILED: write ({type(e).__name__})"
        row.error = str(e)
        return row

    row.status = "OK"
    return row


# ─────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────


def _print_summary(rows: list[AppRow], *, dry_run: bool) -> None:
    """Print a markdown table of per-app results."""
    print()
    print(f"## Summary ({'DRY-RUN' if dry_run else 'LIVE'})")
    print()
    print("| App | Category | Episodes | ~Tokens | States | Transitions | L2 Cats | Lint | Status |")
    print("|-----|----------|---------:|--------:|-------:|------------:|--------:|------|--------|")
    for r in rows:
        print(
            f"| {r.app} | {r.category} | {r.episodes} | {r.est_tokens:,} | "
            f"{r.n_states} | {r.n_transitions} | {r.n_l2_categories} | "
            f"{r.lint} | {r.status} |"
        )
    print()
    n_total = len(rows)
    n_ok = sum(1 for r in rows if r.status == "OK")
    n_dry = sum(1 for r in rows if r.status == "DRY-RUN")
    n_failed = sum(1 for r in rows if r.status.startswith("FAILED"))
    n_lint_pass = sum(1 for r in rows if r.lint == "PASS")
    n_lint_fail = sum(1 for r in rows if r.lint.startswith("FAIL"))
    print(f"## Tally")
    print(f"  apps processed:   {n_total}")
    print(f"  status=OK:        {n_ok}")
    print(f"  status=DRY-RUN:   {n_dry}")
    print(f"  status=FAILED:    {n_failed}")
    print(f"  lint=PASS:        {n_lint_pass}")
    print(f"  lint=FAIL:        {n_lint_fail}")
    print(f"  total ~tokens:    {sum(r.est_tokens for r in rows):,}")
    if any(r.error for r in rows):
        print()
        print("## Failures")
        for r in rows:
            if r.error:
                print(f"  {r.app}: {r.error[:200]}")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trajectory-dir", type=Path, required=True,
                   help="dir containing {template}_seed{N}/ episode dirs")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="where to write per-app FSM artifacts")
    p.add_argument("--splits-yaml", type=Path, required=True,
                   help="configs/splits.yaml — used to enumerate source_pool apps")
    p.add_argument("--categories-csv", type=Path, required=True,
                   help="configs/task_categories.csv — for app→play_category")
    p.add_argument("--dry-run", action="store_true",
                   help="compress + token-estimate only; no API call, no artifacts written")
    p.add_argument("--apps", type=str, nargs="*", default=None,
                   help="optional: only process this subset of apps")
    p.add_argument("--sleep-between", type=float, default=2.0,
                   help="seconds to sleep between apps (default 2)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    apps = _load_source_pool_apps(args.splits_yaml)
    if args.apps:
        apps = [a for a in apps if a in set(args.apps)]
        if not apps:
            logger.error("--apps filter matched none of the source-pool apps")
            return 1
    cats = _load_app_categories(args.categories_csv, apps)

    logger.info("Found %d source-pool apps to process: %s", len(apps), apps)
    logger.info("Mode: %s", "DRY-RUN (no API)" if args.dry_run else "LIVE (LLM calls)")

    rows: list[AppRow] = []
    for i, app in enumerate(apps, start=1):
        logger.info("")
        logger.info("######################################################################")
        logger.info("# App %d/%d: %s (category=%s)", i, len(apps), app, cats[app])
        logger.info("######################################################################")
        try:
            row = _process_one_app(
                app, cats[app], args.trajectory_dir, args.output_dir,
                dry_run=args.dry_run,
            )
        except Exception as e:
            # Defensive — _process_one_app already catches per-step, but we
            # don't want a totally unexpected error to abort the sweep.
            logger.exception("[%s] unexpected error in pipeline orchestration", app)
            row = AppRow(app=app, category=cats[app],
                         status=f"FAILED: orchestration ({type(e).__name__})",
                         error=str(e))
        rows.append(row)

        # Be polite to the API
        if i < len(apps) and not args.dry_run:
            time.sleep(args.sleep_between)

    _print_summary(rows, dry_run=args.dry_run)
    n_failed = sum(1 for r in rows if r.status.startswith("FAILED"))
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

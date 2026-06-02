"""Build per-category L_C from the 12 source-pool FSMs — Story 2.3.

Flow:
  1. Read the source pool from ``configs/splits.yaml`` and group apps
     by their ``category`` (Play Store category).
  2. For each category, load the corresponding per-app FSMs from
     ``artifacts/static_fsms/{app}.json``.
  3. Run ``aggregate_L_C`` (LLM-mediated when multi-app, passthrough
     when single-app).
  4. Run ``lint_L_C`` against the source FSMs — a leak of ANY source
     app's specifics fails the category.
  5. Write the merged ``Layer2`` JSON to ``artifacts/L_C/{slug}.json``.
  6. Print a summary table.

Usage:
    PYTHONPATH=android_world_plus:EvoFSM-RL \\
        python EvoFSM-RL/scripts/build_L_C.py

The script exits non-zero if any category fails lint; the offending
file is still written so the user can inspect it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from evofsm_rl.fsm import FSM, Layer2, aggregate_L_C, category_to_slug, lint_L_C
from evofsm_rl.splits import get_source_pool


FSM_DIR = Path("EvoFSM-RL/artifacts/static_fsms")
OUT_DIR = Path("EvoFSM-RL/artifacts/L_C")


def _group_apps_by_category() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for app, info in get_source_pool().items():
        groups[info.category].append(app)
    # stable alphabetical ordering inside each group for byte-stable prompts
    return {cat: sorted(apps) for cat, apps in groups.items()}


def _load_fsm(app: str) -> FSM:
    path = FSM_DIR / f"{app}.json"
    if not path.exists():
        raise FileNotFoundError(f"FSM not found: {path}")
    return FSM.from_json(json.loads(path.read_text()))


def _write_L_C(category: str, merged: Layer2) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{category_to_slug(category)}.json"
    payload = {
        "category": category,
        "layer2": merged.to_json(),
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--categories", default=None,
        help="Comma-separated Play Store category names to restrict to "
             "(default: all source-pool categories).",
    )
    args = parser.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2

    groups = _group_apps_by_category()
    if args.categories:
        wanted = [c.strip() for c in args.categories.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in groups]
        if unknown:
            print(f"ERROR: unknown categories: {unknown}", file=sys.stderr)
            return 2
        groups = {c: groups[c] for c in wanted}

    print(f"Building L_C for {len(groups)} categories...\n")

    rows: list[dict] = []
    any_lint_fail = False

    for category, apps in groups.items():
        t0 = time.time()
        print(f"=== {category} ({len(apps)} app(s): {', '.join(apps)}) ===")
        fsms = [_load_fsm(app) for app in apps]
        input_cat_count = sum(len(f.layer2.categories) for f in fsms)

        try:
            merged = aggregate_L_C(fsms)
        except Exception as e:
            print(f"  aggregate_L_C FAILED: {e}")
            rows.append({
                "category": category,
                "apps": apps,
                "in_cats": input_cat_count,
                "out_cats": 0,
                "lint": "ERROR",
                "path": "",
                "wall_s": time.time() - t0,
            })
            any_lint_fail = True
            continue

        out_cat_count = len(merged.categories)
        passed, lint_errors = lint_L_C(merged, fsms)
        lint_status = "PASS" if passed else f"FAIL ({len(lint_errors)})"
        if not passed:
            any_lint_fail = True

        out_path = _write_L_C(category, merged)
        wall = time.time() - t0

        print(f"  input categories : {input_cat_count}")
        print(f"  merged categories: {out_cat_count}")
        print(f"  lint             : {lint_status}")
        if not passed:
            for err in lint_errors[:10]:
                print(f"     {err}")
            if len(lint_errors) > 10:
                print(f"     ... and {len(lint_errors) - 10} more")
        print(f"  written to       : {out_path}  ({wall:.1f}s)\n")

        rows.append({
            "category": category,
            "apps": apps,
            "in_cats": input_cat_count,
            "out_cats": out_cat_count,
            "lint": lint_status,
            "path": str(out_path),
            "wall_s": wall,
        })

    _print_summary_table(rows)

    return 1 if any_lint_fail else 0


def _print_summary_table(rows: list[dict]) -> None:
    print("\n=== SUMMARY ===")
    headers = ("Category", "Source Apps", "In", "Out", "Lint", "Wall")
    widths = [
        max(len(headers[0]), max((len(r["category"]) for r in rows), default=0)),
        max(len(headers[1]), max((len(", ".join(r["apps"])) for r in rows), default=0)),
        max(len(headers[2]), 3),
        max(len(headers[3]), 3),
        max(len(headers[4]), 8),
        max(len(headers[5]), 6),
    ]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(
            r["category"],
            ", ".join(r["apps"]),
            str(r["in_cats"]),
            str(r["out_cats"]),
            r["lint"],
            f"{r['wall_s']:.1f}s",
        ))

    n_pass = sum(1 for r in rows if r["lint"] == "PASS")
    print(f"\nPASS: {n_pass} / {len(rows)}    "
          f"FAIL: {len(rows) - n_pass}")


if __name__ == "__main__":
    sys.exit(main())

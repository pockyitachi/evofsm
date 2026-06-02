"""Re-lint already-written L_C files without recomputing them.

Used when the linter rule changes after a merge has already run. Reads
every ``artifacts/L_C/*.json``, resolves its source FSMs via
``configs/splits.yaml``, and reports PASS/FAIL per category.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from evofsm_rl.fsm import FSM, lint_L_C, load_L_C
from evofsm_rl.splits import get_source_pool


FSM_DIR = Path("EvoFSM-RL/artifacts/static_fsms")
LC_DIR = Path("EvoFSM-RL/artifacts/L_C")


def _load_fsm(app: str) -> FSM:
    return FSM.from_json(json.loads((FSM_DIR / f"{app}.json").read_text()))


def main() -> int:
    cat_to_apps: dict[str, list[str]] = defaultdict(list)
    for app, info in get_source_pool().items():
        cat_to_apps[info.category].append(app)
    for v in cat_to_apps.values():
        v.sort()

    any_fail = False
    rows = []
    for lc_file in sorted(LC_DIR.glob("*.json")):
        category, layer2 = load_L_C(lc_file)
        apps = cat_to_apps[category]
        sources = [_load_fsm(a) for a in apps]

        passed, errors = lint_L_C(layer2, sources)
        status = "PASS" if passed else f"FAIL ({len(errors)})"
        rows.append((category, apps, len(layer2.categories), status, errors))
        print(f"=== {category} ({len(apps)} app(s)) — {status} ===")
        for e in errors:
            print(f"  {e}")
        if not passed:
            any_fail = True

    print("\n=== SUMMARY ===")
    for cat, apps, n_cats, status, _ in rows:
        print(f"  {cat:15s}  {len(apps)} app  merged={n_cats:3d}  {status}")
    n_pass = sum(1 for r in rows if r[3] == "PASS")
    print(f"\nPASS: {n_pass} / {len(rows)}    FAIL: {len(rows) - n_pass}")

    return 1 if any_fail else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

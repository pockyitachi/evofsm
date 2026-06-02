"""Generate `configs/task_categories.csv` from splits.yaml + task_metadata.json.

One row per task: (task_name, app, tier, play_category, task_type).
Run: python -m scripts.generate_task_categories_csv  (from EvoFSM-RL/)
"""

from __future__ import annotations

import csv
from pathlib import Path

from evofsm_rl import splits, taxonomy


OUT = Path(__file__).resolve().parent.parent / "configs" / "task_categories.csv"


def main() -> None:
    rows = []

    for app, info in splits.get_source_pool().items():
        cat = taxonomy.play_category_of(app)
        for tpl in info.templates:
            rows.append((tpl, app, "source", cat, taxonomy.task_type_of(tpl)))

    for app, info in splits.get_tier_B_apps().items():
        cat = taxonomy.play_category_of(app)
        for tpl in info.T_adapt:
            rows.append((tpl, app, "tier_B/T_adapt", cat, taxonomy.task_type_of(tpl)))
        for tpl in info.T_eval:
            rows.append((tpl, app, "tier_B/T_eval", cat, taxonomy.task_type_of(tpl)))

    for app, info in splits.get_tier_C_apps().items():
        cat = taxonomy.play_category_of(app)
        for tpl in info.T_adapt:
            rows.append((tpl, app, "tier_C/T_adapt", cat, taxonomy.task_type_of(tpl)))
        for tpl in info.T_eval:
            rows.append((tpl, app, "tier_C/T_eval", cat, taxonomy.task_type_of(tpl)))

    rows.sort(key=lambda r: (r[3], r[1], r[0]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_name", "app", "tier", "play_category", "task_type"])
        w.writerows(rows)

    # Quick stats
    by_type: dict[str, int] = {}
    for _, _, _, _, t in rows:
        by_type[t] = by_type.get(t, 0) + 1
    print(f"Wrote {len(rows)} rows to {OUT}")
    print("By task_type:")
    for k in sorted(by_type, key=lambda x: -by_type[x]):
        print(f"  {k:30s} {by_type[k]}")


if __name__ == "__main__":
    main()

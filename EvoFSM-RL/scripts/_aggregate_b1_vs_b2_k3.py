"""Aggregate B1 K=3 and B2 K=3 summaries and print side-by-side comparison."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def _load(seed_files: list[Path]) -> list[dict]:
    rows = []
    for p in sorted(seed_files):
        with p.open() as fh:
            for line in fh:
                rows.append(json.loads(line))
    return rows


def _stderr(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n) if n > 0 else 0.0


def _sr_of(rows: list[dict]) -> tuple[float, float, int]:
    if not rows:
        return 0.0, 0.0, 0
    n = len(rows)
    k = sum(r["success"] for r in rows)
    p = k / n
    return p, _stderr(p, n), n


def _per_tier(rows: list[dict]) -> dict[str, tuple[float, float, int]]:
    out = {}
    for tier in ("tier_B", "tier_C"):
        g = [r for r in rows if r["tier"] == tier]
        out[tier] = _sr_of(g)
    return out


def _per_app(rows: list[dict]) -> dict[str, tuple[str, float, float, int, bool]]:
    per_app = defaultdict(list)
    for r in rows:
        per_app[r["app"]].append(r)
    out = {}
    for app, g in per_app.items():
        p, se, n = _sr_of(g)
        out[app] = (g[0]["tier"], p, se, n, bool(g[0]["l_c_injected"]))
    return out


def _per_template(rows: list[dict]) -> dict[str, tuple[str, str, float, float, int]]:
    per = defaultdict(list)
    for r in rows:
        per[r["template"]].append(r)
    out = {}
    for tpl, g in per.items():
        p, se, n = _sr_of(g)
        out[tpl] = (g[0]["tier"], g[0]["app"], p, se, n)
    return out


def _fmt_pct_se(p: float, se: float, n: int) -> str:
    return f"{p*100:5.1f}% ± {se*100:4.1f}% (n={n})"


def _diff(p_b2: float, se_b2: float, p_b1: float, se_b1: float) -> str:
    d = p_b2 - p_b1
    se = math.sqrt(se_b1 ** 2 + se_b2 ** 2)
    z = d / se if se > 0 else 0.0
    sign = "+" if d >= 0 else ""
    return f"{sign}{d*100:5.1f}pp  (z={z:+.2f})"


def main() -> int:
    b1_dir = Path("EvoFSM-RL/traces/b1_teval_k3")
    b2_dir = Path("EvoFSM-RL/traces/b2_teval_k3")
    b1 = _load(list(b1_dir.glob("summary_seed*.jsonl")))
    b2 = _load(list(b2_dir.glob("summary_seed*.jsonl")))

    if not b1 or not b2:
        print("missing summaries")
        return 1

    b1_seeds = sorted({r["seed"] for r in b1})
    b2_seeds = sorted({r["seed"] for r in b2})
    print(f"B1 seeds: {b1_seeds}  ({len(b1)} episodes)")
    print(f"B2 seeds: {b2_seeds}  ({len(b2)} episodes)")

    print("\n## Overall (K=3)")
    p1, se1, n1 = _sr_of(b1)
    p2, se2, n2 = _sr_of(b2)
    print(f"  B1 : {_fmt_pct_se(p1, se1, n1)}")
    print(f"  B2 : {_fmt_pct_se(p2, se2, n2)}")
    print(f"  Δ  : {_diff(p2, se2, p1, se1)}")

    print("\n## Per-tier (K=3)")
    t1 = _per_tier(b1)
    t2 = _per_tier(b2)
    for tier in ("tier_B", "tier_C"):
        p1, se1, n1 = t1[tier]
        p2, se2, n2 = t2[tier]
        print(f"  [{tier}]")
        print(f"    B1: {_fmt_pct_se(p1, se1, n1)}")
        print(f"    B2: {_fmt_pct_se(p2, se2, n2)}")
        print(f"    Δ : {_diff(p2, se2, p1, se1)}")

    print("\n## Per-app (K=3)")
    a1 = _per_app(b1)
    a2 = _per_app(b2)
    apps_order = sorted(set(a1) | set(a2), key=lambda a: (a1.get(a, a2[a])[0], a))
    print(f"  {'App':22s}  {'Tier':7s} {'L_C?':4s}  {'B1':>22s}  {'B2':>22s}  {'Δ':>18s}")
    print("  " + "-" * 98)
    for app in apps_order:
        tier, p1, se1, n1, inj = a1.get(app, a2[app])
        _, p2, se2, n2, _ = a2[app]
        marker = "yes" if inj else "no"
        print(f"  {app:22s}  {tier:7s} {marker:4s}  "
              f"{_fmt_pct_se(p1, se1, n1):>22s}  "
              f"{_fmt_pct_se(p2, se2, n2):>22s}  "
              f"{_diff(p2, se2, p1, se1):>18s}")

    print("\n## Per-template (K=3, Tier-B only, sorted by Δ)")
    t1 = _per_template(b1)
    t2 = _per_template(b2)
    entries = []
    for tpl in set(t1) | set(t2):
        tier, app, p1, se1, n1 = t1.get(tpl, (None, None, 0.0, 0.0, 0))
        _, _, p2, se2, n2 = t2.get(tpl, (None, None, 0.0, 0.0, 0))
        if tier != "tier_B":
            continue
        entries.append((tpl, app, p1, p2, p2 - p1, n1, n2))
    entries.sort(key=lambda e: -e[4])
    print(f"  {'Template':40s}  {'App':22s}  {'B1':>6s}  {'B2':>6s}  {'Δ':>7s}")
    print("  " + "-" * 90)
    for tpl, app, p1, p2, d, n1, n2 in entries:
        print(f"  {tpl:40s}  {app:22s}  {p1*100:5.1f}%  {p2*100:5.1f}%  "
              f"{('+' if d >= 0 else '')}{d*100:5.1f}pp")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

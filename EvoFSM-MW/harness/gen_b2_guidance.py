"""Offline B2 guidance renderer — task_name → app_guidance text for MobileWorld T-eval.

Implements `docs/lc_injection_multiapp.md` (per-app resolve → dedup by category →
multi-block render with app labels) at the STATIC (B2) stage: tier order is
app-FSM → category-L_C → nothing. The bootstrap tier needs target-app exploration
trajectories (Phase-2 / TTA) and is intentionally absent here — novel-category
apps (Mastodon / Mattermost / Taodian) contribute no block, so pure Tier-C tasks
get text == "" and degrade to B1 exactly.

Run from the project root with the MAIN venv (needs evofsm_rl importable):

    cd /shared/linqiang/evofsm_project
    PYTHONPATH=EvoFSM-RL .venv/bin/python EvoFSM-MW/harness/gen_b2_guidance.py

Output: EvoFSM-MW/artifacts/b2_guidance.json
    {"meta": {...}, "tasks": {task_name: {"text": str, "blocks": [...]}}}

The JSON is the frozen injection artifact: MobileWorld's B2 agent
(`harness/qwen3vl_b2_agent.py`) only reads this file — it never imports
evofsm_rl, so the MobileWorld venv stays untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

EVOFSM_MW = Path(__file__).resolve().parents[1]
PROJECT = EVOFSM_MW.parent
sys.path.insert(0, str(PROJECT / "EvoFSM-RL"))

from evofsm_rl.fsm.aggregator import category_to_slug, load_L_C  # noqa: E402
from evofsm_rl.fsm.schema import FSM  # noqa: E402

SPLITS_YAML = EVOFSM_MW / "configs" / "mobileworld_splits.yaml"
FSM_DIR = PROJECT / "EvoFSM-RL" / "artifacts" / "static_fsms_v2"
LC_DIR = PROJECT / "EvoFSM-RL" / "artifacts" / "L_C_v2"
OUT_PATH = EVOFSM_MW / "artifacts" / "b2_guidance.json"

# MW app → Play category (docs/dataset_tiers.md §4). Novel categories have no
# L_C file on disk, which is what routes them to the "none" outcome below.
MW_APP_CATEGORY = {
    "Mail": "Communication",
    "Messages": "Communication",
    "Contacts": "Communication",
    "Chrome": "Communication",
    "Files": "Tools",
    "Settings": "Tools",
    "Clock": "Tools",
    "Calendar": "Productivity",
    "Maps": "Maps & Navigation",
    "Camera": "Photography",
    "Gallery": "Photography",
    "Docreader": "Books & Reference",
    "Mastodon": "Social",
    "Mattermost": "Social",
    "Taodian": "Shopping",
}

# The 6 system apps that share the SAME package between AW+ and MobileWorld
# (dataset_tiers.md §2 footnote †) — only these may hit the app-FSM tier.
MW_APP_FSM_KEY = {
    "Chrome": "chrome",
    "Contacts": "contacts",
    "Settings": "system_settings",
    "Clock": "clock",
    "Camera": "camera",
    "Files": "files",
}

INTRO = (
    "The following workflow knowledge was learned from this app or related apps "
    "of the same category. Use it to plan; it describes WHAT to do at a high "
    "level, not exact coordinates — you still must look at the screenshot to "
    "ground each action."
)


def resolve_task_guidance(apps: list[str], mode: str = "full") -> tuple[str, list[dict]]:
    """Per-app resolve → dedup category blocks → render. Returns (text, blocks).

    mode:
      full           — B2: tier-1 injects the FULL app FSM (Layer-1 + Layer-2)
      app-l2         — B2': tier-1 injects ONLY the app FSM's own Layer-2
                       (Layer-1 states/visual_cues never cross the benchmark gap)
      category-only  — B2'': tier-1 disabled; every app resolves via category
                       L_C — the original EvoFSM-RL B2 recipe (+9.3pp run), 1:1
    """
    rendered: list[str] = []
    blocks: list[dict] = []
    seen_categories: dict[str, dict] = {}  # category -> block meta (for dedup)

    for app in apps:
        # ── Tier 1: app-level static FSM (same package on both benchmarks) ──
        fsm_key = MW_APP_FSM_KEY.get(app)
        if mode != "category-only" and fsm_key and (FSM_DIR / f"{fsm_key}.json").exists():
            fsm = FSM.from_json(json.loads((FSM_DIR / f"{fsm_key}.json").read_text()))
            if mode == "app-l2":
                body = fsm.layer2.to_prompt_text(category=MW_APP_CATEGORY[app])
                label = "app workflow knowledge"
            else:
                body = fsm.to_prompt_text()
                label = "app-specific FSM"
            rendered.append(f"## For [{app}]  ({label}, source: pretrained)\n{body}")
            blocks.append({"apps": [app], "tier": "app", "key": fsm_key, "chars": len(body)})
            continue

        # ── Tier 2: category-level L_C, dedup by category ──
        category = MW_APP_CATEGORY[app]
        if category in seen_categories:
            meta = seen_categories[category]
            meta["apps"].append(app)
            idx = meta["render_idx"]
            apps_label = ", ".join(meta["apps"])
            rendered[idx] = (
                f"## For [{apps_label}]  (category: {category}, source: pretrained)\n"
                + rendered[idx].split("\n", 1)[1]
            )
            continue

        lc_path = LC_DIR / f"{category_to_slug(category)}.json"
        if not lc_path.exists():
            # Novel category (Social / Shopping): B2 static has no bootstrap —
            # this app contributes nothing (TTA adds the bootstrap tier later).
            blocks.append({"apps": [app], "tier": "none", "key": None, "chars": 0})
            continue

        _, layer2 = load_L_C(lc_path)
        body = layer2.to_prompt_text(category=category)
        rendered.append(f"## For [{app}]  (category: {category}, source: pretrained)\n{body}")
        meta = {
            "apps": [app],
            "tier": "category",
            "key": category_to_slug(category),
            "chars": len(body),
            "render_idx": len(rendered) - 1,
        }
        seen_categories[category] = meta
        blocks.append(meta)

    if not rendered:
        return "", blocks

    text = "# App guidance\n" + INTRO + "\n\n" + "\n\n".join(rendered)
    for b in blocks:
        b.pop("render_idx", None)
    return text, blocks


def main(mode: str = "full", out_path: Path = OUT_PATH) -> None:
    splits = yaml.safe_load(SPLITS_YAML.read_text())
    tasks: dict[str, dict] = {}
    stats = {"with_guidance": 0, "empty": 0, "by_tier": {}}

    for split_name in ("adapt", "eval"):
        for row in splits[split_name]:
            name, tier, apps = row["task"], row["tier"], row["apps"]
            text, blocks = resolve_task_guidance(apps, mode)
            tasks[name] = {
                "split": split_name,
                "tier": tier,
                "apps": apps,
                "text": text,
                "blocks": blocks,
                "chars": len(text),
            }
            stats["with_guidance" if text else "empty"] += 1
            stats["by_tier"].setdefault(tier, {"with": 0, "empty": 0})
            stats["by_tier"][tier]["with" if text else "empty"] += 1

    stage = {
        "full": "B2-static (no bootstrap tier)",
        "app-l2": "B2'-static app-Layer2-only (no Layer-1)",
        "category-only": "B2''-static category-L_C-only (original EvoFSM-RL B2 recipe)",
    }[mode]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "meta": {
                    "splits_version": splits["meta"]["version"],
                    "fsm_dir": str(FSM_DIR.relative_to(PROJECT)),
                    "lc_dir": str(LC_DIR.relative_to(PROJECT)),
                    "mode": mode,
                    "stage": stage,
                    "stats": stats,
                },
                "tasks": tasks,
            },
            ensure_ascii=False,
            indent=1,
        )
    )

    print(f"wrote {out_path}  (mode={mode})")
    print(f"tasks: {len(tasks)}  with_guidance: {stats['with_guidance']}  empty: {stats['empty']}")
    for tier, c in sorted(stats["by_tier"].items()):
        print(f"  Tier-{tier}: with={c['with']} empty={c['empty']}")
    sizes = sorted((t["chars"], n) for n, t in tasks.items() if t["chars"])
    if sizes:
        print(f"guidance chars: min={sizes[0][0]} median={sizes[len(sizes)//2][0]} max={sizes[-1][0]} ({sizes[-1][1]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render static guidance JSON for B2-family evals")
    ap.add_argument("--mode", choices=["full", "app-l2", "category-only"], default="full")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--lc-dir", type=Path, default=None,
                    help="override the L_C directory (e.g. EvoFSM-RL/artifacts/L_C_v3), "
                         "relative to project root or absolute; default L_C_v2")
    args = ap.parse_args()
    if args.lc_dir is not None:
        LC_DIR = args.lc_dir if args.lc_dir.is_absolute() else PROJECT / args.lc_dir
    main(args.mode, args.out)

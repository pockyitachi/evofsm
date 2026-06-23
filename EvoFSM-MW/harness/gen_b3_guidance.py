"""B3 snapshot guidance: render the round-1 evolved champions into an eval
guidance JSON (same shape as b2p_guidance.json), so the B3-adapted FSMs can be
injected on the frozen 110-task eval split and compared to B1 / B2'.

Per eval task: for each app, take that app's CHAMPION variant's Layer-2 and
render it (the evolved knowledge — incl. bootstrap apps that started empty and
grew content). This is exactly the B2'/app-Layer2 injection shape, but with
evolved champions in place of static seeds. Tier-C bootstrap apps (Mastodon/
Mattermost/Taodian) now contribute non-empty blocks (vs B2' where they were
empty) — the main B3 vs B2' difference.

Run in the skyrl-agent venv (needs the TTAController import):
  cd .../SkyRL-AndroidWorld/skyrl-agent
  .venv/bin/python /shared/.../EvoFSM-MW/harness/gen_b3_guidance.py \
      --run-dir tmp_training/mw_b3_base_r1 \
      --out /shared/.../EvoFSM-MW/artifacts/b3champ_guidance.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import yaml

SKY = Path(__file__).resolve()
# skyrl-agent on path for the controller import
SKYROOT = Path("/shared/linqiang/evofsm_project/SkyRL-AndroidWorld/skyrl-agent")
sys.path.insert(0, str(SKYROOT))

from skyrl_agent.evofsm_tta.controller import TTAController, MW_APP_CATEGORY  # noqa: E402

SPLITS = Path("/shared/linqiang/evofsm_project/EvoFSM-MW/configs/mobileworld_splits.yaml")
INTRO = (
    "The following workflow knowledge was learned (and test-time evolved) for "
    "this app or related apps of the same category. Use it to plan; it describes "
    "WHAT to do at a high level, not exact coordinates — you still must look at "
    "the screenshot to ground each action."
)


def main(run_dir: str, out_path: Path) -> None:
    ctrl = TTAController.get(run_dir)
    splits = yaml.safe_load(SPLITS.read_text())
    tasks: dict = {}
    stats = {"with_guidance": 0, "empty": 0, "by_tier": {}}

    for split_name in ("adapt", "eval"):
        for row in splits[split_name]:
            name, tier, apps = row["task"], row["tier"], row["apps"]
            blocks_txt, blocks_meta = [], []
            for app in apps:
                pop = ctrl.populations.get(app)
                if pop is None:
                    blocks_meta.append({"apps": [app], "champ": None, "chars": 0})
                    continue
                ch = pop.champion
                l2 = ch.fsm.layer2
                cats = getattr(l2, "categories", None) or []
                if not cats:
                    blocks_meta.append({"apps": [app], "champ": ch.id, "chars": 0})
                    continue
                body = l2.to_prompt_text(category=MW_APP_CATEGORY.get(app, ""))
                blocks_txt.append(f"## For [{app}]  (B3 champion {ch.id})\n{body}")
                blocks_meta.append({"apps": [app], "champ": ch.id, "chars": len(body)})
            text = ("# App guidance\n" + INTRO + "\n\n" + "\n\n".join(blocks_txt)) if blocks_txt else ""
            tasks[name] = {"split": split_name, "tier": tier, "apps": apps,
                           "text": text, "blocks": blocks_meta, "chars": len(text)}
            stats["with_guidance" if text else "empty"] += 1
            stats["by_tier"].setdefault(tier, {"with": 0, "empty": 0})
            stats["by_tier"][tier]["with" if text else "empty"] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"meta": {"source": "B3 round-1 champions", "run_dir": run_dir,
                  "stage": "B3-evolved", "stats": stats}, "tasks": tasks},
        ensure_ascii=False, indent=1))
    print(f"wrote {out_path}")
    print(f"tasks: {len(tasks)}  with_guidance: {stats['with_guidance']}  empty: {stats['empty']}")
    for t, c in sorted(stats["by_tier"].items()):
        print(f"  Tier-{t}: with={c['with']} empty={c['empty']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    main(a.run_dir, a.out)

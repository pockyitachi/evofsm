#!/usr/bin/env python3
"""Merge an FSM-champion guidance + a lessons-only guidance into fsm+lessons.

fsm+lessons(task) = FSM-champion text  +  distilled lesson delta (if any).

Both inputs share task keys and the {"meta":..., "tasks":{name:{"text":...}}}
shape (champion from gen_b3_guidance.py, lessons from gen_b3_lessons_guidance.py
--lessons-only). The MAI/qwen B2 agents read tasks[name].text and meta.stats, so
we preserve meta.stats. Per task: concat champion + "\n" + lesson (union of keys;
whichever side is missing contributes nothing).

    python merge_fsm_lessons.py \
      --fsm   artifacts/b3mai_champ_guidance.json \
      --lessons artifacts/b3mai_lesson_only_guidance.json \
      --out   artifacts/b3mai_fsm_lesson_guidance.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsm", type=Path, required=True, help="FSM champion guidance json")
    ap.add_argument("--lessons", type=Path, required=True, help="lessons-only guidance json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    fsm = json.loads(a.fsm.read_text())
    les = json.loads(a.lessons.read_text())
    ft, lt = fsm["tasks"], les["tasks"]

    out_tasks: dict[str, dict] = {}
    stats = {"with_fsm": 0, "with_lessons": 0, "both": 0, "empty": 0}
    for name in sorted(set(ft) | set(lt)):
        f = ft.get(name, {})
        l = lt.get(name, {})
        ftext = (f.get("text") or "").strip()
        ltext = (l.get("text") or "").strip()
        if ftext and ltext:
            text = ftext + "\n" + ltext
        else:
            text = ftext or ltext
        base = f or l  # carry split/tier/apps from whichever exists
        out_tasks[name] = {
            "split": base.get("split"), "tier": base.get("tier"), "apps": base.get("apps"),
            "text": text, "chars": len(text),
            "n_lessons": l.get("n_lessons", 0), "has_fsm": bool(ftext),
        }
        stats["with_fsm"] += 1 if ftext else 0
        stats["with_lessons"] += 1 if ltext else 0
        stats["both"] += 1 if (ftext and ltext) else 0
        stats["empty"] += 1 if not text else 0

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "meta": {"mode": "fsm+lessons",
                 "note": "FSM champion + distilled lessons",
                 "fsm_src": str(a.fsm.name), "lessons_src": str(a.lessons.name),
                 "stats": stats},
        "tasks": out_tasks,
    }, ensure_ascii=False, indent=1))
    print(f"wrote {a.out}  ({len(out_tasks)} tasks; stats={stats})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render eval guidance for B3 lesson-memory mode.

guidance(task) = B2'(app-l2) static prior  +  retrieved VALIDATED lessons.

The prior is identical to what the B2'(v2) baseline injects, so B3 − B2'
isolates the lessons. Lessons come from the per-app tables an adapt run wrote
to {run-dir}/lessons/{app}.json; only validated ones (proven by the
with/without game during adapt) are injected at eval.

    PYTHONPATH=EvoFSM-RL:SkyRL-AndroidWorld/skyrl-agent \
      python EvoFSM-MW/harness/gen_b3_lessons_guidance.py \
        --run-dir SkyRL-AndroidWorld/skyrl-agent/tmp_training/mw_b3_lesson_r1 \
        --out EvoFSM-MW/artifacts/b3lesson_guidance.json
"""
from __future__ import annotations
import argparse
import importlib.util as ilu
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "EvoFSM-RL"))
sys.path.insert(0, str(PROJECT / "SkyRL-AndroidWorld" / "skyrl-agent"))

# reuse the B2' prior renderer + splits
_g_spec = ilu.spec_from_file_location("gen_b2", str(Path(__file__).with_name("gen_b2_guidance.py")))
_g = ilu.module_from_spec(_g_spec)
_g_spec.loader.exec_module(_g)

from skyrl_agent.evofsm_tta import lessons as L  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="adapt run-dir holding lessons/{app}.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=2, help="max lessons injected per task")
    ap.add_argument("--all-lessons", action="store_true",
                    help="inject ALL mined lessons (validated_only=False). Use when "
                         "adapt-time with/without validation could not fire (single-pass "
                         "schedule -> lessons never re-encountered); the 110-eval then IS "
                         "the validation. Default injects only validated lessons.")
    ap.add_argument("--lessons-only", action="store_true",
                    help="inject ONLY the lessons, DROP the verbose B2' prior. The lesson "
                         "is the adapt-distilled, behaviour-grounded compression of the "
                         "prior — inject the distilled form, not the bloat. Tasks with no "
                         "matching lesson get an empty injection (= B1).")
    a = ap.parse_args()

    lessons_dir = Path(a.run_dir)
    if not lessons_dir.is_absolute():
        lessons_dir = PROJECT / lessons_dir
    lessons_dir = lessons_dir / "lessons"

    tables: dict[str, dict] = {}
    for p in sorted(lessons_dir.glob("*.json")):
        tables[p.stem] = json.loads(p.read_text())
    n_val = sum(1 for t in tables.values() for x in t.get("lessons", []) if x.get("validated"))
    print(f"loaded {len(tables)} app tables, {n_val} validated lessons")

    splits = __import__("yaml").safe_load(_g.SPLITS_YAML.read_text())
    out_tasks: dict[str, dict] = {}
    stats = {"with_prior": 0, "with_lessons": 0, "empty": 0}

    for split_name in ("adapt", "eval"):
        for row in splits[split_name]:
            name, tier, apps = row["task"], row["tier"], row["apps"]
            prior, _ = _g.resolve_task_guidance(apps, "app-l2")
            words = L._stems  # noqa
            task_words = __import__("re").sub(r"(?<=[a-z])(?=[A-Z])", " ", name).replace("_", " ")
            # retrieve validated lessons from every app this task touches
            picked: list[dict] = []
            seen = set()
            for app in apps:
                for les in L.retrieve(tables.get(app, {"lessons": []}), task_words,
                                      top_k=a.top_k, validated_only=not a.all_lessons):
                    key = les["applies_when"]
                    if key not in seen:
                        seen.add(key)
                        picked.append(les)
            picked = picked[: a.top_k]
            delta = L.render_lessons(picked) if picked else ""
            if a.lessons_only:
                text = delta  # distilled form replaces the verbose prior
            else:
                text = prior + (("\n" + delta) if delta else "")
            out_tasks[name] = {"split": split_name, "tier": tier, "apps": apps,
                               "text": text, "n_lessons": len(picked), "chars": len(text)}
            stats["with_prior"] += 1 if prior else 0
            stats["with_lessons"] += 1 if picked else 0
            stats["empty"] += 1 if not text else 0

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "meta": {"mode": "b3-lesson", "run_dir": str(a.run_dir),
                 "lc_dir": str(_g.LC_DIR.relative_to(PROJECT)),
                 "validated_lessons": n_val, "stats": stats},
        "tasks": out_tasks,
    }, ensure_ascii=False, indent=1))
    print(f"wrote {a.out}  ({len(out_tasks)} tasks; with_lessons={stats['with_lessons']})")


if __name__ == "__main__":
    main()

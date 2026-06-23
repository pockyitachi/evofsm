"""B2''' guidance: entry-level retrieval on top of the frozen B2' artifact.

Implements fix #1 of docs/qwen3_8b_res.md §4.3: instead of injecting whole
Layer-2 / L_C libraries, keep at most K=3 entries per block — the ones a
retriever judges relevant to the task instruction. Single-variable vs B2':
same blocks, same rendering, same INTRO; only entries are filtered (fixes
#2/#3 deliberately NOT applied).

Retriever: the same Qwen3-VL-8B served on :8001 (text-only chat call,
temperature 0 → deterministic), one call per (task, block). Instructions come
from artifacts/task_goals.json (extracted from the B1 trajectories); the two
tasks without trajectories fall back to a de-camel-cased task name.

Only the 110 eval-split tasks are retrieved; adapt-split rows are copied from
B2' unfiltered and marked, since they are not run by the B-series evals.

Run (main venv):  .venv/bin/python EvoFSM-MW/harness/gen_b2ppp_guidance.py
Output:           EvoFSM-MW/artifacts/b2ppp_guidance.json
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / "artifacts"
SRC = ART / "b2p_guidance.json"
GOALS = ART / "task_goals.json"
OUT = ART / "b2ppp_guidance.json"
TASKLIST = Path(__file__).resolve().parents[1] / "configs" / "teval_tasklist.txt"
VLLM = "http://localhost:8001/v1/chat/completions"
MODEL = "Qwen3-VL-8B-Instruct"
K = 3


def chat(messages: list[dict], max_tokens: int = 150) -> str:
    body = json.dumps(
        {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    ).encode()
    req = urllib.request.Request(VLLM, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def split_blocks(task_text: str) -> tuple[str, list[dict]]:
    """task text -> (intro part, [{header, banner, entries:[(name, text)]}])."""
    pieces = re.split(r"\n(?=## For )", task_text)
    intro, blocks = pieces[0], []
    for piece in pieces[1:]:
        header, _, body = piece.partition("\n")
        segs = re.split(r"\n(?=CATEGORY: )", body)
        banner = segs[0]
        entries = []
        for seg in segs[1:]:
            name = seg.split("\n", 1)[0][len("CATEGORY: "):].strip()
            entries.append((name, seg.rstrip()))
        blocks.append({"header": header, "banner": banner.rstrip(), "entries": entries})
    return intro, blocks


def select_entries(goal: str, header: str, entries: list[tuple[str, str]]) -> tuple[list[str], bool]:
    """Ask the retriever for <=K relevant entry names. Returns (names, fallback)."""
    cand = []
    for name, text in entries:
        m = re.search(r"^  precondition: (.+)$", text, re.M)
        cand.append(f"- {name}: {m.group(1) if m else ''}")
    user = (
        f"Task instruction: {goal}\n\n"
        f"Candidate procedure entries from block \"{header.strip('# ').strip()}\":\n"
        + "\n".join(cand)
        + f"\n\nSelect AT MOST {K} entries whose procedures this task actually needs. "
        "Prefer fewer. If none apply, answer []. "
        "Answer with ONLY a JSON array of entry names."
    )
    msgs = [
        {"role": "system", "content": "You select relevant procedure entries for a mobile GUI task. Answer with ONLY a JSON array of entry names."},
        {"role": "user", "content": user},
    ]
    valid = {n for n, _ in entries}
    for _ in range(3):
        try:
            out = chat(msgs)
            m = re.search(r"\[.*?\]", out, re.S)
            names = [n for n in json.loads(m.group(0)) if n in valid][:K]
            return names, False
        except Exception:
            continue
    return [n for n, _ in entries], True  # fallback: keep whole block, flagged


def decamel(task: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", task.removesuffix("Task")).lower()


def main() -> None:
    src = json.load(open(SRC))
    goals = json.load(open(GOALS))
    run_list = {t.strip() for t in TASKLIST.read_text().replace("\n", "").split(",") if t.strip()}

    tasks_out: dict[str, dict] = {}
    jobs = []  # (task_name, goal, intro, blocks)
    for name, entry in src["tasks"].items():
        if name not in run_list or not entry["text"]:
            e = dict(entry)
            e["retrieval"] = "skipped (not in eval run list)" if name not in run_list else "empty"
            tasks_out[name] = e
            continue
        goal = goals.get(name) or decamel(name)
        intro, blocks = split_blocks(entry["text"])
        jobs.append((name, goal, intro, blocks, entry))

    def work(job):
        name, goal, intro, blocks, entry = job
        new_blocks_meta, rendered = [], []
        # Rendered "## For" sections correspond 1:1, in order, with the
        # non-"none" meta rows; "none" rows have no text and pass through.
        none_rows = [dict(m) for m in entry["blocks"] if m["tier"] == "none"]
        for blk, meta in zip(blocks, [m for m in entry["blocks"] if m["tier"] != "none"]):
            names, fallback = select_entries(goal, blk["header"], blk["entries"])
            kept = [text for n, text in blk["entries"] if n in names]
            bm = dict(meta)
            bm.update(entries_total=len(blk["entries"]), entries_kept=names, fallback=fallback)
            if kept:
                body = blk["banner"] + "\n" + "\n".join(kept)
                rendered.append(blk["header"] + "\n" + body)
                bm["chars"] = len(body)
            else:
                bm["chars"] = 0
            new_blocks_meta.append(bm)
        new_blocks_meta.extend(none_rows)
        text = (intro + "\n" + "\n\n".join(rendered)) if rendered else ""
        return name, {
            "split": entry["split"], "tier": entry["tier"], "apps": entry["apps"],
            "text": text, "blocks": new_blocks_meta, "chars": len(text),
            "instruction_used": goal,
        }

    with ThreadPoolExecutor(max_workers=3) as ex:
        for name, e in ex.map(work, jobs):
            tasks_out[name] = e
            print(f"  {name}: {e['chars']} chars, kept={[(b['tier'], b['entries_kept']) for b in e['blocks'] if 'entries_kept' in b]}", file=sys.stderr)

    retrieved = [t for t, e in tasks_out.items() if t in run_list and "instruction_used" in e]
    fallbacks = [t for t in retrieved if any(b.get("fallback") for b in tasks_out[t]["blocks"])]
    emptied = [t for t in retrieved if tasks_out[t]["chars"] == 0]
    sizes = sorted(tasks_out[t]["chars"] for t in retrieved if tasks_out[t]["chars"])
    meta = {
        "mode": "app-l2-retrieved",
        "base": "b2p_guidance.json (B2')",
        "retriever": {"model": MODEL, "temperature": 0, "max_entries_per_block": K},
        "stats": {
            "retrieved_tasks": len(retrieved),
            "fallback_tasks": fallbacks,
            "emptied_by_retrieval": emptied,
            "chars": {"min": sizes[0] if sizes else 0,
                      "median": sizes[len(sizes) // 2] if sizes else 0,
                      "max": sizes[-1] if sizes else 0},
        },
        "source_meta": src["meta"],
    }
    OUT.write_text(json.dumps({"meta": meta, "tasks": tasks_out}, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}")
    print(json.dumps(meta["stats"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

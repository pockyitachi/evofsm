"""Clean app-specific leakage out of LAYER 2 of each static FSM.

For every FSM under ``artifacts/static_fsms/`` whose ``layer2`` fails
``lint_layer2``, send the offending block + violation list to Claude and
replace the block with a generalized rewrite. Up to ``--max-rounds``
Claude calls per app (default 2). The ``layer1`` part of the FSM is
never touched.

Usage:
    python scripts/clean_layer2.py
    python scripts/clean_layer2.py --apps markor,joplin
    python scripts/clean_layer2.py --max-rounds 3

After each successful rewrite the paired ``{app}.txt`` is regenerated
via ``FSM.to_prompt_text()``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic

from evofsm_rl.fsm.linter import lint_layer2
from evofsm_rl.fsm.schema import FSM, Layer2


ROOT = Path(__file__).resolve().parent.parent
FSM_DIR = ROOT / "artifacts" / "static_fsms"
MODEL_ID = "claude-opus-4-7"


SYSTEM_PROMPT = (
    "You rewrite app-specific references out of abstract workflow "
    "descriptions. You output only valid JSON that matches the provided "
    "schema. No prose, no markdown fences, no extra keys."
)


USER_PROMPT_TEMPLATE = """The following LAYER2 block was extracted from an app-specific FSM but contains app-specific references that violate the transferability constraint.

Here are the lint violations:
{violations}

Here is the current LAYER2:
{layer2_json}

Rewrite LAYER2 so that:
- Every mention of app names, resource IDs, package names, or app-specific widget names is replaced with generic descriptions
- The abstract steps, failure modes, and verification checklists remain equally informative but fully app-agnostic
- Do NOT remove any categories or steps — only generalize the wording
- Output valid JSON matching the original schema, nothing else
"""


def extract_json(text: str) -> dict:
    """Best-effort: strip markdown fences and parse JSON."""
    t = text.strip()
    if t.startswith("```"):
        # Drop first fence line and last fence line.
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def rewrite_layer2(client: anthropic.Anthropic, layer2: dict, violations: list[str]) -> dict:
    """Call Claude to rewrite one layer2 block. Returns the new layer2 dict."""
    msg = client.messages.create(
        model=MODEL_ID,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    violations="\n".join(violations),
                    layer2_json=json.dumps(layer2, indent=2, ensure_ascii=False),
                ),
            },
        ],
    )
    text_parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    if not text_parts:
        raise RuntimeError(f"No text content from Claude for app; stop_reason={msg.stop_reason}")
    return extract_json("\n".join(text_parts))


def clean_one_app(client: anthropic.Anthropic, app: str, max_rounds: int) -> dict:
    """Clean one FSM; return a summary dict."""
    json_path = FSM_DIR / f"{app}.json"
    txt_path = FSM_DIR / f"{app}.txt"

    fsm = FSM.from_json(json.loads(json_path.read_text()))
    ok, errs = lint_layer2(fsm)
    if ok:
        return {"app": app, "status": "PASS", "rounds": 0, "initial_violations": 0,
                "final_violations": 0}

    initial = len(errs)
    round_num = 0
    last_errs = errs
    while not ok and round_num < max_rounds:
        round_num += 1
        t0 = time.time()
        try:
            new_layer2 = rewrite_layer2(client, fsm.layer2.to_json(), last_errs)
            fsm.layer2 = Layer2.from_json(new_layer2)
        except Exception as e:
            return {"app": app, "status": "FAIL", "rounds": round_num,
                    "initial_violations": initial, "final_violations": len(last_errs),
                    "error": f"{type(e).__name__}: {e}"}
        ok, last_errs = lint_layer2(fsm)
        elapsed = time.time() - t0
        print(f"  [{app}] round {round_num}: {len(last_errs)} violations "
              f"({'PASS' if ok else 'still FAIL'}) in {elapsed:.1f}s", flush=True)

    # Save — even if still FAIL we save the partially-improved version so
    # downstream work isn't blocked; the summary tells the caller.
    json_path.write_text(json.dumps(fsm.to_json(), indent=2, ensure_ascii=False) + "\n")
    txt_path.write_text(fsm.to_prompt_text() + "\n")

    return {"app": app, "status": "PASS" if ok else "FAIL", "rounds": round_num,
            "initial_violations": initial, "final_violations": len(last_errs),
            "remaining": last_errs if not ok else []}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apps", default="", help="Comma-separated app list (default: all FAILs)")
    parser.add_argument("--max-rounds", type=int, default=2)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 1
    client = anthropic.Anthropic()

    if args.apps:
        apps = [a.strip() for a in args.apps.split(",") if a.strip()]
    else:
        apps = []
        for p in sorted(FSM_DIR.glob("*.json")):
            fsm = FSM.from_json(json.loads(p.read_text()))
            ok, _ = lint_layer2(fsm)
            if not ok:
                apps.append(p.stem)

    print(f"Cleaning {len(apps)} app(s): {apps}", flush=True)
    results = []
    for app in apps:
        print(f"\n=== {app} ===", flush=True)
        res = clean_one_app(client, app, args.max_rounds)
        results.append(res)

    print("\n\n=== SUMMARY ===", flush=True)
    width = max((len(r["app"]) for r in results), default=10)
    for r in results:
        line = (f"  {r['app']:<{width}}  {r['status']:<4}  "
                f"rounds={r['rounds']}  "
                f"{r['initial_violations']} -> {r['final_violations']} violations")
        if "error" in r:
            line += f"  [error: {r['error']}]"
        print(line, flush=True)

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = len(results) - passed
    print(f"\nPASS: {passed} / {len(results)}    FAIL: {failed}", flush=True)

    summary_path = ROOT / "artifacts" / "_cleanup_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"Detailed summary: {summary_path}", flush=True)

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

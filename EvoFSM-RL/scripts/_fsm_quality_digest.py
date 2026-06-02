"""Dump a per-app digest of FSM + trajectory stats for quality audit.

For each of the 12 source-pool apps, writes a compact summary to
stdout showing:
  - FSM: states (id+desc), transitions (edges), strategies (names),
    dead_ends (count + sample), L2 categories
  - Trajectories: list of templates, per-template success rate over K=5
    seeds, mean step count for success vs failure
  - Isolated states (no in- or out-transition)

Consumed by the quality memo author (human / LLM) as grounding.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from evofsm_rl.fsm.schema import FSM

FSM_DIR = Path("EvoFSM-RL/artifacts/static_fsms")
TRAJ_DIR = Path("EvoFSM-RL/traces/source_pool_trajectories")

APPS = [
    "audio_recorder", "bluecoins", "calculator", "clock", "contacts",
    "files", "joplin", "markor", "pi_music", "simple_sms_messenger",
    "snapseed", "tasks_org",
]


def _template_to_app(template: str) -> str:
    # We can't easily reverse — instead rely on meta.json "app".
    return template


def _load_traj_meta():
    """Group episode metas by app. Returns {app: [meta,...]}."""
    by_app: dict[str, list[dict]] = defaultdict(list)
    for d in sorted(TRAJ_DIR.iterdir()):
        if not d.is_dir():
            continue
        m = d / "meta.json"
        if not m.exists():
            continue
        meta = json.loads(m.read_text())
        meta["_dir"] = d.name
        by_app[meta["app"]].append(meta)
    return by_app


def _dump_traj_sample(app: str, meta_list: list[dict], k_steps: int = 6):
    """For one app, print per-template SR, mean steps, and action traces from 1 success + 1 failure."""
    # per-template aggregation
    per_tpl: dict[str, list[dict]] = defaultdict(list)
    for m in meta_list:
        per_tpl[m["template"]].append(m)

    print(f"  # templates: {len(per_tpl)}, # episodes: {len(meta_list)}")
    for tpl, metas in sorted(per_tpl.items()):
        n = len(metas)
        succ = sum(1 for m in metas if m.get("success", 0) == 1.0)
        avg_steps = sum(m.get("n_steps", 0) for m in metas) / max(1, n)
        succ_steps = [m["n_steps"] for m in metas if m.get("success") == 1.0]
        fail_steps = [m["n_steps"] for m in metas if m.get("success") != 1.0]
        succ_mean = sum(succ_steps) / len(succ_steps) if succ_steps else 0
        fail_mean = sum(fail_steps) / len(fail_steps) if fail_steps else 0
        print(f"  {tpl}: {succ}/{n} SR, succ_mean_steps={succ_mean:.1f}, fail_mean_steps={fail_mean:.1f}")

    # action traces: one success + one failure per template (cap 3 templates)
    print("\n  --- sample action traces (goal + compact actions) ---")
    tpl_budget = 3
    for tpl, metas in sorted(per_tpl.items()):
        if tpl_budget <= 0:
            break
        succ_ex = next((m for m in metas if m.get("success") == 1.0), None)
        fail_ex = next((m for m in metas if m.get("success") != 1.0), None)
        for label, m in [("SUCCESS", succ_ex), ("FAILURE", fail_ex)]:
            if m is None:
                continue
            ep = (TRAJ_DIR / m["_dir"] / "episode.jsonl").read_text().splitlines()
            # extract goal (first step, if present)
            try:
                first = json.loads(ep[0])
                goal = first.get("goal", "")
            except Exception:
                goal = ""
            actions = []
            for line in ep[:k_steps]:
                try:
                    s = json.loads(line)
                    act = s.get("action", s.get("action_json", ""))
                    if isinstance(act, dict):
                        act_s = act.get("action_type", "") + " " + str(act.get("target", act.get("text", ""))[:40] if isinstance(act.get("target", act.get("text", "")), str) else act.get("index", ""))
                    else:
                        act_s = str(act)[:60]
                    actions.append(act_s.strip()[:80])
                except Exception:
                    pass
            print(f"    [{label}] {tpl} ({m['_dir']}, n_steps={m['n_steps']})")
            if goal:
                print(f"       goal: {goal[:120]}")
            for i, a in enumerate(actions):
                print(f"       step{i+1}: {a}")
        tpl_budget -= 1


def main():
    by_app = _load_traj_meta()

    for app in APPS:
        fsm_path = FSM_DIR / f"{app}.json"
        if not fsm_path.exists():
            print(f"\n### {app} — FSM FILE MISSING")
            continue
        fsm = FSM.from_json(json.loads(fsm_path.read_text()))

        print(f"\n{'=' * 80}")
        print(f"### {app} ({fsm.layer1.category})")
        print(f"{'=' * 80}")
        print(f"FSM stats: states={len(fsm.layer1.states)}, "
              f"transitions={len(fsm.layer1.transitions)}, "
              f"strategies={len(fsm.layer1.strategies)}, "
              f"dead_ends={len(fsm.layer1.dead_ends)}, "
              f"l2_categories={len(fsm.layer2.categories)}")

        print("\n-- STATES --")
        for s in fsm.layer1.states:
            cues = "; ".join(s.visual_cues[:2])
            hints = ",".join(s.resource_hints[:2])
            print(f"  {s.id}: {s.description[:80]}")
            if cues:
                print(f"    cues: {cues[:120]}")
            if hints:
                print(f"    rids: {hints[:120]}")

        print("\n-- TRANSITIONS (edges) --")
        for t in fsm.layer1.transitions:
            print(f"  {t.from_state} --[{t.action[:50]}]--> {t.to_state}")
        # isolated states
        in_deg = defaultdict(int); out_deg = defaultdict(int)
        for t in fsm.layer1.transitions:
            out_deg[t.from_state] += 1
            in_deg[t.to_state] += 1
        isolated = [s.id for s in fsm.layer1.states if in_deg[s.id] == 0 and out_deg[s.id] == 0]
        orphan_in = [s.id for s in fsm.layer1.states if in_deg[s.id] == 0 and s.id != fsm.layer1.states[0].id]
        orphan_out = [s.id for s in fsm.layer1.states if out_deg[s.id] == 0]
        print(f"  isolated (no in/out): {isolated}")
        print(f"  no in-edge (excluding entry): {orphan_in}")
        print(f"  no out-edge (terminal-ish): {orphan_out}")

        print("\n-- STRATEGIES --")
        for st in fsm.layer1.strategies:
            print(f"  {st.name}")
            print(f"    pre: {st.preconditions[:120]}")
            for i, step in enumerate(st.steps):
                print(f"    step{i+1}: {step[:120]}")
            print(f"    success_signal: {st.success_signal[:150]}")
            if st.fallback:
                print(f"    fallback: {st.fallback[:120]}")

        print("\n-- DEAD ENDS --")
        for de in fsm.layer1.dead_ends[:20]:
            print(f"  {de}")

        print("\n-- LAYER 2 (abstract categories) --")
        for cat in fsm.layer2.categories:
            print(f"  [{cat.name}]  pre: {cat.precondition[:80]}")
            for i, step in enumerate(cat.abstract_steps):
                print(f"    abs_step{i+1}: {step[:140]}")
            for i, fm in enumerate(cat.failure_modes):
                print(f"    fail_mode{i+1}: {fm[:140]}")
            for i, vc in enumerate(cat.verification_checklist):
                print(f"    verify{i+1}: {vc[:140]}")

        print("\n-- TRAJECTORY COVERAGE --")
        app_metas = by_app.get(app, [])
        if not app_metas:
            print("  (no trajectories found)")
        else:
            _dump_traj_sample(app, app_metas, k_steps=8)


if __name__ == "__main__":
    main()

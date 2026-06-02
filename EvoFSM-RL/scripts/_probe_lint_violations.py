import json
from pathlib import Path
from evofsm_rl.fsm.schema import FSM
from evofsm_rl.fsm.linter import lint_layer2

FSM_DIR = Path("EvoFSM-RL/artifacts/static_fsms")
ALL_APPS = [
    "audio_recorder", "bluecoins", "calculator", "clock", "contacts",
    "files", "joplin", "markor", "pi_music", "simple_sms_messenger",
    "snapseed", "tasks_org",
]

summary = []
for app in ALL_APPS:
    fsm = FSM.from_json(json.loads((FSM_DIR / f"{app}.json").read_text()))
    passed, errors = lint_layer2(fsm)
    status = "PASS" if passed else "FAIL"
    summary.append((app, status, len(errors)))
    print(f"\n=== {app} ({status}, {len(errors)} violations) ===")
    for e in errors:
        print(f"  {e}")

print("\n=== SUMMARY ===")
for app, status, n in summary:
    print(f"  {app:22s} {status}  {n} violations")

n_pass = sum(1 for _, s, _ in summary if s == "PASS")
print(f"\nPASS: {n_pass} / {len(summary)}    FAIL: {len(summary) - n_pass}")

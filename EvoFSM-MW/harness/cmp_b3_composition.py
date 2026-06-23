#!/usr/bin/env python3
"""Composition diff: which tasks MAI-B3 (evolved inj, 1 run) passes vs
MAI-B1 (zero inj) / MAI-B2' (static inj), each 5 runs. Net total ~tied;
this shows whether the evolved injection moved WHICH tasks pass."""
import glob, os

ROOT = "/shared/linqiang/evofsm_project/MobileWorld/traj_logs"
B3_DIR = "b3mai_eval_qwen3vl8b"
B1_DIRS = [f"maib1_qwen3vl8b_r{i}" for i in range(1, 6)]
B2P_DIRS = [f"maib2v_qwen3vl8b_r{i}" for i in range(1, 6)]

def passed_tasks(run_dir):
    """task -> True if result.txt has score 1.0"""
    out = {}
    for rt in glob.glob(os.path.join(ROOT, run_dir, "*", "result.txt")):
        task = os.path.basename(os.path.dirname(rt))
        try:
            txt = open(rt).read()
        except Exception:
            continue
        out[task] = ("score: 1.0" in txt)
    return out

b3 = passed_tasks(B3_DIR)
b1_runs = [passed_tasks(d) for d in B1_DIRS]
b2_runs = [passed_tasks(d) for d in B2P_DIRS]

# union of all tasks seen anywhere
all_tasks = set(b3)
for r in b1_runs + b2_runs:
    all_tasks |= set(r)

def count_pass(runs, task):
    return sum(1 for r in runs if r.get(task, False))
def evaluable(runs, task):
    return sum(1 for r in runs if task in r)

rows = []
for t in sorted(all_tasks):
    b3p = 1 if b3.get(t, False) else 0
    b3seen = t in b3
    b1c, b1n = count_pass(b1_runs, t), evaluable(b1_runs, t)
    b2c, b2n = count_pass(b2_runs, t), evaluable(b2_runs, t)
    rows.append((t, b3seen, b3p, b1c, b1n, b2c, b2n))

# binary "baseline usually passes" = majority of evaluable runs
def maj(c, n):
    return n > 0 and c >= (n + 1) / 2

print(f"tasks total considered: {len(all_tasks)}")
print(f"B3 pass: {sum(r[2] for r in rows if r[1])}/{sum(1 for r in rows if r[1])}")
print(f"B1 mean pass/run: {sum(count_pass(b1_runs,t) for t in all_tasks)/5:.1f}")
print(f"B2' mean pass/run: {sum(count_pass(b2_runs,t) for t in all_tasks)/5:.1f}")
print()

wins, losses, stable_pass, stable_fail, mixed = [], [], [], [], []
for t, b3seen, b3p, b1c, b1n, b2c, b2n in rows:
    if not b3seen:
        continue
    b1m, b2m = maj(b1c, b1n), maj(b2c, b2n)
    base_usually = b1m or b2m          # baseline (either) usually passes
    base_rare = (b1c <= 1) and (b2c <= 1)  # both baselines almost never pass
    if b3p and base_rare:
        wins.append((t, b1c, b1n, b2c, b2n))
    elif (not b3p) and base_usually:
        losses.append((t, b1c, b1n, b2c, b2n))
    elif b3p and base_usually:
        stable_pass.append(t)
    elif (not b3p) and base_rare:
        stable_fail.append(t)
    else:
        mixed.append((t, b3p, b1c, b1n, b2c, b2n))

print(f"=== B3 WINS (B3过, B1&B2'都几乎不过 ≤1/5): {len(wins)} ===")
for t, b1c, b1n, b2c, b2n in wins:
    print(f"  {t}   B1={b1c}/{b1n}  B2'={b2c}/{b2n}")
print(f"\n=== B3 LOSSES (B3没过, 但B1或B2'多数过): {len(losses)} ===")
for t, b1c, b1n, b2c, b2n in losses:
    print(f"  {t}   B1={b1c}/{b1n}  B2'={b2c}/{b2n}")
print(f"\n=== 稳定过(B3过 & baseline多数过): {len(stable_pass)} ===")
print("  " + ", ".join(stable_pass))
print(f"\n=== 边界/混杂(B3与baseline不齐, 非清晰输赢): {len(mixed)} ===")
for t, b3p, b1c, b1n, b2c, b2n in mixed:
    print(f"  {t}  B3={b3p}  B1={b1c}/{b1n}  B2'={b2c}/{b2n}")
print(f"\n净: wins {len(wins)} - losses {len(losses)} = {len(wins)-len(losses)}")

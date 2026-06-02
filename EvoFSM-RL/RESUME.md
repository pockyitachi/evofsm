# RESUME — B4 F1 smoke (handoff to tmux'd claude)

Written 2026-05-12 by the pre-tmux claude session. If you're a new claude reading this:
the b4 smoke is **already running detached** — don't relaunch, just attach to its log.

## What's currently running (will survive ssh disconnect)

| Process | PID at write time | How to find now | Survives? |
|---|---|---|---|
| `python run_b4_evolution.py simple_calendar_pro --n-iterations 6` | 61569 | `pgrep -f run_b4_evolution` | YES (setsid by claude bg) |
| emulator `AWAvd2` snapshot `apps_ready_dec2025` on ports **5710/8710** | 47528 (bash wrapper, PPID=1) | `adb -s emulator-5710 shell echo alive` | YES (nohup, reparented to init) |

## Log files to tail

- **Smoke run log (primary)**: `/shared/linqiang/evofsm_project/EvoFSM-RL/traces/b4_smoke_f1/simple_calendar_pro_log.txt`
- **GRPO metrics JSONL (load-bearing for F1 verification)**: `/shared/linqiang/evofsm_project/EvoFSM-RL/traces/b4_smoke_f1/simple_calendar_pro/grpo_metrics.jsonl` (created once first GRPO fires)
- **Emulator boot log**: `/tmp/awavd2_emu_5710.log`

`tail -F` the smoke log; grep for `Iteration |grad_norm=|adv_std|Traceback|OOM` to filter signal.

## Smoke parameters (already locked, don't change mid-run)

- App: `simple_calendar_pro`
- Iterations: 6 (cheap; goal is F1 validation, not training)
- LoRA: rank=16, lr=1e-4, update-every=5, checkpoint-every=5
- GPU: `CUDA_VISIBLE_DEVICES=2` (H200, 143 GB, ours)
- Output dir: `EvoFSM-RL/traces/b4_smoke_f1/simple_calendar_pro/`

## Why this smoke exists (F1 verification criteria)

CLAUDE.md line ~361 — F1 fix: `grpo_step` now divides per-trajectory loss scale by `T_j`
(trajectory length). Before F1, a 30-step trajectory contributed 30× the gradient of a
1-step one; resulting raw grad_norm 50–270 got squashed to 1.0 by `max_grad_norm=1.0`,
killing the signal. F1 verified at math level (synthetic 2-traj test), **not yet on GPU**.

**Pass criteria (read from `grpo_metrics.jsonl` after at least 1 fire):**

| Metric | v2 baseline (broken) | F1 target |
|---|---|---|
| `grad_norm` | 50–270 (often hits clip) | **O(1), say 0.3–5** |
| `advantage_std` | n/a (didn't exist) | **> 0** on active fires |
| `loss` | -77 to +9, no trend | should have direction |
| `n_active` | 0 in 2 of 6 apps | should be > 0 on simple_calendar_pro |

If grad_norm still 50+ after F1 → F1 didn't take, re-read `evofsm_rl/rl/grpo.py:343-349`.
If grad_norm now O(1) AND adv_std > 0 → F1 confirmed, can plan v3 sweep with same code.

## Expected timeline from write time

- 2026-05-12 21:25:09 — launch
- 2026-05-12 ~21:30 — Qwen3-VL-8B download complete (17 GB to `/shared/huggingface/hub/...`)
- 2026-05-12 ~21:31 — model loaded on GPU 2, iter 0 starts
- 2026-05-12 ~21:31–22:00 — iters 0-2, no GRPO fire yet (mutation buffer filling)
- 2026-05-12 ~22:00–22:15 — first GRPO fire (likely iter 3); first row in `grpo_metrics.jsonl`
- 2026-05-12 ~22:30–22:45 — iter 5 done, second fire, run exits

Total wall: 60-80 min. If still running past 23:30, something's wrong.

## Environment (must be in shell before touching anything)

```bash
cd /shared/linqiang/evofsm_project
source ~/.anthropic_env       # ANTHROPIC_API_KEY (chmod 600, do not echo)
source .venv/bin/activate     # torch 2.11.0+cu128, 8× H200 detected
export ANDROID_HOME=$(pwd)/android-sdk
export TMPDIR=$(pwd)/tmp
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=android_world_plus:EvoFSM-RL
# HF_HOME=/shared/huggingface is already in environment globally
```

## Quick health checks for a new claude session

```bash
# Smoke still alive?
pgrep -af run_b4_evolution || echo "DIED — read tail of log to find cause"

# Emulator still alive?
/shared/linqiang/evofsm_project/android-sdk/platform-tools/adb -s emulator-5710 shell getprop sys.boot_completed
# Expect: 1

# How far through?
tail -20 /shared/linqiang/evofsm_project/EvoFSM-RL/traces/b4_smoke_f1/simple_calendar_pro_log.txt | grep -E "Iteration|grad_norm|adv_std|Done"

# Look at GRPO metrics (only exists after first fire)
cat /shared/linqiang/evofsm_project/EvoFSM-RL/traces/b4_smoke_f1/simple_calendar_pro/grpo_metrics.jsonl 2>/dev/null \
  | python -c "import sys, json; [print(json.loads(l).get('iter'), 'grad_norm=', json.loads(l).get('grad_norm'), 'adv_std=', json.loads(l).get('advantage_std'), 'loss=', json.loads(l).get('loss'), 'n_active=', json.loads(l).get('n_active')) for l in sys.stdin]"
```

## What to do once smoke finishes

1. Read `grpo_metrics.jsonl` and apply the pass criteria table above.
2. If F1 confirmed → write up findings into `EvoFSM-RL/CLAUDE.md` under the existing
   2026-05-12 F1/F2 note (append a "GPU-verified" line), and decide on v3 sweep design.
3. If F1 failed → diagnose from log. Likely suspects in order: (a) traj_scale wiring,
   (b) backward fired twice per traj, (c) grad accumulator zeroing in wrong spot.
4. Either way, do NOT re-run a 6-iter smoke if results are clear — go straight to v3 design.

## Cleanup (do NOT do until smoke is fully analyzed and any v3 plan is committed)

- Emulator: `kill 47528` (or pgrep for `qemu-system-x86_64`) — only if no v3 sweep planned soon
- Output dir: `traces/b4_smoke_f1/` is keepable for forensics; ~100-200 MB

## Files containing F1/F2 changes (so you can verify the edits are still in place)

- `evofsm_rl/rl/grpo.py` — F1 marker at line ~349 (`traj_scale = scale / n_steps_traj`),
  F2 marker at lines 294-295, 388-389 (`advantage_std`, `advantage_abs_max`)
- `evofsm_rl/fsm/evolution.py` — line ~623 (log line uses `adv_std`/`adv_max`)
- `tests/test_evolution.py` — 8 mock dicts updated with F2 fields; 49/49 unit tests pass

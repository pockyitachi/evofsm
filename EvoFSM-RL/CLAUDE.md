# CLAUDE.md — EvoFSM-RL

Project context for Claude. **Every folder also has its own `CLAUDE.md`
(working context) and `README.md` (what it is)** — read those when working in a
folder. This file is the global picture, the runtime/operational essentials, and
current status.

## Rule 0 — 工作准则（最高优先级）

以第一性原理！从原始需求和问题本质出发，不从惯例或模板出发。

- 不要假设我清楚自己想要什么。动机或目标不清晰时，停下来讨论。
- 目标清晰但路径不是最短的，直接告诉我并建议更好的办法。
- 遇到问题追根因，不打补丁。每个决策都要能回答"为什么"。
- 输出说重点，砍掉一切不改变决策的信息。

---

## What this project is

EvoFSM-RL is a research method paper. Goal: a mobile GUI agent that, deployed to
an unseen app, **adapts at test time** by jointly evolving an app-specific FSM
(symbolic) and fine-tuning policy LoRA weights (sub-symbolic), reusing a
per-category abstract-action library `L_C` learned during pretraining. Built on
`android_world_plus` (the parent folder), which extends Google's `android_world`
with 6 apps from BMOCA + AndroidLab.

**Method in one breath** (full spec: `docs/design/algorithm.md`):
- Two-layer FSM — L1 app-specific (non-transferable), L2 category-generic
  (transferable, aggregated into `L_C`), separated by an enforced linter.
- One adaptation loop, two phases — Phase 1 pretrains a shared LoRA on the
  source pool; Phase 3 co-adapts the LoRA + the target app's FSM on `T_adapt`,
  then freezes and evaluates on `T_eval`.
- Four-rung ablation: B1 (zero-shot) → B2 (static `L_C`) → B3 (FSM evolution) →
  B4 (joint LoRA + FSM).

**Current direction — cross-benchmark TTA**: train on AndroidWorld+, test on
**MobileWorld** GUI-only (pure-vision). The MobileWorld side lives in
`../EvoFSM-MW/`.

---

## Repository navigation

| Need | Go to |
|---|---|
| Work inside a folder | that folder's `CLAUDE.md` (gotchas) + `README.md` (what it is) |
| App / task / split inventory + rationale | `docs/design/dataset.md` |
| Algorithm / method definitions | `docs/design/algorithm.md` |
| B1–B4 results | `docs/results/` — see warning below on B4=52.9 |
| Tickets / next steps | `docs/plan/linear_tickets.md` |
| The paper | `paper_draft/` (one `.tex` per section, no `main.tex`) |
| Abandoned routes (PPO+PRM, RFT) | `archive/` (don't revive without a decision) |
| Cross-benchmark MobileWorld work | `../EvoFSM-MW/` |

Dev env: `cd /shared/linqiang/evofsm_project && source .venv/bin/activate`, run
with `PYTHONPATH=android_world_plus:EvoFSM-RL`.

---

## Global gotchas (do not re-question)

- **`docs/results/experiments.md` "B4 = 52.9%" is a cherry-pick** — a per-tier
  oracle (Tier-B from the v3-C checkpoint + Tier-C from v3-B, two different
  models), NOT reproducible by one model. Clean single-model B4 = 48.1%.
- **`chrome/Browser{Maze,Multiply,Draw}` deadlock** — AndroidWorld
  instrumentation gap: the preamble says open the file in "File Manager", but
  `open_app` can't resolve that string (the app is registered as "Files"), so
  the agent loops until step exhaustion and the rollout is dropped. Report chrome
  as a uniform 0% across baselines and footnote the cause; do NOT patch.
- **SQLite FTS4** — stdlib `sqlite3` on the server shipped without FTS4, so
  joplin / broccoli / vlc tasks errored on `tear_down`. Fixed via
  `pip install pysqlite3-binary` + a `try: import pysqlite3 as sqlite3` shim in
  `android_world_plus/.../sqlite_utils.py`.
- **`splits.yaml` is `v1.1-K1`** — `WikipediaDecreaseTextSize50` (commented out
  upstream) was removed; Wikipedia 6→5, grand total 192→191. Change splits only
  by bumping `meta.version`.
- App / task naming facts (phone.py copy-paste bug, OpenTracks = Sports Tracker,
  Broccoli = recipe, WikipediaOpen IS registered, etc.) are pinned in
  `docs/design/dataset.md` — trust that, don't re-count.

---

## Canonical AVD & server bootstrap

Standardize on a single pre-baked AVD with all 14 apps installed: **`AWAvd2`
snapshot `apps_ready_dec2025`**. Don't run `emulator_setup=True` /
`bootstrap_avd.sh` — the snapshot already has the apps.

**Boot (read-only, never modifies the snapshot):**
```bash
ANDROID_AVD_HOME=/shared/linqiang/evofsm_project/android-sdk/avd \
ANDROID_SDK_ROOT=/shared/linqiang/evofsm_project/android-sdk \
nohup /shared/linqiang/evofsm_project/android-sdk/emulator/emulator -avd AWAvd2 \
  -port 5710 -grpc 8710 \
  -snapshot apps_ready_dec2025 -no-snapshot-save -read-only \
  -no-window -no-audio -skip-adb-auth -no-boot-anim \
  > /tmp/awavd2_emu.log 2>&1 &
# verify (prints 1 ~30s later):
adb -s emulator-5710 wait-for-device shell getprop sys.boot_completed
```
Multiple instances coexist on different `-port`/`-grpc` pairs. Bootstrap zip
backup: `/shared/linqiang/AWAvd2_full.zip`.

**Fresh-server recipe:** create a Python 3.12 venv; install from
`requirements.lock.txt` (reproduce) or `requirements.txt` (dev); unzip
`AWAvd2_full.zip` into `android-sdk/avd/` and patch `AWAvd2.ini` `path=`; boot as
above; smoke-test with `scripts/baseline_10task.py`.

**H200 (`gpu-h200-211`) deltas:** (1) tarball + AVD already in place — skip
unzip. (2) No system 3.12 — use `/shared/miniconda3/bin/python3.12`. (3) Use the
project's own SDK at `android-sdk/` (ken's paths are inaccessible). (4) CUDA 13
driver on sm_90; the pinned `torch cu128` wheel is forward-compatible — verify
`torch.cuda.is_available()` after install. (5) **`linqiang` needs the `kvm`
group** (`sudo usermod -aG kvm linqiang`, sysadmin-only) — the x86_64 image
requires hardware acceleration or the emulator crashes immediately. After
cloning files in as root: `sudo chown -R linqiang:linqiang` and restore `+x` on
`.venv/bin/*` and the SDK binaries.

---

## Latest progress (current)

**Direction**: cross-benchmark TTA — train π^pre on AndroidWorld+, test on
MobileWorld GUI-only (pure-vision, native `mobile_use` format). MW work +
results in `../EvoFSM-MW/docs/`.

**State**:
- Phase-1 shared-LoRA pretraining — running (LoRA, language-only; the vLLM
  vision-LoRA prefix bug is patched). Judge learning by paired-template gain +
  zero-gradient %, **never** mean reward.
- Static symbolic layer rebuilt at 25-app scope (`artifacts/static_fsms_v2` +
  `L_C_v2`); injection is 3-tier (app → category → bootstrap).
- MW B-series done (`EvoFSM-MW/docs/qwen3_8b_res.md`): static prior is roughly
  baseline-neutral cross-benchmark; **Layer-1 injection is strictly harmful**;
  best static config = app-Layer-2 + `L_C` (B2').
- B4 = joint (FSM evo + weight evo) on MW — bridge built, gated on Phase-1.

**Durable lessons**:
- Phase-3 LoRA is null/negative under clean ablation — the symbolic FSM/`L_C`
  evolution is the working half. Clean B4 = 48.1% (NOT the cherry-picked 52.9).
- Layer-1 is empirically non-transferable across benchmarks.
- MobileWorld containers do NOT reset app state — never reuse across runs.
- Large-scale RL runs on external verl/SkyRL (replaced the in-repo `grpo.py`).

**Next**: finish Phase-1 → fill MW B3/B4 → paper Level-2 numbers. Session-level
detail lives in auto-memory (`MEMORY.md`).

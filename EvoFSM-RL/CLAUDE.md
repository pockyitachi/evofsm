# CLAUDE.md — EvoFSM-RL project context

## Rule 0 — 工作准则（最高优先级）

以第一性原理！从原始需求和问题本质出发，不从惯例或模板出发。

- 不要假设我清楚自己想要什么。动机或目标不清晰时，停下来讨论。
- 目标清晰但路径不是最短的，直接告诉我并建议更好的办法。
- 遇到问题追根因，不打补丁。每个决策都要能回答"为什么"。
- 输出说重点，砍掉一切不改变决策的信息。

---

This file is the project-scope context for Claude working on EvoFSM-RL. Read it before answering questions or writing code in this directory.

---

## What this project is

**EvoFSM-RL** is a research method paper. Goal: a GUI agent that, when deployed to a new Android app, **adapts at test time** by jointly evolving an app-specific FSM (symbolic) and fine-tuning policy LoRA weights (sub-symbolic), reusing a per-category abstract-action library `L_C` learned during pretraining.

Built on top of `android_world_plus` (this folder's parent), which extends Google's `android_world` benchmark with 6 additional apps from BMOCA + AndroidLab.

---

## Method overview (so Claude doesn't have to re-derive)

Two-phase loop:
- **Phase 1 — multi-source pretraining.** Train per-app L1 FSMs and a per-Play-Store-category `L_C` abstract library across the **source pool** (12 apps).
- **Phase 2 — hand-off.** Given a target app, look up `play_category_of(target)` to decide initial `L_C` row + initial FSM. `L_C` is **read-only** at test time.
- **Phase 3 — TTA.** Same evolve+RL loop as Phase 1, but only mutates `L1_target` and the **target-app row** of `L_C`, plus LoRA fine-tunes the policy. Runs on the target app's `T_adapt` templates.

Two-layer FSM:
- **L1 (app-specific)** — non-transferable, one per app.
- **L2 (generic, organized as L_C)** — transferable, indexed by Play Store category.

Tiers (= "what kind of generalization is this measuring") — both inside the **target-app pool** (the 13 apps we evaluate on; source pool is separate):
- **Tier-B / near-transfer** — target app whose category IS in source pool.
- **Tier-C / far-transfer** — target app whose category is NOT in source pool.

Inside each target app, templates are **template-disjoint** split into `T_adapt` (TTA loop learns here) and `T_eval` (frozen eval — this is the headline number). The app is the same on both sides — only the tasks differ. See `docs/design/dataset.md`.

(Historical note: earlier docs and `configs/splits.yaml` still use the key names `tier_B_held_out` / `tier_C_held_out`. These are synonyms for `tier_B_target` / `tier_C_target`; the code-level rename is a separate pass that requires bumping `splits.yaml.meta.version`.)

---

## Source-of-truth files (don't re-derive these)

| File | What it pins down |
|---|---|
| `docs/design/dataset.md` | Every app, package names, task counts, Play Store categories, source/Tier-B/Tier-C assignment, T_adapt/T_eval split protocol + rationale + citations. Absorbs the old `taxonomy.md` + `splits_protocol.md` + `app_task_inventory.md`. |
| `docs/design/algorithm.md` | Two-layer FSM design, E-SPL paradigm adaptation, four-baseline progression (B1–B4). Companion to `plan/algorithm_design.md` (latter is the raw working document; `design/algorithm.md` is the paper-ready version). |
| `configs/splits.yaml` | K=1 baseline: which apps in source / Tier-B / Tier-C, which templates in `T_adapt` / `T_eval` per target app. **Don't change without updating the version field.** |
| `configs/multi_seed_config.yaml` | K=5 (adapt) / K=3 (eval) seed counts per template. Layered on top of splits.yaml. |
| `docs/plan/adr/001-emulator-path.md` | Why native AVD, not Docker. |
| `docs/plan/adr/002-base-model.md` | Why Qwen3-VL-8B, not 30B-A3B. |
| `docs/plan/adr/003-snapshot-schema.md` | Frozen-checkpoint on-disk format for `T_adapt → T_eval` hand-off. |

If a question is about counts, packages, or splits — the answer is in those files. Don't guess.

## Documentation

All project documentation lives under `EvoFSM-RL/docs/`:

- `docs/PROJECT_OVERVIEW.md` — Project introduction, motivation, contributions, and technical architecture.
- `docs/design/` — Dataset + algorithm design.
  - `dataset.md` — Full dataset spec, splits (source / Tier-B / Tier-C), T_adapt/T_eval protocol, citations.
  - `algorithm.md` — E-SPL background + EvoFSM-RL two-layer FSM method + four-baseline progression.
- `docs/results/` — Experiment results and analysis.
  - `b1_b2_static_baselines.md` — B1 (zero-shot) vs B2 (static L_C) K=3 comparison on T_eval + regression case study + source-FSM quality audit appendix.
  - `b3_evolution.md` — B3 evolution sweep summary + B3 vs B2 T_eval results (Tier-B complete, Tier-C placeholder until the current sweep finishes).
- `docs/plan/` — Planning and decision records.
  - `adr/` — Architecture decision records (emulator path, base model, snapshot schema).
  - `linear_tickets.md` — Epic / Story / Task breakdown and current ticket state.

When writing new documentation, place it in the appropriate subfolder. **Do not create docs outside `EvoFSM-RL/docs/`.** The two files still living at `plan/` (`algorithm_design.md`, `project_plan.md`) are the raw working documents the reorganized `docs/` was distilled from — treat them as lower-priority sources; if they contradict `docs/design/*.md`, trust `docs/`.

---

## Conventions

- **App labels** are the snake_case keys used in `splits.yaml` (e.g. `simple_calendar_pro`, `pi_music`, `system_settings`, `simple_sms_messenger`). These are NOT the AndroidWorld internal label strings — they're our own canonical names.
- **Template names** are PascalCase class names from the registry (e.g. `MarkorCreateNote`, `BluecoinsAddExpense`).
- **Total counts to remember**: 25 active primary-task apps; 116 vanilla AndroidWorld + 78 Plus = **194 tasks**; 192 of those are app-attributable (the other 2 are `OpenApp` generic and `SaveCopyOfReceipt` composite).
- **Run protocol baseline**: K=1 (one seed per template, `task_random_seed=30`) — matches AndroidWorld leaderboard.
- **Cluster** is a deprecated term — use **Play Store category** instead. Old memos may still say "cluster"; treat as a synonym for category.

---

## Verified facts (do not re-question)

- `phone.py` in vanilla AndroidWorld has `app_names=("markor",)` — a copy-paste bug. Phone has 0 tasks in metadata. Excluded from the active app list.
- `WikipediaOpen` IS registered in `registry.py:287` → Wikipedia has 6 tasks (not 5 as the onboarding doc says).
- "Sports Tracker" = OpenTracks (`de.dennisguse.opentracks`).
- Recipe app = Broccoli (`com.flauschcode.broccoli`).
- Retro Music package = `code.name.monkey.retromusic`.
- `walmart.py` is a stub with 0 tasks — not active.
- AudioRecorder tasks need the recordings directory to exist on the device, otherwise `task.initialize_task()` raises `RuntimeError: Failed to inspect recordings directory`. The canonical `AWAvd2` snapshot (see "Canonical AVD" section) already has it. If you ever boot a fresh AVD the workaround is `adb -s <emulator> shell mkdir -p /storage/emulated/0/Android/data/com.dimowner.audiorecorder/files/Music/records` — verified 2026-04-17 to flip `AudioRecorderRecordAudio` from infra-error → success=1.
- **`chrome/Browser{Maze,Multiply,Draw}` deadlock (AndroidWorld instrumentation gap, not our bug)**. The `BrowserTask` base class `preamble` (in `android_world/task_evals/single/browser.py`) says: *"Open the file task.html in Downloads in the file manager; when prompted open it with Chrome."*. The agent correctly emits `{"action_type": "open_app", "app_name": "File Manager"}` (following the preamble's wording verbatim), but AndroidWorld's `open_app` resolver has no entry mapping the natural-language string `"File Manager"` to any package — the actual file-manager app on the AVD is registered as `"Files"` (package `com.simplemobiletools.filemanager.pro`). ADB then runs `monkey -p "File Manager"` → exit 255. The agent re-emits the same action until `max_steps_multiplier × budget` is exhausted; the rollout returns `result.error != None` ("Reached max number of steps") which causes the B4 rollout wrapper to **skip `save_episode` and `get_trajectory_data`** for that trajectory. Empirical impact on B4 Phase 3 sweep of chrome (2026-05-14): 40 potential rollouts, only 2 entered the GRPO buffer (5%), 0 mutations succeeded over 20 iters, champion μ stayed at 25.0 (population size 1 throughout). Net: B4-chrome ≈ B2/B3-chrome ≈ B1-chrome ≈ 0% on `BrowserMultiply` $T_{\text{eval}}$. **Decision (2026-05-14)**: do NOT patch. Report chrome as a uniform 0% across all four baselines and footnote the instrumentation cause; or exclude chrome from the Tier-B aggregate with a footnote. Both choices preserve methodological cleanliness.
- **SQLite FTS4 was unavailable on the H200 server's stdlib `sqlite3` (fixed 2026-05-14)**. The Python 3.12 stdlib `sqlite3` (version 3.51.0) shipped without `ENABLE_FTS4=1`, so `sqlite_utils.delete_all_rows_from_table` raised `sqlite3.OperationalError: no such module: fts4` on any task whose `initialize_task` / `tear_down` clears an FTS4 virtual table. Affected source-pool / target apps: **joplin** (4 templates, all info-retrieval), **broccoli** (8 recipe templates), **vlc** (1 playlist template). Symptom: every rollout `result.error != None` → trajectory never enters the GRPO buffer → 20-iter B4 sweep finishes in seconds with 0 fires, e.g. broccoli completed 20 iters in 8 seconds with 0/20 reward > 0.5 and population stayed at the root. **Fix**: `pip install pysqlite3-binary` (version 3.51.1 with FTS4 compiled in) and patch `android_world_plus/android_world/task_evals/utils/sqlite_utils.py` to `try: import pysqlite3 as sqlite3 / except ImportError: import sqlite3`. Verified end-to-end: `CREATE VIRTUAL TABLE t USING fts4(x); INSERT; DELETE` works. The B4 sweep results on joplin / broccoli / vlc before this fix should be re-run; the Tier-B sweep results on the other 5 apps and the Phase 1 v1 pilot are unaffected because they do not touch FTS4 tables (Phase 1 v2 sampled joplin → those iters wasted compute but did not corrupt LoRA, since failed `initialize_task` produces no trajectory_data and contributes nothing to GRPO).
- **`splits.yaml` had one template not in the AndroidWorld registry (fixed 2026-05-14, v1.0-K1 → v1.1-K1)**. `tier_C_held_out > wikipedia > T_adapt` listed `WikipediaDecreaseTextSize50`, whose class exists in `task_evals/single/wikipedia.py` but is commented out in upstream `registry.py` alongside 7 other Wikipedia Disable* tasks. Harness raised `KeyError: "Unknown template 'WikipediaDecreaseTextSize50'"` on iterations that sampled it (8 such iters out of 20 in the Tier-C B4 sweep). **Fix**: removed the template from `splits.yaml`, bumped `meta.version` to `v1.1-K1`, recorded the change in `meta.revision_notes`. Wikipedia template count 6 → 5, Tier-C total 47 → 46, benchmark grand total 192 → 191. Verified by re-running the splits-vs-registry sanity script: 0 missing templates after the change.

---

## Current ticket status (2026-04-26)

| Story | Design | Code | Notes |
|---|---|---|---|
| 0.1 — App taxonomy & per-task labels | ✅ | ✅ | `evofsm_rl/taxonomy.py` + `configs/task_categories.csv` |
| 0.2 — Source pool / Tier-B / Tier-C allocation | ✅ | ✅ | `evofsm_rl/splits.py` loads `configs/splits.yaml` |
| 0.3 — Target-app T_adapt / T_eval split | ✅ | ✅ | K=1 alphabetical 60/40 baked into `splits.yaml`; loader + tests |
| **Epic 0 closed.** 12/12 tests pass. | | | |
| 1.1 — AVD bootstrap + smoke test | ✅ | ✅ | **Superseded by AWAvd2 snapshot** (see "Canonical AVD" section). Don't run `bootstrap_avd.sh`/`emulator_setup=True`; just boot the snapshot. |
| 1.2 — Base model loader | ✅ | ✅ | `evofsm_rl/model/loader.py`. Fingerprint lock at `configs/model_fingerprint.lock.json`. |
| 1.3 — Snapshot module | ✅ | — | **NOT YET WRITTEN.** Needed before full TTA loop. ADR-003 schema: per-(app,seed) dir with `snapshot.json` + `fsm_L1.json` + `L_C_target_row.json` + `lora_delta/`. (B4 v2 sweep ran without it — LoRA adapters are saved per-iter under `lora_checkpoints/iter_NNNN/`, but no formal snapshot bundle.) |
| 1.4 — Zero-shot baseline eval (single-shot v0) | ✅ | ✅ | **SUPERSEDED by 1.5.** v0 traces archived to `traces/archive_singleshot_v0/`. See "Baseline lineage". |
| 1.5 — Qwen3-VL-M3A agent (code) | ✅ | ✅ | Two-phase action+summary loop, Qwen3-VL-8B base. `evofsm_rl/agent/{prompts,rollout,action,a11y}.py` rewritten 2026-04-16. 94/94 unit tests pass. |
| 1.5 — source-pool 50-task baseline | ✅ | ✅ | `traces/m3a_50task_v01/` — 27/50 = **54%** SR. Report in `docs/results/b1_b2_static_baselines.md` §7. |
| 1.5 — T_eval headline baseline | ✅ | ✅ | `traces/m3a_teval_v01/` — 35 templates. Tier-B **58.3%**, Tier-C **35.3%**, overall 47.1%. **Paper Table 1 zero-shot row.** |
| 2.0 — Trajectory persistence | ✅ | ✅ | `Qwen3VLAgent.save_episode()` + flags in `baseline_10task.py`. |
| 2.1 — Trajectory collection | ✅ | ✅ | `traces/source_pool_trajectories/` — **480 episodes** (96 templates × K=5 seeds 30..34). SR 40.4%. |
| 2.2 — FSM builder (LLM synthesis) | ✅ | ✅ | `evofsm_rl/fsm/builder.py` writes per-app `F^0_a` + `L_C` initial library. Driven by `scripts/build_all_fsms.py` and `scripts/build_L_C.py`. |
| B1 — Zero-shot baseline (T_eval K=3) | ✅ | ✅ | `traces/m3a_teval_v01/` repurposed as B1. Report `docs/results/b1_b2_static_baselines.md`. |
| B2 — Static L_C baseline | ✅ | ✅ | `scripts/run_b2_eval.py` + `traces/b2_teval_*`. Tier-B +9.3pp; Tier-C null. |
| B3 — Evolution (FSM-only, no LoRA) | ✅ | ✅ | `scripts/run_b3_evolution.py` + `run_b3_teval.py`. `traces/b3_evolution/` + `traces/b3_teval/`. Tier-B +3.7pp overall. Report `docs/results/b3_evolution.md`. |
| B4 — Evolution + GRPO LoRA | ✅ | ⚠️ | `scripts/run_b4_evolution.py` + `run_b4_sweep.sh`. **v1 (2026-04-22) hit 4× OOM**, **v2 (2026-04-23) ran 6/6 apps × 20 iters with 0 OOM**, BUT **no learning signal observed** — see "B4 sweep lineage" below. |

Epic 1 decisions (settled 2026-04-15, see `docs/plan/adr/`):
- **ADR-001** — emulator: native Mac AVD on M5 Pro (arm64), x86_64 AVD on rented A100 box. No Docker.
- **ADR-002** — base model: **Qwen3-VL-8B-Instruct** (fits 24 GB Mac inference + single-A100 LoRA training; 30B-A3B deferred to optional appendix).
- **ADR-003** — snapshot schema: per-(app,seed) directory with `snapshot.json` + `fsm_L1.json` + `L_C_target_row.json` + `lora_delta/`. Frozen on `T_adapt` end, read-only on `T_eval`.

### Baseline lineage (read before touching the agent or quoting numbers)

Two distinct agent designs exist in this repo's history. Don't conflate them:

- **single-shot v0 (Story 1.4, deprecated 2026-04-16).** One LLM call per
  step, JSON-only output, no chain-of-thought, no per-step summary. Used
  CC-style prompts in the original `prompts.py`. Trace data lives under
  `traces/archive_singleshot_v0/` (see the README there). Dominant failure
  mode: agent emits `navigate_home` for the entire episode and never
  emits `status: complete` (3/49 emit rate on 50-task sweep). **Do NOT
  use these numbers as the paper baseline.** Keep them only for the
  ablation that motivates self-reflection.
- **Qwen3-VL-M3A (Story 1.5, current).** Mirrors `android_world.agents.m3a`
  1:1, with Qwen3-VL-8B substituted for GPT-4o. Two LLM calls per step
  (action selection → summary), `Reason: …\nAction: {…}` format, NL
  summary history. Trace data lives under `traces/m3a_*`. **This is the
  paper's baseline going forward.** All downstream methods (LoRA SFT,
  RL, FSM) compare against this.

If you re-read old conversation memory or commit messages mentioning
`SYSTEM_PROMPT_V0`, `ACTION_SCHEMAS`, `build_messages`, `build_user_turn`,
`build_system_prompt`, or `format_history` — those symbols were removed
in the Story 1.5 rewrite. They no longer exist in `evofsm_rl/agent/prompts.py`.

Hardware context (J, 2026-04-16): dev on M5 Pro Mac 24 GB; A100 server at 202.78.161.193 (shared, user linqiang, project at `/shared/linqiang/evofsm_project`).

### Mac vs A100 role split (IMPORTANT)
- **Mac (MPS, fp16, max_pixels=802816)**: development only. Smoke tests, prompt iteration, parser unit tests. Numbers from Mac are NOT for paper.
- **A100 (CUDA, bf16, max_pixels=1605632)**: canonical eval + LoRA training. All reportable numbers come from this path.
- MPS generate() is extremely slow (first-call kernel compilation takes minutes). Generation-dependent tests (coord_probe, rollout, baseline eval) should run on A100.

Dev environment setup (Mac):
```bash
cd /Users/apigo/Desktop/Projects/android_world_plus
source .venv/bin/activate          # Python 3.12 venv (pyenv local 3.12)
export ANDROID_HOME="$HOME/Library/Android/sdk"
PYTHONPATH=android_world_plus:EvoFSM-RL python3 -c "from android_world.env import env_launcher; print('ok')"
```

Dev environment setup (A100 server, current):
```bash
cd /shared/linqiang/evofsm_project
source .venv/bin/activate
PYTHONPATH=android_world_plus:EvoFSM-RL python3 -c "from evofsm_rl.model import resolve_device; print(resolve_device())"  # should print "cuda"
```

### Project layout (top-level `evofsm_project/`)

```
evofsm_project/
├── EvoFSM-RL/                       # ★ THIS PROJECT — all code, configs, traces, docs
│   ├── evofsm_rl/                   # Python package (agent, env, fsm, model, splits, taxonomy)
│   ├── tests/                       # pytest-style unit tests (also runnable as plain scripts)
│   ├── scripts/                     # CLI entry points (baseline_10task.py, run_rollout.py, ...)
│   ├── configs/                     # splits.yaml, model.yaml, task lists, multi-seed config
│   ├── plan/                        # algorithm_design.md, linear_tickets.md, project_plan.md
│   ├── docs/                        # Consolidated docs: PROJECT_OVERVIEW.md, design/, results/, plan/ (incl. adr/)
│   ├── traces/                      # all trajectory & summary outputs (gitignored, ~2.2 GB)
│   ├── apks/                        # 6 Plus-app APK files (gitignored; download instructions in apks/README.md)
│   ├── requirements.txt             # curated >= deps (install spec)
│   ├── requirements.lock.txt        # exact pins from current .venv (replay spec)
│   └── CLAUDE.md                    # ← you are here
├── android_world_plus/              # Submodule — AW benchmark + 6 Plus apps. SMALL (14 MB).
├── android-sdk/                     # AVD home (AWAvd2.avd). REGENERABLE from /shared/linqiang/AWAvd2_full.zip
│   └── avd/AWAvd2.avd/              # canonical AVD with apps_ready_dec2025 snapshot
├── .venv/                           # Python 3.12 venv (~9 GB). REGENERABLE from requirements.lock.txt
├── tmp/                             # HF + torchinductor caches. REGENERABLE.
└── (linqiang's other projects: docker_AW_plus/, android_world_generalization/, ...)
```

**External to project, but critical:**
- `/shared/linqiang/AWAvd2_full.zip` (~13 GB) — bootstrap zip for the AVD. Without this you cannot rebuild the emulator on a fresh box.

### Setup on a fresh server (clean-room recipe)

Assumes you have the project tarball + `AWAvd2_full.zip` in hand.

```bash
# 1. Untar the project
mkdir -p /shared/<you>
cd /shared/<you>
tar xzf evofsm_backup_YYYYMMDD.tar.gz       # restores evofsm_project/ + AWAvd2_full.zip

# 2. Create venv (Python 3.12 required)
cd evofsm_project
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install deps. Two paths:
#    (a) reproducibility — exact pins from when paper was written
uv pip install --python .venv/bin/python -r EvoFSM-RL/requirements.lock.txt
#    (b) dev — curated forward-compat deps (preferred for new feature work)
pip install -r android_world_plus/requirements.txt
pip install -r EvoFSM-RL/requirements.txt
# IMPORTANT: if your CUDA version differs from the lock-file's cu128, see
# the CUDA WARNING block at the top of requirements.lock.txt and override
# torch / torchvision before pip-installing the rest.

# 4. Restore the canonical AVD from the bootstrap zip
mkdir -p android-sdk/avd
unzip -o /shared/<you>/AWAvd2_full.zip -d android-sdk/avd/
# Patch the AVD ini path (zip's ini still points at Ken's old path)
sed -i 's|/shared/ken/.android/avd/AWAvd2.avd|'$(pwd)'/android-sdk/avd/AWAvd2.avd|' \
  android-sdk/avd/AWAvd2.ini

# 5. Boot the emulator (read-only, never overwrites the snapshot)
ANDROID_AVD_HOME=$(pwd)/android-sdk/avd \
ANDROID_SDK_ROOT=/shared/ken/.android \
nohup /shared/ken/.android/emulator/emulator -avd AWAvd2 \
  -port 5710 -grpc 8710 \
  -snapshot apps_ready_dec2025 -no-snapshot-save -read-only \
  -no-window -no-audio -skip-adb-auth -no-boot-anim \
  > /tmp/awavd2_emu.log 2>&1 &

# 6. Verify (should print "1" once boot finishes ~30s later)
adb -s emulator-5710 wait-for-device shell getprop sys.boot_completed

# 7. Smoke test the agent end-to-end (1 quick task)
export ANDROID_HOME=$(pwd)/android-sdk
export TMPDIR=$(pwd)/tmp
export CUDA_VISIBLE_DEVICES=0     # whichever GPU is yours
PYTHONPATH=android_world_plus:EvoFSM-RL python EvoFSM-RL/scripts/baseline_10task.py \
  --console-port 5710 --grpc-port 8710 \
  --adb-path $ANDROID_HOME/platform-tools/adb \
  --output-dir EvoFSM-RL/traces/smoke \
  --tasks SystemBrightnessMinVerify:easy \
  --max-steps-multiplier 5
# Expect success=1.00 in ~30s (baseline confirmed working).
```

**On the current A100 server**, the emulator binary is `/shared/ken/.android/emulator/emulator` and `ANDROID_SDK_ROOT=/shared/ken/.android`. If you're on a new box without these, install Android SDK first (`sdkmanager "platform-tools" "system-images;android-33;google_apis;x86_64" "emulator"`).

### H200 server bootstrap notes (2026-05-12, `gpu-h200-211`)

Project was migrated from the A100 box (202.78.161.193) to `gpu-h200-211`. The fresh-server recipe above mostly works, with three deltas to apply when re-bootstrapping:

1. **Skip recipe steps 1 + 4** — the project tarball + AWAvd2 zip are already in place: `/shared/linqiang/AWAvd2_full.zip` (13 GB), `/shared/linqiang/evofsm_project/android-sdk/avd/AWAvd2.avd/` (already unzipped, ~9 GB), and `AWAvd2.ini` already points at the linqiang path (no `sed` needed).
2. **No system Python 3.12** — `/usr/bin/python3` on H200 is 3.10 only. Use `/shared/miniconda3/bin/python3.12 -m venv .venv` instead of `python3.12 -m venv .venv` in step 2.
3. **Don't use ken's emulator path.** CLAUDE.md's main "Canonical AVD" boot command points at `/shared/ken/.android/emulator/emulator` and `ANDROID_SDK_ROOT=/shared/ken/.android` — both are inaccessible on H200 (ken's binary is mode `-rw-r--r--` and we can't chmod it). The project ships its own complete SDK at `/shared/linqiang/evofsm_project/android-sdk/` (emulator, platform-tools, build-tools, cmdline-tools, system-images). After chown, `find android-sdk/{emulator,platform-tools,build-tools,cmdline-tools} -type f -user linqiang -exec file {} + | grep -E "executable\|ELF" | awk -F: '{print $1}' | xargs -r chmod +x` restores +x on the binaries (verified working: `emulator -version` → 35.3.11.0). Boot with `ANDROID_SDK_ROOT=/shared/linqiang/evofsm_project/android-sdk` and `/shared/linqiang/evofsm_project/android-sdk/emulator/emulator -avd AWAvd2 ...`, not ken's paths.
4. **CUDA driver is 13.0** (driver) on a sm_90 H200; lock file pins `torch==2.11.0+cu128`. The bundled +cu128 runtime is forward-compatible with a 13.0 driver, so the wheel should work — but verify with `python -c "import torch; print(torch.cuda.is_available())"` immediately after install. If false, swap to `--index-url https://download.pytorch.org/whl/cu130` per the lock-file warning header.

**One-time chown reminder:** new files copied/cloned in as `root` need `sudo chown -R linqiang:linqiang /shared/linqiang/evofsm_project` before `linqiang` can edit them. The 2026-05-12 session hit this on `evofsm_rl/rl/grpo.py` and the `.venv/bin/*` binaries — separately, the .venv binaries arrived without +x (cause unclear; possibly a `cp` that didn't preserve mode), so after chown also run `find /shared/linqiang/evofsm_project/.venv/bin -type f -exec chmod +x {} +` to restore.

5. **`linqiang` needs `kvm` group membership for the emulator.** `/dev/kvm` is mode `crw-rw---- root kvm` and the AWAvd2 x86_64 image strictly requires hardware acceleration (no software fallback — that's an arm64-only capability). Without KVM access the emulator crashes immediately with "x86_64 emulation currently requires hardware acceleration!". Fix is sysadmin-only: `sudo usermod -aG kvm linqiang`, then re-login (or `newgrp kvm`) so the new group membership takes effect. As of 2026-05-12 the kvm group on this box is `kvm:x:109:jeslyn,yuanqi` — those are the only users who can boot emulators here.

### Canonical AVD — `AWAvd2` snapshot `apps_ready_dec2025` (USE THIS GOING FORWARD)

As of 2026-04-17 the project standardizes on a single pre-baked AVD with
**all 14 apps already installed** (vanilla AW + Plus apps). This replaces the
old "boot fresh AVD then run `emulator_setup=True`" flow, which is brittle:
`setup.setup_apps()` crashes at the Joplin step (it tries to pull
`/data/data/net.cozic.joplin/databases` before Joplin has been launched), and
in any case can't reach OsmAnd / Broccoli / OpenTracks / Retro Music on most
fresh emulators (4 apps the public install pipeline misses).

**Location:**
- AVD home: `/shared/linqiang/evofsm_project/android-sdk/avd/AWAvd2.avd/`
- `.ini` registration: `/shared/linqiang/evofsm_project/android-sdk/avd/AWAvd2.ini`
- Original zip backup: `/shared/linqiang/AWAvd2_full.zip` (~13 GB compressed; restore by `unzip -o /shared/linqiang/AWAvd2_full.zip -d /shared/linqiang/evofsm_project/android-sdk/avd/` then patch `AWAvd2.ini` `path=` to the linqiang path)
- Snapshots inside the AVD: `clean` (fresh boot, May 2025) and `apps_ready_dec2025` (with all apps, Dec 2025) — **always boot from `apps_ready_dec2025`**

**Boot command (read-only, never modifies the snapshot):**
```bash
ANDROID_AVD_HOME=/shared/linqiang/evofsm_project/android-sdk/avd \
ANDROID_SDK_ROOT=/shared/ken/.android \
nohup /shared/ken/.android/emulator/emulator -avd AWAvd2 \
  -port 5710 -grpc 8710 \
  -snapshot apps_ready_dec2025 -no-snapshot-save -read-only \
  -no-window -no-audio -skip-adb-auth -no-boot-anim \
  > /tmp/awavd2_emu.log 2>&1 &
```
- `-read-only` puts emulator changes in a per-process scratch overlay; the
  underlying `userdata-qemu.img.qcow2` is never written.
- `-no-snapshot-save` is belt-and-suspenders so we cannot accidentally
  overwrite `apps_ready_dec2025` on exit.
- Multiple instances can be booted concurrently on different `-port` /
  `-grpc` pairs (5710 / 8710, 5712 / 8712, …) — they all read from the
  same snapshot but cannot collide.

**Verified packages in the snapshot** (all present, ready to use):
markor, audio recorder (with `Music/records` dir already created, owner
u0_a176), simple calendar pro, simple draw pro, simple gallery pro, simple
sms messenger, AndroidWorld helper, MapsMe, Wikipedia, Bluecoins
(pro_expense), OsmAnd, Broccoli (recipe), OpenTracks (sports tracker),
Retro Music. Plus chrome / dialer / contacts / files / clock / camera /
clipper from the system image.

**What you don't need to do anymore:**
- Don't run `harness.connect(emulator_setup=True)` — the snapshot already
  has the apps. Use `emulator_setup=False` (the default).
- Don't run the `mkdir -p /storage/emulated/0/Android/data/com.dimowner.audiorecorder/files/Music/records` workaround — already in snapshot.
- Don't run `./scripts/bootstrap_avd.sh` or `./scripts/install_plus_apks.sh`.
  Those scripts are the legacy fresh-AVD path; they remain for reference but
  the canonical path is the snapshot above.

**Don't redo work that already used a different AVD.** The Story 1.5
50-task baseline (`traces/m3a_50task_v01/`) was run against a now-killed
`emulator-5700` instance with a partial app install. Those numbers stand
as the canonical M3A baseline — do not re-run them on the snapshot AVD
just for purity. Re-running would burn 2 hours for the same result.

### B4 sweep lineage + GRPO debugging (condensed 2026-06-01; full history in `CLAUDE.md.bak_20260601`)

Two B4 sweeps ran on the hand-written `grpo.py`: **v1** (`traces/b4_evolution/`, 2026-04-22, ABORTED — 4× OOM from per-step `replay/*.pt` caching) and **v2** (`traces/b4_evolution_v2/`, COMPLETE — fixed by reading text-only `episodes/*.jsonl`; 120/120 iters, 0 OOM, but **pipeline-stable / learning-null**).

Durable GRPO lessons (carry into the verl migration):
- **Normalize loss per trajectory length `T_j`** (F1, 2026-05-12) — else long trajectories dominate, grad_norm blows to 50–270 then gets squashed by clip. After fix grad_norm settles to O(1).
- **Read `advantage_std`, NOT `mean_advantage`** as the learning-signal indicator (F2) — group-mean subtraction makes `mean_adv ≡ 0` by construction.
- **Group advantages by `(fsm_variant_id, task_name)` tuple** (F5, 2026-05-13), per `algorithm_design.md` §3.2 → requires **N≥2 rollouts per (app,task) per iter**.
- Three textbook-GRPO deviations still in the code (detail in the 2026-06-01 entry below): dropout-on during grad forward, advantage not std-normalized, sequence-level (not per-token) KL. These likely forced β as high as 0.05 → **verl re-sweep starts from 0.001**.

Tier-C B4 design: shipped **(a) pure-LoRA TTA** (no L_C, no FSM mutation) for the first paper row; bootstrap-FSM variant (b) is queued (adds `--enable-bootstrap-fsm` / `--disable-mutation` → a clean 2×2 {evolution}×{LoRA} on Tier-C). Clean ablation (2026-06-01): Phase-3 LoRA is **null/negative** (v3-B init: frozen 49.0 → trained 43.3 = −5.7pp) — the TTA-via-weight-adaptation claim is currently unsupported; see the 2026-06-01 entry.

### Dense reward design rule (2026-05-20 — partial credit ablation)

We are building **opt-in dense reward** (partial credit for multi-row tasks) as an ablation experiment. The implementation must follow strict isolation rules. Future Claude sessions touching this code path MUST respect these:

1. **Never modify `is_successful()`** on any AndroidWorld task evaluator. `is_successful` is the binary {0, 1} signal that T_eval reports. Modifying it would invalidate all prior baseline numbers (B1 ≈ 38.6%, B2 ≈ 43.3%, B3 ≈ 46.7%, B4 v2 ≈ 45.2%, B4 K=4 ≈ 48.1%).

2. **Add `get_dense_reward()` as a parallel method**, not a replacement. Default in `task_eval.py` base class is `get_dense_reward = lambda env: float(self.is_successful(env))` (binary fallback). Only multi-row task subclasses (`AddMultipleRows`, `DeleteMultipleRows`, `DeleteDuplicateRows` and their concrete subclasses) override it to return float partial credit.

3. **CLI flag `--use-dense-reward` is OFF by default.** All existing training commands without this flag must produce byte-identical behavior to current code (binary reward path). Adding new training commands that need dense reward must explicitly opt in.

4. **`run_b4_teval.py` NEVER uses dense reward.** T_eval always extracts `task.is_successful(env)` directly. This guarantees paper numbers remain comparable to AndroidWorld leaderboard / B1-B3 baselines.

5. **Plumbing path**:
   - `task_evals/single/*.py` — only multi-row task subclasses override `get_dense_reward` / implement `_count_matching_rows`
   - `task_evals/common_validators/sqlite_validators.py` — `AddMultipleRows` / `DeleteMultipleRows` base class adds `get_dense_reward` that delegates to subclass `_count_matching_rows`
   - `task_evals/task_eval.py` — base class fallback `get_dense_reward = is_successful`
   - `evofsm_rl/env/harness.py` — `run_template(..., use_dense_reward=False)`; True → try `task.get_dense_reward(env)`, fallback to `is_successful` if not implemented
   - `scripts/run_b4_evolution.py` + `scripts/run_phase1_pretraining.py` — `--use-dense-reward` flag, default False
   - `scripts/run_b4_teval.py` — DO NOT add this flag

6. **Verification before claiming "dense reward works"**:
   - All existing unit tests must pass (binary path unchanged)
   - Smoke test 1 app twice: once without flag (reward dist should be `{0.0, 1.0}` only) and once with flag (reward dist should include `{0.33, 0.67, ...}` for multi-row apps)
   - Then full ablation: K=4 sweep with `--use-dense-reward` vs without, compare T_eval (T_eval still binary, only training changes)

7. **Why this matters**: B4 v2 (45.2%) ≈ B3 (46.7%) is plateau-ish. Dense reward is the most cited reason in literature for breaking through sparse-reward plateaus (Lightman et al. 2023 PRM, OpenAI process supervision). We're testing whether trajectory-level partial credit alone is enough (no step-level process model). Result decides whether the paper writes "dense reward gives +X pp" or "trajectory-level partial credit didn't help, step-level PRM is left as future work".

### 2026-06-01 — RL framework audit + results-attribution recheck + verl/SkyRL spike

Three findings from a from-raw-data recheck this session. All numbers below re-aggregated directly from each run's `results.csv` (0 bad rows, 0 dup template/seed, episode counts verified 105 = 54 Tier-B + 51 Tier-C). Trust these over the prose in `docs/project_progress_2026_05.md`, which over-claims (see #2).

**1. The headline "B4 = 52.9% / +14.3pp" does NOT stand.** It is reconstructed exactly as Tier-B 70.4 (from `phase1_v3c_teval`) + Tier-C 34.3 (from `pi_pre_v3b_standalone_teval`), weighted (70.4×54 + 34.3×51)/105 = 52.9. Two problems: (a) **per-tier oracle** — picks the best init per tier using the test set; (b) **both components are STANDALONE frozen π^pre with ZERO Phase-3 LoRA training**, and both use `b4_k4_v3binit`'s evolved L_C. So B4's defining step (Phase-3 LoRA weight adaptation) is absent from the headline number. **Use the clean single-pipeline B4 = 48.1%** (`b4_k4_teval`, pilot init) as the real B4. `docs/results/experiments.md` (2026-05-28) already has it right.

**2. Phase-3 LoRA training is null/negative under a clean ablation.** The only L_C-held-constant comparison (v3-B init; both standalone and full use `traces/b4_k4_v3binit/.../l_c_champion.json` — identical L_C, only LoRA-trained-or-not differs): frozen π^pre 49.0 → Phase-3-trained 43.3 = **−5.7pp (hurts)**. pilot (+5.7) and v3-C (−2.9) deltas are CONFOUNDED (standalone uses v3binit L_C, full uses each run's own evolved L_C — so can't attribute to LoRA). Net: the B3→B4 +1.4pp may be L_C evolution, not LoRA. **The TTA-via-weight-adaptation claim (B4's selling point, see memory `project-b4-is-tta-centerpiece`) is currently unsupported by a clean ablation.**

**3. RL framework audit (`evofsm_rl/rl/grpo.py`).** The hand-written GRPO is **REINFORCE + group-mean baseline + KL anchor**, NOT the PPO-clip GRPO that `method.tex` / `docs/project_progress_2026_05.md §2.3` write (`grpo.py:496/500` is `-adv·traj_scale·log π_θ`; no importance ratio ρ, no clip, cached old-logprobs unused). **Under strict on-policy single-step-per-fire (buffer collected over 3 iters with no weight update in between → ρ≡1, one `optimizer.step()` per fire) this is mathematically correct for what we run** — but the paper equation is wrong relative to code (fix the writeup, or implement ratio+clip). Three real deviations/bugs, none numerically tested (`test_grpo.py` only covers `compute_reward`/`compute_advantages` + a fake-zero-model plumbing check; `grpo_step`'s loss/grad has NO ground-truth test):
  - **dropout-on during the grad forward** — `grpo.py:412` `model.train()` (for gradient-checkpointing) also enables LoRA dropout=0.05, so the gradient logprob ≠ the eval-mode logprob the action was sampled under. Injects noise.
  - **advantage not std-normalized** (`grpo.py:164` is `r-mean`; standard GRPO is `(r-mean)/(std+eps)`).
  - **KL computed at sequence level** (sum of token logprobs → fragile, needs the `clip=10` band-aid); per-token is standard.
  These likely partly explain why β had to be as high as 0.05 (`experiments.md` Table 6) to stay stable.

**Phase 1 is online RL** — `scripts/run_phase1_pretraining.py` is the SAME emulator-live GRPO loop as Phase 3 (`_build_pretrain_rollout_fn` mirrors `_build_b4_rollout_fn`; `harness.run_template` each iter), differing only: source pool apps, KL ref=**base** (`--anchor-to-base`), FSMs frozen (`fsm_variant_id="static_{app}"`, no Opus mutation), K=2. It is the compute大头 (~4800 traj) and trained飞 twice historically (v2/v3). π^pre is NOT offline/SFT (that's RFT = 41%, separate).

**Decision parked → spike plan written: `docs/plan/spike_verl_migration.md`.** Candidate is the [SkyRL-AndroidWorld](https://github.com/Guliisgreat/SkyRL-AndriodWorld) fork (backend = **verl**, the vetted part). It already ships the AndroidWorld↔SkyRL env bridge + a 16-parallel Dockerized emulator pool (= the parked concurrent-rollout speedup, prebuilt; directly attacks our ~70%-emulator-bound bottleneck). Env feasibility confirmed on this box (2026-06-01): `docker 20.10.24` + nvidia runtime; `linqiang` ∈ `kvm`+`docker` groups; box already runs OSWorld Docker GUI containers. **Spike is Phase-1-first** (pure RL, no Opus/L_C in loop → zero-glue fit; KL ref=base = verl default; biggest compute + historically unstable → most to gain). Gaps to handle: fork example uses Qwen2-VL-7B (we need Qwen3-VL-8B, multimodal-packing risk), LoRA not shown (verl supports, must add/verify), and the Opus/L_C/FSM symbolic half stays in our outer loop (inject L_C into the env server's prompt between rounds). verl KL is built-in (`use_kl_loss` + `kl_loss_type=low_var_kl` = our k3, but per-token; don't double-apply with `kl_coef`); Phase-3 ref=π^pre and a β re-sweep (from 0.001, NOT our 0.05) are the only KL items we own. Time-boxed 2–3 days, M3 outputs (A) full migrate / (B) Phase-1 only / (C) don't migrate — do the numerical-equivalence test + fix the 3 deviations instead.

### 2026-06-01 (session 2) — Direction pivot: cross-benchmark TTA, eval on MobileWorld; pure-vision harness; M0 verl-env built

**RESUME HERE. Status = under discussion, NOT all locked.** Read this block first next login.

**The pivot (new target pipeline).** Prior EvoFSM idea already validated on AndroidWorld → now push to **cross-benchmark generalization**:
- **Train** (Phase 1 → π^pre) on **android_world_plus** (our 193-task set).
- **T-adapt** (B4 TTA, evolve FSM + LoRA) on the target app. ⚠️ **OPEN — confirm with user:** does adapt run on the MobileWorld target apps (my read: yes, TTA adapts on the eval-domain app), or on AW+?
- **T-eval** on **all MobileWorld GUI-only tasks**.
- Adapt reuses **same-category** `L_C` knowledge when the target app's Play-category is in our source pool; else falls back to **app-level** adaptation from scratch (user's words: "adapt 用同 cat 的 or app 的知识").

**Harness decision (settled this session): drop a11y → pure-vision; adopt MobileWorld's `general_e2e` harness.**
- We no longer use a11y elements. The current `Qwen3VLAgent` (M3A clone: a11y tree + 2-phase action+summary + element-index actions) is **to be REPLACED**.
- Why forced: a11y element-index actions don't transfer across benchmarks; **pixel-coordinate grounding is benchmark-agnostic** → required for AW+→MobileWorld transfer + train/eval consistency.
- **FSM/L_C symbolic injection survives** — goes into the general prompt (EvoFSM thesis intact). Cost: all existing a11y-trained LoRAs must be **retrained** on the vision format.

**MobileWorld repo added** at project top-level: `MobileWorld/` (git, fork `pockyitachi/MobileWorld`, branch main, 274 MB). Its `general_e2e` agent (`src/mobile_world/agents/implementations/general_e2e_agent.py` + prompt `.../utils/prompts/general_e2e.py`): screenshot-only input, single prompt, single forward/step; output `Thought:…\nAction:{json}`; **pixel-coordinate** actions (rel 0–999 → abs): click/double_tap/long_press/drag/input_text/answer/scroll/status/wait/navigate_*/ask_user/keyboard_enter **+ MCP tool calls** (GUI+MCP hybrid — drop MCP for our GUI-only use); history = last-3 raw screenshots + responses (not M3A NL summary).

**MobileWorld GUI-only task set (extracted this session):** **161 GUI-only** tasks (of 201 total; 40 use MCP), across **15 apps**. Mapping to our 12 Play-Store-category taxonomy:

| MobileWorld app (task count) | → our category | same-cat prior? |
|---|---|---|
| Mail(43), Messages(26), Contacts(11) | Communication | ✅ |
| Calendar(27) | Productivity | ✅ |
| Files(31) | Tools/Productivity | ✅ |
| Docreader(10) | Books & Reference | ✅ |
| Gallery(11), Camera(3) | Photography | ✅ |
| Maps(9) | Maps & Navigation | ✅ |
| Clock(7), Settings(7), Chrome(10) | Tools | ✅ |
| **Mastodon(41), Mattermost(17)** | Social | ❌ novel category |
| **Taodian(15)** | Shopping/电商 | ❌ novel category |

→ 12 apps have same-category pretraining support (tests **transfer**); Mastodon/Mattermost/Taodian are novel categories (tests **pure app-level adaptation**). Set gives BOTH TTA regimes — good for the paper. Eval uses MobileWorld's own harness + reward (no porting needed). (Counts are per-app occurrence; tasks may span 2 apps.)

**M0 (verl env) DONE:** built `androidworld:evofsm-tasks193` (193 tasks = 116 vanilla + 77 plus; build ctx `build_evofsm_tasks/`). See memory `project_m0_smoke_verified_recipe` + `docs/plan/spike_verl_migration.md` (M0 section). ⚠️ The harness inside this image is still M3A — needs swap to `general_e2e` per the decision above.

**Prompt decision (settled): use Qwen3-VL's NATIVE `mobile_use` format, don't design from scratch.** MobileWorld ships two prompts — `general_e2e` (generic markdown table, coord 0–999) and **`qwen3vl`** (`prompts/qwen3vl.py`: `MOBILE_QWEN3VL_PROMPT`, the official Qwen `mobile_use` tool-call schema — `<tools>`/`<tool_call>` XML, actions click/long_press/swipe/type/answer/system_button/wait/terminate/ask_user, 999×999 coord space). Base on **`qwen3vl`** because Qwen3-VL was pretrained on exactly this schema → **strongest zero-shot grounding → least RL to bootstrap** (this is why the "pure-vision grounding is a new/risky skill" worry is mostly moot — it's Qwen3-VL's native ability). Our only real design work = (a) graft the FSM/L_C injection slot into the template, (b) drop the MCP part (GUI-only), (c) action-name translation layer.

**Env coordinate-action support CONFIRMED (hard prereq for pure-vision training, env needs ZERO changes).** In `androidworld:evofsm-tasks193`, `env/actuation.py` executes coordinate actions natively (with `actuation_test.py` coverage): `click`/`long_press`/`double_tap` consume `JSONAction.x/y` → `adb_utils.tap_screen(x,y)`; `swipe` consumes `direction=[x,y,x2,y2]` (4-elem list) → `generate_swipe_command` ("Precise pixel-coordinate swipe"). `JSONAction` has both `index` AND `x`/`y` fields. So the only glue needed = a thin **`mobile_use` JSON → `JSONAction`** translation in the rollout harness (swipe coords → `direction` list; `system_button` Back/Home/Menu/Enter → navigate_back/navigate_home/keycode; `terminate(status)` → `status(goal_status)`). Not env work.

**Online-RL data clarification (don't re-debate):** Phase 1/3 are online RL — rollouts ARE the training data, generated fresh on-policy by the current policy on the live emulator each iter; **no pre-prepared training dataset needed** (just task set + env + model). Old a11y-era RL trajectories are NOT reusable for the RL loop (off-policy + a11y-index→coordinate space change + **bbox was never persisted**, only index+text+screenshot, so index→coord is unrecoverable from stored data). What DOES carry over: **L_C/FSM** (abstract NL, no a11y/coord coupling → inject into the new prompt as-is) + optional warm-start SFT only if you replay old action-index sequences in the emulator to re-derive coords (not required if Qwen3-VL native grounding is good enough).

**Workspace created (2026-06-02): `../EvoFSM-MW/`** (option A — thin orchestration + new-harness layer that **imports** EvoFSM-RL's symbolic core, does NOT fork it). Scaffold only: `README.md` (authoritative plan + reuse boundary), `harness/{prompt,qwen3vl_agent,action_translation}.py` (stubs marking the contract), `orchestration/ configs/ results/ notes/`. NOT git-init'd (user deferred). Read its `README.md` to resume building.

**Open questions to resume on:**
1. Where T-adapt runs (MobileWorld target apps vs AW+) — confirm.
2. Mastodon/Mattermost/Taodian: position as "works w/o same-cat prior", or add pretraining categories?
3. Multi-app MobileWorld tasks (e.g. Calendar+Messages): how `L_C` category knowledge applies across 2 apps in one task.
4. **M1 (Phase-1 verl training) BLOCKED on GPU.** Usable 4 & 7 both occupied: GPU4 by `dawood` (external, 122 GB, active); GPU7 by teammate `ziqiang` (vLLM, 99 GB, up 25h, looks idle — compute/CPU ~0). GPU2 empty but off-limits. Realistic lever = ask ziqiang to free GPU7. Meanwhile GPU-free track = wire verl GRPO config (KL: `use_kl_loss` + `low_var_kl`, ref=base, β from 0.001) + new general_e2e rollout + Phase-1 pilot task config.

Note on task_type labels:
- Vanilla AndroidWorld tasks (116) → resolved from `task_metadata.json` tags via priority rule.
- Plus-repo tasks (78) → hand-annotated in `_PLUS_TASK_TYPES` inside `taxonomy.py`.
- 12 tasks still resolve to `generic` (5 are honest pure-navigation/open tasks, 7 are vanilla tasks whose only tag is `requires_setup` or empty).

---

## How to navigate this folder

- Asked about "what apps / tasks do we have?" → `docs/design/dataset.md` (tables + inventory)
- Asked about "how do we split? why?" → `docs/design/dataset.md` (§T_adapt/T_eval protocol) + `configs/splits.yaml`
- Asked about "why this taxonomy / why these citations?" → `docs/design/dataset.md`
- Asked about "what's the algorithm?" → `docs/design/algorithm.md` (paper-ready) or `plan/algorithm_design.md` (raw working doc)
- Asked about "what are B1/B2 results?" → `docs/results/b1_b2_static_baselines.md`
- Asked about "what are B3 evolution results?" → `docs/results/b3_evolution.md`
- Asked about "what are we doing next?" → `docs/plan/linear_tickets.md` + `plan/project_plan.md`
- Writing code → `evofsm_rl/` — active modules:
  - `evofsm_rl/model/loader.py` — base model loader (Story 1.2)
  - `evofsm_rl/model/lora.py` — LoRA wrap/unwrap (B4)
  - `evofsm_rl/agent/prompts.py` — M3A prompts + Qwen3-VL multi-image chat-template packers (Story 1.5)
  - `evofsm_rl/agent/action.py` — JSON→JSONAction parser with error recovery (Story 1.5)
  - `evofsm_rl/agent/a11y.py` — UI-element JSON list + SoM marker
  - `evofsm_rl/agent/rollout.py` — `Qwen3VLAgent.step()` mirrors M3A 1:1 + `save_episode()` writer (Stories 1.5, 2.0)
  - `evofsm_rl/env/harness.py` — thin AndroidWorld env wrapper (Story 1.1)
  - `evofsm_rl/fsm/builder.py` — per-app FSM synthesis via Anthropic API (Story 2.2)
  - `evofsm_rl/rl/grpo.py` — GRPO trainer for B4 LoRA fine-tune. Reads `episodes/*.jsonl`, computes group advantages, fires when buffer is full (every 3 iters by default). **Known issue**: mean_advantage→0, no grad clip, no learning signal in B4 v2 sweep.
  - `scripts/model_smoke.py` — forward-pass smoke test
  - `scripts/run_rollout.py` — single-task baseline eval CLI
  - `scripts/baseline_10task.py` — N-task sweep CLI with persistence flags
  - `scripts/coord_probe.py` — coordinate calibration probe
  - `scripts/build_all_fsms.py` + `scripts/build_L_C.py` — Story 2.2 FSM/L_C synthesis CLIs
  - `scripts/run_b2_eval.py` — B2 static-L_C eval
  - `scripts/run_b3_evolution.py` + `scripts/run_b3_teval.py` — B3 evolution sweep + T_eval
  - `scripts/run_b4_evolution.py` + `scripts/run_b4_sweep.sh` — B4 evolution + GRPO LoRA sweep
  - `scripts/plot_convergence.py` — convergence plots for evolution sweeps
- Task-list configs → `configs/`:
  - `configs/baseline_50task.txt` — original source-pool 50-task list (used for `m3a_50task_v01`)
  - `configs/source_pool_96.txt` — full 96-template source pool (used for `m3a_traj` 480-episode collection)
  - `configs/t_eval/teval_v01.txt` — 35 held-out templates (Tier-B + Tier-C T_eval)
- Looking at trace data → `traces/` (current sizes 2026-04-26, total ~27 GB):
  - `traces/m3a_50task_v01/` — source-pool 50-task baseline (summary only)
  - `traces/m3a_teval_v01/` — T_eval 35-task B1 headline baseline (summary only)
  - `traces/source_pool_trajectories/` — **480 per-episode dirs** with Story 2.0 schema (meta.json + episode.jsonl + step PNGs). 2.1 GB. Source for Story 2.2 FSM builder.
  - `traces/source_pool_summary/seed{30..34}.json` — per-seed aggregate summaries
  - `traces/b1_teval_k3/` (770 MB) — B1 K=3 sweep
  - `traces/b2_teval_v01/` (326 MB) + `traces/b2_teval_k3/` (744 MB) — B2 static-L_C eval
  - `traces/b3_evolution/` (1.3 GB) — B3 FSM-only evolution sweep
  - `traces/b3_teval/` (1.8 GB) — B3 T_eval sweep
  - `traces/b4_smoke/` (119 MB) + `traces/b4_smoke_v3/` (99 MB) — pre-sweep smoke runs
  - `traces/b4_evolution/` (**17 GB, ABORTED v1**) — 4× OOM, kept for forensics. See "B4 sweep lineage" before quoting anything from this dir. Candidate for deletion to reclaim disk.
  - `traces/b4_evolution_v2/` (2.1 GB, COMPLETE v2) — 6 apps × 20 iters, 0 OOM, but no GRPO learning signal. **Pipeline-stable but learning-null.** See "B4 sweep lineage".
  - `traces/m3a_audio_rerun/`, `traces/m3a_smoke_1task/`, `traces/m3a_traj_smoke/` — small smoke runs
  - `traces/archive_singleshot_v0/` — deprecated single-shot v0 runs (ablation only)
- Reading paper-style baseline reports → `docs/results/`:
  - `b1_b2_static_baselines.md` — B1 zero-shot + B2 static-L_C K=3 (Tier-B +9.3pp, Tier-C null).
  - `b3_evolution.md` — B3 evolution sweep + B3 vs B2 T_eval (Tier-B +3.7pp overall).
  - **B4 report not yet written.** v2 sweep complete but learning-null — needs the "what next" decision before writing up.

---

## Things to NOT do

- Do not invent new taxonomies or split protocols without checking that they're cite-able. This is a research paper — methods need precedent (see `docs/design/dataset.md` for the bar).
- Do not re-count apps or tasks from scratch — the inventory is verified, trust it.
- Do not change `splits.yaml` without bumping the `version` field in `meta` and noting the change in this CLAUDE.md.
- Do not put project files outside `EvoFSM-RL/` — keep the parent `android_world_plus` repo clean.
- Do not quote numbers from `traces/archive_singleshot_v0/` as the paper baseline. That's the deprecated single-shot v0 agent, not the current Qwen3-VL-M3A. See "Baseline lineage" above.
- Do not "restore" the legacy single-shot agent code (`SYSTEM_PROMPT_V0`, `ACTION_SCHEMAS`, `build_messages`, `build_system_prompt`, `build_user_turn`, `format_history`, `UIElementsView`, `build_ui_elements_view`, `describe_ui_element`). They were removed in the Story 1.5 rewrite. If a future caller still references them, fix the caller — don't re-add the symbol.

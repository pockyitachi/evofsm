# Reproduce

Two studies, two harnesses. The **within-benchmark** ladder runs against a
pre-baked Android emulator; the **cross-benchmark** ladder serves the backbone on
vLLM and drives fresh MobileWorld containers. Both share the same symbolic core
(FSM / `L_C`), imported from `EvoFSM-RL/evofsm_rl/` via `PYTHONPATH`.

!!! note "Full flags live in the subprojects"
    The commands below are the reproducible skeleton. Every script takes more
    flags than shown — pass `--help`, and read each subproject's `README.md` /
    `harness/README.md` (and the top docstring of each script) for the exact,
    authoritative wiring.

## Within-benchmark (AndroidWorld+)

### 1 — Environment

Python 3.12 venv; put both the benchmark and the package on the path.

```bash
cd /shared/linqiang/evofsm_project && source .venv/bin/activate
export PYTHONPATH=android_world_plus:EvoFSM-RL
```

### 2 — Boot the pre-baked emulator

A single pre-baked AVD — **`AWAvd2`, snapshot `apps_ready_dec2025`** — ships with
all apps installed. Boot it **read-only** so the snapshot is never modified. Do
**not** run `emulator_setup=True` / `bootstrap_avd.sh`; the snapshot already has
the apps.

```bash
ANDROID_AVD_HOME=$PWD/android-sdk/avd ANDROID_SDK_ROOT=$PWD/android-sdk \
  $PWD/android-sdk/emulator/emulator -avd AWAvd2 -port 5710 -grpc 8710 \
  -snapshot apps_ready_dec2025 -no-snapshot-save -read-only \
  -no-window -no-audio -no-boot-anim &

# verify (prints 1 ~30s after boot):
adb -s emulator-5710 wait-for-device shell getprop sys.boot_completed
```

Multiple instances coexist on different `-port`/`-grpc` pairs.

### 3 — Run the ablation (B1 → B4)

Each rung adds exactly one mechanism. Every script takes
`--console-port 5710 --grpc-port 8710`; pass `--help` for the full flag set.

```bash
# B1 — zero-shot baseline on T_eval (M3A, Qwen3-VL-8B)
python EvoFSM-RL/scripts/baseline_10task.py        --console-port 5710 --grpc-port 8710 ...

# B2 — + static category L_C injection
python EvoFSM-RL/scripts/run_b2_eval.py            --console-port 5710 --grpc-port 8710 ...

# B3 — evolve L_C on T_adapt (per Tier-B app), then frozen eval on T_eval
python EvoFSM-RL/scripts/run_b3_evolution.py       --app pro_expense ...
python EvoFSM-RL/scripts/run_b3_teval.py           --console-port 5710 --grpc-port 8710 ...

# B4 — joint LoRA + FSM (full method): Phase-1 pretrain, then per-app adapt + eval
python EvoFSM-RL/scripts/run_phase1_pretraining.py ...
python EvoFSM-RL/scripts/run_b4_evolution.py       --app pro_expense ...
python EvoFSM-RL/scripts/run_b4_teval.py           --console-port 5710 --grpc-port 8710 ...
```

The FSM / `L_C` knowledge layer is pre-built under `artifacts/`. To regenerate it,
set `ANTHROPIC_API_KEY` and run `scripts/build_all_fsms.py` + `scripts/build_L_C.py`.

## Cross-benchmark (MobileWorld)

This study reuses the symbolic core but swaps in a pure-vision Qwen3-VL
`mobile_use` harness and the MobileWorld benchmark. The agents run in
MobileWorld's own venv (`mw eval ...`); the guidance generators run from the
project root in the main venv with `PYTHONPATH=EvoFSM-RL`.

### 1 — Serve the backbone and boot fresh containers

Serve the backbone on vLLM at `localhost:8001`, then bring up **fresh** MobileWorld
containers for the run.

```bash
# backbone served on localhost:8001 (Qwen3-VL-8B / MAI-UI-8B)
# image: mobile_world:reset   ·   network: mwnet
```

!!! warning "Never reuse a container"
    MobileWorld containers do **not** reset app state between tasks, so a
    container that has already run an eval carries dirty state. Every run gets a
    fresh pool on network `mwnet` (the host's default docker bridge is gone).
    Only tear down containers you created. MCP-401 noise at startup is harmless
    for the GUI-only 110-task list.

### 2 — Build the injection guidance

The B2-family guidance is pre-built under `artifacts/`. To regenerate the
strongest static config (**B2′**, app-level Layer-2 + category `L_C`, no Layer-1):

```bash
python harness/gen_b2_guidance.py --mode app-l2     # -> artifacts/b2p_guidance.json
```

!!! note "Layer-1 is strictly harmful cross-benchmark"
    Source-environment Layer-1 state descriptions displace grounded behaviour, so
    the strongest static config is `--mode app-l2` (Layer-2 only). Do not add
    Layer-1 back. Other modes: `--mode full` (B2, full app FSM) ·
    `--mode category-only` (B2″, category `L_C`).

### 3 — Evaluate on the 110-task `T_eval`

Run `mw eval` over the 110-task eval list. B1 = stock agent; B2′ = the file-path
agent (`harness/qwen3vl_b2_agent.py`) pointed at the guidance JSON via
`EVOFSM_B2_GUIDANCE`.

```bash
mw eval ... --max_round 50 --max-concurrency 3 --enable_mcp --enable_user_interaction
```

Eval traces land in `MobileWorld/traj_logs/<run>/<task>/result.txt` — `score: 1.0`
is a pass, denominator 110. Exact per-arm commands and container port bands are in
`EvoFSM-MW/harness/README.md` and the setup block of `docs/qwen3_8b_res.md`.

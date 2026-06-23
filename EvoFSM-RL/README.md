<div align="center">

# EvoFSM-RL

**Test-time adaptation for mobile GUI agents on _unseen_ apps**

Adapt to a new Android app on a deployment-time budget of a few dozen
trajectories — by **jointly** evolving a two-layer FSM prior (symbolic) and
fine-tuning a LoRA policy (sub-symbolic).

Built on [`android_world_plus`](../android_world_plus) · backbone Qwen3-VL-8B

</div>

---

## 🎯 The problem

Mobile GUI agents handle multi-step tasks well on apps they were trained on, but
stay brittle on apps they have never seen. The headline 60–80% success rates on
today's benchmarks are measured where train and test share the same task
templates — they say little about the deployment-relevant question:

> When a user installs a banking app, a niche notes tool, or a ride-hailing
> service the agent has never encountered, can it become reliable from a handful
> of adaptation examples?

Per-app data synthesis and large-budget online RL are both off the table at
deployment time. **EvoFSM-RL** targets exactly this small-budget, unseen-app
regime.

## 🧩 Approach

The agent's prior knowledge is a **two-layer finite state machine**, and
adaptation co-evolves it with the policy weights in one loop:

```
PRIOR ── two-layer FSM
  ├─ LAYER 1  app-specific UI graph (screens, transitions, visual cues)   non-transferable
  └─ LAYER 2  category-generic archetypes (ADD_ENTRY, QUERY_INFO, …)      transferable
                 │  a linter rejects any app token in Layer 2  →  guaranteed transfer
                 └─ same-category apps aggregate Layer 2 ──►  L_C   (per-category library)

LOOP (one adaptation iteration)
  rollouts under FSM variants ─► TrueSkill rank ─► frozen LLM proposes a Layer-2 diff   (prompt channel)
                              └─► GRPO within-(FSM,task) advantage ─► LoRA update         (weight channel)

PHASES   Phase 1  pretrain a shared LoRA on the source pool
         Phase 3  co-adapt LoRA + the target app's FSM on T_adapt ─► freeze ─► eval on T_eval
```

The **same loop** runs at source-pool pretraining and at target-app
deployment, which makes "test-time adaptation" a well-defined operation rather
than an ad-hoc fine-tune.

## 📊 Results on AndroidWorld+

**Benchmark.** AndroidWorld extended with 6 BMOCA/AndroidLab apps → **25 apps,
12 Play-Store categories**. Three *independent* disjoint splits stacked together
give an honest generalization measurement (absolute numbers are lower than
leaderboard by design):

| Split | Measures |
|---|---|
| source pool ↔ target apps | cross-app transfer |
| near-transfer (Tier-B) ↔ far-transfer (Tier-C) | cross-category transfer |
| `T_adapt` ↔ `T_eval` templates | within-app generalization (not template memorization) |

**Ablation ladder** — each rung adds exactly one mechanism (`T_eval`, K=3 seeds):

| Arm | Mechanism added | Tier-B | Tier-C | Overall |
|---|---|:--:|:--:|:--:|
| **B1** | zero-shot (M3A, Qwen3-VL-8B) | 47.2 | 29.4 | 38.6 |
| **B2** | + static category `L_C` | 56.5 | 29.4 | 43.3 |
| **B3** | + online `L_C` evolution | **63.9** | 31.4 | **48.1** |
| **B4** | + joint LoRA & FSM | **70.4** | **34.3** | **52.9** |

**Reading the deltas (Tier-B):**
- **B2 − B1 = +9.3 pp** — static category knowledge transfers across same-category apps.
  Tier-C delta is exactly **0.0 pp** (no `L_C` to inject → prompt is byte-identical to B1): a built-in **null control** confirming the gain is mechanism-bound.
- **B3 − B2 = +3.7 pp** — letting Layer 2 evolve on the target app's adaptation set adds more on top.

**Where it helps** — gains concentrate where the category prior structurally
overlaps the target: `pro_expense` 33→67% (+33 pp, Finance `L_C`),
`system_settings` 75→92% (Tools `L_C`). Apps whose UI diverges from the source
pool benefit less from static injection alone — which is what online evolution
is for.

> **B4 — the full joint method** brings both channels together (joint LoRA +
> evolved FSM): **38.6 → 52.9 overall (+14.3 pp)** end-to-end over the B1
> zero-shot floor, and +6.5 pp Tier-B over B3.

## 📁 Repository

| Folder | What | | Folder | What |
|---|---|---|---|---|
| `evofsm_rl/` | core package (agent · FSM · model · GRPO) | | `paper_draft/` | the paper, one `.tex` per section |
| `scripts/` | B1–B4 CLI entry points | | `artifacts/` | static FSM / `L_C` knowledge layer |
| `configs/` | `splits.yaml` (source of truth) | | `archive/` | abandoned routes (PPO+PRM, RFT) |
| `docs/` | design · results · plan | | `tests/` | unit tests |

Every folder has its own `README.md` (what it is) and `CLAUDE.md` (working
context). Start at [`CLAUDE.md`](CLAUDE.md) for the project-wide picture and
operational/bootstrap details.

## 🔁 Reproduce

**1 — Environment** (Python 3.12; canonical setup + fresh-server recipe in [`CLAUDE.md`](CLAUDE.md)):
```bash
cd /shared/linqiang/evofsm_project && source .venv/bin/activate
export PYTHONPATH=android_world_plus:EvoFSM-RL
```

**2 — Boot the pre-baked emulator** (read-only, never modifies the snapshot):
```bash
ANDROID_AVD_HOME=$PWD/android-sdk/avd ANDROID_SDK_ROOT=$PWD/android-sdk \
  $PWD/android-sdk/emulator/emulator -avd AWAvd2 -port 5710 -grpc 8710 \
  -snapshot apps_ready_dec2025 -no-snapshot-save -read-only -no-window &
adb -s emulator-5710 wait-for-device shell getprop sys.boot_completed   # → 1
```

**3 — Run the ablation** (each script takes `--console-port 5710 --grpc-port 8710`; `--help` for full flags):
```bash
# B1  zero-shot baseline on T_eval
python EvoFSM-RL/scripts/baseline_10task.py   --console-port 5710 --grpc-port 8710 ...

# B2  static L_C injection
python EvoFSM-RL/scripts/run_b2_eval.py       --console-port 5710 --grpc-port 8710 ...

# B3  evolve L_C on T_adapt (per Tier-B app), then frozen eval on T_eval
python EvoFSM-RL/scripts/run_b3_evolution.py  --app pro_expense ...
python EvoFSM-RL/scripts/run_b3_teval.py      --console-port 5710 --grpc-port 8710 ...

# B4  joint LoRA + FSM (full method): Phase-1 pretrain, then per-app adapt + eval
python EvoFSM-RL/scripts/run_phase1_pretraining.py ...
python EvoFSM-RL/scripts/run_b4_evolution.py  --app pro_expense ...
python EvoFSM-RL/scripts/run_b4_teval.py       --console-port 5710 --grpc-port 8710 ...
```
The FSM/`L_C` are pre-built under `artifacts/`; to regenerate them set
`ANTHROPIC_API_KEY` and run `scripts/build_all_fsms.py` + `scripts/build_L_C.py`.

## 🔭 Cross-benchmark (where this is going)

The same method extends from *within*-benchmark transfer (here) to
*cross*-benchmark transfer: pretrain on AndroidWorld+, deploy on an
independently authored benchmark (**MobileWorld**, GUI-only). That work lives in
[`../EvoFSM-MW`](../EvoFSM-MW).

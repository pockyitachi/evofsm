<div align="center">

# EvoFSM-MW

**Cross-benchmark test-time adaptation for mobile GUI agents**

Pretrain on **AndroidWorld+**, deploy on **MobileWorld** — an independently
authored benchmark. The symbolic prior (two-layer FSM / `L_C`) transfers by Play
Store category; the agent adapts at test time on a small per-app budget.

Reuses the symbolic core of [`../EvoFSM-RL`](../EvoFSM-RL) · pure-vision Qwen3-VL `mobile_use` harness

</div>

---

## 🎯 The problem

[`../EvoFSM-RL`](../EvoFSM-RL) measures *within*-benchmark generalization (held-out
apps of the same benchmark). The harder, deployment-faithful question is
*cross*-benchmark: train on one benchmark and deploy on a **separately
constructed** one, where even "seen" categories are instantiated by different
apps with different UIs, longer horizons, and many multi-app tasks. That is what
EvoFSM-MW evaluates — AndroidWorld+ → MobileWorld.

## 🧩 Approach

- **Reuse, don't fork.** The symbolic core (FSM / `L_C` builder, evolution loop,
  taxonomy, splits) is **imported** from `../EvoFSM-RL/evofsm_rl/` via
  `PYTHONPATH` — not copied.
- **Pure-vision harness.** A new harness drives the agent in Qwen3-VL's native
  `mobile_use` tool-call format (accessibility tree dropped, GUI-only). The
  FSM/`L_C` symbolic prior is spliced into the prompt and transfers across the
  benchmark boundary by Play Store category.
- **Same B1→B4 ladder.** B1 zero-shot · B2 static prior · B3 online symbolic
  evolution · B4 joint symbolic + weight (LoRA) adaptation.

## 📊 Results on MobileWorld

110-task GUI-only eval, 5-run variance, two models (a weak base and the
benchmark's near-ceiling first-party agent):

| Model | B1 (zero-shot) | B2′ (best static prior) | B3 (symbolic TTA) | B4 |
|---|--:|--:|--:|:--:|
| **EvoFSM-8B** (qwen3-VL) | 8.2 ± 0.8 | 9.2 ± 1.6 | 10 – 10.4 | _headroom_ |
| **MAI-UI-8B** (first-party) | 26.2 ± 2.4 | 26.4 ± 2.0 | 25.6 – 29 | _headroom_ |

**What the numbers say:**
- **Static-injection benefit is capability-dependent** — +1.0 on the weak base,
  +0.2 (net-zero) on the strong one. The stronger the model, the less a
  hand-distilled prior adds.
- **Cross-benchmark Layer-1 injection is strictly harmful** — source-environment
  state descriptions displace grounded behaviour; the best static config is
  **B2′** (app-level Layer-2 + category `L_C`, no Layer-1).
- **Symbolic TTA (B3) tops out at the static prior on both models** — neither FSM
  evolution nor distilled lessons cleanly beats B2′. The one reportable positive
  is *efficiency*: distilled "lessons-only" matches the prior at ~1/20 the text.
- **→ The headroom is the B4 weight channel.** This is the thesis the
  cross-benchmark setting sharpens: static and symbolic-adaptive knowledge are
  capability-bounded; weight adaptation is the open axis.

Full results: [`docs/qwen3_8b_res.md`](docs/qwen3_8b_res.md) (8B B-series) ·
[`docs/mai_ui_8b_res.md`](docs/mai_ui_8b_res.md) (MAI-UI + cross-model + B3).

## 🗂 Dataset & splits

**Train** on AndroidWorld+ (193 tasks → Phase-1 shared LoRA `π^pre`). **Test** on
MobileWorld's **GUI-only** subset — 161 of 201 tasks (the MCP-tool tasks are
excluded), pure-vision, hard-coded single instances (no seed, K=1).

The 161 GUI-only tasks are split **task-disjoint** (a task is wholly in adapt or
eval). Tiers are **category-level** — defined by whether the apps a task touches
fall in Play categories seen during AndroidWorld+ pretraining:

| Tier | Meaning | Tasks | `T_adapt` | `T_eval` |
|---|---|--:|--:|--:|
| **Tier-B** | near-transfer — all apps in source-seen categories | 91 | 33 | 58 |
| **Tier-C** | far-transfer — novel categories (Social: Mastodon/Mattermost · Shopping: Taodian) | 43 | 18 | 25 |
| **Tier-A** | mixed — multi-app tasks combining seen + novel categories | 27 | 0 | 27 |
| **Total** | | **161** | **51** | **110** |

Tier-A is held entirely in eval — it is the headline test of *composing* seen and
novel knowledge, and has no clean single-app split. The adaptation loop runs on
the 51-task `T_adapt`; the frozen headline number is the 110-task `T_eval`.
Source of truth: [`configs/mobileworld_splits.yaml`](configs/mobileworld_splits.yaml)
(rationale in [`docs/dataset_tiers.md`](docs/dataset_tiers.md) /
[`docs/mobileworld_split.md`](docs/mobileworld_split.md)).

## 📁 Repository

| Folder | What |
|---|---|
| `harness/` | Pure-vision Qwen3-VL `mobile_use` eval harness — agents, guidance generators, action translation, eval drivers |
| `configs/` | `mobileworld_splits.yaml` (source of truth) + the 110-task eval list |
| `docs/` | Results (`qwen3_8b_res`, `mai_ui_8b_res`) · design (`b3_b4_mw_tta_design`, `b3_lesson_memory_*`) · data (`dataset_tiers`, `mobileworld_split`) |
| `artifacts/` | Injection guidance JSONs (durable) + run logs / eval-log transcripts (disposable) |
| `results/` · `orchestration/` · `notes/` | Outputs · outer-loop scaffolding · design notes |

Every folder has its own `README.md` (what it is) and `CLAUDE.md` (working
context).

## 🔗 Reuse boundary (do **not** duplicate)

| Concern | Where | Mode |
|---|---|---|
| Symbolic core (FSM/`L_C`, evolution, taxonomy, splits) | `../EvoFSM-RL/evofsm_rl/` | **import** via `PYTHONPATH` |
| Eval benchmark + reward | `../MobileWorld/` | use its harness/reward as-is |
| RL training backend (verl GRPO) | external SkyRL | drive via its server |

## 🔁 Reproduce

Serve the backbone on vLLM (`localhost:8001`); boot **fresh** MobileWorld
containers per run (image `mobile_world:reset`, network `mwnet` — containers do
**not** reset app state, so never reuse them). Then, per arm:

```bash
# B2-family guidance is pre-built under artifacts/; to regenerate the strongest:
python harness/gen_b2_guidance.py --mode app-l2          # -> artifacts/b2p_guidance.json (B2')

# Eval on the 110-task T_eval list (B1 = stock agent, B2' = file-path agent + guidance):
mw eval ... --max_round 50 --max-concurrency 3 --enable_mcp --enable_user_interaction
```
Exact harness wiring and per-arm commands are in
[`harness/README.md`](harness/README.md) and the setup block of
[`docs/qwen3_8b_res.md`](docs/qwen3_8b_res.md).

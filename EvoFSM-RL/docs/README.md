# docs/ — the paper trail

Everything that explains EvoFSM-RL in prose: the dataset/method specs the paper
derives from, the experiment reports behind every number, and the planning +
decision records that got us here.

## Layout

| Path | Role |
|---|---|
| `design/` | Paper-ready specs — `dataset.md` (apps, packages, task counts, source/Tier-B/Tier-C splits, T_adapt/T_eval protocol + citations) and `algorithm.md` (E-SPL background + the two-layer FSM method + the B1–B4 progression). |
| `results/` | Experiment reports — `experiments.md` (B1→B4 main results table), `b1_b2_static_baselines.md` (zero-shot vs static `L_C`), `b3_evolution.md` (online `L_C` evolution), plus the B4 diagnosis / index / data-format / ablation / audit deep-dives and `figures/`. |
| `plan/` | Working planning docs — `adr/` (architecture decision records: emulator path, base model, snapshot schema), `linear_tickets.md` (Epic→Story→Task breakdown), `spike_verl_migration.md` (verl/SkyRL spike). |
| `PROJECT_OVERVIEW.md` | One-read intro: motivation, contributions, technical architecture, current-progress table. Start here. |
| `method_report.md` | Self-contained walk-through of dataset + FSM + how each baseline B1–B4 is run; a synthesis of the design + results docs. |
| `project_progress_2026_05.md` | Dated tech-lead progress report (Chinese). |

## Where do I look for X

- **What apps / tasks / how are they split, and why?** → `design/dataset.md`
- **What is the method / algorithm?** → `design/algorithm.md` (or `method_report.md` for the whole picture in one read)
- **What are the experiment numbers?** → `results/experiments.md` (main table), then the per-rung reports
- **A high-level "what is this project?"** → `PROJECT_OVERVIEW.md`
- **Why was decision Y made (emulator, base model, schema)?** → `plan/adr/`
- **What's planned / ticket status?** → `plan/linear_tickets.md`

See `CLAUDE.md` in this directory for AI working context, and `../CLAUDE.md` for
the project-wide picture.

## Ablation results at a glance

Numbers so a reader doesn't have to open each report. All on the `T_eval`
protocol (35 templates × K=3 seeds = 105 episodes; Tier-B 54 + Tier-C 51),
binary success rate. Source of each row in the right column.

### Main ladder (B1 → B4, overall `T_eval`)

| Rung | Method | Overall | Source |
|---|---|:---:|---|
| **B1** | zero-shot Qwen3-VL-M3A (no FSM, no LoRA) | **38.6** | [`b1_b2_static_baselines.md`](results/b1_b2_static_baselines.md) §2 / [`experiments.md`](results/experiments.md) T1 |
| **B2** | + static category `L_C` (frozen) | **43.3** | [`b1_b2_static_baselines.md`](results/b1_b2_static_baselines.md) §2 / [`experiments.md`](results/experiments.md) T1 |
| **B3** | + online `L_C` evolution (Opus, frozen at eval) | **46.7–48.1** | [`b3_evolution.md`](results/b3_evolution.md) Part B / [`experiments.md`](results/experiments.md) T1 |
| **B4** | + joint Phase-3 LoRA (clean, **single-model**) | **48.1** | [`experiments.md`](results/experiments.md) T1 |

- B3 reads **46.7** in `experiments.md` Table 1 and **48.1** in `b3_evolution.md`
  Part B (seeds `[40,41,42]`, 50.5/105); the spread is K=3 seed variance, both
  within noise of each other.
- Clean single-model **B4 = 48.1 = B3**: under a clean ablation the Phase-3 LoRA
  step adds **nothing over the symbolic (FSM/`L_C`) half**.
- **The 52.9% in the paper draft / top-level README is NOT a single-model
  result.** It is a *provisional per-tier cherry-pick*: Tier-B taken from the
  v3-C checkpoint + Tier-C taken from the v3-B checkpoint — two different models
  stitched together (an oracle upper bound). The honest clean single-model B4 is
  **48.1**. (See `../CLAUDE.md` and `docs/CLAUDE.md` "Don't".)

### Phase-1 init & standalone-vs-Phase-3 ablation

Same B4 K=4 Phase-3 pipeline; only the π^pre LoRA init changes. "Standalone" =
frozen π^pre + per-app `L_C` (the `L_C` evolved by `b4_k4_v3binit`), with **no
Phase-3 LoRA training at all**.

| π^pre init | Phase-1 spec | Phase-3 (full) | Standalone (no Phase-3 LoRA) | Δ from Phase-3 LoRA | Source |
|---|---|:---:|:---:|:---:|---|
| pilot | 4 apps × 200 iter, no KL | **48.1** | 42.4 | +5.7 | [`experiments.md`](results/experiments.md) T2 |
| v3-B | 12 apps × 200 iter, KL β=0.05 | 43.3 | **49.0** | **−5.7** | [`experiments.md`](results/experiments.md) T2 |
| v3-C | 4 apps × 600 iter, KL β=0.05 | 45.7 | 48.6 | −2.9 | [`experiments.md`](results/experiments.md) T2 |

- **The key result**: **v3-B frozen standalone = 49.0 BEATS clean B4 = 48.1.**
  i.e. π^pre + `L_C` with *no Phase-3 LoRA training* outperforms the full joint
  method. Under clean ablation **Phase-3 LoRA TTA is null/negative** — the
  working contribution is the **symbolic (FSM/`L_C`) half**, not LoRA weight
  adaptation.
- Phase-3 LoRA only helps the *weakest* init (pilot, +5.7); on the stronger
  v3-B / v3-C inits it actively hurts.
- Also tried: **K=2 vs K=4** (K=4 +5.9 pp on Tier-C cold-start bootstrap;
  `experiments.md` T7) and a **dense-reward** variant (**collapsed the LoRA**,
  subset 40.5 not comparable; `experiments.md` T5).

For the full per-app / per-template breakdowns and statistical caveats, see the
per-rung reports linked above.

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

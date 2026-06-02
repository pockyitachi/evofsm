# Dataset design — AndroidWorld-Plus

This document is the source-of-truth for the benchmark split used by every
EvoFSM-RL experiment. The machine-readable form is `configs/splits.yaml`
(version `v1.0-K1`); this document is the human-readable form and must be
kept in sync with it.

## 1. Overview

AndroidWorld-Plus is a 25-app, 192-template, 6-Play-Store-category Android
benchmark built on two components:

- **AndroidWorld** (Rawles et al., 2024) — 19 active primary-task apps
  covering 114 app-attributable templates plus 2 generic/composite templates
  (`OpenApp`, `SaveCopyOfReceipt`).
- **Plus additions** — 6 apps imported from BMOCA (Calculator, Snapseed,
  Wikipedia) and AndroidLab (Bluecoins, Maps.me, Pi Music) contributing 78
  templates.

Totals: **25 active primary-task apps**, **194 templates** in
`task_metadata.json` overall, of which **192 are app-attributable** (the
remaining 2 are generic/composite and excluded from the per-app split).

Categories are **Google Play Store categories** taken directly from each
app's Play listing. The Play taxonomy is used (rather than an invented
clustering) because it is externally validated, cite-able against Android
Control (Li et al., 2024), and provides a natural equivalence class for the
"category-generic" LAYER 2 knowledge that EvoFSM-RL transfers.

The 25 apps span 12 active Play categories. Six of those categories have
multiple apps and are eligible to anchor the source pool; the other six have
a single app each and necessarily form the Tier-C far-transfer pool.

## 2. Source pool (12 apps, 96 templates)

The source pool is used for Phase-1 pretraining: per-app static FSM
extraction (Story 2.2) and per-category `L_C` synthesis (Story 2.3). All
96 templates are used; no target-set split is carved out of the source
pool. Seeds
`[30, 31, 32, 33, 34]` (K=5) give 480 source-pool episodes.

| App | Category | # templates |
|---|---|---:|
| markor | Productivity | 14 |
| tasks_org | Productivity | 6 |
| joplin | Productivity | 4 |
| calculator | Tools | 19 |
| clock | Tools | 3 |
| files | Tools | 2 |
| bluecoins | Finance | 15 |
| pi_music | Music & Audio | 12 |
| audio_recorder | Music & Audio | 2 |
| snapseed | Photography | 11 |
| simple_sms_messenger | Communication | 6 |
| contacts | Communication | 2 |
| **Total** | **6 categories** | **96** |

Six Play Store categories are represented: Productivity (3 apps),
Tools (3), Finance (1), Music & Audio (2), Photography (1), Communication
(2). These six are the categories for which a matching `L_C` will be
available at test-time hand-off.

## 3. Tier-B target apps (6 apps, 50 templates)

The Tier-B target pool is **near-transfer**: every app's Play category
is represented in the source pool, so a matching `L_C` exists and can seed
the target-app FSM. Total templates in the pool: 50 (32 `T_adapt` + 18
`T_eval`).

| App | Category | # T_adapt | # T_eval |
|---|---|---:|---:|
| simple_calendar_pro | Productivity | 11 | 6 |
| system_settings | Tools | 9 | 6 |
| pro_expense | Finance | 6 | 3 |
| retro_music | Music & Audio | 3 | 1 |
| camera | Photography | 1 | 1 |
| chrome | Communication | 2 | 1 |
| **Total** | | **32** | **18** |

Each category in this table matches a category in the source pool
(Productivity, Tools, Finance, Music & Audio, Photography, Communication).
Near-transfer therefore tests whether category-generic knowledge learned
from one source-pool app transfers to a target app of the same category.

## 4. Tier-C target apps (7 apps, 46 templates)

The Tier-C target pool is **far-transfer**: no app's Play category is
represented in the source pool, so no matching `L_C` exists. Under B3
(evolved `L_C`), Tier-C apps fall back to the B1 (no-injection) path
because `resolve_l_c_for_app()` returns `None`. Total templates: 46
(29 `T_adapt` + 17 `T_eval`).

| App | Category | # T_adapt | # T_eval |
|---|---|---:|---:|
| maps_me | Maps & Navigation | 9 | 6 |
| osmand | Maps & Navigation | 2 | 1 |
| broccoli | Food & Drink | 8 | 5 |
| opentracks | Health & Fitness | 4 | 2 |
| wikipedia | Books & Reference | 4 | 2 |
| vlc | Video Players & Editors | 1 | 1 |
| simple_draw_pro | Art & Design | 1 | 0 |
| **Total** | | **29** | **17** |

`simple_draw_pro` has N=1 and is assigned entirely to `T_adapt`;
its `T_eval` is empty and it is excluded from the Tier-C aggregate
`T_eval` SR. This is flagged explicitly in `splits.yaml` and in the
splits protocol notes.

## 5. `T_adapt` / `T_eval` split protocol

Within every target app, templates are partitioned into two
**template-disjoint** sets (not just seed-disjoint — the task-template
classes themselves are different). The app is the same across the two
sets; only the templates differ.

- **`T_adapt`** — used for online adaptation during B3 evolution and
  B4 joint TTA. FSM mutation and LoRA updates happen on `T_adapt` only.
- **`T_eval`** — used for a single, frozen evaluation pass after
  adaptation has converged. No updates occur during `T_eval`. This is
  the **headline number** reported per tier.

### Partition rule

Deterministic alphabetical **60/40 split** per app:

- N ≥ 3: round 60% of the sorted template list into `T_adapt`, the
  remainder into `T_eval`.
- N = 2: split 1/1.
- N = 1: entire single template into `T_adapt`; `T_eval` empty
  (degenerate case; currently only `simple_draw_pro`).

Alphabetical was chosen over stratified-by-difficulty because most apps
have N ≤ 6 templates, so stratification reduces to enumeration, and the
alphabetical convention is fully reproducible without a split-sampling
script.

### Seed protocol

- `T_adapt` seeds: `[30, 31, 32, 33, 34]` (K=5). Shared with Phase-1 so
  adaptation-time distributions align with pretraining distributions.
- `T_eval` seeds: `[40, 41, 42]` (K=3). **Disjoint from `T_adapt`
  seeds** so that even if a `T_eval` template shared parameter values
  with a `T_adapt` template, the realized initial state would differ.

Bumping `K_eval` 3 → 5 is reserved for the camera-ready if reviewers
press on CI width (see `configs/multi_seed_config.yaml`).

## 6. Why this design

The split decomposes into three decisions, each with a citation anchor
in the project's reference list:

- **Source apps vs target apps.** Standard train/test separation at
  the app level. Apps in the target pool are never part of Phase-1
  pretraining in any form; no gradient from their trajectories flows
  through the pretrained artifact. (Earlier drafts called these
  "held-out" apps; we renamed to "target" once test-time adaptation
  on the same apps was introduced, since "held-out" wrongly implied
  the apps were never touched.)
- **Tier-B vs Tier-C.** Measuring near-transfer and far-transfer
  separately is more informative than a single conflated transfer
  number — the two regimes have qualitatively different failure modes
  and we expect different method gains. Structurally this mirrors
  Android Control's `unseen-app` (= Tier-B) vs `unseen-category` (=
  Tier-C) split (Li et al., 2024).
- **`T_adapt` vs `T_eval`, template-disjoint.** Following the
  support/query protocol of meta-learning (Finn et al., 2017 MAML) and
  the train-level / test-level protocol of generalization-focused RL
  benchmarks (Cobbe et al., 2020 Procgen). Template-disjointness —
  rather than just seed-disjointness — is what makes `T_eval` a
  measurement of adaptation generality rather than parameter
  robustness on memorized templates.

The paper-ready composite citation in the methods section is:
*"Within each target app, templates are partitioned into a disjoint
adaptation set `T_adapt` and evaluation set `T_eval`, mirroring the
support/query protocol of meta-learning (Finn et al., 2017) and the
train-level / test-level protocol of generalization-focused RL
benchmarks (Cobbe et al., 2020). Template disjointness ensures that
`T_eval` measures within-app generalization of the adaptation
procedure rather than parameter robustness on memorized tasks."*

Anticipated reviewer objections are addressed by the same set of
citations: TTA as a paradigm (Sun et al., 2020; Wang et al., 2021);
template-disjoint vs seed-disjoint (Li et al., 2024 — Android
Control's `unseen-task` analog).

## 7. Category mapping (all 25 apps)

| App | Play Store category | Pool |
|---|---|---|
| markor | Productivity | source |
| tasks_org | Productivity | source |
| joplin | Productivity | source |
| simple_calendar_pro | Productivity | tier_B |
| calculator | Tools | source |
| clock | Tools | source |
| files | Tools | source |
| system_settings | Tools | tier_B |
| bluecoins | Finance | source |
| pro_expense | Finance | tier_B |
| audio_recorder | Music & Audio | source |
| pi_music | Music & Audio | source |
| retro_music | Music & Audio | tier_B |
| snapseed | Photography | source |
| camera | Photography | tier_B |
| simple_sms_messenger | Communication | source |
| contacts | Communication | source |
| chrome | Communication | tier_B |
| maps_me | Maps & Navigation | tier_C |
| osmand | Maps & Navigation | tier_C |
| broccoli | Food & Drink | tier_C |
| opentracks | Health & Fitness | tier_C |
| wikipedia | Books & Reference | tier_C |
| vlc | Video Players & Editors | tier_C |
| simple_draw_pro | Art & Design | tier_C |

(Chrome's Play Store listing is in Communication; that is the assignment
used by the split. The listing category of a bundled Android app is the
one Google publishes on its Play page, not the tag used inside the
AndroidWorld registry.)

## 8. Data scale per experiment

| Experiment | Episodes |
|---|---:|
| K=1 vanilla leaderboard-style sweep (all 192 templates) | 192 |
| K=3 `T_eval` sweep (B1, B2, or B3), Tier-B + Tier-C apps with non-empty T_eval: 35 templates × 3 seeds | 105 per arm |
| K=5 source-pool Phase-1 pretraining (96 templates × 5 seeds) | 480 |
| K=5 Tier-B `T_adapt` (32 templates × 5 seeds) | 160 |
| K=3 Tier-B `T_eval` (18 templates × 3 seeds) | 54 |
| K=5 Tier-C `T_adapt` (29 templates × 5 seeds) | 145 |
| K=3 Tier-C `T_eval` (17 templates × 3 seeds) | 51 |
| **Grand total, full protocol** | **890** |

For the B1 / B2 / B3 headline numbers reported in
`docs/results/b1_b2_static_baselines.md` and
`docs/results/b3_evolution.md`, the evaluation unit is the K=3 `T_eval`
sweep: 52 templates with non-empty T_eval × 3 seeds = 156 episodes per
arm under the full protocol, or 105 per arm when counting only the 35
templates currently instrumented.

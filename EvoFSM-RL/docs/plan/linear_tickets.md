# EvoFSM-RL — Linear Tickets (Epic → Story → Task)

Project: **EvoFSM-RL** (method + benchmark for app-specific adaptation in GUI agents)

Each Epic below maps to one Phase of the implementation roadmap (§4 of `project_plan.md`). No timelines attached — order is dependency-driven.

Linear tip: paste each Epic's block as a new issue with label `Epic`, then paste each Story as a sub-issue (label `Story`), then each Task as a sub-issue of its Story (label `Task`).

---

## EPIC 0 — Benchmark Design & Project Setup

**Labels:** `epic`, `benchmark`, `setup`
**Goal:** Freeze the benchmark split, source-pool / target-pool allocation, and task taxonomy so that all downstream epics operate on a stable contract.

**Definition of done**
- App taxonomy (Play Store category, sourced from Google Play / Android Control) checked in.
- Source pool `A_src`, Tier-B (near-transfer) and Tier-C (far-transfer) target pools finalized.
- Target-app `T_adapt` / `T_eval` split procedure specified, deterministic, and template-disjoint.

---

### Story 0.1 — App taxonomy & per-task labels

**Description:** Adopt the **Play Store category** scheme as the app-level taxonomy (12 categories actually used across our 25 active apps), citing Android Control (Li et al. 2024). For task-level labels, reuse the existing 15 multi-label `tags` from `task_metadata.json` (e.g. `data_entry`, `screen_reading`, `search`) and collapse them to a single `task_type` per task via a fixed priority rule (see `docs/taxonomy.md` §4.2).
**Labels:** `story`, `benchmark`
**Acceptance criteria:**
- `evofsm_rl/taxonomy.py` exposes `play_category_of(app)` and `task_type_of(task_name)`.
- Every registered AndroidWorld + Plus task resolves to exactly one `task_type`.
- 1-page memo in `EvoFSM-RL/docs/taxonomy.md` referencing Android Control.
- `EvoFSM-RL/configs/task_categories.csv` lists every task with `(task_name, app, play_category, task_type)`.

- **Task 0.1.1** — Draft Play-Store-category taxonomy with Android Control cross-ref. [S] ✅
- **Task 0.1.2** — Generate per-task `task_type` CSV via priority rule. [S]
- **Task 0.1.3** — Implement `play_category_of` / `task_type_of` helpers with unit tests. [S]
- **Task 0.1.4** — Review memo with team and merge. [XS]

---

### Story 0.2 — Source pool / Tier-B / Tier-C allocation

**Description:** Partition the 25 active apps into a source pool `A_src`, a Tier-B target pool (Play category present in `A_src`), and a Tier-C target pool (Play category absent from `A_src`). The split is fixed in `EvoFSM-RL/configs/splits.yaml` and is the benchmark's primary contract for the paper.
**Labels:** `story`, `benchmark`
**Acceptance criteria:**
- `evofsm_rl/splits.py` exposes `get_source_pool()`, `get_tier_B_apps()`, `get_tier_C_apps()` and returns deterministically from `splits.yaml`.
- Every target app falls in exactly one of Tier-B / Tier-C.
- Rationale documented in `EvoFSM-RL/docs/splits_protocol.md`.

- **Task 0.2.1** — Propose source-pool candidates (≥1 app per source category). [S] ✅
- **Task 0.2.2** — Assign target apps to Tier-B vs Tier-C by category presence in `A_src`. [S] ✅
- **Task 0.2.3** — Implement `splits.py` loader (read `splits.yaml`). [S]
- **Task 0.2.4** — Document split rationale (in `splits_protocol.md`). [XS] ✅

---

### Story 0.3 — Target-app T_adapt / T_eval split

**Description:** Within each target app, partition templates into `T_adapt` and `T_eval` (template-disjoint). Baseline ratio: **60/40 alphabetical** (deterministic by template name; chosen for reproducibility over stratification given many apps have N≤6 templates). Degenerate handling: N=1 → adapt-only; N=2 → 1/1. Multi-seed support via `multi_seed_config.yaml` (K_adapt=5, K_eval=3). Citations supporting the protocol live in `docs/citations.md`.
**Labels:** `story`, `benchmark`
**Acceptance criteria:**
- `evofsm_rl/splits.py::get_T_adapt(app)` and `get_T_eval(app)` return template lists per `splits.yaml`.
- Unit test: every target app has `T_adapt ∩ T_eval = ∅`.
- Unit test: `splits.yaml` round-trips deterministically.

- **Task 0.3.1** — Implement deterministic 60/40 alphabetical split (already baked into `splits.yaml`); expose loader. [S]
- **Task 0.3.2** — Add unit tests for template-disjointness + determinism. [XS]

---

## EPIC 1 — Infrastructure

**Labels:** `epic`, `infrastructure`
**Goal:** Stand up the runnable substrate — emulator, base model, single-task rollout, snapshot I/O, baseline eval — so Epic 2+ can plug methods in and immediately get numbers. All three architectural decisions are pinned in ADRs (`docs/adr/`); this Epic executes them.
**Dependencies:** Epic 0 (taxonomy + splits) ✅

**Definition of done**
- `scripts/bootstrap_avd.sh` sets up an AVD on dev (arm64) and cloud (x86_64); smoke test runs ≥10 random templates on both.
- Qwen3-VL-8B loads locally (Mac 8-bit) and on A100 (bf16); can emit an action given one screenshot.
- `evofsm_rl/snapshots.py` implements ADR-003 save/load/validate, with unit tests.
- **Baseline eval artifact:** vanilla Qwen3-VL-8B + AndroidWorld M3A agent, K=1, full 192 app-attributable tasks → CSV in `/results/baseline_zero_shot/`. This is Table 1 row 1 of the paper.

**Scope cuts** (explicitly deferred, don't do in Epic 1)
- Multi-emulator parallelism → Epic 2 if single-emulator eval wall-clock hurts.
- CI (ruff / mypy / GitHub Actions) → backlog; research iteration speed > lint pedantry for now.
- Evolution / FSM / LoRA / GRPO → Epic 2+.

---

### Story 1.1 — Emulator bootstrap + smoke test [per ADR-001]

**Labels:** `story`, `infrastructure`, `emulator`
**Description:** Implement the native-AVD path decided in ADR-001 — arm64 image on dev Mac, x86_64 image on rented A100 box, same `Pixel_6_API_33 google_apis` contract on both. Docker is out of scope per ADR.
**Acceptance criteria:**
- `scripts/bootstrap_avd.sh {arm64|x86_64}` installs the system image and creates the AVD idempotently.
- `tests/test_smoke_emulator.py` boots the AVD, runs 10 random AndroidWorld templates via `env_launcher.load_and_setup_env()`, and asserts `TaskEval.is_successful()` returns a bool (not an exception). No success-rate assertion — this is a wiring test, not a quality test.
- Smoke test passes on both arm64 (Mac) and x86_64 (A100) hosts.

- **Task 1.1.1** — Write `scripts/bootstrap_avd.sh` parameterized by arch. [S]
- **Task 1.1.2** — Wrapper `evofsm_rl/env/harness.py::run_template(template_cls, seed)` returning `(trajectory, success)`. [M]
- **Task 1.1.3** — `test_smoke_emulator.py` (arm64 on Mac). [S]
- **Task 1.1.4** — Run smoke test on rented A100 box (x86_64). [S]

---

### Story 1.2 — Base model loader [per ADR-002]

**Labels:** `story`, `infrastructure`, `model`
**Description:** Load `Qwen/Qwen3-VL-8B-Instruct` — 8-bit on Mac, bf16 on A100. No 30B-A3B path, no Tinker — per ADR-002.
**Acceptance criteria:**
- `configs/model.yaml` pins `{name, hf_revision, dtype_mac, dtype_cuda}`.
- `evofsm_rl/model/loader.py::load_base_model(device)` returns `(model, processor)` with a deterministic revision check (SHA match or fail loud).
- Smoke: `python -m scripts.model_smoke` loads the model, takes one AndroidWorld screenshot, runs forward, prints first-token log-prob. Fits in 24 GB on Mac with the emulator running.

- **Task 1.2.1** — `configs/model.yaml` + revision pin. [XS]
- **Task 1.2.2** — Loader with device-aware dtype/quant config. [M]
- **Task 1.2.3** — `scripts/model_smoke.py` sanity run. [S]
- **Task 1.2.4** — Memory-budget report (Mac peak RSS with emulator concurrent). [S]

---

### Story 1.3 — Snapshot module [per ADR-003]

**Labels:** `story`, `infrastructure`, `snapshot`
**Description:** Implement the frozen-checkpoint format from ADR-003. Used later by Epic 2+'s adapt loop (writer) and by every eval in the paper (reader). Epic 1 only ships the I/O primitive — no one calls it in anger yet.
**Acceptance criteria:**
- `evofsm_rl/snapshots.py` exposes `save(dir, artifacts, metadata)`, `load(dir) → Snapshot`, `validate(Snapshot)`, `migrate(snapshot, target_schema_version)`.
- All seven invariants from ADR-003 enforced by `validate()` with clear error messages; tests cover each violation path.
- Round-trip test: `save` then `load` yields bit-identical metadata and recovers FSM / L_C row / LoRA weight equality.

- **Task 1.3.1** — Schema dataclasses + JSON (de)serialization. [S]
- **Task 1.3.2** — LoRA adapter dir round-trip (stub PEFT adapter is fine). [S]
- **Task 1.3.3** — `validate()` implementing all invariants with pytest coverage. [M]
- **Task 1.3.4** — `migrate()` stub (v1→v1 no-op); placeholder for future bumps. [XS]

---

### Story 1.4 — Zero-shot baseline eval (Table 1 row 1)

**Labels:** `story`, `eval`, `baseline`
**Description:** Run the vanilla AndroidWorld M3A agent + Qwen3-VL-8B on all 192 app-attributable tasks, K=1 (`task_random_seed=30`). This is the "no method" floor — every later number in the paper is measured relative to this row. Also shakes out every pipeline bug before we add FSM/LoRA complexity.
**Acceptance criteria:**
- `scripts/run_zero_shot_eval.py` iterates the 192 tasks, writes one row per `(task, seed)` to `/results/baseline_zero_shot/runs.csv` with columns `task_name, app, tier, play_category, task_type, seed, success, n_steps, wall_s`.
- Aggregate script emits `summary.csv`: per-app SR, per-category SR, overall SR, with 95% Wilson CIs.
- Compared against AndroidWorld leaderboard published numbers (vanilla M3A + Qwen-family) — our vanilla-task SR sits within a plausible band (flag if off by >10pp for sanity).

- **Task 1.4.1** — `run_zero_shot_eval.py` driver (uses Story 1.1 harness + Story 1.2 model). [M]
- **Task 1.4.2** — Aggregation + 95% CI script. [S]
- **Task 1.4.3** — Cross-check vanilla tasks vs AndroidWorld leaderboard; sanity memo. [S]
- **Task 1.4.4** — Commit `results/baseline_zero_shot/` + one-page findings note. [S]

---

## EPIC 2 — Knowledge Extraction & Static Baselines

**Labels:** `epic`, `knowledge-extraction`, `baselines`
**Goal:** Produce `{F^0_a}` for the 25 active apps via T3A + LLM synthesis, and ship Baselines B1 (zero-shot) and B2 (static FSM) adaptation-curve artifacts (flat lines) as the lower bounds for the main results.
**Dependencies:** Epic 1

**Definition of done**
- `{F^0_a}` FSMs exist in the repo for every app, linted for two-layer schema compliance.
- B1 and B2 have produced adaptation curves on both near-transfer and far-transfer target pools.

---

### Story 2.1 — T3A trajectory collection

**Labels:** `story`, `knowledge-extraction`
**Description:** Run GPT-4o-backed T3A on source-app training tasks to collect trajectories. Concern (§3 open Q): T3A ≈ 30% SR, so trajectory quality may be low — mitigation is LLM synthesis that tolerates this.
**Acceptance criteria:**
- Trajectories stored in canonical JSONL format (obs, action, reward, timestamp).
- SR of T3A reported per app for sanity.

- **Task 2.1.1** — Wire GPT-4o T3A agent into our harness. [M]
- **Task 2.1.2** — Collect trajectories for all source-pool apps (training task split). [L]
- **Task 2.1.3** — Per-app T3A SR report. [S]

---

### Story 2.2 — FSM builder (LLM synthesis)

**Labels:** `story`, `knowledge-extraction`
**Description:** Given per-app trajectories, synthesize a static FSM `F^0_a` with the two-layer schema (LAYER1 APP_SPECIFIC / LAYER2 GENERIC by category).
**Acceptance criteria:**
- `builder.py::build_fsm(trajectories, app) → FSM` produces valid two-layer JSON.
- Linter rejects LAYER2 blocks that contain app names or resource-id substrings.
- All 25 active apps have an `F^0_a` in `artifacts/static_fsms/`.

- **Task 2.2.1** — FSM schema (two-layer) + JSON serializer. [M]
- **Task 2.2.2** — LLM synthesis prompt for `F^0_a`. [M]
- **Task 2.2.3** — LAYER2 linter (no app-specific strings). [S]
- **Task 2.2.4** — Run builder on all 25 apps. [S]
- **Task 2.2.5** — Manual spot-check 5 FSMs; file quality memo. [S]

---

### Story 2.3 — B1 (zero-shot) adaptation-curve runs

**Labels:** `story`, `baselines`, `eval`
**Acceptance criteria:**
- B1 curves (flat lines) produced on all Tier-B and Tier-C target apps.
- Results logged in `/results/B1/`.

- **Task 2.3.1** — B1 runner (no FSM, no weight update). [S]
- **Task 2.3.2** — Run B1 across all target apps × 3 seeds. [M]

---

### Story 2.4 — B2 (static FSM) adaptation-curve runs

**Labels:** `story`, `baselines`, `eval`
**Acceptance criteria:**
- B2 curves (flat lines, higher than B1) on all target apps.
- Per-target comparison vs B1 in `/results/B2/`.

- **Task 2.4.1** — B2 runner (FSM = `F^0_{a_tgt}`, frozen weights). [S]
- **Task 2.4.2** — Run B2 across target apps × 3 seeds. [M]
- **Task 2.4.3** — B1 vs B2 comparison plot. [XS]

---

## EPIC 3 — Self-Evolving FSM, Single-App Loop

**Labels:** `epic`, `evolution`, `baselines`
**Goal:** Implement the single-app EvoFSM loop from §3.2 (TrueSkill + layered reflection + mutation + within-app crossover) and produce B3 adaptation curves on target apps (evolution only; weights frozen at base).
**Dependencies:** Epic 2

**Definition of done**
- Single-app loop converges on at least one source app within budget.
- B3 curves produced on all target apps; should visibly rise above B2.

---

### Story 3.1 — TrueSkill + population manager

**Labels:** `story`, `evolution`
**Acceptance criteria:**
- `PopulationManager` supports append, select-top-M, TrueSkill updates.
- Unit tests for TrueSkill math vs reference library.

- **Task 3.1.1** — TrueSkill wrapper (reuse existing lib). [S]
- **Task 3.1.2** — Population manager with sliding window `K`. [M]
- **Task 3.1.3** — Unit tests for update/selection. [S]

---

### Story 3.2 — Layered reflection prompt + mutation operator

**Labels:** `story`, `evolution`, `prompt-design`
**Description:** Mutation prompt must produce FSM diffs tagged `@LAYER1` / `@LAYER2`. Critical for transfer — lint enforces tag presence.
**Acceptance criteria:**
- `mutate(F, trajectories, category) → F'` returns valid two-layer diff.
- Lint check in CI: any mutation diff without proper tags is rejected.

- **Task 3.2.1** — Design layered reflection prompt. [M]
- **Task 3.2.2** — `apply_fsm_diff` implementation. [M]
- **Task 3.2.3** — Diff linter. [S]
- **Task 3.2.4** — Manual validation: 20 mutations, rate separation quality. [S]

---

### Story 3.3 — Within-app crossover operator

**Labels:** `story`, `evolution`
**Acceptance criteria:**
- `fsm_crossover(past_tasks, R, Ω, π_θ, π_ref) → F_cross`.
- Ablation hook: `p_crossover` configurable; 0 disables.

- **Task 3.3.1** — Crossover prompt design. [M]
- **Task 3.3.2** — Implementation + config. [M]

---

### Story 3.4 — Single-app loop wiring

**Labels:** `story`, `evolution`
**Acceptance criteria:**
- `run_single_app(a, π_θ_frozen, T_a, iterations) → S_a`.
- Logs μ, σ, V_i per iteration to wandb / tensorboard.

- **Task 3.4.1** — Main loop glue code. [M]
- **Task 3.4.2** — Logging + checkpointing. [S]
- **Task 3.4.3** — Smoke test: converges on Bluecoins in ≤ 30 iterations. [S]

---

### Story 3.5 — B3 adaptation-curve runs (evolution only)

**Labels:** `story`, `baselines`, `eval`
**Acceptance criteria:**
- B3 runner uses frozen base weights (no LoRA) and runs the single-app loop during Phase 3 on `T_adapt`.
- Curves produced on all target apps (near + far) × 3 seeds.
- B3 curve visibly rises above B2 on at least some targets.

- **Task 3.5.1** — B3 runner: FSM-only Phase 3. [S]
- **Task 3.5.2** — Run B3 × all target apps × 3 seeds. [L]
- **Task 3.5.3** — Compare B2 vs B3 slopes. [S]

---

## EPIC 4 — Full EvoFSM-RL

**Labels:** `epic`, `rl`, `baselines`
**Goal:** Add GRPO + LoRA weight updates to the single-app loop (Baseline 4 — core contribution). Produce B4 adaptation curves under the near-transfer condition on at least one near-transfer target (full multi-source pretraining comes in Epic 5).
**Dependencies:** Epic 3

**Definition of done**
- GRPO + LoRA on top of B3 runs end-to-end.
- B4 curve visibly exceeds B3 on at least one target.
- Path A (LoRA frozen) consistency check passes (≈ B3).

---

### Story 4.1 — LoRA + GRPO plumbing

**Labels:** `story`, `rl`
**Acceptance criteria:**
- LoRA rank configurable (16/32 default).
- GRPO update implemented with β=0, group advantage normalization.

- **Task 4.1.1** — Attach LoRA adapters to Qwen3-VL agent. [M]
- **Task 4.1.2** — GRPO update op (TRL-based or custom; Tinker rejected per Epic 1 hardware decision). [L]
- **Task 4.1.3** — Reward function (success + efficiency + exploration). [M]
- **Task 4.1.4** — Path A / B / C update-frequency knob. [S]

---

### Story 4.2 — Evolution + RL joint loop

**Labels:** `story`, `rl`, `evolution`
**Acceptance criteria:**
- `run_evofsm_rl(a, π_θ, T_a, iterations)` updates θ and `S_a` together.
- Logs RL loss, FSM mutation count, μ trajectory.

- **Task 4.2.1** — Integrate GRPO into the §3.4 loop. [M]
- **Task 4.2.2** — End-to-end smoke test on Bluecoins. [S]

---

### Story 4.3 — B4 runs + Path consistency check

**Labels:** `story`, `baselines`, `rl`
**Acceptance criteria:**
- B4 curves produced on a pilot near-transfer target.
- Path A (LoRA frozen) curve matches B3 within noise — consistency check.
- Path C default confirmed via ablation.

- **Task 4.3.1** — Path C run on pilot target. [M]
- **Task 4.3.2** — Path A consistency run. [S]
- **Task 4.3.3** — Path B (every-step) ablation. [M]

---

### Story 4.4 — Component ablations (LoRA rank, group size, reward)

**Labels:** `story`, `rl`, `ablation`
**Acceptance criteria:**
- LoRA rank sweep 8/16/32/64 on one app.
- GRPO group size sweep.
- Evolution-only / RL-only / joint comparison.

- **Task 4.4.1** — LoRA rank sweep. [M]
- **Task 4.4.2** — GRPO group size sweep. [S]
- **Task 4.4.3** — Evolution-only vs RL-only vs joint. [M]

---

## EPIC 5 — Unified Pretraining + Transfer Experiments

**Labels:** `epic`, `transfer`, `pretraining`
**Goal:** Run the single unified Phase-1 multi-source joint training (one run, produces `(π^pre_θ, {S^pre_a}, {L_C})`), then evaluate all four baselines under both the near-transfer and far-transfer conditions. This produces the paper's main figure.
**Dependencies:** Epic 4, Epic 0

**Definition of done**
- One unified pretrained artifact shipped and frozen.
- Two-panel main figure (near / far × 4 baselines) produced with K_adapt=5 / K_eval=3 seeds per `multi_seed_config.yaml`.
- Hand-off ablation table filled.

---

### Story 5.1 — Cross-app crossover operator

**Labels:** `story`, `transfer`, `evolution`
**Description:** New operator (§3.3 extension) — collect per-app champions' LAYER2 per category, synthesize `F_C_abstract`, store in `L_C`. Never push back into any `S_a`.
**Acceptance criteria:**
- `CrossAppCrossover(A_src, {S_a}, C, π_ref) → F_C_abstract`.
- `L_C` keyed by (category, timestamp, rating).
- Geometric-schedule trigger (every `2^n` iterations) by default.

- **Task 5.1.1** — Cross-app crossover prompt design. [M]
- **Task 5.1.2** — Operator implementation. [M]
- **Task 5.1.3** — `L_C` library + retrieval policy (most-recent highest-rated). [S]
- **Task 5.1.4** — Linter: `F_C_abstract` contains no app-specific strings. [S]

---

### Story 5.2 — Multi-source joint training loop

**Labels:** `story`, `pretraining`
**Acceptance criteria:**
- `run_multisource(A_src, T_all, iterations)` updates shared LoRA θ and per-app `S_a`, with cross-app crossover triggering.
- Per-source-app validation monitored; if any regresses past threshold, halt and alert.

- **Task 5.2.1** — Sampling: app + task + category per iteration. [S]
- **Task 5.2.2** — Shared-LoRA gradient accumulation across apps. [M]
- **Task 5.2.3** — Per-app validation monitor + alarm. [M]
- **Task 5.2.4** — Run Phase-1 pretraining end-to-end; produce artifact. [L]

---

### Story 5.3 — Deployment-time hand-off (near / far branch)

**Labels:** `story`, `transfer`
**Acceptance criteria:**
- `deploy(a_tgt, artifact)` branches on `play_category_of(a_tgt) ∈ source_pool_categories` → near-transfer (Tier-B) or far-transfer (Tier-C) hand-off.
- Near branch: translate nearest same-category source LAYER2.
- Far branch: specialize `F_C_abstract` from `L_{C_tgt}`.
- Cold-start exploration (5–10 rollouts of π^pre_θ) wired in.

- **Task 5.3.1** — LAYER2-translate prompt (near). [M]
- **Task 5.3.2** — LAYER2-ground prompt (far, from `F_C_abstract`). [M]
- **Task 5.3.3** — `deploy()` branching logic. [S]
- **Task 5.3.4** — Cold-start exploration helper. [S]

---

### Story 5.4 — Run the 4 × 2 main experiment matrix

**Labels:** `story`, `eval`, `transfer`
**Acceptance criteria:**
- For each baseline B1–B4 × each condition (near, far) × each target app × 3 seeds → adaptation curve.
- Two-panel main figure generated.

- **Task 5.4.1** — Runner script: all baselines × both conditions. [M]
- **Task 5.4.2** — Compute: schedule + queue. [M]
- **Task 5.4.3** — Aggregate + plot main figure. [S]

---

### Story 5.5 — Hand-off ablations (H-none / H-weights-only / H-single-source / H-stale-L_C)

**Labels:** `story`, `ablation`, `transfer`
**Description:** All reuse the same pretrained artifact — no new training runs.
**Acceptance criteria:**
- Each hand-off variant produces an adaptation curve on the same target pool.
- Table filled in results doc.

- **Task 5.5.1** — H-none runs (static FSM only). [S]
- **Task 5.5.2** — H-weights-only runs. [S]
- **Task 5.5.3** — H-single-source (near only) runs. [S]
- **Task 5.5.4** — H-stale-L_C (far only) runs. [S]

---

## EPIC 6 — Analysis & Paper

**Labels:** `epic`, `analysis`, `paper`
**Goal:** Aggregate all results, produce tables/figures, write the paper (target: NeurIPS Dataset Track or equivalent), and prepare the code release.
**Dependencies:** Epic 5

**Definition of done**
- Paper draft complete through Discussion.
- Anonymized code + benchmark + results bundle ready for submission.

---

### Story 6.1 — Aggregate results and compute metrics

**Labels:** `story`, `analysis`
**Acceptance criteria:**
- Metric sheet covers `SR@0`, `SR@K_max`, slope, AUC, Δ vs zero-shot, knowledge transfer ratio, adaptation cost.
- CSV + parquet artifacts.

- **Task 6.1.1** — Metric computation module. [S]
- **Task 6.1.2** — Per-condition × per-baseline aggregation. [S]
- **Task 6.1.3** — Variance across seeds + confidence bands. [S]

---

### Story 6.2 — Figures & tables

**Labels:** `story`, `analysis`, `paper`
**Acceptance criteria:**
- Main figure: 2-panel × 4-curve adaptation plot.
- Secondary figures: source-category → target-app heatmap (near), FSM-size-over-k, LAYER2 edit-distance.
- All tables as LaTeX + matplotlib figures as PDF.

- **Task 6.2.1** — Main figure (near / far × 4 baselines). [S]
- **Task 6.2.2** — Near-transfer source→target heatmap. [S]
- **Task 6.2.3** — FSM convergence diagnostics plot. [S]
- **Task 6.2.4** — Hand-off ablation table. [XS]

---

### Story 6.3 — Paper writing

**Labels:** `story`, `paper`
**Acceptance criteria:**
- Draft sections 1–6 complete.
- Related-work gap filed vs Agent-SAMA / ActionEngine / SPlanner.

- **Task 6.3.1** — Abstract + Introduction. [M]
- **Task 6.3.2** — Related Work. [M]
- **Task 6.3.3** — Method (EvoFSM-RL + unified framing). [L]
- **Task 6.3.4** — Benchmark section (taxonomy, splits, metrics). [M]
- **Task 6.3.5** — Experiments (with 2-panel narrative). [L]
- **Task 6.3.6** — Discussion + Conclusion. [M]
- **Task 6.3.7** — Appendix (hyperparameters, streaming sanity check, hand-off ablations). [M]

---

### Story 6.4 — Code, benchmark, and results release prep

**Labels:** `story`, `release`
**Acceptance criteria:**
- Anonymized fork created.
- README + reproduction recipe.
- Benchmark data bundle (splits + static FSMs + L_C artifact).

- **Task 6.4.1** — Repo cleanup + anonymization. [M]
- **Task 6.4.2** — Reproduction README (one-command pretraining + one-command eval). [M]
- **Task 6.4.3** — Release artifact bundle (HF or Zenodo). [S]

---

## Cross-cutting / Risk-driven Stories (file under relevant Epic as discovered)

These are risks from §3.6.5 / §8.6; create as dedicated stories in the relevant Epic if they fire.

- **R1: LAYER2 reflection brittleness.** Add strict linter + reject diffs with app-specific strings. (Epic 3, Story 3.2)
- **R2: Visual negative transfer (near condition).** B2 may be below zero-shot. Report separately per condition. (Epic 5, Story 5.5)
- **R3: Multi-source interference (hurts far most).** Per-app validation monitor during Phase 1; halt and shrink `|A_src|` if a source regresses. (Epic 5, Story 5.2)
- **R4: `L_C` staleness.** Default most-recent highest-rated; H-stale-L_C ablation tests. (Epic 5, Story 5.5)
- **R5: Cross-app crossover cadence.** Geometric schedule `2^n`; too frequent wastes compute, too rare collapses to one-shot. (Epic 5, Story 5.1)
- **R6: Rollout cost.** N=1 is noisy; push to N=3 via multi-emulator parallelism. (Epic 1, Story 1.4)
- **R7: Unbounded FSM growth.** Add size cap or periodic LLM compression. (Epic 3, Story 3.2)

---

## Estimate legend (relative complexity, not time)

- XS: trivial
- S: small
- M: medium
- L: large

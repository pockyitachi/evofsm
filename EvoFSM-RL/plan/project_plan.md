# EvoFSM-RL: Self-Evolving App Knowledge for GUI Agents via Joint FSM Evolution and Reinforcement Learning

## Concrete Project Plan — April 2026 (Revised after team discussion)

---

## 1. Positioning & Differentiation

### Core Thesis

GUI agents fail on unseen apps not because they lack general reasoning, but because they lack **app-specific knowledge** — the navigation structure, UI idioms, and workflow patterns unique to each app. We propose **EvoFSM-RL**, a method that jointly evolves app-specific FSM knowledge (declarative) and model weights (procedural) via a combined evolutionary + RL loop inspired by E-SPL. We also contribute a benchmark for evaluating cross-app and cross-category knowledge transfer.

### Focus: Method + Benchmark

The project is a **method paper with a benchmark contribution**. The core method is EvoFSM-RL (joint FSM evolution + RL weight updates). The benchmark provides standardized evaluation tiers for cross-app and cross-category transfer.

### Why This Isn't Agent-SAMA / ActionEngine / SPlanner

| Dimension | Agent-SAMA | ActionEngine | SPlanner | **Ours (EvoFSM-RL)** |
|-----------|-----------|--------------|----------|----------------------|
| FSM construction | Real-time per-execution | Offline crawling | Pre-defined EFSMs | Static extraction + **self-evolution** |
| Knowledge reuse | Mentor agent (single method) | SMG memory (single method) | Prompt injection (single method) | **4 baselines systematically compared** |
| Cross-app transfer | Not studied | Not studied | Not studied | **Tier B: explicit cross-app eval** |
| Cross-category transfer | Not studied | Not studied | Not studied | **Tier C: explicit cross-category eval** |
| Self-evolution | No | Failure-triggered repair | No | **E-SPL-inspired evolutionary + RL co-optimization** |
| Weight updates | No | No | No | **Joint FSM evolution + RL weight updates (GRPO)** |
| Contribution type | Method paper | Method paper | Method paper | **Method + benchmark paper** |

**Our unique contributions:**
1. **EvoFSM-RL**: First method to jointly evolve app-specific FSMs and model weights for GUI agents (inspired by E-SPL)
2. First benchmark for cross-app and cross-category knowledge transfer in GUI agents
3. Clean ablation story: zero-shot → static FSM → evolving FSM → evolving FSM + RL
4. Principled app similarity taxonomy for transfer evaluation

---

## 2. Benchmark Design

### 2.1 Task Universe

**Base: android_world_plus (194 tasks across 26 apps)**

| Source | Apps | Tasks |
|--------|------|-------|
| Original AndroidWorld | 20 | 116 |
| New (BMOCA) | Calculator, Snapseed, Wikipedia | 35 |
| New (AndroidLab) | Bluecoins, MAPS.ME, Pi Music | 43 |
| **Total** | **26** | **194** |

### 2.2 Evaluation Tiers

> **Note (Team Decision):** Tier A (same-app unseen tasks) has been **removed**. Our goal is test-time adaptation for cross-app and cross-category transfer — "trained on A but works well on B." Tier A doesn't test transfer and is therefore out of scope.

#### Tier B — Cross-App Transfer (Similar Apps)

**Goal:** Does knowledge from one app help with a similar but different app?

**Split strategy:**
- Pair apps by similarity (see taxonomy below)
- Train on source app(s), evaluate on target app (all tasks)
- Also evaluate: how much does transfer degrade vs. training on the target app directly?

**App similarity taxonomy (three dimensions):**

| Dimension | Description | Example |
|-----------|-------------|---------|
| **Functional** | Same domain/purpose | Bluecoins ↔ Expense (finance) |
| **Structural** | Similar UI patterns (lists, forms, settings) | Wikipedia settings ↔ System settings |
| **Task-level** | Same task type (add, delete, query, navigate) | MarkorCreateNote ↔ CalendarAddEvent |

**Proposed app clusters:**

| Cluster | Apps | Rationale |
|---------|------|-----------|
| Finance | Bluecoins, Expense | Income/expense CRUD, query |
| Note-taking | Markor, Simple Draw Pro | Create/edit content files |
| Media | Retro Music, VLC, Pi Music | Playlist, playback control |
| Navigation | OsmAnd, MAPS.ME | Map, routing, places |
| Settings-heavy | Wikipedia, System settings, Clock | Toggle settings, preferences |
| Calendar/Contacts | Simple Calendar, Contacts | CRUD structured entries |
| Image | Camera, Simple Gallery, Snapseed | Capture, browse, edit images |

**Transfer experiments:**
- Within-cluster transfer (e.g., train on Bluecoins → test on Expense)
- Cross-cluster transfer as negative control (e.g., train on Bluecoins → test on Camera)

#### Tier C — Cross-Category Transfer

**Goal:** Does knowledge about a task *type* transfer across different apps?

**Task categories:**

| Category | Description | Example tasks |
|----------|-------------|---------------|
| **CRUD-Add** | Create/add a new entry | CalendarAddEvent, ContactsAddContact, MarkorCreateNote, BluecoinsAddExpense |
| **CRUD-Delete** | Remove entry/entries | CalendarDeleteEvents, ExpenseDeleteSingle, MarkorDeleteNote |
| **Query** | Retrieve/answer from app state | BluecoinsQuerySpending, PiMusicQuerySongAlbum |
| **Settings** | Toggle/modify app settings | WikipediaDisablePreview, SystemWifiTurnOn, SystemBrightness |
| **Navigation** | Navigate to location/page | MapsMeNavigateToLocation, OsmAndFavorite |
| **Media-Control** | Playback, playlist management | RetroCreatePlaylist, VlcCreatePlaylist, PiMusicPauseAndSeek |
| **Multi-step Composite** | Cross-app or multi-phase tasks | MarkorCreateNoteAndSms, TurnOnWifiAndOpenApp |

**Split strategy:**
- For each category: 70% train / 30% test, stratified across apps
- The agent learns about "how to add things" from some apps, tested on adding things in target apps

### 2.3 Metrics

Our evaluation is **test-time-adaptation (TTA) centric**. The central object is not a single success rate but an **adaptation curve** — SR on a target-app evaluation set as a function of the online adaptation budget `k` spent on that same target app.

| Metric | Description |
|--------|-------------|
| **Adaptation Curve `SR_eval(k)`** | **Primary metric.** SR on held-out eval tasks `T_eval` of the target app, measured every `k` online adaptation tasks performed on `T_adapt` of the same target app. Reported as a curve, not a single scalar. |
| **SR@0** | Initial success rate on target app before any online adaptation (tests quality of pretrained artifact / transferred FSM). |
| **SR@K_max** | Asymptotic success rate after the full adaptation budget is exhausted. |
| **Adaptation Slope / AUC** | Shape summary of the adaptation curve. Large early slope = fast adaptation. AUC up to budget K_max. |
| **Δ SR vs. zero-shot** | Improvement of SR@K_max over the frozen zero-shot baseline — the headline adaptation signal. |
| **Step Efficiency** | Average steps to completion (fewer = better adaptation) |
| **Knowledge Transfer Ratio** | SR@0 (with pretrained transfer) / SR@K_max (target-only training from scratch). Quantifies how much of the target-app competence came "for free" from transfer. |
| **Adaptation Cost** | Online compute / API calls per adaptation step. |

**Rationale for curve-based evaluation.** Under TTA, reporting a single final SR conflates (a) the quality of pretrained knowledge, (b) how efficiently the system absorbs new target-app data, and (c) the final ceiling. An adaptation curve separates these: SR@0 measures (a), slope measures (b), SR@K_max measures (c). This is also the more realistic deployment view — a user cares "how good am I after 5 / 20 / 50 tasks of interaction with the new app."

---

## 3. Method & Baselines (Revised — Team Decision)

### Base Model: Qwen3-VL-8B

All experiments use **Qwen3-VL-8B** as the base model. This is a multimodal VLM that can process both screenshots and text UI descriptions.

### 3.1 Four Baselines (Reinterpreted Under TTA)

All four baselines are now interpreted as **different deployment-time adaptation strategies** applied to the same target app. Each produces an adaptation curve `SR_eval(k)` rather than a single scalar. The ablation story is: how much of the curve comes from evolving knowledge, how much from evolving weights, and how much from pretraining on source apps.

| # | Baseline | FSM at deployment | Weights at deployment | Expected curve shape |
|---|----------|-------------------|-----------------------|----------------------|
| **1** | Qwen3-VL-8B zero-shot | None | Frozen | Flat line at SR@0 — no adaptation (lower bound). |
| **2** | Qwen3-VL-8B + static FSM | Fixed (extracted offline from target app, never updates) | Frozen | Flat line, higher than B1 by the value of declarative knowledge. |
| **3** | Qwen3-VL-8B + self-evolving FSM | **Evolves online on target app** (mutation + crossover driven by `T_adapt` rollouts) | Frozen | Rising curve — prompt-only TTA. Tests whether FSM evolution alone can adapt to a new app. |
| **4** | Qwen3-VL-8B + EvoFSM-RL | **Evolves online on target app** | **Updates online via GRPO on target app** | Rising curve with higher slope / higher ceiling — full TTA. |

This creates a clean ablation: each baseline adds one online-adaptation component to the same pretrained artifact, isolating its contribution to the adaptation curve. Baselines 3 and 4 both perform TTA; Baseline 3 isolates the contribution of prompt-only evolution, and Baseline 4 adds weight updates on top.

**Note on Baselines 3 and 4 paths.** Baseline 4 has three design variants for *how often* LoRA is updated relative to FSM:
- **Path A** — FSM evolves every step, LoRA frozen (this collapses B4 to B3, serves as internal consistency check).
- **Path B** — FSM and LoRA both update at every adaptation step.
- **Path C** — FSM updates every step, LoRA updates only every `K` steps (default plan).

We default to Path C for compute reasons and report Path B as an ablation.

### 3.2 Method 0 — FSM Construction (Training Phase)

**Inspired by:** GUI-Xplore, ActionEngine

**What:** Before any adaptation, run a capable model (GPT-4o T3A agent) on source app tasks to collect trajectories. An LLM synthesizes these into structured FSMs per app.

**Output:** Per-app FSM files (markdown). These are used as the static FSM (Baseline 2) and as the seed for evolution (Baselines 3 & 4).

### 3.3 Baseline 2 — Static FSM Injection

**What:** Serialize the app's FSM directly into the agent's system prompt. The FSM is generated once during training and never updated.

### 3.4 Baseline 3 — Self-Evolving FSM (No Weight Updates)

**Inspired by:** GEPA (ICLR 2026 Oral), E-SPL prompt evolution component

**What:** Maintain a population of FSM variants per app. Evolve them via LLM-driven mutation and crossover, scored by task performance. Model weights stay frozen.

**The loop (simplified — no Phase 3 from full algorithm):**
```
Population P = {F_0, Mutate(F_0), ...}   # Initialize from static FSM
For each iteration:
    1. Select K FSM variants by TrueSkill rating
    2. Run rollouts with Qwen3-VL-8B (frozen) + each FSM variant
    3. Score FSMs by task success rate
    4. Update TrueSkill ratings
    5. Evolve: mutation + crossover → new generation
Output: Best FSM F*
```

### 3.5 Baseline 4 — EvoFSM-RL (Full Method — Core Contribution)

**Inspired by:** E-SPL (arxiv 2602.14697), SkillRL, SAGE

**What:** The headline contribution. Jointly evolve app-specific FSMs (declarative knowledge) and update Qwen3-VL-8B weights (procedural knowledge) using a combined evolutionary + RL loop.

**Key insight from E-SPL:** There is a natural division of labor — declarative knowledge (app structure, navigation rules) lives in the evolving FSM, while procedural knowledge (action selection patterns, visual grounding) lives in the model weights. Both should co-evolve.

**Full algorithm: See `espl_fsm_algorithm_design.md` for detailed pseudocode.**

**Summary:**
```
For each iteration:
    Phase 1: Select K FSM variants from population (TrueSkill-weighted)
    Phase 2: Run parallel rollouts (Qwen3-VL-8B + each FSM on task batch)
    Phase 3: RL weight update (GRPO on collected trajectories)  ← NEW vs Baseline 3
    Phase 4: Update TrueSkill ratings for FSM variants
    Phase 5: Evolve FSM population (mutation + crossover)
Output: Updated model M_θ* + best FSM F*
```

**RL details:**
- Algorithm: GRPO (Group Relative Policy Optimization)
- Fine-tuning: LoRA/QLoRA on Qwen3-VL-8B (practical for 8B model)
- Reward: AndroidWorld task evaluator (binary success + step efficiency)
- Group: same task across different FSM variants → relative advantage

**Evolution details:**
- Population: N_pop = 8-16 FSM variants per app
- Selection: TrueSkill rating (μ, σ) with exploration bonus
- Mutation: LLM-driven FSM refinement using failed trajectory feedback
- Crossover: LLM-driven merge of complementary FSM variants

### 3.6 Overall Pipeline, Transfer, and Test-Time Adaptation

**Framing.** Our approach is fundamentally a **test-time-adaptation** (TTA) approach. The model and the FSM are *never frozen forever*: after any amount of source-pool pretraining, deployment on a new target app continues the same co-evolution loop online, using a small budget of target-app tasks. **We produce a single pretrained artifact** from multi-source joint training over a broad source pool, and **Tier B / Tier C are two evaluation conditions on that same artifact** — they are not two independent training tracks. What differs between them is only which target apps we deploy to: near-transfer targets (Tier B) share a cluster with a source app; far-transfer targets (Tier C) do not. The main metric (§2.3) is the adaptation curve `SR_eval(k)` on target-app tasks, reported separately under each condition.

#### 3.6.0 Five-Phase Pipeline

| Phase | Scope | Apps | What happens | Outputs |
|-------|-------|------|--------------|---------|
| **0. Offline static extraction** | Run once | All 26 apps | Run GPT-4o T3A on source-app training tasks, synthesize per-app static FSM `F^0_a`. | `{F^0_a}` (used as static FSM baseline and as seed for evolution). |
| **1. Unified source-pool pretraining** | One run | Source pool `A_src` (multiple clusters, multiple categories) | Run **multi-source joint EvoFSM-RL** over the entire source pool with **shared LoRA** and **per-app FSM populations**, plus **cross-app crossover** writing to per-category abstract libraries `{L_C}` (see §3.6.3). This is a single training run — **not** per-tier. | Single pretrained artifact used for all target apps in both evaluation conditions: `π^pre_θ`, per-source-app populations `{S^pre_a}`, per-category abstract libraries `{L_C}`. |
| **2. Cold-start on target app** | Once per target app | Target app `a_tgt` | (i) 5–10 exploration rollouts of `π^pre_θ` on `a_tgt` → exploration report. (ii) **Condition-dependent hand-off** (§3.6.4): near-transfer translates the nearest same-cluster source app's LAYER2; far-transfer specializes the matching `F_C_abstract` from `{L_C}`. (iii) Seed `S^init_{a_tgt}` from mutations of the resulting FSM. (iv) Split target-app tasks `T_{a_tgt}` into `T_adapt` (70%) / `T_eval` (30%), stratified by sub-category. | `π^pre_θ`, `S^init_{a_tgt}`, `T_adapt`, `T_eval`. |
| **3. Online adaptation on target app** | Per-target-app TTA | `T_adapt` only | Run the same EvoFSM-RL main loop as Phase 1, but drawing only from `T_adapt`. Save `(π_θ, S_{a_tgt})` snapshots every `k_step` adaptation tasks. This is the TTA phase — the curve `k ∈ {0, k_step, …, K_max}` is traced here. | Snapshot sequence `{(π^k_θ, F^k_{a_tgt})}_{k=0..K_max}`. |
| **4. Held-out evaluation on target app** | Per snapshot | `T_eval` only | Evaluate each snapshot on the **frozen** held-out `T_eval`. Snapshots are frozen at evaluation time (no weight / FSM update during eval). Plot `SR_eval(k)`. | Adaptation curve per target app × per baseline × per condition (near / far). |

**Why this structure and not pure online RL.** We considered the alternative of *not* splitting target-app tasks and running pure online RL on all target-app tasks (reporting streaming SR over the run). We choose the split-with-held-out-eval design for three reasons:

1. **Generalization measurement.** Pure online reports whether the system fits the exact task sequence it has seen. The split lets us ask whether adaptation on 30 observed tasks **generalizes** to 13 unseen tasks of the same app — which is the quantity a deployer actually cares about.
2. **Task-order robustness.** Online-streaming SR is sensitive to the order in which tasks happen to arrive, which makes seed-to-seed variance huge. A frozen `T_eval` gives a stable comparison point across seeds and baselines.
3. **Cleaner story.** "Our TTA adapts in 20 tasks and transfers to the remaining held-out tasks" is a strictly stronger claim than "our online SR goes up during a streaming run."

A streaming variant (no held-out) can additionally be reported in the appendix as a sanity check.

**Source apps are not themselves split for evaluation.** We use essentially all source-app tasks for Phase 1 pretraining. A small fraction (10–15%) is held out only as a *sanity check* to verify that the pretrained artifact has not collapsed on source apps. The main comparison lives entirely on target apps.

**Target pool is split into two evaluation conditions.** Every target app is pre-assigned to exactly one condition based on whether its cluster is represented in the source pool (§3.6.4). The two conditions are evaluated on disjoint pools of target apps — a given target app never appears in both.

**Compute budget knobs.** `K_max` (total adaptation budget on target app) defaults to `K_max ≈ |T_adapt|` (one pass) for near-transfer targets, and can be raised to `K_max ≈ 2·|T_adapt|` for far-transfer targets (harder cold start, more adaptation expected). `k_step` (snapshot granularity) is 5–10 tasks.

#### 3.6.1 Wrapping the single-app loop for transfer

**Motivation.** The EvoFSM-RL main loop in §3.5 is a *single-app optimizer* — it drives both π_θ and F deeper into one app's task distribution. This is sufficient for within-app learning but **does not itself produce transferable knowledge** in a form that cold-starts a new target app well. Cross-app and cross-category transfer require us to wrap the single-app loop with additional mechanisms (two-layer FSM schema, layered reflection prompt, multi-source joint pretraining, cross-app crossover). The rest of §3.6 makes those mechanisms explicit.

#### 3.6.2 Two-Layer FSM Schema (required for transfer)

Standard E-SPL evolves a flat system prompt. We augment FSMs with **two explicit layers**:

| Layer | Contents | Transferable? |
|-------|----------|---------------|
| **App-specific** | Button positions, field names, color cues, exact navigation paths | No (per-app) |
| **Generic / task-type** | Abstract task strategies, UI idioms, common failure modes, error recovery patterns | Yes (across apps) |

The self-reflection prompt given to π_ref during mutation is modified to *explicitly* ask for both layers. Example reflection prompt instruction:

> "When proposing edits, separate what you learned into two parts:
> (a) facts that only apply to this specific app (write under `APP_SPECIFIC`),
> (b) patterns that likely generalize to other apps in the same cluster or to other apps with the same task type (write under `GENERIC`)."

This is a critical design change from vanilla E-SPL. Without it, mutation would bake transferable insights into per-app detail and lose them at transfer time.

#### 3.6.3 Phase 1 — Unified Multi-Source Joint Training

Phase 1 is a **single training run** over the full source pool `A_src`. The source pool is chosen to span multiple clusters and multiple task categories so that the pretrained artifact is simultaneously useful for near-transfer deployment (when a same-cluster source exists) and far-transfer deployment (when only category-level abstractions apply). We do **not** run separate "Tier B training" and "Tier C training" — single-source training is a degenerate special case of multi-source (take `|A_src|=1`) and is subsumed by this design.

**Multi-source joint training loop:**

```
Inputs:
  - Source pool A_src = {a_1, ..., a_K}  spanning multiple clusters / categories
  - Task categories C_set = {C_1, ..., C_M}
  - Per-app task sets {T_a}, each tagged by category
  - Per-app FSM populations {S_{a_1}, ..., S_{a_K}}
  - Shared LoRA weights θ (single set across all source apps)
  - Abstract libraries L_C initialized empty for each C

for iteration = 1 to T:
    # Sample an app and a task of any category from that app
    a ← sample_uniform(A_src)
    t ← sample(T_a)
    C ← category_of(t)

    # Select M FSMs from THIS app's population
    select {(F_i, μ_i, σ_i)}^M from S_a

    # Parallel rollouts on emulator (shared π_θ + app-specific FSMs)
    for i = 1 to M:
        τ_{i,j} = rollout(π_θ, F_i, t) for j = 1..N
        r_{i,j}, V_i as usual

    # Update SHARED LoRA weights with GRPO on these rollouts
    θ ← GRPO_update(θ, {τ, r, F_i}, app=a)

    # Update ONLY the FSM population of app a (TrueSkill, mutation, within-app crossover)
    update S_a as in single-app loop

    # Cross-app crossover (stochastic, per category) — writes to L_C
    if p_cross_app > rand():
        C_pick ← sample_eligible_category(A_src, C_set)
        F_{C_pick}_abstract ← CrossAppCrossover(A_src, {S_a}, category=C_pick, π_ref)
        L_{C_pick}[timestamp] = F_{C_pick}_abstract
```

**Key property.** The shared LoRA receives gradients from *all* source apps across *all* categories. This biases `π_θ` toward patterns that are common across apps within a category (helpful for near-transfer) and across categories (helpful for far-transfer). Each app's FSM population `S_a` specializes declaratively to its own app.

**Cross-App Crossover (new operator — extends E-SPL's within-distribution crossover):**

```
Inputs: best FSMs per source app for category C: {F_a*.LAYER2[C] : a ∈ A_src}

1. Collect the category-C LAYER2 blocks from each source app's champion FSM.
2. Build a crossover prompt for π_ref:
     "Here are the {C}-strategy LAYER2 blocks from {K} different apps: [...].
      Identify the COMMON abstract structure, drop any app-specific strings,
      and produce a single APP-AGNOSTIC generic strategy for category {C}."
3. π_ref outputs F_C_abstract (a task-type template, no specific app bindings).
4. Store F_C_abstract in L_C (keyed by category, timestamp, rating).
```

This operator is an **extension beyond standard E-SPL** — the original paper assumes a single problem distribution, while we are explicitly distilling across distributions. The abstract library `{L_C}` is *never* pushed back into any `S_a`; it is only used by the Phase-2 hand-off for far-transfer targets.

#### 3.6.4 Phase 2 — Deployment-Time Hand-off (Near vs Far Conditions)

Once Phase 1 has produced the single pretrained artifact `(π^pre_θ, {S^pre_a}, {L_C})`, every target app `a_tgt` enters Phase 2 via the same hand-off routine. The routine branches on whether `a_tgt`'s cluster is represented in the source pool. This branch is the **only** per-condition difference in the entire pipeline — Phase 3 (TTA) and Phase 4 (eval) are identical.

**The Two Evaluation Conditions.**

- **Near-transfer condition (Tier B).** `a_tgt`'s cluster is represented in `A_src`, i.e., there exists at least one same-cluster source app `a_near` (e.g., target = Expense, source pool includes Bluecoins). Hand-off: translate `a_near`'s LAYER2 onto `a_tgt`'s exploration report. Tests *within-cluster* transfer across different UIs.
- **Far-transfer condition (Tier C).** `a_tgt`'s cluster is **not** represented in `A_src` (e.g., target = Simple Calendar, no calendar-cluster source). Hand-off: specialize the most-recent highest-rated `F_C_abstract` from `L_{C_tgt}`. Tests *category-level* transfer across unrelated clusters.

A given target app belongs to exactly one condition by construction of the benchmark split.

**Hand-off procedure:**
```
Deploy(a_tgt, π^pre_θ, {S_a}, {L_C}, A_src, π_ref):

  1. Run 5–10 exploration rollouts of π^pre_θ on a_tgt → F_{a_tgt}.LAYER1 skeleton.
  2. Determine C_tgt (the category we are evaluating on).
  3. if cluster(a_tgt) is represented in A_src:          # ── near-transfer
         a_near = nearest same-cluster source app
         L2_src = S_{a_near}.champion.LAYER2[C_tgt]
         F_{a_tgt}_init = LAYER2_translate(L2_src, a_tgt, π_ref)
     else:                                               # ── far-transfer
         F_C_abstract = most_recent_highest_rated( L_{C_tgt} )
         F_{a_tgt}_init = LAYER2_ground(F_C_abstract, a_tgt, π_ref)
  4. Seed S^init_{a_tgt} from mutations of F_{a_tgt}_init.
  5. Split T_{a_tgt} ∩ C_tgt → T_adapt (70%) / T_eval (30%).
```

**Online adaptation (Phase 3)** is identical in both conditions: run the EvoFSM-RL main loop on `T_adapt`, taking snapshots.

**Baselines as Phase-3 update policies (evaluated under both conditions).** Each of B1–B4 from §3.1 produces **two** adaptation curves — one averaged over near-transfer target apps, one averaged over far-transfer target apps. The paper's main figure is two panels (near / far) side-by-side, each with four baseline curves. The near-vs-far gap at any fixed baseline tells us how quickly transfer quality degrades with target-cluster distance.

**Reading the curves.**
- `SR@0` measures the value of the Phase-2 hand-off *before any TTA*.
- Slope of the curve during Phase 3 measures how efficiently TTA closes the gap that Phase 2 left behind.
- `SR@K_max` is the ceiling after the adaptation budget is exhausted.

**Expected qualitative story.**
- In the **near-transfer** panel we expect B2 (weight-only) to be small or negative for visually different clusters, B1 (static FSM) to give a meaningful `SR@0` bump with a modest slope, and B4 (EvoFSM-RL) to give both a better `SR@0` and a steeper early slope — the "transfer buys you adaptation speed" result.
- In the **far-transfer** panel we expect all baselines to start lower at `SR@0` (no in-cluster source), but B4's slope to remain positive — thanks to the shared LoRA's procedural priors and `F_C_abstract`'s declarative priors — closing the near-vs-far gap as `k` grows.

**Optional hand-off ablations (same pretrained artifact, alternative step-3 choices).** These spawn *no* new training runs; each reuses the Phase-1 artifact. Run only if the main result calls for it.

| Ablation | Hand-off (Phase 2 step 3) | Purpose |
|----------|----------------------------|---------|
| H-none | skip step 3; static `F^0_{a_tgt}` | lower bound (≈ B1 under each condition) |
| H-weights-only | skip step 3 LAYER2 injection; seed from `F^0_{a_tgt}` | isolates shared-LoRA contribution (≈ B2) |
| H-single-source (near only) | random same-cluster source instead of nearest | tests sensitivity to source choice |
| H-stale-L_C (far only) | oldest `L_C` entry instead of most-recent | tests `L_C` staleness claim (§3.6.5) |
| H-full (default) | as specified above | the reported number |

#### 3.6.5 Risks of the Transfer Story

We are deliberately transparent about three risks that the 4-baseline comparison is designed to surface:

1. **Reflection prompt brittleness.** If π_ref fails to cleanly separate GENERIC from APP_SPECIFIC during mutation, the generic layer will be contaminated with Bluecoins-isms and translation to Expense will fail. *Mitigation:* Iterate on reflection prompt design in Phase 3 before committing to Phase 5 transfer experiments. Validate on held-out "generic layer quality" metric (human rating on 20 samples).

2. **Visual negative transfer (near-transfer condition).** Even when source and target are in the same cluster, the shared LoRA's visual priors may hurt on target apps with very different screens (dark vs light, icon-style vs text-style). *Mitigation:* B1 (static FSM) vs B2 (weights only) vs B4 (full) on the near-transfer panel will surface this directly. A negative B2 result is still a publishable finding ("weight transfer alone does not help in visually diverse GUI settings").

3. **Multi-source interference (hurts far-transfer most).** A shared LoRA trained over many source apps may dilute app-specific competence or, in the worst case, produce gradients that cancel on genuinely app-specific behaviors — producing a pretrained policy worse than any single-source LoRA on any one app. This hurts far-transfer most, because far-transfer has no in-cluster source to fall back on. *Mitigation:* monitor per-source-app validation throughout Phase 1; if any source regresses past a threshold, reduce `|A_src|`. The H-weights-only and H-none ablations (§3.6.4) also let us read off the interference cost directly.

4. **Abstract library staleness.** `L_C` is written during Phase 1 and read at Phase 2 hand-off. Early `F_C_abstract` entries can become stale relative to the shared LoRA's later state. *Mitigation:* default to the most-recent highest-rated entry; the H-stale-L_C ablation tests sensitivity.

---

## 4. Implementation Roadmap

### Phase 1: Infrastructure (Weeks 1–2)

| Task | Description | Files to modify/create |
|------|-------------|----------------------|
| 1.1 | Create project scaffolding | `adaptation/` directory with `__init__.py`, `config.py` |
| 1.2 | Implement task split logic for Tiers B/C | `adaptation/splits.py` — train/test split functions |
| 1.3 | Build app similarity taxonomy | `adaptation/taxonomy.py` — app clusters, similarity scores |
| 1.4 | Extend `registry.py` with task metadata | Add category tags, complexity, app cluster to each task |
| 1.5 | Create evaluation harness | `adaptation/evaluator.py` — run adapted vs. baseline, compute metrics |
| 1.6 | Set up Qwen3-VL-8B inference pipeline | `adaptation/model/qwen_agent.py` — Qwen3-VL-8B agent wrapper |
| 1.7 | Set up parallel rollout infrastructure | Multiple emulator support for faster data collection |

### Phase 2: Knowledge Extraction + Static Baselines (Weeks 3–4)

| Task | Description | Files to create |
|------|-------------|----------------|
| 2.1 | Trajectory collection pipeline | Run GPT-4o T3A on source apps, record full traces |
| 2.2 | Implement FSM construction from trajectories | `adaptation/fsm/builder.py` |
| 2.3 | Define FSM schema + serialization | `adaptation/fsm/schema.py`, `adaptation/fsm/serializer.py` |
| 2.4 | Generate static FSMs for all 26 apps | Scripts in `scripts/build_fsms.py` |
| 2.5 | **Baseline 1:** Qwen3-VL-8B zero-shot evaluation | On Tier B and Tier C splits |
| 2.6 | **Baseline 2:** Qwen3-VL-8B + static FSM evaluation | Inject FSMs into system prompt |

### Phase 3: FSM Evolution (Weeks 5–7)

| Task | Description | Files to create |
|------|-------------|----------------|
| 3.1 | Implement TrueSkill rating library | `adaptation/evolution/trueskill.py` |
| 3.2 | Implement LLM-driven mutation operator | `adaptation/evolution/mutation.py` |
| 3.3 | Implement LLM-driven crossover operator | `adaptation/evolution/crossover.py` |
| 3.4 | Build evolutionary loop (no RL) | `adaptation/evolution/evo_loop.py` |
| 3.5 | FSM population management | `adaptation/evolution/population.py` |
| 3.6 | **Baseline 3:** Self-evolving FSM evaluation | Run evolutionary loop, evaluate best FSM |

### Phase 4: Full EvoFSM-RL (Weeks 8–11)

| Task | Description | Files to create |
|------|-------------|----------------|
| 4.1 | Set up LoRA/QLoRA for Qwen3-VL-8B | `adaptation/training/lora_setup.py` |
| 4.2 | Implement GRPO weight update | `adaptation/training/grpo.py` |
| 4.3 | Integrate RL into evolutionary loop | `adaptation/evolution/evofsm_rl.py` |
| 4.4 | Reward design (success + efficiency + exploration) | `adaptation/training/reward.py` |
| 4.5 | **Baseline 4:** Full EvoFSM-RL evaluation | Joint evolution + RL, evaluate |
| 4.6 | Ablation experiments | Vary population size, K, RL vs no-RL, etc. |

### Phase 5: Transfer Experiments (Weeks 12–14)

Phase 1 multi-source joint pretraining happens **once** over the source pool and yields the single artifact `(π^pre_θ, {S^pre_a}, {L_C})`. All of E1–E5 below reuse this artifact; they vary only the target-app pool (near-transfer vs far-transfer) and the Phase-2 hand-off step.

| Experiment | Evaluation condition | Baselines | Description |
|-----------|------|-----------|-------------|
| E1: Near-transfer adaptation curves | Tier B | 1, 2, 3, 4 | Deploy on target apps whose cluster is represented in `A_src`; same-cluster source LAYER2 translation. |
| E2: Far-transfer adaptation curves | Tier C | 1, 2, 3, 4 | Deploy on target apps whose cluster is **not** in `A_src`; specialize `F_C_abstract` from `L_C`. |
| E3: Hand-off ablations (H-none / H-weights-only / H-full) | Tier B, Tier C | 4 | §3.6.4 hand-off table; isolates declarative vs procedural contributions at Phase-2. |
| E4: Source-choice sensitivity | Tier B | 4 | H-single-source vs H-full (nearest) on near-transfer targets. |
| E5: `L_C` staleness | Tier C | 4 | H-stale-L_C vs H-full on far-transfer targets. |
| E6: Convergence analysis | Tier B, Tier C | 3, 4 | Plot FSM evolution and LoRA update curves over Phase-3 `k`. |

**Estimated compute:**
- Rollouts: ~20-40 min per iteration per app, 50-100 iterations → ~16-66 hours/app
- LoRA fine-tuning: ~50-100 GPU-hours on A100 (or equivalent)
- Parallelizable across multiple emulators

### Phase 6: Analysis & Paper (Weeks 14–16)

| Task | Description |
|------|-------------|
| 6.1 | Aggregate all results, compute metrics |
| 6.2 | Generate tables and figures |
| 6.3 | Analyze cross-app transfer patterns (heatmaps) |
| 6.4 | FSM convergence + quality analysis |
| 6.5 | Write paper (see outline below) |
| 6.6 | Code cleanup and release preparation |

---

## 5. Experiment Design Details

### 5.1 Baselines (Under TTA Framing)

All baselines below are deployed on the same target app and evaluated as adaptation curves `SR_eval(k)` on `T_eval`. What differs is **what is allowed to update at deployment time**.

| # | Baseline | What's allowed to update during Phase 3 (target app) |
|---|----------|------------------------------------------------------|
| **1** | **Qwen3-VL-8B zero-shot** | Nothing — flat line. |
| **2** | **Qwen3-VL-8B + static FSM** | Nothing — flat line at higher SR. |
| **3** | **Qwen3-VL-8B + self-evolving FSM** | FSM only (mutation + crossover + TrueSkill on `T_adapt`). Weights frozen. |
| **4** | **Qwen3-VL-8B + EvoFSM-RL** | FSM + LoRA weights (joint update on `T_adapt`, Path C by default). |

Each of the four is evaluated under **both** the near-transfer condition (Tier B) and the far-transfer condition (Tier C). Both conditions use the *same* pretrained artifact from Phase 1; they differ only in the Phase-2 hand-off step (§3.6.4), which is determined by whether the target app's cluster is represented in the source pool. The paper's main figure is a 2-panel × 4-curve plot: near-transfer panel on the left, far-transfer panel on the right, each showing the four baselines' adaptation curves.

### 5.2 Ablation Studies

For Baseline 2 (static FSM):
- Full FSM vs. task-relevant subgraph
- Text description vs. structured format vs. natural language
- With/without pre/post-conditions

For Baseline 3 (self-evolving FSM):
- Population size: 4, 8, 16
- Number of evolution iterations: 10, 25, 50, 100
- Mutation only vs. mutation + crossover
- TrueSkill vs. simple reward-based selection

For Baseline 4 (EvoFSM-RL):
- LoRA rank: 8, 16, 32, 64
- GRPO group size effects
- Evolution-only vs. RL-only vs. joint (key ablation)
- Adaptive scheduling: early evolution heavy → late RL heavy

### 5.3 Analysis Questions

1. **Does each component help under TTA?** Clean ablation on adaptation curves under both conditions: B1 flat → B2 flat-higher → B3 rising (FSM-only TTA) → B4 rising-steeper (joint TTA).
2. **Does knowledge transfer across apps within a cluster?** `SR@0` and early-slope lift on the **near-transfer panel** compared to H-none (static-only hand-off).
3. **Does task-type knowledge transfer across clusters?** `SR@0` and early-slope behaviour on the **far-transfer panel** compared to H-none — isolating the contribution of the abstract library `L_C`.
4. **How much does transfer quality degrade with target-cluster distance?** Gap between the near-transfer and far-transfer panels at each baseline.
5. **Does the FSM converge during TTA?** Plot FSM size, edit distance, and TrueSkill ratings along `k` in both conditions.
6. **Does joint TTA outperform FSM-only TTA?** Baseline 4 vs. Baseline 3 curve gap in both panels.
7. **Adaptation cost-performance tradeoff.** `SR_eval(k)` per unit compute.
8. **Which source→target pairs transfer best?** Heatmap of near-transfer `SR@0` by source-cluster → target-app pair.
9. **Declarative vs. procedural knowledge.** Decompose curve gains via H-weights-only vs. H-full hand-off ablations.

---

## 6. Paper Outline

**Title:** EvoFSM-RL: Self-Evolving App Knowledge for GUI Agents via Joint FSM Evolution and Reinforcement Learning

### Abstract
GUI agents fail on unseen apps because they lack app-specific knowledge. We propose EvoFSM-RL, a method that jointly evolves app-specific finite state machine (FSM) knowledge and model weights through a combined evolutionary + RL loop inspired by E-SPL. The FSM captures declarative app knowledge (states, transitions, strategies) that evolves via LLM-driven mutation and crossover, while the model (Qwen3-VL-8B) weights capture procedural action-selection knowledge updated via GRPO. We also introduce a benchmark for evaluating cross-app and cross-category knowledge transfer for GUI agents, built on AndroidWorld (194 tasks, 26 apps). Our four-baseline ablation (zero-shot → static FSM → evolving FSM → EvoFSM-RL) cleanly isolates the contribution of each component. We find that [key results].

### 1. Introduction
- GUI agents are brittle on unseen apps (cite T3A/M3A results on new apps: 48.1% vs 7.8%)
- Recent work proposes FSM-based knowledge (Agent-SAMA, ActionEngine, SPlanner) but no standard evaluation for cross-app transfer
- E-SPL shows joint prompt evolution + RL outperforms either alone in reasoning tasks
- We contribute: (1) EvoFSM-RL for GUI agents, (2) benchmark with cross-app (Tier B) and cross-category (Tier C) transfer evaluation, (3) clean ablation study

### 2. Related Work
- GUI agent architectures (M3A, T3A, SeeAct, CogAgent, Qwen-VL)
- FSM-based approaches (Agent-SAMA, ActionEngine, SPlanner)
- Evolutionary prompt optimization (E-SPL, GEPA, EvoPrompt, PromptBreeder)
- Self-evolving agents (EvolveR, SAGE, SkillRL, AppAgentX)
- RL for GUI agents (DigiRL, MobileRL, ADAGRPO)

### 3. EvoFSM-RL: Method
- 3.1 FSM representation and serialization
- 3.2 Evolutionary FSM population management (TrueSkill selection)
- 3.3 LLM-driven mutation and crossover operators
- 3.4 GRPO weight updates conditioned on FSM variants
- 3.5 Joint optimization loop
- 3.6 Cross-app FSM bootstrapping for transfer

### 4. Benchmark Design
- 4.1 Task universe and app inventory (194 tasks, 26 apps)
- 4.2 Evaluation tiers (B: cross-app, C: cross-category)
- 4.3 App similarity taxonomy
- 4.4 Metrics

### 5. Experiments
- 5.1 Setup (Qwen3-VL-8B, compute, seeds)
- 5.2 Main results (4 baselines × 2 tiers)
- 5.3 Cross-app transfer analysis (Tier B heatmaps)
- 5.4 Cross-category transfer analysis (Tier C)
- 5.5 FSM convergence and quality analysis
- 5.6 Declarative vs. procedural knowledge contribution
- 5.7 Ablation studies
- 5.8 Computational cost analysis

### 6. Discussion
- Why joint optimization works (E-SPL's division of labor applied to GUI agents)
- When FSM evolution helps most vs. when RL helps most
- Limitations (compute cost, emulator speed, app coverage)
- Implications for deployable GUI agents

### 7. Conclusion

### Appendices
- A: Full task list with tier assignments
- B: App similarity matrix
- C: FSM examples and evolution traces
- D: Mutation and crossover prompt templates
- E: Full per-task results
- F: Hyperparameter sensitivity

---

## 7. Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| API costs exceed budget | High | Prioritize T3A (cheaper, text-only), scope Method C as optional |
| FSM extraction is noisy | Medium | Multiple extraction passes, human validation on subset |
| Self-evolution doesn't converge | High | Fallback: show convergence analysis even if final SR is modest |
| Cross-app transfer shows no signal | Medium | This is still a valid finding — "negative results" are publishable |
| AndroidWorld emulator instability | Medium | Use checkpointing (already built), Docker for reproducibility |
| Related work publishes similar benchmark | High | Move fast; our multi-method comparison and Tier B/C are hard to replicate quickly |

---

## 8. Recommended Execution Order

**Build incrementally — each phase validates the next:**

1. **Week 1-2:** Infrastructure + Qwen3-VL-8B agent setup (Phase 1)
2. **Week 3-4:** Trajectory collection + static FSM generation → **Baselines 1 & 2** (Phase 2)
3. **Week 5-7:** Evolutionary loop implementation → **Baseline 3** (Phase 3)
4. **Week 8-11:** GRPO integration + LoRA setup → **Baseline 4 / EvoFSM-RL** (Phase 4)
5. **Week 12-14:** Cross-app and cross-category transfer experiments (Phase 5)
6. **Week 14-16:** Analysis, ablations, paper writing (Phase 6)

---

## 9. Codebase Structure

```
android_world_plus/
├── android_world/              # Existing core (mostly unchanged)
│   ├── agents/                 # M3A, T3A, SeeAct + new agent
│   │   └── qwen_agent.py          # NEW: Qwen3-VL-8B agent wrapper
│   ├── env/
│   ├── task_evals/
│   └── utils/
├── adaptation/                 # NEW: all adaptation code
│   ├── __init__.py
│   ├── config.py               # Experiment configs
│   ├── splits.py               # Tier B/C split logic
│   ├── taxonomy.py             # App similarity taxonomy
│   ├── evaluator.py            # Evaluation harness
│   ├── fsm/
│   │   ├── schema.py           # FSM data structures
│   │   ├── builder.py          # FSM construction from trajectories
│   │   ├── serializer.py       # FSM ↔ text serialization
│   │   └── subgraph.py         # Task-relevant FSM extraction
│   ├── evolution/
│   │   ├── population.py       # FSM population management
│   │   ├── trueskill.py        # TrueSkill rating system
│   │   ├── mutation.py         # LLM-driven FSM mutation
│   │   ├── crossover.py        # LLM-driven FSM crossover
│   │   ├── evo_loop.py         # Evolutionary loop (Baseline 3)
│   │   └── evofsm_rl.py        # Full EvoFSM-RL loop (Baseline 4)
│   ├── training/
│   │   ├── lora_setup.py       # LoRA/QLoRA config for Qwen3-VL-8B
│   │   ├── grpo.py             # GRPO weight update implementation
│   │   └── reward.py           # Reward function design
│   └── transfer/
│       ├── fsm_bootstrap.py    # Cross-app FSM initialization
│       └── weight_transfer.py  # Cross-app weight transfer
├── knowledge/                  # NEW: extracted knowledge artifacts
│   ├── {app_name}/
│   │   ├── fsm.md              # App FSM (markdown format)
│   │   └── trajectories/       # Recorded trajectories
├── experiments/                # NEW: experiment configs and results
│   ├── configs/
│   └── results/
├── scripts/
│   ├── collect_trajectories.py # NEW: collect training data
│   ├── build_fsms.py           # NEW: generate static FSMs
│   ├── run_evolution.py        # NEW: run evolutionary loop
│   ├── run_evofsm_rl.py        # NEW: run full EvoFSM-RL
│   └── run_experiments.py      # NEW: main experiment runner
├── server/
├── web_ui/
├── run.py
└── Dockerfile
```

---

## 10. Target Venues

| Venue | Deadline (estimated) | Fit |
|-------|---------------------|-----|
| **NeurIPS 2026 Datasets & Benchmarks** | June 2026 | Ideal — benchmark paper track |
| **ICLR 2027** | Oct 2026 | If we have strong method results (Method D) |
| **EMNLP 2026** | June 2026 | Good for the benchmark + adaptation analysis |
| **AAAI 2027** | Aug 2026 | Backup |
| **Lifelong Agents Workshop** | Varies | Good for early/partial results |

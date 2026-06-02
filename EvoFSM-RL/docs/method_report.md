# EvoFSM-RL — Method Report

Last revised 2026-05-13. This document is a self-contained walk-through of how the dataset is split, what the FSM is (with real examples), and how each of the four baselines B1–B4 is set up and run. It distills `docs/design/dataset.md`, `docs/design/algorithm.md`, `plan/algorithm_design.md`, `docs/results/b1_b2_static_baselines.md`, `docs/results/b3_evolution.md`, and the source code under `evofsm_rl/` + `scripts/`. For authoritative reference, consult those files; this document is a synthesis for someone who needs the whole picture in one read.

---

## 1. Dataset and splits

### 1.1 Benchmark composition

The benchmark sits on top of AndroidWorld (Rawles et al., 2024) with six additional apps imported from BMOCA (Calculator, Snapseed, Wikipedia) and AndroidLab (Bluecoins, Maps.me, Pi Music). Final composition: **25 active primary-task apps, 194 templates** (192 app-attributable; 2 generic/composite templates `OpenApp` and `SaveCopyOfReceipt` are excluded from per-app splits).

The 25 apps span 12 active Google Play Store categories. We use Play categories directly as the equivalence class for category-generic knowledge — they are externally validated and cite-able against Android Control (Li et al., 2024). Six categories have multiple apps and are eligible to anchor the source pool; the other six have a single app each and necessarily fall into Tier-C.

### 1.2 Source pool — 12 apps, 96 templates, 6 categories

The source pool is used for Phase-1 pretraining: per-app static FSM extraction (Story 2.2), per-category `L_C` synthesis (Story 2.3), and the shared-LoRA pretraining for B4 (now in progress).

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
| **Total — 6 categories** | | **96** |

Category counts: Productivity 3 apps, Tools 3, Communication 2, Music & Audio 2, Finance 1, Photography 1. These six categories are exactly the ones for which an `L_C` exists at test-time hand-off — Tier-C target apps have no category match by construction.

Seeds for the source pool are `[30, 31, 32, 33, 34]` (K=5), giving **480 source-pool episodes** in `traces/source_pool_trajectories/`. All 96 templates are used; no train/eval split is carved out *inside* the source pool because the source pool itself is the training side of the larger train/eval split between source apps and target apps.

### 1.3 Tier-B target pool — 6 apps, 50 templates (near-transfer)

Every Tier-B app's Play category is also represented in the source pool, so a matching `L_C` exists and can seed the target-app FSM at hand-off. All counts below are **templates**, not episodes; episodes = templates × K seeds (K=5 for `T_adapt`, K=3 for `T_eval`).

| App | Category | T_adapt templates | T_eval templates |
|---|---|---:|---:|
| simple_calendar_pro | Productivity | 11 | 6 |
| system_settings | Tools | 9 | 6 |
| pro_expense | Finance | 6 | 3 |
| retro_music | Music & Audio | 3 | 1 |
| camera | Photography | 1 | 1 |
| chrome | Communication | 2 | 1 |
| **Total — templates** | | **32** | **18** |
| **Total — episodes** | | **160** (32 × 5) | **54** (18 × 3) |

### 1.4 Tier-C target pool — 7 apps, 46 templates (far-transfer)

No Tier-C app's Play category is in the source pool, so `resolve_l_c_for_app()` returns `None` for every Tier-C app. Under B2/B3, the Tier-C arm therefore degrades byte-identically to the no-injection B1 path — this is what makes Tier-C a built-in null control for the L_C injection mechanism. Counts are templates; episodes = templates × K seeds.

| App | Category | T_adapt templates | T_eval templates |
|---|---|---:|---:|
| maps_me | Maps & Navigation | 9 | 6 |
| osmand | Maps & Navigation | 2 | 1 |
| broccoli | Food & Drink | 8 | 5 |
| opentracks | Health & Fitness | 4 | 2 |
| wikipedia | Books & Reference | 4 | 2 |
| vlc | Video Players & Editors | 1 | 1 |
| simple_draw_pro | Art & Design | 1 | 0 |
| **Total — templates** | | **29** | **17** |
| **Total — episodes** | | **145** (29 × 5) | **51** (17 × 3) |

`simple_draw_pro` has only one template and is excluded from the Tier-C aggregate eval number because its `T_eval` is empty.

### 1.5 `T_adapt` / `T_eval` protocol

Within every target app, templates are partitioned into two **template-disjoint** sets. `T_adapt` is used for online adaptation — FSM mutation in B3 and LoRA updates in B4 happen on `T_adapt` only. `T_eval` is the held-out, frozen-evaluation set used once after adaptation converges; no updates happen during `T_eval`. The headline number reported per tier comes from `T_eval`.

The partition rule is a deterministic alphabetical 60/40 split per app. For N=2 templates → 1/1; for N=1 → all in `T_adapt`, `T_eval` empty. Seeds: `T_adapt` uses K=5 `[30–34]`; `T_eval` uses K=3 `[40–42]`. The two seed lists are disjoint so that even if a `T_eval` template shared parameter values with a `T_adapt` template the realized initial state would differ.

Why template-disjoint and not just seed-disjoint? Because `T_eval` is meant to measure within-app generalization of the adaptation procedure, not parameter robustness on memorized templates. This mirrors meta-learning's support/query split (Finn et al., 2017) and Procgen-style train-level / test-level RL generalization (Cobbe et al., 2020). Template-disjointness is the load-bearing methodological commitment.

### 1.6 Data scale per experiment

| Experiment | Episodes |
|---|---:|
| K=1 leaderboard-style sweep (all 192 templates) | 192 |
| K=5 source-pool Phase-1 pretraining (96 × 5) | 480 |
| K=5 Tier-B `T_adapt` (32 × 5) | 160 |
| K=3 Tier-B `T_eval` (18 × 3) | 54 |
| K=5 Tier-C `T_adapt` (29 × 5) | 145 |
| K=3 Tier-C `T_eval` (17 × 3) | 51 |
| **Grand total, full protocol** | **890** |

---

## 2. The FSM — two-layer in-context knowledge structure

The agent's "knowledge of how to operate an app" is encoded as a two-layer Finite State Machine. The FSM is rendered to text and injected into the agent's **action-selection prompt** as context; it is *not* baked into the model weights (that is LoRA's job, in B4). The two layers separate **non-transferable** (app-specific UI knowledge) from **transferable** (category-generic workflow patterns).

### 2.1 LAYER 1 — app-specific (non-transferable)

LAYER 1 contains states, transitions, visual cues, resource hints, element locations, and concrete dead-ends — all tied to one app's UI. Its strings name specific buttons, dialog titles, and field names. LAYER 1 is meaningless outside that exact app.

**Concrete example — `markor.txt` LAYER 1** (selected states and transitions):

```
APP: markor
CATEGORY: Productivity

STATES:
  S0: HOME_SCREEN — Android home screen
    visual_cues: ['Phone/Chrome/Settings icons', 'Search bar', 'clock widget']
    resource_hints: ['Home', 'Phone', 'Search']
  S1: FILE_LIST — Markor file browser showing files and folders
    visual_cues: ['red plus FAB bottom-right', "'Markor' title",
                  'Sort by/Search/More options in top bar', 'list of files/folders']
    resource_hints: ['Create a new file or folder', 'Markor', 'Go to', 'Sort by']
  S3: NEW_FILE_DIALOG — Dialog to create new file/folder
    visual_cues: ["Name text field pre-filled with 'my_note'", 'extension field',
                  'Type radio group', 'OK/CANCEL/FOLDER buttons']
    resource_hints: ['Name', 'my_note', '.md', 'OK']
  S4: FILE_SELECTED_MODE — File list with one or more items selected via long-press
    visual_cues: ['red checkmark next to file',
                  'action bar shows Favourite/Rename/Details/Delete']
  S8: EDITOR_VIEW — Note editor opened for a file
    visual_cues: ['filename in title bar', 'Undo/Redo/Save/Search in toolbar',
                  'text area with cursor', 'keyboard']

TRANSITIONS:
  S0 --open_app(Markor)--> S1     [pre: Markor installed]  [post: file list shown]
  S1 --click(fab_plus)--> S3      [post: dialog with default 'my_note' appears]
  S3 --click(OK)--> S8            [pre: name entered]  [post: new file opened in editor]
  S1 --click(file_row)--> S8      [pre: file exists]  [post: file opened in editor]
  S1 --long_press(file_row)--> S4 [post: file selected with checkmark]
  S4 --click(Delete)--> S5        [post: confirm-delete dialog]
  S5 --click(OK)--> S1            [post: file deleted]
```

Every visual cue and resource hint above contains a Markor-specific string. The state machine is also Markor-specific: a "FILE_LIST" with a "red plus FAB" is meaningful only for Markor.

### 2.2 LAYER 2 — category-generic (transferable)

LAYER 2 contains abstract workflow patterns, failure modes, and verification checklists written **without any app-specific strings**. Each LAYER 2 block has four fields:

- `precondition` — when this archetype applies
- `abstract_steps` — sequence of app-agnostic moves
- `failure_modes` — known cross-app failure patterns to watch
- `verification_checklist` — post-condition checks the agent should perform before terminating

LAYER 2 categories are task-shape archetypes (`ADD_ENTRY`, `QUERY_INFO`, `REMOVE_ENTRY`, `FILTER_LIST`, etc.) — they describe *what* the task shape is, not *which* app it happens in. A linter (`lint_layer2`) rejects any LAYER 2 diff that contains an app-name or resource-id substring; this is the structural guarantee that category-generic knowledge does not quietly absorb app-specific strings.

**Concrete example — `L_C/productivity.json`, `QUERY_INFO` category** (this `L_C` was aggregated from `markor`, `joplin`, `tasks_org` LAYER 2 blocks):

```json
{
  "name": "QUERY_INFO",
  "precondition": "Target data-bearing screen reachable from launcher; target item/query specifies a filter (date, status, priority, attribute, etc.); app may not be in foreground",
  "abstract_steps": [
    "Launch the relevant app from the launcher",
    "Navigate to the primary list/overview screen or to the named item by browsing the list or using an in-app lookup affordance",
    "Resolve any relative terms in the query (today, next week, after next) to absolute values using the device/task context",
    "Scroll through the full list to enumerate all candidate items",
    "Inspect the visible marker/attribute (icon, prefix, checkbox, color) that indicates the attribute in question",
    "Apply the query's filter predicate to each item using observable visual attributes",
    "Aggregate matching items into the requested output format (list, count, yes/no)",
    "Emit a single answer action, then immediately emit a terminal completion signal"
  ],
  "failure_modes": [
    "Declaring the task infeasible from the launcher without attempting to open the app",
    "Answering based on a screen that does not actually contain the target item",
    "Answering from a partial view without scrolling the full list",
    "Looping on the same answer action without finalizing with a completion status",
    "Confusing relative date labels with the queried absolute date",
    "Guessing semantic attributes (priority, category) from unrelated visual cues",
    "Miscounting due to duplicated or partially visible rows after scrolling"
  ],
  "verification_checklist": [
    "Did I reach the list screen that actually contains the data?",
    "Did I map all relative terms to concrete values consistent with the current date?",
    "Did I scroll until the end of the list (no new content appears)?",
    "Does each returned item satisfy every part of the filter (date AND status AND priority)?",
    "Is the answer format exactly what the user requested (comma-separated / count / None)?",
    "A single completion/status signal is emitted after the answer"
  ]
}
```

Notice that nowhere does this `QUERY_INFO` block name Markor, Joplin, or Tasks_org. The abstraction has been laundered: only generic UI primitives ("primary list/overview screen", "scroll through the full list", "visible marker/attribute") appear. This is what lets the same `L_C` be reused on a target app in the same category (e.g., `simple_calendar_pro`) without any app-specific contamination from the source apps.

The full Productivity `L_C` contains **14 such abstract categories** (`ADD_ENTRY`, `QUERY_INFO`, `REMOVE_ENTRY`, etc.); the agent's action-selection prompt is augmented with the entire `L_C` text block in B2 / B3 / B4.

### 2.3 Per-app static FSM construction (Story 2.2)

For each of the 12 source-pool apps, we synthesize one static FSM from that app's bundle of K=5 trajectories. Pipeline:

```
traces/source_pool_trajectories/{app}_*_seed{30..34}/episode.jsonl
                                  │
                                  ▼
   compress_trajectories(app, traj_dir)   produces a single text bundle of
                                          per-step (screen, action, result)
                                  │       chunks for all K=5 trajectories
                                  ▼
   build_fsm(app, category, compressed)   Claude Opus 4.7 (1M context) reads
                                          the bundle and synthesizes a
                                  │       two-layer FSM in JSON
                                  ▼
   lint_layer2(fsm)                       rejects FSMs whose LAYER 2 contains
                                          any app-name / resource-id leakage
                                  │
                                  ▼
   artifacts/static_fsms/{app}.json   +   structured form
   artifacts/static_fsms/{app}.txt        human-readable serialization
```

Driver: `scripts/build_all_fsms.py`. Synthesis model: Claude Opus 4.7 (1M context) — large context lets it ingest all 5 seeds of trajectory text in one call. All 12 source apps have static FSMs. An audit of the 12 (Story 2.2 → 2.3 gate) returned 6 PASS / 6 WARN / 0 FAIL; the WARNs are surface-level coverage gaps (e.g., dangling states around external apps) and none are abstraction leakage.

### 2.4 Per-category `L_C` construction (Story 2.3)

Each of the 6 source-pool categories contributes one `L_C` file — a **union of the LAYER 2 blocks** of same-category source apps with duplicates merged and contradictions resolved. For Productivity, the three contributors are `markor`, `joplin`, `tasks_org`; their respective LAYER 2 blocks are passed to Claude Opus with a merge prompt that retains only claims supported by multiple source apps, explicitly enumerates cross-source failure modes, and strips every app-specific string.

```
artifacts/static_fsms/{markor,joplin,tasks_org}.json
                            │
                            ▼
       aggregate_L_C(fsms)           Claude Opus 4.7 union prompt
                            │
                            ▼
       lint_L_C(merged, source_fsms) hard fail if ANY source app's
                            │       specifics leak into the merged file
                            ▼
       artifacts/L_C/productivity.json
```

Driver: `scripts/build_L_C.py`. The 6 resulting `L_C` files:

| Category | # source apps | # abstract categories in `L_C` |
|---|:---:|:---:|
| Productivity | 3 | 14 |
| Tools | 3 | 15 |
| Music & Audio | 2 | 11 |
| Communication | 2 | 7 |
| Finance | 1 | 7 |
| Photography | 1 | 6 |

Single-source categories (Finance, Photography) pass `aggregate_L_C` trivially — no union needed; the per-app LAYER 2 *is* the `L_C` for that category.

At deployment time, `resolve_l_c_for_app(a_target)` looks up `play_category_of(a_target)` and returns the matching `L_C`, or `None` if the category is not in the source pool (the Tier-C condition).

---

## 3. The four baselines

The four baselines form an **ablation ladder**, each rung isolating one mechanism of EvoFSM-RL. The headline metric for every baseline is success rate on the K=3 `T_eval` set, reported separately per tier.

### 3.1 B1 — zero-shot

B1 is the floor: a clean Qwen3-VL-8B-Instruct on the M3A two-phase loop, with no FSM and no `L_C` injected anywhere. Per step the agent does one LLM call to pick an action (`Reason: … Action: {JSON}`), executes it on the emulator via ADB, then does a second LLM call to summarize the step into history. Generation is bf16, temperature 0.0, `max_pixels=1605632` on the A100/H100 path. Success is AndroidWorld's rule-based `is_successful(env)` at episode end — self-reported `status:complete` does not count on its own. Step budget is `max_steps_multiplier=10.0` × per-template default.

Code path: `scripts/run_b2_eval.py --no-l-c` is the canonical way to run B1 (B1 is B2 with the injection turned off).

**Headline result.** Tier-B `T_eval` K=3 SR = **47.2% ± 6.8%** (n=54).

### 3.2 B2 — static `L_C` injection

B2 keeps the B1 agent and generation config unchanged. The only difference is that when `resolve_l_c_for_app(app)` returns a category match, that category's `L_C` text is injected into the **action-selection prompt only**, after the `PROMPT_PREFIX` and before the dynamic goal/history/UI block. The injection sits in the KV-cache-stable prefix so latency overhead is ~zero after warm-up. The summary prompt is unchanged — LAYER 2 abstraction lives in the planning context and stays out of post-hoc rationalization.

`L_C` is **frozen** in B2: no evolution, no LoRA update. For Tier-C apps `resolve_l_c_for_app()` returns `None`, so the Tier-C action-selection prompt is byte-identical to B1's — Tier-C therefore acts as a built-in null control for the injection mechanism.

Code path: `scripts/run_b2_eval.py`.

**Headline result.** Tier-B `T_eval` K=3 SR = **56.5% ± 6.7%** (n=54), **+9.3 pp over B1**. Tier-C SR matches B1 exactly (29.4% on both arms — null control passes).

### 3.3 B3 — evolved `L_C` (FSM evolves, weights frozen)

B3 replaces B2's static `L_C` with an *evolved population* of FSM variants. Per target app, an online adaptation pass on `T_adapt` grows a tree of FSMs rooted at the `L_C`-seeded initial FSM. LoRA is not attached in B3 — only the FSM (the in-context knowledge) evolves; the model weights stay frozen at base Qwen3-VL-8B.

#### How the population works

The population `S_a` starts with a single root variant: `{(F_root, μ₀=25.0, σ₀=8.33)}`. `F_root` is the category-matched source-pool `L_C` injected directly as the LAYER 2 block; the target app's LAYER 1 is not extracted ahead of time. (In `plan/algorithm_design.md` §8, Phase 2 specifies 5–10 exploration rollouts of `π^pre_θ` on the target app to build a target-app LAYER 1 + then hand off to evolution. The current implementation skips that exploration step and lets the agent learn the target app's LAYER 1 from in-context vision during the B3 loop itself. This is a deliberate scope reduction — LAYER 2 is what we transfer; LAYER 1 is rediscovered online.)

Each variant carries a **TrueSkill rating** `(μ, σ)`: μ is the variant's estimated skill, σ is the uncertainty in that estimate. New children inherit the parent's μ and get σ inflated by `Δσ ≈ 1.5` to give them exploration credit. No elitism, no explicit pruning — only a sliding window of the latest K=15 variants is selectable.

#### One iteration walked through, end to end

Concretely, here is what happens at iteration `i` on `simple_calendar_pro`:

**Step 1 — sample a task.** `t ~ T_adapt`. Suppose `t = SimpleCalendarAddOneEventRelativeDay`, seed = `100 + i`.

**Step 2 — select M=2 variants.** Optimistic softmax over the latest 15: `p_i ∝ exp((μ_i + 1.5·σ_i) / 1.0)`. The `1.5·σ_i` term biases selection toward high-uncertainty (newly-introduced) variants, giving them a chance to accumulate evidence before being abandoned. Suppose this picks `gen0_root` and `gen1_mut_1`.

**Step 3 — roll out each variant once (N=1) on the SAME task with the SAME seed.** Both variants get `rollout_seed = base + i·100 + 0`. The only source of reward difference is therefore the FSM content, not the task initial state. We get `r_root = 0.0` and `r_mut_1 = 1.0` (say).

**Step 4 — update TrueSkill.** `mut_1` beat `root` this tournament round → μ_mut_1 climbs, μ_root drifts down, both σ values shrink (more evidence).

**Step 5 — identify the best.** `k = argmax_i V_i = mut_1`.

**Step 6 — decide whether to mutate.** Mutate the best variant if `best_reward > 0` (there is signal to reflect on) OR if `i % 3 == 0` (force periodic exploration even if everyone is failing). Both conditions are satisfied here.

**Step 7 — mutate.** Call Claude Opus 4.7 as a **frozen reference policy π_ref** (this is *not* the trainable Qwen3-VL-8B agent — using the same model for both rollouts and mutation introduces drift that degrades edit quality; E-SPL ablates this in their Fig. 15 and we follow that finding):

```
reflection_prompt = build_layered_reflection_prompt(
    fsm = F_mut_1,
    trajectories = [(t, τ_mut_1, r=1.0)],
    category = "Productivity",
)
ℓ_L1, ℓ_L2 ~ π_ref(reflection_prompt)          # self-reflection on what worked / failed
diff       ~ π_ref(F_mut_1, ℓ_L1, ℓ_L2)         # propose structured FSM edit
F_child    = apply_fsm_diff(F_mut_1, diff)      # apply the diff
S_a.append((F_child, μ_mut_1, σ_mut_1 + 1.5))
```

**Step 8 — what the diff actually looks like.** The mutation operator is restricted to `layer2_only=True` — only LAYER 2 blocks may be edited; LAYER 1 (app-specific) is frozen during B3 because it was extracted offline. A typical diff:

```json
{
  "ops": [
    {
      "op": "add_verification_check",
      "category": "QUERY_INFO",
      "value": "If the overview shows abbreviated/truncated item text, drill into the item-detail screen before answering"
    },
    {
      "op": "add_failure_mode",
      "category": "QUERY_INFO",
      "value": "Reading truncated overview text as the final answer when month-view truncates titles"
    }
  ]
}
```

This is exactly the kind of edit needed for the `simple_calendar_pro` month-view truncation bug documented in `b1_b2_static_baselines.md` §5. (Whether B3 actually proposes such a diff inside the 20-iter budget is a separate question — the B3 sweep results in `b3_evolution.md` show it sometimes does, sometimes doesn't.)

**Step 9 — stochastic crossover (p ≈ 0.3).** With low probability, sample M variants and call Claude Opus to synthesize a child that combines complementary strengths of two parents across a replay batch of past tasks. The child enters the population with a precision-weighted-average μ and σ inflated by Δσ.

#### Aggregate B3 sweep configuration

- 20 iterations per app
- M=2 variants per iter, N=1 rollout each
- Mutation model: Claude Opus 4.7 (Anthropic API)
- Tier-B only (Tier-C has no `L_C` to evolve and degrades to B1)
- Total: 6 apps × 20 × 2 × 1 = **240 adaptation rollouts**, +55 successful mutation calls + 1 mutation error (recoverable) over 14h 6 min wall

#### What gets handed off to evaluation

After 20 iterations, the population may contain ~7–15 variants. The champion (highest μ) is saved as `l_c_champion.json`. The `T_eval` evaluation pass uses this frozen champion's LAYER 2 as the static `L_C` injection — same prompt-injection mechanism as B2, just with the evolved champion instead of the original source-pool `L_C`.

Code paths: `scripts/run_b3_evolution.py` (per-app evolution), `scripts/run_b3_teval.py` (frozen-champion T_eval), `evofsm_rl/fsm/evolution.py::run_l_c_evolution` (loop body), `evofsm_rl/fsm/builder.py` + Anthropic API client (mutation).

**Headline result.** Tier-B `T_eval` K=3 SR = **63.9%**, **+3.7 pp over B2**. Signal concentrated in `system_settings` (+11.1 pp on n=18) and `retro_music` (+33.3 pp on n=3); other apps tied or within n=3 noise. Tier-C tied at ~0 pp (null control verified — Tier-C reuses the no-injection path).

### 3.4 B4 — joint LoRA + L_C co-evolution (TTA centerpiece)

B4 is B3 plus a second moving part: **the policy's LoRA adapter is updated jointly with the FSM**. Both axes co-adapt during the test-time-adaptation pass on each target app's `T_adapt`. This is the paper's TTA selling point — inspired by E-SPL (Zhang et al., 2026) which showed that, on agentic settings, the combination of context evolution and weight RL beats either axis alone (agentic search: base 6.6% → context 34.0% → RL 44.2% → both 48.6%).

To make this work in practice, B4 is structured in **two phases**, each running the *same* joint EvoFSM-RL algorithm but on different data pools and with different compute budgets.

#### Phase 1 — pretraining (one-shot, over the source pool)

The first phase produces the **shared LoRA adapter `π^pre_θ`** by running the joint loop across the 12 source-pool apps. Conceptually:

- Source pool = "training set" — 12 apps, 96 templates, 5 seeds each.
- Each Phase 1 iteration samples one app uniformly, samples one of its tasks, runs N rollouts under the app's static FSM, and accumulates the trajectories into a GRPO buffer that is shared across all 12 apps.
- When the buffer fills, **one GRPO step updates the single shared LoRA** — gradients from rollouts on `markor` and `bluecoins` and `calculator` all flow into the same adapter.
- The static FSMs are frozen during Phase 1 (per the minimal-Phase-1 scope; full Phase 1 would also evolve the FSMs and aggregate `L_C` cross-app — those products already exist via simpler pipelines, so we skip).

Phase 1 runs once and is a fixed cost. The output `π^pre_θ` is then frozen and reused at every Tier-B and Tier-C target-app deployment. A "minimal Phase 1" pilot of 4 apps × 200 iter is the first run we are now executing; the full 12-app version is a follow-up if the pilot shows the shared LoRA actually learns.

#### Phase 3 — TTA on a single target app

The second phase is what happens at deployment time on a new target app. **It is the same joint loop, but on the target app's `T_adapt` and starting from `π^pre_θ`** (not random LoRA). The FSM half of the loop is the B3 evolution from §3.3; the LoRA half is GRPO continuing to update from where Phase 1 left off. Per target app, Phase 3 runs for 20 iterations (matching the B3 sweep budget) and ends by freezing both the LoRA adapter and the FSM champion. Evaluation on `T_eval` runs with both frozen — same hand-off mechanic as B3.

The reason Phase 3 needs `π^pre_θ` and cannot start from base model: 20 iterations × M=2 × N=2 ≈ 80 rollouts is far below the compute budget needed to train LoRA from scratch. v2's no-learning result came from exactly this — random LoRA, 20 iters, no learning even before F1/F5 were applied. The Phase-1 → Phase-3 split is what makes joint TTA tractable at deployment-time compute scale.

#### The GRPO update — how a weight step actually happens

When the GRPO buffer fills (K trajectories accumulated), one weight step does the following.

**Reward.** Each rollout `τ` on task `t` gets

```
R(t, τ) = success(t, τ) + 0.2 · max(0, 1 − n_steps(τ) / step_budget(t))
```

where `success ∈ {0.0, 0.5, 1.0}` from AndroidWorld's rule-based grader and the 0.2 efficiency bonus is small enough that success dominates. Source: `evofsm_rl/rl/grpo.py::compute_reward`.

**Advantage.** GRPO has no critic. The baseline is the within-group mean reward. **Groups are keyed by `(fsm_variant_id, task_name)` tuple**: for every (FSM, task) cell seen in the buffer, the advantage of rollout j is

```
A_{i,j} = r_{i,j} − V_i,   V_i = mean over rollouts in group i
```

This matches the original §3.2 design: each iteration runs M FSMs × N rollouts on a *single sampled task*, and each (FSM, task) cell provides its own baseline. **N ≥ 2 rollouts per (FSM, task) is required** because under this grouping a singleton group's advantage is constructively 0 (no peer to subtract). A historical note: through v1/v2 the implementation grouped by `fsm_variant_id` alone, which collapsed the design to a REINFORCE-with-per-FSM baseline; the (FSM, task) grouping was restored on 2026-05-13 (the F5 fix) and is now GPU-verified.

**Loss.** For active trajectories (group size ≥ 2),

```
L = −Σ_j  (1 / T_j) · A_j · Σ_t  log P(a_t | s_t, F, task)
                            └────────────┬───────────┘
                                fresh forward pass through the
                                LoRA-bearing model recomputes
                                each step's log-prob with gradient
```

The `1 / T_j` factor (the F1 fix from 2026-05-12) normalizes each trajectory's contribution by its length, so a 30-step rollout and a 1-step rollout contribute equally to the gradient direction. Without F1, long trajectories dominate and the resulting `grad_norm` lands in 50–270, then `max_grad_norm = 1.0` clips it 50–270× — squashing the actual learning direction. After F1, smoke shows `grad_norm ≈ 0.2–0.4` (well below the clip), so the clip is now an inactive safety net.

**Optimizer.** `torch.optim.AdamW` over LoRA-only trainable parameters. The base Qwen3-VL-8B weights are frozen; only the LoRA A and B matrices on `q_proj` and `v_proj` update. LoRA rank=16, α=32, dropout=0.05, LR=3e-4. No KL penalty (LoRA's implicit regularization replaces it).

#### Putting one Phase-1 iteration together, end to end

```
# Pseudocode for one Phase 1 iter:
a   ~ Uniform({markor, bluecoins, calculator, contacts})
t   ~ Uniform(templates[a])
F   = static_fsm[a]                            # frozen, app-specific
for j in 1..N:                                  # N=2
    τ_j = rollout(π_θ, system_prompt=F, task=t, seed=seed_base + iter·100 + j)
    r_j = R(t, τ_j)
buffer.add(trajectories)

if buffer.size >= K:                            # K=10
    advantages = compute_advantages(buffer)     # grouped by (F.app, t)
    # build a single backward over all active groups:
    L = 0
    for τ_j, A_j in zip(buffer, advantages):
        L += (1 / T_j) · A_j · Σ_t log P(a_t | s_t, F, t)   # F1 scaling
    L.backward()
    clip_grad_norm_(LoRA_params, max=1.0)
    optimizer.step()                            # AdamW update on shared LoRA
    optimizer.zero_grad()
    buffer.clear()

every 50 iters: save_lora_checkpoint(model)
```

Phase 3 differs in only two places: (i) the app and task pool are restricted to one target app's `T_adapt`; (ii) the FSM is no longer frozen — between buffer-fire steps, the B3 evolution machinery (TrueSkill + mutation + crossover) also runs, growing a per-target-app population. The same GRPO step continues to update the same LoRA, now starting from `π^pre_θ` instead of random init.

#### B4 current status (2026-05-13)

| Step | Status |
|---|---|
| B4 framework (`scripts/run_b4_evolution.py`) | Done (2026-04-22) |
| B4 v1 sweep | Aborted, OOM in 4/6 apps (2026-04-22) |
| B4 v2 sweep | Completed, pipeline stable but no learning signal (2026-04-24) |
| F1 — loss scaling by `1/T_j` | Applied + GPU-verified (2026-05-12) |
| F2 — `adv_std` diagnostic | Applied (2026-05-12) |
| F5 — `(FSM, task)` group key | Applied + GPU-verified (2026-05-13) |
| `--init-lora-from <path>` on B4 runner | Added 2026-05-13 |
| **Phase 1 pilot run (4 apps × 200 iter)** | **Running in tmux `phase1_pilot`, ~13 h ETA** |
| Phase 3 TTA from `π^pre_θ` | Pending Phase 1 completion |
| B4 vs B3 paper numbers | Pending Phase 3 |

The v2 sweep's "no learning" result is not evidence that B4 doesn't work as an algorithm — it reflects four layered issues now addressed: (1) F1 was missing, so the gradient signal was being clipped; (2) F2 was missing, so we were reading the wrong metric (`mean_advantage`, constructively 0) instead of `advantage_std`; (3) F5 was missing, so the group key was wrong; (4) `π^pre_θ` was missing, so per-app LoRA started from random init with only 20 iters of compute — far below the budget needed. All four are addressed in the current run.

### 3.5 Results so far — B1, B2, B3 on T_eval K=3

All numbers below are mean success rate on the K=3 `T_eval` set, mean ± binomial SE. The B1/B2 paired arm ran on seeds `[30, 31, 32]` (`docs/results/b1_b2_static_baselines.md`); the B2/B3 paired arm rebuilt B2 on seeds `[40, 41, 42]` to match the canonical `T_eval` seed list and produce a like-for-like B3 comparison (`docs/results/b3_evolution.md`). The two B2 numbers (56.5% vs 60.2%) bound the n=54 seed-set noise floor.

#### Aggregate per tier

| Arm | Tier-B SR (n=54) | Tier-C SR (n=51) | Overall (n=105) |
|---|:---:|:---:|:---:|
| B1 (zero-shot) | 47.2% ± 6.8% | 29.4% ± 6.4% | 38.6% ± 4.8% |
| B2 — paired with B1 (seeds 30–32) | 56.5% ± 6.7% | 29.4% ± 6.4% | 43.3% ± 4.8% |
| B2 — paired with B3 (seeds 40–42) | 60.2% | 29.4% | 45.2% |
| B3 (evolved `L_C`) | **63.9%** | 31.4% | **48.1%** |
| B4 (evolved `L_C` + LoRA) | **TBD** | **TBD** | **TBD** |

Tier-C is the **null control** — under B2/B3, Tier-C apps get no `L_C` injection (no category match) and degrade byte-identically to the no-injection B1 path. The tiny +2.0 pp B2 → B3 Tier-C drift comes from a single template × single seed flip on `broccoli/RecipeDeleteMultipleRecipesWithNoise` (paired-run variance from emulator + bf16 CUDA non-determinism), not a methodological effect.

#### Ablation deltas

| Comparison | Mechanism isolated | Tier-B Δ | Tier-C Δ |
|---|---|---:|---:|
| B2 − B1 | Static category-knowledge transfer | **+9.3 pp** | +0.0 pp (null) |
| B3 − B2 | Online evolution of LAYER 2 knowledge | **+3.7 pp** | +2.0 pp (null, expected ≈0) |
| B4 − B3 | Joint weight co-adaptation at TTA | **TBD** | **TBD** |

#### Per-app Tier-B breakdown

Numbers are SR (T_eval K=3). `T_eval` columns: templates × 3 seeds = episodes. B2 column is the version paired with the report each row's right-hand baseline lives in — B2 vs B1 uses seeds 30–32 (b1_b2 report); B3 vs B2 uses seeds 40–42 (b3 report).

| App | T_eval templates | T_eval episodes (K=3) | B1 SR | B2 SR (vs B1) | Δ B2−B1 | B2 SR (vs B3) | B3 SR | Δ B3−B2 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| simple_calendar_pro | 6 | 18 | 38.9% | 38.9% | +0.0 | 50.0% | 50.0% | +0.0 |
| system_settings | 6 | 18 | 75.0% | 88.9% | +13.9 | 80.6% | **91.7%** | **+11.1** |
| pro_expense | 3 | 9 | 33.3% | **66.7%** | **+33.3** | 66.7% | 66.7% | +0.0 |
| retro_music | 1 | 3 | 66.7% | 50.0% | −16.7 | 33.3% | **66.7%** | **+33.3** |
| camera | 1 | 3 | 0.0% | 0.0% | +0.0 | 66.7% | 33.3% | −33.3 |
| chrome | 1 | 3 | 0.0% | 0.0% | +0.0 | 0.0% | 0.0% | +0.0 |
| **Tier-B overall** | **18** | **54** | **47.2%** | **56.5%** | **+9.3** | **60.2%** | **63.9%** | **+3.7** |

#### Reading the deltas

- **B2 − B1 = +9.3 pp Tier-B.** Static category knowledge helps. The signal concentrates on apps whose source-pool category had strong structural overlap with the target: `pro_expense` (+33.3 pp, Finance `L_C` from `bluecoins`) and `system_settings` (+13.9 pp, Tools `L_C` from `calculator`/`clock`/`files`). The Tier-C null delta is exactly +0.0 pp — confirms no spurious behavioral drift outside the intended scope.
- **B3 − B2 = +3.7 pp Tier-B.** Online evolution narrows the residual gap; concentrated on `system_settings` (+11.1 pp at n=18) and `retro_music` (+33.3 pp at n=3, within single-seed swing). The aggregate is fragile: removing `system_settings` drops the Tier-B Δ to roughly +0 pp across the remaining 5 apps. This is **a single-anchor result** and must always be reported with the per-app breakdown.
- **`simple_calendar_pro` is B3's biggest open problem.** It is the largest-n Tier-B app (n=18 across `T_eval`) and tied at 50.0% B2→B3. The expected gain site was the month-view truncation bug documented in `b1_b2_static_baselines.md` §5; the 12 mutations on this app did not converge on a fix inside the 20-iter budget. This is where B4's weight co-adaptation is most likely to help — the truncation handling may be procedural (a learned visual habit) rather than something LAYER 2 prose can describe.
- **`chrome` and `camera` tied at 0% / regressed.** These are structural failures upstream of `L_C`: `chrome`'s sole T_eval template hits an `open_app: "File Manager"` hallucination loop at the harness level (parser bug, not a knowledge gap); `camera/CameraTakeVideo` involves precise gesture timing that `L_C` text cannot describe. Neither is fixable by FSM evolution alone; both are candidates for B4 if reward shaping is added later.

#### Statistical caveat

K=3 is underpowered for declaring significance. For B2−B1 Tier-B, the two-sample z = +0.97 (p ≈ 0.17 one-sided); for B3−B2 Tier-B, z ≈ 0.4. Direction is consistent across both sweeps but n=54 per arm is the bottleneck. Bumping to K=5 (n=90) is reserved for the camera-ready if reviewers press on CI width.

Tier-C is where B4 is expected to matter most — there is no `L_C` to lean on, so weight co-adaptation has to carry the entire transfer load. The B1/B2/B3 Tier-C numbers all hover around 29–31% (essentially the same value, since none of those three methods touch Tier-C in a methodologically distinguishable way). B4 will be the first arm to exercise an actual learned mechanism on Tier-C.

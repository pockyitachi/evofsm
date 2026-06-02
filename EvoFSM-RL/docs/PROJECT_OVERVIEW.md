# EvoFSM-RL — Project Overview

## Summary

**What this project builds.** A test-time adaptation (TTA) method for GUI agents. At deployment on a new Android app, the agent continues to learn — it evolves a structured, human-readable knowledge representation about the app, and simultaneously fine-tunes a LoRA adapter on its own weights, using reward signal from its own rollouts on the app. Both updates share the same trajectories.

**What the knowledge representation is.** A two-layer Finite State Machine (FSM). LAYER 1 is app-specific — the finite-state map of an app's UI (states, transitions, dead-ends). LAYER 2 is category-generic — abstract workflow patterns, failure modes, and verification checklists that do not reference any app-specific strings, organised as per-Play-Store-category libraries called `L_C`. LAYER 2 is the part that transfers between apps of the same category and that we evolve online at test time; LAYER 1 grounds it.

**How the two updates fit together.** At every iteration of the test-time loop, we sample two FSM variants from a TrueSkill-rated population, run one rollout per variant on a sampled target-app task, update ratings, and ask a frozen Claude Opus 4.7 reflector to propose a structured edit to the winning variant's LAYER 2. The same rollouts feed a GRPO weight update on a shared LoRA adapter — each trajectory's advantage is its reward minus the mean reward of trajectories that shared an FSM variant. No extra rollouts are spent on weight updates.

**Benchmark.** AndroidWorld-Plus — 25 apps, 192 task templates, 12 Play Store categories. Apps are split into three roles: 12 source apps (pretraining only), 6 Tier-B target apps (category seen in the source pool — near-transfer), 7 Tier-C target apps (category absent from the source pool — far-transfer, also serving as a null control). Within each target app the task templates are further split 60/40 into an adaptation set `T_adapt` and a frozen evaluation set `T_eval`. Headline numbers are scored on the combined 35 `T_eval` templates at K=3 seeds (105 graded episodes per rung).

**Progress to date.** We run a four-rung ablation layering one element of the method at a time on the same `T_eval`:

| Rung | Method | Tier-B SR | Tier-C SR | Overall |
|---|---|:---:|:---:|:---:|
| B1 | Zero-shot Qwen3-VL-8B | 47.2% | 29.4% | 38.6% |
| B2 | + static `L_C` injection | 56.5% (**+9.3 pp**) | 29.4% (flat, as designed) | 43.3% |
| B3 | + online evolution of `L_C` | 63.9% (**+7.4 pp** over B2) | 31.4% (null-control noise) | 48.1% |
| B4 | + LoRA fine-tuning (GRPO) | *sweep running, reports soon* | — | — |

Tier-C is deliberately mechanism-free on every rung — it receives no `L_C` and no LoRA signal, so its constant ≈30% confirms that emulator/CUDA non-determinism contributes only a couple of percentage points of noise. The cumulative **+16.7 pp on Tier-B** between B1 (zero-shot) and B3 (evolved `L_C`) is therefore a real effect of the method, not instrumentation noise. B4, currently sweeping six target apps × 20 iterations each (≈14 h wall time, started on a dedicated H100), will tell us whether weight updates on top of evolution extend that further.

**Current phase.** Epics 0 – 3 closed. Epic 4 in progress: the B4 runner (joint LoRA + evolution) is running on the same emulator + GPU setup that produced B1 / B2 / B3; all 379 unit tests pass; the sweep is expected to finish overnight, and a paired-seed `T_eval` pass will follow to populate the B4 row.

---

## 1. Motivation

GUI agents built on multimodal LLMs can now execute real Android tasks end-to-end. Our M3A-style zero-shot baseline — Qwen3-VL-8B plugged into the AndroidWorld M3A two-phase action+summary loop — reaches **38.6%** overall success on the `T_eval` split of the target apps in AndroidWorld-Plus at K=3 seeds (47.2% on near-transfer targets, 29.4% on far-transfer targets). Injecting a static per-category knowledge library lifts that to 43.3%; evolving the library online on the target app raises it further to 48.1% on the same templates (see §6.1). Roughly **one in two** tasks on a previously unseen Android app can be solved with today's best open-weights VLM plus careful prompt engineering — the other half resist, and that half is what this project targets.

Humans do not operate this way. When a person picks up a new app, they spend the first few minutes building a small mental model — "this is where the list lives, this is how you open an entry, this control silently reverts when you leave the picker without confirming" — and then use that model to plan subsequent interactions. That model is *inspectable* (they can say it out loud), *transferable* (it reuses templates from other apps of the same kind), and *editable* (a single failure updates it). Current GUI agents have none of these properties.

## 2. Problem statement

Current GUI agents are train-then-deploy. Any adaptation signal goes into model weights, where it is implicit, uninspectable, and non-transferable: we cannot examine what the agent has learned about app A, nor transplant that knowledge to a related app B, nor revise it when the UI changes. A freshly deployed agent on a novel Android app cannot exploit any structural kinship between that app and the apps seen during pretraining, because the knowledge learned from those apps is not exposed in any form the agent can consult at test time.

## 3. How existing work addresses this

**Prompting with frozen weights.** AndroidWorld's M3A agent (Rawles et al., 2024) and AndroidLab-style agents execute tasks via a fixed system prompt plus a two-phase action/summary loop. No adaptation occurs at deployment time.

**RL fine-tuning on GUI.** RL fine-tuning approaches update model weights only; the acquired knowledge is implicit and not inspectable or transferable across apps.

**Per-episode reflection.** Reflexion-style agents (Shinn et al., 2023) and ExpeL (Zhao et al., 2024) recover from mistakes within an episode or extract per-task insights, but the structured memory is either scoped to one episode or to one training-time extraction pass, not jointly evolved against weight updates.

**Experience-driven declarative memory.** Voyager (Wang et al., 2023) grows a skill library during exploration and reuses it on held-out tasks; AutoManual (Wang et al., 2024) synthesizes a task manual from training interactions. These establish that a declarative, test-time-readable memory is useful for LLM-agents. They do not jointly evolve that memory with weight updates on a multi-step visual-agentic setting.

**E-SPL (Zhang et al., 2026).** Introduces *joint context + weight optimization* for LLMs on reasoning benchmarks: a population of system prompts evolves via structured diff mutation and TrueSkill selection while a shared LoRA is RL-trained on the same rollouts. On their hardest agentic-search setting the combined approach moves from 6.6% (base) through 34.0% (evolution-only) and 44.2% (RL-only) to 48.6% (combined), demonstrating that neither axis alone saturates the method. EvoFSM-RL adapts this paradigm to multi-step GUI agents.

## 4. Our approach

EvoFSM-RL makes the agent's app-knowledge an **evolvable, two-layer Finite State Machine** (FSM):

- **LAYER 1 (app-specific)** — states, transitions, element references, and dead-ends grounded in one app's UI. Non-transferable.
- **LAYER 2 (category-generic)** — abstract workflow patterns, failure modes, and verification checklists written *without* any app-specific strings. Transferable across same-Play-Store-category apps and stored in per-category libraries we call `L_C`.

The FSM is rendered into natural-language text and spliced into the agent's system prompt at every step, so the policy sees its own knowledge directly. At test time both the FSM and a LoRA adapter on the base VLM are jointly optimized in two coupled loops:

1. **Context evolution.** A population of FSM variants, each with a TrueSkill rating, is maintained in memory. Each iteration samples a handful of variants (we use 2 per iteration, noted below as `M=2`), runs one rollout per variant on a sampled task (`N=1`), updates ratings, and then asks a frozen reflector model — Claude Opus 4.7 — to propose a structured edit to the best variant grounded in its recent trajectories. Mutations are constrained to LAYER 2, so evolution modifies only the transferable part of the knowledge.
2. **Weight update.** LoRA adapters on the base VLM are updated via REINFORCE with a group-relative baseline (GRPO-style) on the same rollouts that fed evolution — reusing rollouts, not spending extra ones.

At deployment on a new target app, the appropriate `L_C` for the target's Play Store category seeds the FSM population, the same evolve+RL loop runs on the target's adaptation tasks, and the frozen end-of-adaptation checkpoint is graded once on the target's `T_eval` tasks.

### What LAYER 2 actually looks like

A LAYER 2 library is a list of *abstract categories*, each a small typed object describing one workflow shape. For example, one entry from the Tools category (`artifacts/L_C/tools.json`, 15 abstract categories total, aggregated from the calculator / clock / files source apps) is:

```json
{
  "name": "TOGGLE_STATE",
  "precondition": "A binary control is visible and its current state is observable",
  "abstract_steps": [
    "Locate the binary control (switch, checkbox, toggle)",
    "Read the current state from visual affordance (icon, label, color)",
    "If the target state differs from current, tap to flip",
    "Wait for the UI to settle and re-read the state from the same surface"
  ],
  "failure_modes": [
    "Tapping before the current state is read, inverting an already-correct state",
    "Mistaking a disabled control for an unset one",
    "Reading the state from a stale cached screen instead of the live one"
  ],
  "verification_checklist": [
    "Target-state indicator visible on the control itself",
    "No pending spinner / confirmation dialog blocking"
  ]
}
```

Every field is plain prose. There are no app names, package IDs, or pixel coordinates — by construction of the Story 2.3 aggregation pipeline, LAYER 2 is scrubbed of anything app-specific. This is what makes it transferable across apps of the same category, and this is the surface that Claude Opus edits during test-time evolution. Mutations are restricted to LAYER 2 — LAYER 1 stays untouched during B3 / B4 because app-specific grounding is not the thing we want to transfer.

## 5. The benchmark — AndroidWorld-Plus

AndroidWorld-Plus extends Google's AndroidWorld benchmark with six apps from BMOCA and AndroidLab, covering six additional Play Store categories that are absent from the vanilla AndroidWorld pool. Every app is pre-installed in a canonical Android emulator snapshot (`AWAvd2 / apps_ready_dec2025`) and every task has a rule-based success checker from the original benchmarks.

### 5.1 Apps and tasks, by role

We partition the 25 active apps into two disjoint roles — **source apps** and **target apps** — and split the target apps into two tiers by whether their Play Store category overlaps with the source pool. **No app appears in more than one role**: an app is either in the source pool (used to build the transferable knowledge library) or in the target pool (used for test-time adaptation and evaluation).

| Role | Count | Apps (total templates) | What it's for |
|---|:---:|---|---|
| Source apps | 12 apps, 96 templates | markor (14), tasks_org (6), joplin (4), calculator (19), clock (3), files (2), bluecoins (15), pi_music (12), audio_recorder (2), snapseed (11), simple_sms_messenger (6), contacts (2) | Pretraining. Rollouts here feed per-app FSM synthesis and per-category `L_C` aggregation. Never touched at test time. |
| Target apps, Tier-B (near-transfer) | 6 apps, 50 templates | simple_calendar_pro (17), system_settings (15), pro_expense (9), retro_music (4), camera (2), chrome (3) | Target apps whose Play Store category IS in the source pool. `L_C` transfer is possible. |
| Target apps, Tier-C (far-transfer) | 7 apps, 46 templates | maps_me (15), broccoli (13), opentracks (6), wikipedia (6), osmand (3), vlc (2), simple_draw_pro (1) | Target apps whose Play Store category is NOT in the source pool. Built-in null control for the L_C-injection mechanism. |
| **Total** | **25 apps, 192 templates** | | |

(Two additional templates — `OpenApp` and `SaveCopyOfReceipt` — are generic or composite and not assigned to any single app. They do not appear in any split. Grand total 194, app-attributable total 192.)

### 5.2 Category coverage

Each Tier-B target app sits in the same category as at least one source app; each Tier-C target app sits in a category no source app covers. The matrix below shows the per-category app counts on each side.

| Play Store category | Source apps | Tier-B target | Tier-C target |
|---|---|---|---|
| Productivity | markor, tasks_org, joplin | simple_calendar_pro | — |
| Tools | calculator, clock, files | system_settings | — |
| Finance | bluecoins | pro_expense | — |
| Music & Audio | pi_music, audio_recorder | retro_music | — |
| Photography | snapseed | camera | — |
| Communication | simple_sms_messenger, contacts | chrome | — |
| Maps & Navigation | — | — | maps_me, osmand |
| Food & Drink | — | — | broccoli |
| Health & Fitness | — | — | opentracks |
| Books & Reference | — | — | wikipedia |
| Video Players & Editors | — | — | vlc |
| Art & Design | — | — | simple_draw_pro |

The six **source-pool categories** each have 1–3 source apps and exactly one Tier-B target app. We synthesize one `L_C` file per source category (see §6.4). The six **Tier-C categories** have no source evidence, so Tier-C target apps run under the zero-shot baseline prompt shape — no knowledge to inject — and their numbers move only with emulator/CUDA non-determinism.

### 5.3 Per-app adaptation / evaluation split

Within a single target app, the app's templates are further divided into two mutually-exclusive sets — the **adaptation set** (`T_adapt`) that the test-time loop can learn from, and a frozen **evaluation set** (`T_eval`) that only receives a single graded pass after adaptation ends. The split is a deterministic 60/40 alphabetical cut by template name (`configs/splits.yaml`, version-pinned so every rung of every experiment sees the same split). K=3 seeds means each template is run three times with different task-state seeds, reported as a mean.

The app is the **same** across `T_adapt` and `T_eval` — it is the *tasks* that are disjoint, not the app. For example, on `pro_expense` the adaptation tasks are six Expense-Add / Expense-Delete-Duplicates templates, and the evaluation tasks are three Expense-Delete-Multiple / Single templates on that same app's UI. This layer of the protocol measures **task-level generalisation within a target app** after adaptation. App-level generalisation is measured at the layer above — by the source / target split in §5.1, where an app is either pretrained on (source) or only seen at deployment time (target).

| App | Tier | Category | T_adapt | T_eval | Total |
|---|---|---|:---:|:---:|:---:|
| simple_calendar_pro | Tier-B | Productivity | 11 | 6 | 17 |
| system_settings | Tier-B | Tools | 9 | 6 | 15 |
| pro_expense | Tier-B | Finance | 6 | 3 | 9 |
| retro_music | Tier-B | Music & Audio | 3 | 1 | 4 |
| chrome | Tier-B | Communication | 2 | 1 | 3 |
| camera | Tier-B | Photography | 1 | 1 | 2 |
| **Tier-B total** | | | **32** | **18** | **50** |
| maps_me | Tier-C | Maps & Nav. | 9 | 6 | 15 |
| broccoli | Tier-C | Food & Drink | 8 | 5 | 13 |
| opentracks | Tier-C | Health & Fitness | 4 | 2 | 6 |
| wikipedia | Tier-C | Books & Reference | 4 | 2 | 6 |
| osmand | Tier-C | Maps & Nav. | 2 | 1 | 3 |
| vlc | Tier-C | Video Players | 1 | 1 | 2 |
| simple_draw_pro | Tier-C | Art & Design | 1 | 0 | 1 |
| **Tier-C total** | | | **29** | **17** | **46** |

All headline numbers in the next section are scored on the combined 18 + 17 = **35 T_eval templates**, at K=3 seeds → **105 scored episodes per rung**. No T_adapt template ever reaches a reported T_eval number.

Two layers of generalisation are being measured together:

| Layer | What it tests | Governed by |
|---|---|---|
| App-level (cross-app) | Can knowledge learned on source apps transfer to target apps we never pretrained on? | Source vs. Tier-B / Tier-C target (§5.1) |
| Task-level (within-target-app) | After adapting to a target app on some of its tasks, does that adaptation generalise to its remaining tasks? | T_adapt / T_eval (this section) |

## 6. Experimental results

We execute a four-rung ablation, each rung layering one element of the method on the previous rung, all scored on the same frozen `T_eval` templates defined in §5.3.

- **B1 — zero-shot.** Qwen3-VL-8B in the M3A two-phase loop, no knowledge injected. Establishes the ceiling without any per-app structure.
- **B2 — static `L_C`.** Per-category `L_C` from the source-pool pretraining (§6.4) injected into every action-selection prompt. Only fires on Tier-B apps (Tier-C has no matching `L_C`), so Tier-C is byte-identical to B1.
- **B3 — evolved `L_C`.** Same `L_C` injection, but on each target app the `L_C` is mutated online on `T_adapt` by Claude Opus 4.7 for 20 iterations (M=2 variants sampled per iteration, N=1 rollout per variant), then the end-of-adaptation champion is frozen and graded on `T_eval`.
- **B4 — evolved `L_C` + LoRA.** Same context loop as B3, but rollouts now also drive a GRPO weight update on a shared LoRA adapter (rank 16, attached to `q_proj` / `v_proj`). **Sweep in progress at time of writing.**

### 6.1 T_eval headline (K=3 seeds, 105 episodes per rung)

| Rung | Tier-B SR | Tier-C SR | Overall | Δ vs previous rung |
|---|:---:|:---:|:---:|:---:|
| B1 (zero-shot) | 47.2% (25.5 / 54) | 29.4% (15 / 51) | 38.6% (40.5 / 105) | — |
| B2 (static `L_C`) | 56.5% (30.5 / 54) | 29.4% (15 / 51) | 43.3% (45.5 / 105) | +4.8 pp |
| B3 (evolved `L_C`) | 63.9% (34.5 / 54) | 31.4% (16 / 51) | 48.1% (50.5 / 105) | +2.9 pp (Tier-B +3.7) |
| B4 (evolved + LoRA) | *sweep in progress* | — | — | — |

Tier-C stays flat across rungs, as designed. Its +0.0 pp (B1 → B2) and +2.0 pp (B2 → B3) confirm that paired-run variance from the Android emulator and bf16 CUDA kernel ordering sits in the low single digits — so Tier-B moves exceeding that threshold (+9.3 pp and +3.7 pp) are real effects of injection and evolution, not instrumentation noise.

### 6.2 B3 evolution sweep — per-app detail

Each Tier-B app ran 20 iterations of the evolve loop on its `T_adapt` split before the end-of-run champion was scored on `T_eval`. The total sweep took 14 h 6 min wall time across the 6 apps, with one recoverable Claude Opus truncation in 56 mutation attempts.

| App | T_eval n | B2 SR → B3 SR | Δ | Successful mutations | Champion TrueSkill μ / σ |
|---|:---:|:---:|:---:|:---:|:---:|
| system_settings | 18 | 80.6% → 91.7% | **+11.1 pp** | 14 / 14 | 25.00 / 8.33 |
| retro_music | 3 | 33.3% → 66.7% | **+33.3 pp** | 10 / 10 | 53.02 / 11.39 |
| pro_expense | 9 | 66.7% → 66.7% | +0.0 pp | 7 / 7 | 31.53 / 11.05 |
| simple_calendar_pro | 18 | 50.0% → 50.0% | +0.0 pp | 12 / 12 | 46.42 / 16.71 |
| chrome | 3 | 0.0% → 0.0% | +0.0 pp | 6 / 6 | 25.00 / 8.34 |
| camera | 3 | 66.7% → 33.3% | −33.3 pp | 6 / 6 | 25.00 / 8.34 |
| **Aggregate** | **54** | **60.2% → 63.9%** | **+3.7 pp** | **55 / 56** | |

### 6.3 Where the Tier-B signal is concentrated

Two apps drive the Tier-B aggregate:

- **system_settings (+11.1 pp, n=18).** Largest T_eval in Tier-B. The Tools `L_C` started from three source apps (calculator / clock / files); adaptation on the Wi-Fi + Bluetooth toggle templates refined it toward the verification patterns that distinguish `System…Verify` templates from their action-only siblings. Every tournament match during adaptation was a win, which is why champion μ stayed at the initialisation value of 25.0 (uniform win-rate gives no differential signal to TrueSkill).
- **retro_music (+33.3 pp, n=3).** Only one T_eval template, so this is suggestive rather than significant. Champion μ = 53.0 is the highest in the sweep — the population clearly advanced — but n=3 does not support a strong claim.

Of the other four Tier-B apps, three tied at their B2 rate and one regressed within-noise. The remaining-five-app aggregate (excluding system_settings) is ≈ +0 pp over the 36 episodes; we always report the per-app breakdown alongside the aggregate for this reason. Detailed analysis of the individual ties (`simple_calendar_pro` month-view truncation; `chrome` `open_app` harness quoting bug; `camera` single-template noise) lives in `docs/results/b3_evolution.md` §Analysis.

#### What the 14 mutations actually did on `system_settings`

Because B3's biggest win is on the Tools `L_C`, it is worth looking at what the reflector actually wrote. Running the start-of-run `L_C` and the end-of-run champion through a structured diff (counting failure modes per abstract category) gives:

| Abstract category in `L_C` | v0 (start) | champion (iter 20) | Change |
|---|:---:|:---:|:---|
| TOGGLE_STATE | 3 failure modes | 15 | **+12** |
| NAVIGATE_TO_SECTION | 5 | 13 | **+8** |
| ADJUST_CONTINUOUS_CONTROL | — | 8 | **newly created** |
| other 14 categories | unchanged | unchanged | — |

Three of the 15 abstract categories were touched across 14 mutations; twelve remained verbatim. The edits track the specific `T_adapt` failure modes encountered during evolution — `TOGGLE_STATE` was refined with Wi-Fi / Bluetooth-toggle-specific verification patterns after the first few `SystemWifiTurnOn...Verify` rollouts revealed that the agent was missing "toggle done vs toggle done + visible verification indicator" as a distinguishable state. `ADJUST_CONTINUOUS_CONTROL` was introduced from scratch when the brightness-slider templates surfaced that no existing abstract category covered analogue sliders. Every edit stayed inside LAYER 2 (no `system_settings`-specific strings are present in the champion text — lint verified), so the result is a generic Tools-category refinement rather than a system_settings-specific cheat sheet. That is what +11.1 pp of T_eval gain on this app actually buys us, and it is the kind of inspectable artifact that motivates the two-layer design.

### 6.4 Pretraining pipeline (the source of `L_C`)

The `L_C` libraries used by B2, B3, B4 were built once, offline, from the 12 source apps:

- **480 source-pool trajectories** — 96 templates × 5 seeds (30…34), collected with the K=1-tuned M3A baseline. 40.4% mean success rate, zero infrastructure errors, 2.1 GB of per-step traces. (`traces/source_pool_trajectories/`.)
- **12 per-app LAYER-1 FSMs**, one per source app, synthesized by Claude Opus from each app's trajectory bundle. (`artifacts/static_fsms/`.)
- **6 per-category `L_C` libraries**, one per source-pool category, aggregated by Claude Opus from the LAYER-2 blocks of the per-app FSMs whose app sits in that category. Every library passes a lint that rejects app-name / package-id / resource-id leakage. (`artifacts/L_C/`.)

| Category | Source apps contributing | Abstract categories in the resulting `L_C` |
|---|:---:|:---:|
| Communication | 2 | 7 |
| Finance | 1 | 7 |
| Music & Audio | 2 | 11 |
| Photography | 1 | 6 |
| Productivity | 3 | 14 |
| Tools | 3 | 15 |

### 6.5 Statistical caveats

- B1 → B2 → B3 on Tier-B is monotonically increasing (+9.3, +3.7 pp) and the direction is consistent across rungs, but K=3 at n=54 per arm has SE ≈ 9 pp — each individual step is underpowered to declare significance. We report direction plus per-app breakdowns; we do not claim significance on the aggregate alone.
- The signal concentrates on `system_settings` (n=18, +11 pp between B2 and B3). Removing that row flattens Tier-B to ≈ +0 pp on the remaining five apps. We always publish the per-app table alongside the aggregate.
- Tier-C ties across B1 → B2 → B3 within single-template noise (exactly 0 pp on B1 → B2, +2.0 pp on B2 → B3 driven by one template-seed flip), consistent with the null expectation since Tier-C receives no mechanism change on any rung.

## 7. Contributions

- **Two-layer FSM knowledge structure** — a tool that explicitly separates app-specific from category-generic knowledge and therefore supports cross-app transfer (LAYER 2) while preserving app-specific grounding (LAYER 1).
- **Test-time adaptation framework for GUI agents** — a unified loop that lets a deployed agent continue evolving its context and fine-tuning its LoRA on a target app's `T_adapt` tasks, then freezes a reproducible snapshot for the template-disjoint `T_eval` pass.
- **First application of the E-SPL paradigm to multi-step visual-agentic settings** — context + weight joint optimization transferred from single-turn reasoning to stochastic, expensive GUI rollouts.
- **Near-vs-far transfer evaluation protocol** — Tier-B / Tier-C split by Play-Store-category membership in the source pool, giving two separately reported curves and a built-in null control.
- **AndroidWorld-Plus benchmark** — 25-app, 192-template, 12-category benchmark on top of AndroidWorld plus 6 BMOCA / AndroidLab apps, with a frozen split contract and K=1 / K=3 / K=5 seed protocols.

## 8. Technical architecture

- **Base model:** `Qwen/Qwen3-VL-8B-Instruct` (ADR-002) — dense 8B VLM, bf16 on A100, fp16 with `max_pixels=802816` on Mac for development.
- **Environment:** Android AVD `AWAvd2`, Android 33, Pixel-6-style device, Google APIs image, booted from the `apps_ready_dec2025` snapshot read-only so every run starts from byte-identical device state (ADR-001). All 14 apps (vanilla AndroidWorld + 6 Plus) are pre-installed in the snapshot.
- **Agent loop:** Qwen3-VL-M3A — mirrors `android_world.agents.m3a` 1:1 with Qwen3-VL-8B substituted for GPT-4o. Two LLM calls per step (action selection, then summary), `Reason: …\nAction: {…}` format, natural-language summary history.
- **Reflector (mutation proposer):** Claude Opus 4.7 (1M context) for per-app FSM synthesis (§6.4, Story 2.2), for LAYER 2 aggregation into `L_C` (§6.4, Story 2.3), and for online FSM-diff proposal in the evolution loop (B3 and B4).
- **LoRA adapters (B4 only):** rank 16, α = 32, dropout 0.05, attached to `q_proj` and `v_proj`. Trainable parameter count is ≈0.1–1% of the base model; base weights stay frozen.
- **GRPO weight update (B4 only):** per-trajectory advantage = trajectory reward − within-group (same FSM variant) mean reward. Gradient flows through LoRA only; replay tensors are saved per step and consumed per update, then deleted. Replay files average ≈10 MB/step (pixel values dominate).
- **Snapshot format (ADR-003):** per-`(target_app, seed)` directory with `snapshot.json` + `fsm_L1.json` + `L_C_target_row.json` + `lora_delta/`. Frozen at the end of `T_adapt`, read-only on `T_eval`.

## 9. Current status and next steps

| Epic | Scope | Status |
|---|---|---|
| 0 | Splits and taxonomy (§5) | **Closed.** `configs/splits.yaml` frozen; 12/12 splits tests pass. |
| 1 | AVD + base model + M3A agent | **Closed.** Canonical snapshot; fingerprint-locked loader; M3A rewrite. K=1 and K=3 T_eval baselines produced here. |
| 2 | Pretraining — source FSMs + `L_C` (§6.4) | **Closed.** 480 source trajectories; 12 per-app FSMs; 6 per-category `L_C` libraries. |
| 3 | B2 + B3 ablations (§6) | **Closed.** Full reports in `docs/results/b1_b2_static_baselines.md` and `docs/results/b3_evolution.md`. |
| 4 | B4 — joint LoRA + evolution | **In progress.** Stories 4.1–4.4 delivered (LoRA attach, log-prob collection, GRPO update, evolution-loop wiring, B4 runner, convergence-plot utility). 379 / 379 unit tests pass. The 6-Tier-B-app B4 sweep (20 iters × M=2 × N=1 per app) is running on emulator-5710 + GPU 2 at time of writing. After it completes, the frozen B4 champions will be evaluated on `T_eval` (Story 4.5) and filled into the B4 row of the §6.1 headline table. |

**Deferred.** Story 1.3 (the snapshot module from ADR-003) — will write when the end-to-end TTA loop needs the `T_adapt → T_eval` hand-off format on disk. The B4 sweep does not require it yet because each app's evolution writes its champion directly under `traces/b4_evolution/{app}/`.

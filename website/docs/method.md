# Method

EvoFSM is a **test-time adaptation (TTA)** method for mobile GUI agents. When the
agent is deployed on a previously unseen Android app, it keeps learning: it
**evolves a structured, human-readable knowledge representation** of the app and
**fine-tunes a LoRA adapter** on its own weights, both driven by reward from its
own rollouts on that app. The two updates share the same trajectories — no extra
rollouts are spent on the weight channel.

The design adapts the joint context+weight optimization of **E-SPL** (Zhang et
al., 2026) — which evolves a free-text system prompt while RL-training a shared
LoRA on single-turn reasoning — to a **multi-step, visual, stochastic,
expensive-rollout** GUI setting. Two things change to make that transfer work: a
*structured* knowledge format (the two-layer FSM) replaces free text, and the
per-iteration budget shrinks (M = 2 variants, N = 1–2 rollouts) because each
Android episode costs 1–3 minutes of wall-clock.

---

## 1. The two-layer FSM

The agent's "knowledge of how to operate an app" is a two-layer **Finite State
Machine (FSM)**, rendered to text and spliced into the action-selection prompt at
every step — the policy reads its own knowledge directly. The two layers separate
what *cannot* transfer from what *can*.

!!! abstract "Layer 1 — app-specific (non-transferable)"
    The finite-state map of one app's UI: states, transitions, visual cues,
    element locations, resource hints, and concrete dead-ends. Every string names
    a specific button, dialog, or field, so the layer is meaningless outside that
    exact app.

    ```text
    APP: markor   CATEGORY: Productivity

    STATES:
      S1: FILE_LIST — Markor file browser
        visual_cues: ['red plus FAB bottom-right', "'Markor' title", 'list of files']
      S3: NEW_FILE_DIALOG — Dialog to create a new file/folder
        visual_cues: ["Name field pre-filled with 'my_note'", 'OK/CANCEL/FOLDER buttons']

    TRANSITIONS:
      S1 --click(fab_plus)--> S3   [post: dialog with default 'my_note' appears]
      S3 --click(OK)--> S8         [pre: name entered]  [post: new file opened in editor]
    ```

!!! abstract "Layer 2 — category-generic (transferable)"
    Abstract workflow patterns written **without any app-specific strings**,
    indexed by Play Store category. Categories are task-shape archetypes —
    `ADD_ENTRY`, `QUERY_INFO`, `REMOVE_ENTRY`, `FILTER_LIST`, `TOGGLE_STATE`, …
    Each block has four fields:

    | Field | Meaning |
    |---|---|
    | `precondition` | when this archetype applies |
    | `abstract_steps` | sequence of app-agnostic moves |
    | `failure_modes` | known cross-app failure patterns to watch |
    | `verification_checklist` | post-condition checks before terminating |

    ```json
    {
      "name": "TOGGLE_STATE",
      "precondition": "A binary control is visible and its state is observable",
      "abstract_steps": [
        "Locate the binary control (switch, checkbox, toggle)",
        "Read the current state from visual affordance (icon, label, color)",
        "If the target state differs from current, tap to flip",
        "Wait for the UI to settle and re-read the state from the same surface"
      ],
      "failure_modes": [
        "Tapping before the state is read, inverting an already-correct state",
        "Reading the state from a stale cached screen instead of the live one"
      ],
      "verification_checklist": [
        "Target-state indicator visible on the control itself",
        "No pending spinner / confirmation dialog blocking"
      ]
    }
    ```

!!! note "A linter is the structural guarantee of transfer"
    A linter (`lint_layer2`) **rejects any Layer-2 edit that contains an app-name
    or resource-id substring**. This is what stops generic insight gleaned from
    one app's trajectories from being silently absorbed into app-specific strings
    and lost. Because Layer 2 is scrubbed of anything app-specific, **it is the
    unit of cross-app transfer** — and the only surface the test-time evolution is
    allowed to edit. Layer 1 grounds it but stays frozen during adaptation.

### Category library `L_C`

Same-category apps' Layer-2 blocks are aggregated into a single **per-category
library `L_C`**. During pretraining a reflector model (Claude Opus) unions the
Layer-2 blocks of all source apps in a category, keeping only claims that held
across multiple apps, enumerating cross-source failure modes, and stripping every
app-specific string — verified by the same linter. At deployment,
`resolve_l_c_for_app(app)` looks up the target app's Play category and returns its
`L_C`, or `None` if the category was never seen during pretraining (the
far-transfer condition).

```text
markor.L2  ┐
joplin.L2  ├──►  aggregate + lint  ──►  L_C / productivity.json   (e.g. QUERY_INFO,
tasks_org.L2 ┘                                                     ADD_ENTRY, … )
```

---

## 2. The joint adaptation loop

A **single deployment-time loop** co-adapts two channels on the target app's
adaptation tasks. Both consume the *same* rollouts.

```text
                 ┌──────────────  one iteration  ──────────────┐
  sample task t  │  select M=2 FSM variants (optimistic softmax)│
                 │  roll N rollouts per variant on t (same seed)│
                 │                                              │
   ┌─────────────┤  ┌─ PROMPT channel ─┐   ┌─ WEIGHT channel ─┐ │
   │ TrueSkill   │  │ frozen π_ref      │   │ GRPO step on the │ │
   │ rate & pick │  │ mutates winner's  │   │ shared LoRA from │ │
   │ best variant│  │ Layer-2 (FSM diff)│   │ the same rollouts│ │
   └─────────────┤  └───────────────────┘   └──────────────────┘ │
                 └─────────────────────────────────────────────┘
```

**Prompt channel — FSM mutation by a frozen reference LLM.** A population of FSM
variants is held in memory, each carrying a **TrueSkill** rating `(μ, σ)`. After
the rollouts, a **frozen reference policy `π_ref`** (Claude Opus, *not* the
trainable agent) reads the recent trajectory window — successes and failures
together — and proposes a structured **FSM diff** restricted to Layer 2. The child
is appended with the parent's `μ` and `σ` inflated by `Δσ`.

!!! tip "Why the mutator is frozen"
    Using the RL-updating policy to also propose edits introduces drift that
    degrades edit quality; E-SPL ablates this. EvoFSM follows that finding — the
    reflector that writes FSM diffs is held fixed throughout.

**Weight channel — group-relative policy optimization (GRPO).** The same
trajectories feed a LoRA update on the base VLM. GRPO uses no critic; the baseline
is the **within-group mean reward**, and groups are keyed by the
**`(FSM_variant, task)` tuple**:

```text
A_{i,j} = r_{i,j} − V_i ,    V_i = mean reward over rollouts in group i
```

This is why **N ≥ 2 rollouts per (FSM, task) cell** are needed: a singleton group
has no peer to subtract, so its advantage is constructively zero. Reward is
rule-based episode success (`{0, 0.5, 1.0}` from AndroidWorld's grader) plus a
small step-efficiency bonus; each trajectory's contribution is normalized by its
length (`1/T_j`) so a 30-step and a 1-step rollout weigh equally. Gradients flow
through **LoRA only** (rank 32, α 32, language-only `q_proj`/`v_proj`); base weights stay
frozen and LoRA's implicit regularization replaces an explicit KL penalty.

**Selection — TrueSkill chooses FSM variants.** Each iteration is a tournament
round: M = 2 variants are sampled from a sliding window of the latest K via an
**optimistic softmax** `p_i ∝ exp((μ_i + λσ_i) / T)`. The `λσ_i` term biases
selection toward high-uncertainty (newly introduced) variants, giving them a
chance to accumulate evidence before being dropped. After the round, one Bayesian
TrueSkill update adjusts the ratings from the pairwise outcome; the highest-`μ`
variant is the one mutated.

---

## 3. Two phases — the same loop, run twice

The identical joint loop runs at two scales. Only the data pool and compute budget
differ.

| | Phase 1 — source-pool pretraining | Phase 2 — target-app deployment |
|---|---|---|
| **Pool** | all source apps (a "training set") | one unseen target app's `T_adapt` |
| **LoRA** | trains the **shared `π^pre`** from scratch | continues from `π^pre`, keeps updating |
| **FSM** | (minimal scope: source FSMs frozen; `L_C` built offline) | the Layer-2 evolution of §2 runs live |
| **Cadence** | runs **once**, a fixed cost | runs per target app, then freezes |

The output of Phase 1 — the shared adapter **`π^pre`** — is frozen and reused as
the LoRA initialization at every deployment.

!!! warning "Why deployment cannot start from the base model"
    A deployment budget of ≈ 20 iterations × M × N rollouts is far below what
    training a LoRA from scratch needs — starting from random LoRA at that scale
    shows no learning. The Phase-1 → Phase-2 split is precisely what makes joint
    TTA tractable at deployment-time compute. After Phase 2, **both** the LoRA
    adapter and the FSM champion are frozen and graded once on the held-out
    `T_eval` tasks.

---

## 4. The B1 → B4 ablation ladder

EvoFSM is evaluated as a ladder where each rung **adds exactly one mechanism**, so
each delta isolates that mechanism's contribution. (Definitions here; numbers live
in the [Within-benchmark](within-benchmark.md) and
[Cross-benchmark](cross-benchmark.md) studies.)

| Rung | Mechanism added | What the rung is |
|---|---|---|
| **B1** | — | **Zero-shot.** Base VLM in the standard action+summary loop, no FSM, no `L_C`. The floor every mechanism must clear. |
| **B2** | static category transfer | **Static `L_C`.** The category-matched `L_C` is injected into the action-selection prompt and held **frozen** — no evolution, no weight update. |
| **B3** | online context evolution | **Online `L_C` evolution.** The `L_C`-seeded FSM becomes the rated population root; the Layer-2 evolution loop of §2 runs on `T_adapt`, then the champion is frozen for `T_eval`. Weights stay frozen. |
| **B4** | joint weight co-adaptation | **Joint LoRA + FSM.** B3 plus the GRPO weight channel of §2, starting from `π^pre`. Both channels co-adapt, then both freeze. |

Reading the deltas decomposes the test-time-adaptation benefit into three
separable parts:

- **B2 − B1** — value of *static* category-knowledge transfer.
- **B3 − B2** — value of *evolving* that transferable knowledge online.
- **B4 − B3** — value of *weight co-adaptation* on top of the evolved context.

!!! info "Far-transfer is a built-in null control"
    Target apps whose category was never in the source pool receive `L_C = None`,
    so under B2/B3 their prompt is byte-identical to B1 — any movement there is
    pure emulator/CUDA non-determinism. This makes far-transfer both a null
    control for the injection mechanism and the regime where the **B4 weight
    channel** must carry the entire transfer load, since no `L_C` exists to lean
    on.

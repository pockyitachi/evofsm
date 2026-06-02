# Algorithm design — EvoFSM-RL

The authoritative reference for the algorithm is `plan/algorithm_design.md`.
This document is the distilled methods writeup that the paper's method
section derives from.

## 1. Background: E-SPL (Zhang et al., 2026)

**Paper.** *Evolutionary System Prompt Learning for Reinforcement Learning
in LLMs* (Zhang, Chen, Stadie; arXiv 2602.14697, Feb 2026).

**Problem.** Given an LLM with policy π_θ and a reward R on single-turn
reasoning tasks (AIME / HMMT math, agentic web search), the goal is to
improve end-to-end performance by simultaneously evolving the **system
prompt** and updating **model weights** via RL.

**Key insight.** Context (declarative knowledge: "what to do in this
domain") and weights (procedural knowledge: "how to execute it") carry
complementary information. The paper demonstrates empirically that joint
optimization beats either axis alone:

| Setting | Base | Evo only | RL only | Evo + RL |
|---|---:|---:|---:|---:|
| DAPO100 → AIME25 | 56.3% | 57.3% | 56.3% | 60.6% |
| HMMT 23/24 → 25 | 47.0% | 48.6% | 47.3% | 52.7% |
| AIME → BeyondAIME | 36.5% | 40.0% | 38.8% | 45.1% |
| Agentic Search (gpt-oss-120b) | 6.6% | 34.0% | 44.2% | 48.6% |

The agentic-search row is the closest analog of our setting: the
procedural gap is large (RL alone adds +38 pp over base), context alone
adds +27 pp, and their combination is the best.

**Mechanism.** The system-prompt population `S = {(s_i, μ_i, σ_i)}`
starts from a single root and grows as an evolutionary tree:

1. **Selection.** Sample M prompts from S with optimistic softmax
   probability `p_i ∝ exp((μ_i + λσ_i) / T)`, restricted to the latest
   K prompts via a sliding window.
2. **Parallel rollouts.** Under each selected prompt, take N rollouts
   on one sampled problem; compute average reward `V_i`.
3. **RL weight update.** Policy gradient with group-relative advantage
   `A_{i,j} = r_{i,j} − V_i`, batched across sampled problems. With LoRA,
   KL is implicit (β = 0).
4. **TrueSkill update.** Rank the M prompts by `V_i` and run one Bayesian
   update — each iteration is a tournament round.
5. **Mutation.** Identify the best prompt `k = argmax_i V_i`. Use the
   **frozen reference** policy π_ref (not π_θ) to produce a
   self-reflection `ℓ` over the trajectories under `s_k`, then propose a
   git-diff edit. Apply to `s_k` to get `s_mutate`; append with parent μ
   and σ inflated by Δσ.
6. **Crossover (stochastic).** With probability `p_crossover`, build a
   score matrix over a replay batch of past problems, identify per-prompt
   differential strengths, and synthesize a child.

Two design points are critical: **only the best prompt per iteration is
mutated** (strong evolutionary pressure), and **mutation uses frozen
π_ref** (the RL-updating policy introduces drift that degrades edit
quality; the paper ablates this in Fig. 15).

## 2. From E-SPL to EvoFSM-RL

E-SPL is formulated for single-turn reasoning with free-text system
prompts. Our setting is multi-step, stochastic, visual, expensive-rollout
GUI agents. Four dimensions change:

| Aspect | E-SPL | EvoFSM-RL |
|---|---|---|
| Context format | Free-text system prompt | **Two-layer FSM** (LAYER 1 app-specific + LAYER 2 category-generic) |
| Task unit | Single-turn math / QA problem | **Multi-step Android GUI episode** (10-80 actions, screenshots per step) |
| Eval unit | One output per rollout | **Episode-level rule-based `is_successful(env)`** |
| New challenges | — | Visual input; discrete action space over UI elements; emulator stochasticity; ~1-3 min wall per rollout — forces small N, M |

The structured FSM (vs free text) brings both opportunities and
constraints: FSM diffs are semantically typed (add_state, modify_transition,
etc.), so mutation produces lint-checkable edits rather than arbitrary
text rewrites; but the operators must preserve FSM structural validity,
and the serialization format must be stable across mutations.

Expensive rollouts force the per-iteration budget down: M = 2 selected
FSMs and N = 1 rollout per FSM per task, vs E-SPL's typical M ≈ 4-8,
N ≈ 8-16. To compensate, we run more iterations and lean harder on
TrueSkill's uncertainty-aware rating.

## 3. Two-layer FSM knowledge structure

The two-layer schema is what makes transfer possible. Without it,
generic insight from one app's trajectories would get silently absorbed
into app-specific strings and lost to transfer.

**LAYER 1 (APP_SPECIFIC).** States, transitions, visual cues, element
locations, resource hints, and concrete dead-ends tied to one app's UI.
Example: `S1: ADD_TRANSACTION, visual_cues: ["amount field at top",
"SAVE button top-right"]`. Non-transferable.

**LAYER 2 (GENERIC, indexed by Play Store category).** Abstract workflow
patterns, failure modes, and verification checklists written without any
app-specific strings. Categories are task-shape archetypes like
`ADD_ENTRY`, `QUERY_INFO`, `REMOVE_ENTRY`, `FILTER_LIST`. Each LAYER 2
block has `precondition`, `abstract_steps`, `failure_modes`,
`verification_checklist`. Transferable across same-category apps.

Linting enforces the separation: LAYER 2 diffs are rejected if they
contain any app-name or resource-id substring.

### `L_C` construction pipeline

Per-app LAYER 2 content is aggregated per Play Store category into a
library `L_C` at the end of Phase-1 pretraining:

1. **Per-app static FSMs** (Story 2.2). Claude Opus 4.7 synthesizes
   one two-layer FSM per source app from its bundle of K=5 trajectories
   in `traces/source_pool_trajectories/`. See the source-FSM quality
   audit in `docs/results/b1_b2_static_baselines.md` Appendix A for
   per-app diagnostics.
2. **Per-category aggregation** (Story 2.3). For each Play category
   represented in the source pool, Claude Opus 4.7 unions the LAYER 2
   blocks of same-category source FSMs into a single `L_C` file, keeping
   only claims that held across multiple source apps, explicitly listing
   cross-source failure modes, and containing no app-specific strings.

At deployment, `resolve_l_c_for_app(a_tgt)` returns the `L_C` for
`play_category_of(a_tgt)` if that category is in the source pool, else
`None`.

## 4. Context evolution mechanism (B3)

B3 is the single-app evolution loop with frozen base weights. Per
target app, an online adaptation pass on `T_adapt` grows an FSM
population `S_a` rooted at the initial `L_C`-seeded FSM.

**Population.** A growing tree from the root. Each variant carries a
TrueSkill rating `(μ, σ)`. Children inherit parent `μ` and
`σ + Δσ`. No elitism / no explicit pruning — a sliding window of the
latest K = 15 variants is the only recency constraint.

**Selection.** Optimistic softmax:
`p_i ∝ exp((μ_i + λσ_i) / T)`. Two FSM variants are sampled per
iteration (M = 2). The `λσ_i` term biases selection toward
high-uncertainty (newly introduced) variants, giving them a chance to
accumulate evidence before being dropped for lack of it.

**Mutation.** Claude Opus 4.7 (acting as π_ref, frozen) analyzes the
recent trajectory window — successes and failures together — and
proposes a structured FSM diff in JSON. The mutation operator enforces
`layer2_only=True`: only LAYER 2 blocks may be edited, so evolution
is bounded to the transferable part of the knowledge. The diff is
applied, the child is added to S with μ inherited and σ inflated by
Δσ.

**Fitness.** Episode success rate on T_adapt tasks, with **paired
seeds across variants** to reduce per-seed variance in the TrueSkill
update. Each iteration is one tournament round over M variants on one
sampled task.

**Iteration loop.** Per iteration:

1. Select M = 2 variants from the latest-K window.
2. Roll one episode per variant (N = 1) on the sampled task.
3. Update TrueSkill from the pairwise outcome.
4. Mutate the best variant if either `best_reward > 0` (there is
   signal to reflect on) or every 3rd iteration (force exploration
   otherwise). Append the child.
5. Repeat until budget exhausted.

In our Tier-B sweep each target app ran for 20 iterations (60 total
steps counting selection/rollout/mutation), producing on average
9 mutation events per app.

## 5. Joint context + weight optimization (B4)

B4 = B3 + LoRA fine-tuning of the base VLM on the same rollouts used
for evolution. This is the direct application of E-SPL's joint mode.
The motivating expectation is E-SPL's finding that evolution + RL
beats either axis alone — particularly on agentic settings where the
procedural gap is large (base 6.6% → RL-only 44.2% in their agentic
search result).

Two axes of update frequency are defined (ADR-003 field `lora_delta`):

- **Path A** — LoRA frozen (consistency check: collapses to B3, within
  noise).
- **Path B** — LoRA updates every adaptation step.
- **Path C** — LoRA updates every K adaptation steps (default).

`[TODO: future work — B4 not yet run; numbers pending.]`

## 6. Four-baseline progression

EvoFSM-RL is evaluated as an ablation ladder, each rung isolating one
mechanism:

- **B1 — zero-knowledge baseline.** Pure Qwen3-VL-M3A. No FSM. No `L_C`
  injection. Represents the zero-shot floor against which every
  mechanism must clear its cost. Current Tier-B `T_eval` SR at K=3:
  47.2%.
- **B2 — static `L_C` injection.** The matching-category `L_C` is
  injected into the action-selection prompt between the `PROMPT_PREFIX`
  and the dynamic goal/history/UI content (so the injection sits in the
  KV-cache-stable prefix). Frozen throughout — no evolution, no weight
  update. Tier-C apps receive `None` and degrade to B1 byte-identically,
  which serves as a built-in null control for injection-mechanism
  spuriousness. Current Tier-B `T_eval` SR at K=3: 56.5%.
- **B3 — evolved `L_C`.** The same two-layer FSM is now the rated
  population root; the evolution loop of §4 runs on `T_adapt`, LAYER 2
  mutation only; the end-of-adaptation champion is frozen and
  evaluated on `T_eval`. Current Tier-B `T_eval` SR at K=3: 63.9%
  (see `docs/results/b3_evolution.md`).
- **B4 — evolved `L_C` + LoRA.** Adds joint LoRA weight update per §5.
  `[TODO: not yet run.]`

### Ablation delta interpretation

- `B2 − B1` = value of **static category knowledge transfer**. Tier-B
  delta: +9.3 pp (K=3, n=54 per arm). Tier-C delta: +0.0 pp (null
  control passes — the mechanism adds no spurious drift outside its
  intended scope).
- `B3 − B2` = value of **online evolution of the transferable
  knowledge**. Tier-B delta: +3.7 pp. Signal concentrated in
  system_settings (+11.1 pp) and retro_music (+33.3 pp on n=3);
  other apps tied or within n=3 noise.
- `B4 − B3` = value of **weight co-adaptation** on top of the evolved
  context. `[TODO: pending B4 run.]`

Reading these deltas together decomposes "test-time adaptation
benefit" into three contributions: static transfer, online context
evolution, and joint weight update. The narrative the paper targets
is that each rung adds a real, separable increment — and that
far-transfer (Tier-C, where no `L_C` exists) is the regime where
joint weight update will matter most.

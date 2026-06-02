# E-SPL × FSM: Joint Self-Evolving FSM + RL Weight Updates for GUI Agents

## Algorithm Design Document — April 2026 (Updated with full paper details)

---

## 1. E-SPL: Complete Algorithm from the Paper

**Paper:** "Evolutionary System Prompt Learning for Reinforcement Learning in LLMs"
**Authors:** Lunjun Zhang, Ryan Chen, Bradly C. Stadie (U of Toronto, Northwestern, Bridgewater AIA Labs)
**arxiv:** 2602.14697, Feb 2026

### 1.1 Core Idea

E-SPL jointly optimizes **system prompts** (declarative knowledge) via evolutionary search and **model weights** (procedural knowledge) via RL. The population of system prompts starts from a single root and grows into an **evolutionary tree** through mutation and crossover. Key insight: prompts encode high-level strategies/rules while RL fine-tunes the model's ability to execute them.

### 1.2 Algorithm 1: E-SPL Main Loop (Exact from Paper)

```
Input:
  - Problem set D(·), reward function R(·,·)
  - RL policy π_θ(·); reference policy π_ref(·)
  - RL hyper-parameters: N (rollouts per prompt), α (learning rate), β (KL coeff)
  - Initial system prompt s_root; M (prompts per iteration); p_crossover
  - TrueSkill init: (μ_0, σ_0); Δσ (uncertainty increment); λ (optimism); T (temperature)

Initialize: S = {(s_root, μ_0, σ_0)}    # Population starts with single root

while not converged do:
    Sample a problem x ~ D

    # ── SELECTION ──
    Sample M system prompts {(s_i, μ_i, σ_i)}^M from S
      each with probability p_i ∝ exp((μ_i + λσ_i) / T)    # Eq (4): OPTIMISTIC selection

    # ── PARALLEL ROLLOUTS ──
    for i = 1 to M:
        Sample N solutions {y_{i,j}}^N:  y_{i,j} ~ π_θ(· | s_i, x)
        Reward:  r_{i,j} = R(x, y_{i,j})
        Value:   V_i = (1/N) Σ_j r_{i,j}                    # Eq (2)
    end for

    # ── RL WEIGHT UPDATE ──
    Policy gradient for problem x:
        J(θ) = (1/NM) Σ_i Σ_j (r_{i,j} - V_i) ∇_θ log π_θ(y_{i,j} | s_i, x)
    
    Batch update on problems B ~ D:
        θ ← θ + α · (1/|B|) Σ_{x∈B} ∇_θ( J(θ,x) - β·KL(π_θ, π_ref) )    # Eq (3)
    
    Note: With LoRA, KL regularization is implicit (β = 0) since π_θ
          cannot deviate far from π_ref by construction.

    # ── TRUESKILL RATING UPDATE ──
    v ← argsort([V_1, ..., V_M])            # Rank prompts by value this iteration
    {(μ_i, σ_i)}^M ← TrueSkill.update({(μ_i, σ_i)}^M, v)    # Bayesian rating update
    # This treats each iteration as a TOURNAMENT round among the M selected prompts

    # ── MUTATION (only the BEST prompt gets mutated) ──
    k ← argmax_i V_i                        # Eq (5): identify best prompt this round
    
    Create mutation self-reflection prompt Ψ_reflect using:
        agent trajectories under s_k: [(x, y_{k,1}, r_{k,1}), ..., (x, y_{k,N}, r_{k,N})]
    
    ℓ ~ π_ref(· | Ψ_reflect)                # Self-reflect on mistakes (uses FROZEN π_ref!)
    diff ~ π_ref(· | s_k, ℓ)                # Propose edits in git diff format
    s_mutate ← git.apply(s_k, diff)          # Apply diff to parent prompt
    S.append((s_mutate, μ_k, σ_k + Δσ))     # Child: inherits parent μ, increased σ

    # ── CROSSOVER (stochastic, with probability p_crossover) ──
    if p_crossover > random.rand() then
        Let B = {x_b}^B be a batch of B past problems
        Let Ω = {(s_i, μ_i, σ_i)}^M
        (s', μ', σ') = Crossover(B, R, Ω, π_θ, π_ref)       # Algorithm 2
        S.append((s', μ', σ'))
    end if
end while
```

### 1.3 Algorithm 2: Crossover Operator (Exact from Paper)

```
Input: List of B problems B; reward R(·,·); M system prompts Ω; π_θ; π_ref

# ── BUILD SCORE MATRIX ──
Let Φ ∈ R^{M×B}    # Φ_{i,b} = average reward of prompt s_i on problem x_b

for (s_i, μ_i, σ_i) in Ω:
    for x_b in B:
        Sample {y_{i,b,j}}^N from π_θ(· | s_i, x_b)    # Reuse rollouts from RL!
        Φ_{i,b} = (1/N) Σ_j R(x, y_{i,b,j})
    end for
end for

# ── IDENTIFY OVERALL TOP PROMPT ──
k ← argmax_i Σ_b Φ_{i,b}                 # s_k is the overall best prompt

# ── ANALYZE DIFFERENTIAL STRENGTHS ──
φ = []                                     # Collection of (prompt, problems_where_it_excelled)
ρ_i = []  for each s_i
for x_b in B:
    i = argmax_j Φ_{j,b}                  # Which prompt was BEST on this specific problem?
    ρ_i.append(x_b)                        # Track problems where each prompt shines
end for

for each (s_i, ρ_i):
    φ.append((s_i, ρ_i))                  # Analyze WHY s_i did best on ρ_i
end for

# ── CROSSOVER SELF-REFLECTION ──
Create crossover prompt Ψ_cr using φ       # Shows each prompt's differential strengths
diff_cr ~ π_ref(· | s_k, Ψ_cr)            # Propose edits to s_k based on analysis
s_crossover ← git.apply(s_k, diff_cr)

# ── CHILD RATING: PRECISION-WEIGHTED AVERAGE OF PARENTS ──
μ' = (Σ_i μ_i/σ_i²) / (Σ_i 1/σ_i²)
σ' = √(1/(Σ_i 1/σ_i²) + Δσ²)

Return (s_crossover, μ', σ')
```

### 1.4 Critical Design Decisions from the Paper

1. **Use π_ref (frozen reference policy) for self-reflection, NOT π_θ (training policy)**
   - Fig 15 shows this is important: using the changing RL policy π_θ for evolution degrades performance
   - Rationale: a fixed reference policy provides a stable basis for self-reflection, mutation, and crossover
   - The RL-updating model would introduce instability in quality of evolutionary edits

2. **Only mutate the BEST prompt per iteration** (Eq 5: k = argmax_i V_i)
   - This adds strong evolutionary pressure — only the winner of each tournament gets to reproduce
   - Population grows as an evolutionary tree from s_root

3. **git diff format for mutations** — structured edits, not full rewrites
   - The LLM proposes a diff: additions, deletions, modifications
   - This is applied with `git.apply()` to produce the child prompt

4. **TrueSkill as tournament system** (not simple reward averaging)
   - Each RL iteration = one tournament round among M selected prompts
   - Scores are RELATIVE (within iteration), not absolute — handles non-stationarity as weights change
   - Ratings aggregate evidence across iterations despite changing model weights

5. **Crossover uses PAST problem batch** (replay buffer), not just current iteration
   - Computes a full M×B score matrix to identify differential strengths
   - This grounds crossover in concrete performance differences, not abstract prompt analysis

6. **Population starts from single root, grows into a tree** (not fixed size)
   - No elitism/pruning explicitly described — population grows
   - Selection via sliding window of size K (only sample from latest K prompts)

7. **LoRA with rank 32** for weight updates
   - With LoRA, KL regularization is implicit (β = 0) since weights can't deviate far
   - This is practically important for us with Qwen3-VL-8B

### 1.5 Key Experimental Results

| Setting | Base → E-SPL | Evo Only | RL Only (GRPO) |
|---------|-------------|----------|----------------|
| DAPO100→AIME25 | 56.3% → **60.6%** | 57.3% | 56.3% |
| HMMT 23/24→25 | 47.0% → **52.7%** | 48.6% | 47.3% |
| AIME→BeyondAIME | 36.5% → **45.1%** | 40.0% | 38.8% |
| Agentic Search (gpt-oss-120b) | 6.6% → **48.6%** | 34.0% | 44.2% |

**Key findings:**
- Evolution + RL always beats either alone
- Evolution learns FASTER early on; RL improves ASYMPTOTIC performance (Fig 12)
- On agentic search: RL >> Evolution alone (44.2% vs 34.0%), but combined is best (48.6%)
- On math: Evolution alone can sometimes match or beat RL alone
- Crossover accelerates early progress but mixed effect on final performance (Fig 14)
- Emerged patterns in prompts: generative self-verification, codified domain expertise, heuristics

### 1.6 Emergent Patterns (Relevant for Our FSM Design)

The paper found that evolved system prompts naturally develop:
1. **Generative self-verification** — workflows that verify answers before committing
2. **Codified domain expertise** — domain-specific heuristics (e.g., "use CRT for modular arithmetic")
3. **Failure mode avoidance** — explicit instructions to avoid common mistakes
4. **Critical formatting rules** — precise output format specifications

**Implication for our FSMs:** These patterns map directly to what we want FSMs to encode:
- Self-verification → FSM states for checking task completion before declaring done
- Domain expertise → App-specific navigation strategies and shortcuts
- Failure avoidance → Error recovery transitions and known-bad-path avoidance
- Formatting → Action formatting specifications for the agent

---

## 2. Mapping E-SPL to Our GUI Agent Framework

### The Key Analogy

| E-SPL Component | Our Project Component | Notes |
|----------------|----------------------|-------|
| System prompt (free-form text) | **App-specific FSM** (structured text) | FSMs are more structured than free-form prompts |
| Prompt population S | **FSM population** per app | Grows as evolutionary tree |
| Problem x ~ D | **Task t ~ T_a** sampled for app a | From AndroidWorld task set |
| Rollout y_{i,j} ~ π_θ(·\|s_i, x) | Agent trajectory on Android emulator | Multi-step, not single output |
| Reward R(x, y) | AndroidWorld `is_successful()` + step efficiency | Binary + continuous |
| RL policy π_θ | **Qwen3-VL-8B** with LoRA | Multimodal, sees screenshots |
| Reference policy π_ref | **Frozen Qwen3-VL-8B** (pre-LoRA) | Used for self-reflection! |
| Mutation via π_ref | FSM refinement via frozen model | Add states, fix transitions |
| Crossover via π_ref | FSM merging via frozen model | Combine complementary strategies |
| git diff edits | **FSM diff operations** | Add/remove states, modify transitions |
| TrueSkill ratings | Per-FSM-variant ratings | Same mechanism |

### Why FSM as "System Prompt" Works

1. **FSMs are text-serializable**: Fits in context window, exactly like system prompts
2. **FSMs are modular per-app**: Each app has its own FSM, naturally app-specific
3. **FSMs support git-diff-style edits**: Add state, modify transition, remove condition — structured diffs
4. **FSMs have bounded complexity**: Finite state space per app → evolution converges
5. **FSMs encode exactly the "declarative knowledge" E-SPL targets**: navigation rules, screen structure, task strategies

---

## 3. Our Combined Algorithm: **EvoFSM-RL**

### 3.1 Notation

- `π_θ`: Qwen3-VL-8B with LoRA parameters θ (the training model)
- `π_ref`: Frozen Qwen3-VL-8B (used for self-reflection in evolution)
- `F_i`: FSM variant i for a given app (serialized as structured text)
- `S = {(F_i, μ_i, σ_i)}`: Population of FSM variants with TrueSkill ratings
- `T_a`: Task set for app a
- `M`: Number of FSM variants selected per iteration
- `N`: Number of rollouts per FSM per task
- `λ, T`: Optimism and temperature for selection
- `Δσ`: Uncertainty increment for children
- `p_crossover`: Probability of crossover per iteration

### 3.2 Algorithm: EvoFSM-RL (Faithful to E-SPL Structure)

```
Algorithm: EvoFSM-RL (Joint FSM Evolution + RL Weight Updates for GUI Agents)
═══════════════════════════════════════════════════════════════════════════════

Input:
  - Qwen3-VL-8B π_θ (with LoRA); frozen reference π_ref
  - App a with task set T_a
  - Initial FSM F_root for app a (from static extraction)
  - Hyper-parameters: M, N, α, p_crossover, λ, T, Δσ, (μ_0, σ_0)

Initialize:
  S = {(F_root, μ_0, σ_0)}    # Population starts with single root FSM

while not converged do:

    # ═══ SELECTION ═══
    Sample task t ~ T_a
    Sample M FSM variants {(F_i, μ_i, σ_i)}^M from S
      with p_i ∝ exp((μ_i + λσ_i) / T)          # Optimistic, explores high-σ FSMs
    Also apply sliding window: only sample from latest K FSMs in population

    # ═══ PARALLEL ROLLOUTS ON ANDROID EMULATOR ═══
    for i = 1 to M:
        for j = 1 to N:
            Serialize F_i → system_prompt text
            τ_{i,j} = rollout(π_θ, system_prompt=F_i, task=t)     # Agent acts on emulator
            r_{i,j} = R(t, τ_{i,j})                                # AndroidWorld evaluator
        end for
        V_i = (1/N) Σ_j r_{i,j}                                   # Average reward for FSM i
    end for

    # ═══ RL WEIGHT UPDATE (GRPO with LoRA) ═══
    # Policy gradient objective for task t:
    J(θ, t) = (1/NM) Σ_i Σ_j (r_{i,j} - V_i) ∇_θ log π_θ(τ_{i,j} | F_i, t)
    
    # Batch update over task batch B ⊂ T_a:
    θ ← θ + α · (1/|B|) Σ_{t∈B} ∇_θ J(θ, t)
    # With LoRA: KL regularization is implicit (β = 0)

    # ═══ TRUESKILL RATING UPDATE ═══
    v ← argsort([V_1, ..., V_M])                 # Rank FSMs by performance this round
    {(μ_i, σ_i)}^M ← TrueSkill.update({(μ_i, σ_i)}^M, v)

    # ═══ MUTATION (only the BEST FSM gets mutated) ═══
    k ← argmax_i V_i                              # Best FSM this iteration

    # Step 1: Self-reflect on agent trajectories under F_k (using FROZEN π_ref)
    Ψ_reflect = build_reflection_prompt(
        fsm=F_k,
        trajectories=[(t, τ_{k,1}, r_{k,1}), ..., (t, τ_{k,N}, r_{k,N})]
    )
    ℓ ~ π_ref(· | Ψ_reflect)                      # Self-reflection: identify mistakes & strategies

    # Step 2: Propose FSM edits based on reflection (using FROZEN π_ref)
    diff ~ π_ref(· | F_k, ℓ)                      # Propose structured FSM diff

    # Step 3: Apply diff and add to population
    F_mutate ← apply_fsm_diff(F_k, diff)           # Add states, fix transitions, etc.
    S.append((F_mutate, μ_k, σ_k + Δσ))            # Child inherits parent μ, increased σ

    # ═══ CROSSOVER (stochastic) ═══
    if p_crossover > random.rand() then
        B_past = sample_past_tasks(B_size)           # Replay buffer of past tasks
        Ω = {(F_i, μ_i, σ_i)}^M                     # Current selected FSMs
        (F_cross, μ', σ') = FSM_Crossover(B_past, R, Ω, π_θ, π_ref)
        S.append((F_cross, μ', σ'))
    end if

end while

Output:
  - Updated Qwen3-VL-8B π_θ* (with trained LoRA)
  - Best FSM: F* = argmax_{F_i ∈ S} μ_i
  - Deploy as: (π_θ*, F*)
```

### 3.3 FSM Crossover (Adapted from Algorithm 2)

```
FSM_Crossover(B_past, R, Ω, π_θ, π_ref):

    # Build score matrix: which FSM is best on which task?
    Φ ∈ R^{M × |B_past|}
    for (F_i, μ_i, σ_i) in Ω:
        for t_b in B_past:
            Run N rollouts: {τ_{i,b,j}} using π_θ with F_i on t_b
            Φ_{i,b} = (1/N) Σ_j R(t_b, τ_{i,b,j})
        end for
    end for

    # Find overall best FSM
    k ← argmax_i Σ_b Φ_{i,b}
    F_k is the overall top FSM

    # Identify differential strengths
    For each task t_b: which FSM was the best?
    For each FSM F_i: collect tasks ρ_i where F_i was top performer
    
    # Build crossover context
    φ = [(F_i, ρ_i, "why F_i excelled on ρ_i") for each F_i]

    # Self-reflection + crossover (using FROZEN π_ref)
    Ψ_cr = build_crossover_prompt(F_k, φ)
    diff_cr ~ π_ref(· | F_k, Ψ_cr)
    F_crossover ← apply_fsm_diff(F_k, diff_cr)

    # Child rating: precision-weighted average of parents
    μ' = (Σ_i μ_i/σ_i²) / (Σ_i 1/σ_i²)
    σ' = √(1/(Σ_i 1/σ_i²) + Δσ²)

    return (F_crossover, μ', σ')
```

### 3.4 FSM Diff Operations

Following E-SPL's git-diff approach, our FSM diffs are structured operations:

```
# Example FSM diff format
--- FSM v3 (Bluecoins)
+++ FSM v4 (Bluecoins)

+ ADD_STATE S5: SEARCH_RESULTS
+   Elements: [result_list, filter_bar, back_btn]
+   Note: Appears after using search in transaction list

~ MODIFY_TRANSITION S0→S1
-   Action: tap(add_button)
+   Action: tap(floating_add_button) OR tap(menu→"Add transaction")
+   Note: Some screens show floating button, others use menu

+ ADD_TRANSITION S0→S5
+   Action: tap(search_icon) → input_text(query)
+   Pre: home screen visible
+   Post: filtered transaction list

~ MODIFY_STRATEGY ADD_EXPENSE
-   S0→S1 (enter amount) →S2 (select category) →S1 (save) →S0
+   S0→S1 (enter amount) →S2 (select category) →S1 (set date if needed) →S1 (save) →S0
+   Fallback: If save fails, check required fields (amount, category, account)

+ ADD_STRATEGY SEARCH_TRANSACTION
+   S0→S5 (search by keyword) → tap result → view details
```

### 3.5 Mutation Self-Reflection Prompt (for π_ref)

```
You are analyzing an Android GUI agent's performance on app "{app_name}".
The agent used the following FSM (app knowledge map) to guide its actions:

{serialize(F_k)}

Here are the agent's trajectories on task "{task_description}":

TRAJECTORY 1 (reward: {r_{k,1}}):
  Step 1: Screen=HOME, Action=tap(add_button), Result=ADD_TRANSACTION screen
  Step 2: Screen=ADD_TRANSACTION, Action=input_text(amount="50"), Result=amount entered
  Step 3: Screen=ADD_TRANSACTION, Action=tap(save), Result=ERROR: category required
  ... [agent got stuck here, tried save 3 more times]

TRAJECTORY 2 (reward: {r_{k,2}}):
  Step 1: Screen=HOME, Action=tap(add_button), Result=ADD_TRANSACTION screen
  Step 2: ...

Analyze these trajectories and identify:
1. Where did the FSM lead the agent astray or fail to provide guidance?
2. What screens/states did the agent encounter that aren't in the FSM?
3. What transitions are missing or incorrect?
4. What error recovery strategies should be added?
5. What task strategies need refinement?

Provide your analysis as a structured reflection.
```

### 3.6 Crossover Self-Reflection Prompt (for π_ref)

```
You are combining the strengths of multiple FSM variants for app "{app_name}".

OVERALL BEST FSM (F_k):
{serialize(F_k)}

DIFFERENTIAL ANALYSIS:
{for each (F_i, ρ_i) in φ:}
  FSM variant {i} was the BEST performer on these tasks: {ρ_i}
  Its unique strengths compared to the overall best:
    {analyze why F_i outperformed F_k on tasks ρ_i}

Based on this analysis, propose edits to F_k that incorporate the unique strengths
of each variant. The goal is to produce a child FSM that inherits the best traits
from all parents. Output as a structured FSM diff.
```

### 3.7 Two-Layer FSM Schema (Required for Transfer)

**Motivation.** Sections 3.1–3.6 describe EvoFSM-RL as a *single-app* optimizer: the population S, the rollouts, the mutation, and the RL update are all anchored to one app `a`. In this form, the learned FSM is a monolithic blob of knowledge that gets better and better at app `a` — but there is no reason to believe any of it would help on a different app. Tier B (cross-app) and Tier C (cross-category) cannot be solved by simply running the single-app loop.

To enable transfer, we impose a **two-layer schema** on every FSM from the start. Each FSM `F_i` is serialized as two clearly-separated blocks:

```
# ═══════════════════════════════════════════════
# LAYER 1: APP_SPECIFIC  (non-transferable)
# ═══════════════════════════════════════════════
APP: Bluecoins
STATES:
  S0: HOME
    visual_cues: ["bottom nav with Wallet/Stats/Budgets", "floating + button bottom-right"]
    resource_hints: ["com.rammigsoftware.bluecoins:id/fab_main"]
  S1: ADD_TRANSACTION
    visual_cues: ["amount field at top", "category dropdown", "SAVE button top-right"]
  S2: CATEGORY_PICKER
    visual_cues: ["alphabetical category list", "create-new button"]
  ...
TRANSITIONS:
  S0 --tap(fab_main)--> S1
  S1 --tap(category_dropdown)--> S2
  S2 --tap(category_row)--> S1
  ...

# ═══════════════════════════════════════════════
# LAYER 2: GENERIC  (transferable, app-agnostic)
# ═══════════════════════════════════════════════
CATEGORY: ADD_ENTRY
  precondition: "a home-like screen is visible; a primary add affordance exists"
  abstract_steps:
    1. Locate primary add affordance (floating button OR overflow menu OR tab)
    2. Enter required numeric/text fields in order top-to-bottom
    3. For any dropdown field, tap to open picker, select matching value, return
    4. Confirm via explicit save/submit control; do NOT rely on back-gesture to save
    5. Verify the new entry appears in the primary list before declaring success
  failure_modes:
    - "tapping save before all required fields are filled"
    - "confusing the back button with save"
    - "leaving the picker without selecting, which silently reverts the field"
  verification_checklist:
    - "new row visible in the main list with the entered values"

CATEGORY: EDIT_ENTRY
  ...

CATEGORY: FILTER_LIST
  ...
```

**Why this separation matters.**

1. **Layer 1 (APP_SPECIFIC)** holds everything that is tied to the visual and structural identity of one app: state names, visual cue strings, resource ids, concrete transitions. This layer is **not** expected to transfer. When we move to a new app, it must be re-learned or re-extracted.

2. **Layer 2 (GENERIC)** holds category-indexed, app-agnostic patterns: high-level workflows, failure modes, and verification checklists. This layer is what we **do** expect to transfer, and it is what Tier B and Tier C ablations actually evaluate.

3. **The reflection prompt is modified** to explicitly ask π_ref to separate its insights into the two layers. Every mutation diff must specify which layer it edits. This discipline keeps generic insights from being accidentally tangled with Bluecoins-specific strings.

4. **The GENERIC layer is what gets exported and imported across apps.** A trained FSM `F_a*` for Bluecoins contributes `F_a*.LAYER2` to downstream Tier B / Tier C procedures; its LAYER1 is discarded on transfer.

### 3.8 Layered Reflection Prompt (modification of §3.5)

The mutation reflection prompt is modified so that π_ref is forced to emit its analysis in two buckets. This is the single most important change that makes transfer possible at the algorithmic level.

```
You are analyzing an Android GUI agent's performance on app "{app_name}".
The agent used the following two-layer FSM to guide its actions:

{serialize(F_k)}                 # Shows BOTH LAYER 1 and LAYER 2

Here are the agent's trajectories on task "{task_description}"
(category: {task_category}):

TRAJECTORY 1 (reward: {r_{k,1}}):
  ... step-by-step trace with screenshots ...

Analyze these trajectories and produce TWO SEPARATE LISTS of insights:

═══ LAYER 1 INSIGHTS (APP_SPECIFIC) ═══
Things that are true about "{app_name}" specifically:
- Missing states, wrong visual cues, incorrect resource ids
- Concrete transitions that don't match observed behavior
- App-specific UI quirks (e.g., this app's save button is in the top-right, not bottom)

═══ LAYER 2 INSIGHTS (GENERIC, category = {task_category}) ═══
Things that are true about the category "{task_category}" across apps:
- Abstract workflow patterns for this category
- Failure modes that would also apply to *other* apps in this category
- Verification signals that would generalize
IMPORTANT: Do NOT mention "{app_name}" or any app-specific widget names in Layer 2.
If you cannot say it without naming the app, it belongs in Layer 1.

Then output a structured FSM diff. Every diff block MUST be tagged @LAYER1 or @LAYER2.
```

This tagging requirement is what turns mutation from "make Bluecoins FSM better" into "extract whatever in this round's failure is actually about the add-entry category, and store it in a place that Tier B/C can access later". Without this separation, the generic signal in every trajectory would be quietly absorbed into app-specific strings and lost to transfer.

---

## 4. Reward Design

### 4.1 Task-Level Reward

```
R(t, τ) = w_success · R_success + w_efficiency · R_efficiency

where:
  R_success  = 1.0 if is_successful(task) else 0.0       # AndroidWorld evaluator
  R_efficiency = max(0, 1 - steps_taken / step_budget)    # Fewer steps = better

Suggested weights: w_success = 1.0, w_efficiency = 0.2
```

### 4.2 Value Estimate and Advantage (Following E-SPL Eq 2)

```
V_i = (1/N) Σ_j R(t, τ_{i,j})          # Average reward for FSM i on task t

Advantage for trajectory j under FSM i:
  A_{i,j} = r_{i,j} - V_i               # Relative to same-FSM average (REINFORCE baseline)
```

The key: advantages are computed **within each FSM group** (subtracting V_i), not across FSMs. This is standard REINFORCE with baseline. The **cross-FSM comparison** happens via TrueSkill, not via the policy gradient.

---

## 5. How This Maps to Our 4 Baselines

| Baseline | FSM | Weights | Maps to E-SPL variant |
|----------|-----|---------|----------------------|
| **1. Zero-shot** | None | Qwen3-VL-8B frozen | Base model |
| **2. Static FSM** | Fixed F_root | Qwen3-VL-8B frozen | Fixed system prompt |
| **3. Self-evolving FSM** | Evolving (evolution only) | Qwen3-VL-8B frozen | "Evolution" in Fig 11 |
| **4. EvoFSM-RL** | Evolving + RL | Qwen3-VL-8B + LoRA | "Evolution + RL" in Fig 11 |

**Expected behavior based on E-SPL results:**
- Baseline 3 should learn fast early (evolution is sample-efficient)
- Baseline 4 should achieve better asymptotic performance (RL improves procedural knowledge)
- The gap between 3 and 4 tells us how much procedural knowledge matters for GUI tasks

**The paper's agentic search results (Fig 13) are most analogous to our setting:**
- Base model: 6.6% → Evolution: 34.0% → RL: 44.2% → **Evolution + RL: 48.6%**
- This suggests RL matters A LOT for agentic tasks (vs math where evolution alone is competitive)
- Our GUI agent task is also highly agentic → expect significant RL contribution

---

## 6. Critical Implementation Details (From Paper's Ablations)

### 6.1 Use Frozen Reference for Evolution (Fig 15)

**This is the single most important design decision.**

The paper shows that using π_ref (frozen) instead of π_θ (RL-updating) for the self-reflection step in mutation/crossover consistently improves final performance. On HMMT: 0.510 (frozen) vs 0.527 (frozen) and on BeyondAIME: 0.416 (RL policy) vs 0.425 (frozen).

**For us:** Use the original frozen Qwen3-VL-8B (without LoRA) for the self-reflection steps in mutation and crossover. The LoRA-updated model is only used for task rollouts.

### 6.2 Crossover Effect (Fig 14)

Crossover accelerates early learning (faster convergence) but has **mixed effects on final performance**:
- On BeyondAIME: Mutation-only achieves 45.1%, Mutation+Crossover gets 42.5%
- On HMMT: Mutation+Crossover slightly better (52.7% vs 51.6%)

**Both variants outperform RL-only.** The paper suggests crossover's premature convergence effect could be mitigated by "niching" techniques (fitness sharing, mating restriction, island model).

**For us:** Start with mutation-only, add crossover as an ablation. If crossover helps on GUI tasks (where app knowledge is more structured than math knowledge), keep it.

### 6.3 Population Management

The paper uses a **growing tree** (not fixed population), with a **sliding window of size K** for selection. This means only the latest K prompts are eligible for selection, providing a natural recency bias.

**For us:** 
- Start with sliding window K = 10-20 FSM variants
- Population grows organically through mutation + crossover
- Selection naturally prunes underperformers (low TrueSkill → never selected)

### 6.4 Reusing RL Rollouts for Evolution

A key efficiency feature: **E-SPL reuses the same trajectories generated by RL for evolutionary fitness evaluation.** The score matrix Φ in crossover, and the trajectory analysis in mutation, all use data already generated during RL rollouts. This means evolution adds minimal computational overhead beyond self-reflection LLM calls.

**For us:** This is especially important since Android emulator rollouts are expensive (~1-2 min each). We should never run separate rollouts for evolution evaluation.

---

## 7. Adaptation for GUI Agents: Key Differences from Original E-SPL

### 7.1 Multi-step Trajectories vs. Single-Output Problems

E-SPL's math tasks produce a single output y per rollout. Our GUI tasks produce multi-step trajectories with 10-25 actions. This affects:

- **Policy gradient:** Need to sum over all actions in the trajectory, not just one output
  ```
  ∇_θ log π_θ(τ | F_i, t) = Σ_{step=1}^{T} ∇_θ log π_θ(a_step | state_step, F_i, t)
  ```
- **Self-reflection:** Much richer signal — can pinpoint exactly which step failed and why
- **FSM relevance:** Each step can be grounded in an FSM state/transition

### 7.2 Structured vs. Free-Form System Prompts

E-SPL evolves free-form text prompts. Our FSMs are structured (states, transitions, strategies). This creates both opportunities and constraints:

**Opportunities:**
- FSM diffs are more semantically meaningful than text diffs
- Crossover can combine specific state subgraphs (not just text segments)
- FSM quality can be partially validated (e.g., is the graph connected? are transitions consistent?)

**Constraints:**
- Evolution operators must preserve FSM structural validity
- Need to handle FSM size limits (can't grow unboundedly)
- Serialization format must be consistent across mutations

### 7.3 Expensive Rollouts

E-SPL rollouts are cheap (LLM inference). Our rollouts are expensive (Android emulator interaction, ~1-2 min each). This means:

- **Fewer rollouts per iteration:** N = 1-3 (not ~8-16 as in math tasks)
- **Fewer FSM variants per iteration:** M = 2-4 (not ~4-8)
- **More iterations needed** for statistical significance in TrueSkill
- **Parallelization across emulators** is critical

### 7.4 Visual Grounding

Qwen3-VL-8B processes screenshots. The FSM's declarative knowledge (states, transitions) must be grounded in what the model can see. This means:

- FSM state descriptions should reference visual elements (buttons, menus, etc.)
- Mutation should update visual descriptions when screens change
- The model needs to learn to match FSM states to current screenshots (procedural knowledge via RL)

---

## 8. Unified Pretraining and Two Test-Time Evaluation Conditions

**Design framing (updated).** Earlier drafts of this section described Tier B and Tier C as *two independent training tracks*, each with its own pretraining regime (Tier B = single-source, Tier C = multi-source joint). We now unify them: **there is exactly one pretrained artifact**, produced by multi-source joint training over a broad source pool, and **Tier B / Tier C are two different evaluation conditions on the same artifact**, distinguished only by the relationship between the target app and the source pool at deployment time.

This matches how real deployment would work: you train once and deploy anywhere, with some target apps similar to what you pretrained on ("near transfer", Tier B condition) and others far outside that distribution ("far transfer", Tier C condition). The base single-app optimizer of §3.1–3.6 alone is insufficient for either case — we additionally need the layered schema (§3.7), layered reflection (§3.8), multi-source joint training (§8.2), cross-app crossover (§8.3), and the deployment-time hand-off (§8.4). None of these are optional for transfer.

**Test-time adaptation (TTA) is the core capability.** The single-app EvoFSM-RL loop of §3.2 is the *same* algorithm that runs both during source-pool pretraining (Phase 1) *and* during target-app deployment (Phase 3). The difference between "training" and "deployment" is only **which app's task set we draw from** and **how much compute budget we spend**. This is what we mean by TTA: once deployed on a new target app, both the FSM population `S_{a_tgt}` and the LoRA `π_θ` continue to evolve online as more target-app tasks arrive. The main metric of the paper is an **adaptation curve** `SR_eval(k)` over the target-app online-adaptation budget `k`, reported separately under the near-transfer and far-transfer evaluation conditions.

### 8.0 Unified Pipeline (5 Phases)

All four baselines share the same 5-phase pipeline, and **the pipeline is the same for both evaluation conditions (near-transfer / far-transfer)**. What changes between near and far is only the specific source app chosen in the Phase-2 hand-off (§8.4); Phase 1 is a single, unified run that produces one pretrained artifact used for all downstream evaluations.

| Phase | Scope | Apps | What happens | Output |
|-------|-------|------|--------------|--------|
| **Phase 0** — Offline static extraction | Once, ahead of everything | All apps | GPT-4o T3A runs on source-pool training tasks; an LLM synthesizes per-app static FSMs `F^0_a` (§3.0 / `builder.py`). | `{F^0_a}`, used as the static-FSM baseline and as seed of evolutionary populations. |
| **Phase 1** — Unified source-pool pretraining | One run | Source pool `A_src` (multiple clusters, multiple categories) | Multi-source joint EvoFSM-RL (§8.2) over the entire source pool, with cross-app crossover (§8.3) writing to per-category abstract libraries `{L_C}`. | Single pretrained artifact: `π^pre_θ`, per-source-app populations `{S^pre_a}`, and abstract libraries `{L_C}`. **Used for all target apps in both evaluation conditions.** |
| **Phase 2** — Target-app cold start | Once per target app | Target app `a_tgt` | (1) 5–10 exploration rollouts of `π^pre_θ` on `a_tgt` → exploration report. (2) Build `F_{a_tgt}.LAYER1` from exploration. (3) **Condition-dependent hand-off** (§8.4): near-transfer translates the nearest same-cluster source's LAYER2; far-transfer specializes the matching `F_C_abstract` from `L_C`. (4) Seed `S^init_{a_tgt}` from mutations of the resulting FSM. (5) Split target-app tasks `T_{a_tgt}` into `T_adapt` (70%) / `T_eval` (30%), stratified by sub-category. | `S^init_{a_tgt}`, `T_adapt`, `T_eval`. |
| **Phase 3** — Online adaptation on target app | TTA | `T_adapt` only | Same single-app EvoFSM-RL loop as §3.2, drawing tasks only from `T_adapt`. After every `k_step` adaptation tasks, save a *snapshot* of `(π_θ, S_{a_tgt})`. This is the adaptation phase whose curve we report. | Snapshot sequence `{(π^k_θ, S^k_{a_tgt})}_{k=0..K_max}`. |
| **Phase 4** — Held-out evaluation | Per snapshot | `T_eval` only | For each snapshot `k`, evaluate the policy `(π^k_θ, F^k)` where `F^k = argmax_F μ_F ∈ S^k_{a_tgt}` on the frozen held-out `T_eval`. **No updates during evaluation.** Plot `SR_eval(k)` and derived statistics. | Adaptation curves per baseline × per evaluation condition × per target app. |

**Why splitting the target app (Phase 2 step 5) rather than pure streaming RL.** We considered an alternative where the target app is not split and the entire target-app task set is consumed online, with streaming SR reported over the run. We rejected this in favor of the split-with-held-out design for three reasons:

1. **Generalization measurement.** Pure streaming reports how well the system fits the exact sequence of tasks it saw. The split lets us ask whether adaptation on `T_adapt` *generalizes* to unseen `T_eval` tasks of the same app — which is what a user-facing deployment actually demands.
2. **Task-order robustness.** Streaming SR is very sensitive to the sampling order of tasks; variance across seeds becomes large and comparisons across baselines become noisy. A frozen `T_eval` yields a stable comparison point.
3. **Cleaner paper story.** "TTA in 20 tasks transfers to 13 held-out tasks of the same app" is strictly stronger than "online SR goes up during a streaming run." The streaming-SR number can still be reported as an appendix sanity check.

**What goes into `T_adapt` vs `T_eval`.** Stratified 70/30 split of the target app's category-filtered task set. Stratification is by sub-category (e.g., ADD / QUERY / DELETE / FILTER) so that each split covers the same task shapes. Seeds: we run 3 seeds per (tier × target app × baseline) by re-drawing the split.

**Snapshot granularity.** `k_step = 5` adaptation tasks in early Phase 3 (where the curve is steep), coarsening to `k_step = 10` later. Snapshots are cheap (they are just refs to `π_θ` and `S_{a_tgt}`) — the only cost is the Phase 4 evaluation pass per snapshot on `T_eval`.

**Baselines as TTA variants.** Under this pipeline the four baselines (§5) become different update policies during Phase 3. Each baseline is evaluated under *both* near-transfer and far-transfer conditions, producing two adaptation curves per baseline. The "Tier B vs Tier C" comparison in the paper is across-condition at fixed baseline (same model, same update policy, different target-app regime).

| # | Phase 3 update policy | Phase 4 curve shape |
|---|------------------------|---------------------|
| **B1** Qwen3-VL zero-shot | Nothing updates; FSM = none. | Flat line. |
| **B2** + static FSM | Nothing updates; FSM = `F^0_{a_tgt}` from Phase 0. | Flat line, higher than B1. |
| **B3** + self-evolving FSM | FSM population evolves via §3.2 without RL; `π_θ` frozen at `π^pre_θ`. | Rising curve. |
| **B4** EvoFSM-RL | FSM evolves *and* `π_θ` updates via GRPO. Three paths for LoRA update frequency: **Path A** LoRA frozen (collapses to B3, consistency check), **Path B** LoRA updates every adaptation step, **Path C** LoRA updates every `K` adaptation steps (default). | Steepest curve, highest ceiling. |

### 8.1 The Two Evaluation Conditions

After Phase 1 produces the unified pretrained artifact `(π^pre_θ, {S^pre_a}, {L_C})` over a source pool `A_src`, each target app `a_tgt` is classified into one of two conditions based on its relationship to the source pool. The assignment is made *ahead of time* as part of the benchmark split, not at training time.

**Near-transfer condition (Tier B).** The target app's *cluster* is represented in the source pool, i.e., there exists at least one same-cluster source app `a_nearest ∈ A_src` (e.g., target = Expense Manager, source pool contains Bluecoins). Same-cluster means they share both task categories and broad UI idioms. At Phase-2 hand-off (§8.4), we use `F_{a_nearest}*.LAYER2` for translation. This condition tests *near-distance* transfer — within-cluster knowledge carried across different UIs.

**Far-transfer condition (Tier C).** The target app's *cluster* is not represented in the source pool, or the target itself is held out entirely across its cluster (e.g., target = Simple Calendar, no calendar app in the source pool). The only knowledge the model can draw on is the cross-app abstract library `L_C` built during Phase 1. At Phase-2 hand-off (§8.4), we specialize the best `F_C_abstract ∈ L_C` to the target. This condition tests *far-distance* transfer — category-level abstractions carried across unrelated clusters.

**A single target app belongs to exactly one condition.** Apps in the target pool are split into a near-transfer subset (same-cluster representative is in `A_src`) and a far-transfer subset (cluster is held out in `A_src`). This assignment is part of the benchmark split and is fixed for the paper.

**What we report.** Each baseline B1–B4 produces two adaptation curves — one averaged over near-transfer target apps, one averaged over far-transfer target apps. Paper main figure: two panels (near / far) side-by-side, each with four baseline curves. The near-vs-far gap at any given baseline tells us how quickly transfer quality degrades with target-cluster distance.

### 8.2 Phase 1 — Unified Multi-Source Joint Training

Phase 1 is **a single training run** over the full source pool `A_src` — we do not run separate "Tier B training" and "Tier C training". The source pool is chosen to span multiple clusters and multiple task categories, so that the pretrained artifact is simultaneously useful for near-transfer deployment (when a same-cluster source exists) and far-transfer deployment (when only category-level abstractions apply).

```
Algorithm: MultiSource-EvoFSM-RL   (unified Phase-1 pretraining)
═══════════════════════════════════════════════════════════════

Input:
  - π_θ (single shared LoRA over base VLM), frozen π_ref
  - Source pool A_src = {a_1, ..., a_K} spanning multiple clusters
  - Task categories C_set = {C_1, ..., C_M} (e.g., ADD_ENTRY, QUERY, ...)
  - Per-app task sets {T_{a_k}}, each tagged by category
  - Per-app initial FSMs {F_{a_k, root}}  (layered, §3.7)
  - Hyper-parameters as in §3.2, plus p_cross_app

Initialize:
  For each a in A_src:
     S_a = {(F_{a, root}, μ_0, σ_0)}     # one population per app
  Abstract libraries:
     L_{C_m} = {}  for each C_m in C_set  # one per category

while not converged do:

    # ═══ PER-ITERATION APP & CATEGORY SAMPLING ═══
    a ← sample_uniform(A_src)              # pick an app this round
    t ← sample(T_a)                        # any task in this app (any category)
    C ← category_of(t)                     # which category this task belongs to

    # ═══ STANDARD SINGLE-APP INNER STEP (uses S_a only) ═══
    Select M FSMs {(F_i, μ_i, σ_i)}^M from S_a
    for i=1..M, j=1..N:
        τ_{i,j} = rollout(π_θ, system_prompt=F_i, task=t)
        r_{i,j} = R(t, τ_{i,j})
    V_i = mean_j r_{i,j}

    # ═══ RL UPDATE — SHARED LoRA ═══
    # Gradient is accumulated into the SAME θ regardless of which app was sampled.
    J(θ, t, a) = (1/NM) Σ_i Σ_j (r_{i,j} - V_i) ∇_θ log π_θ(τ_{i,j} | F_i, t)
    θ ← θ + α · ∇_θ J(θ, t, a)

    # ═══ TRUESKILL + LAYERED MUTATION (inside S_a only) ═══
    update {μ_i, σ_i} for S_a via TrueSkill
    k ← argmax_i V_i
    (ℓ_L1, ℓ_L2) ~ π_ref( layered_reflection_prompt(F_k, trajectories, category=C) )
    diff ~ π_ref( F_k, ℓ_L1, ℓ_L2 )          # diff carries @LAYER1 / @LAYER2 tags
    F_mutate ← apply_fsm_diff(F_k, diff)
    S_a.append((F_mutate, μ_k, σ_k + Δσ))

    # ═══ STANDARD SAME-APP CROSSOVER (stochastic, as in §3.3) ═══
    if p_crossover > rand():
        (F_cross, μ', σ') = FSM_Crossover(sample_past_tasks(a), R, Ω_a, π_θ, π_ref)
        S_a.append((F_cross, μ', σ'))

    # ═══ CROSS-APP CROSSOVER (stochastic, per category, §8.3) ═══
    if p_cross_app > rand():
        C_pick ← sample_eligible_category(A_src, C_set)   # one whose champions are mature
        F_{C_pick}_abstract ← CrossAppCrossover(A_src, {S_{a_k}}, category=C_pick, π_ref)
        L_{C_pick}[timestamp] = F_{C_pick}_abstract

end while

Output (single unified artifact):
  - π_θ*         (shared LoRA, trained across full source pool)
  - {F_{a_k}*}   (per-source-app best FSMs)
  - {L_{C_m}}    (one abstract library per task category)
```

Two things are worth calling out about this loop.

First, **a single shared LoRA is updated from rollouts across all source apps and all categories**. This is what gives π_θ the opportunity to learn both cross-app-within-category procedural patterns (helpful for near-transfer) and cross-category cross-app abstractions (helpful for far-transfer). It also introduces the multi-source interference risk (§8.6): gradient signals from diverse apps may partially cancel on truly app-specific behaviors.

Second, **each app keeps its own population `S_a`, and each category keeps its own abstract library `L_C`**. We do not attempt to cross-pollinate LAYER1 content across apps — that would just produce nonsense diffs. Cross-pollination happens at the LAYER2 level only, per category, via the cross-app crossover operator below.

### 8.3 Cross-App Crossover Operator (extends §3.3)

Standard E-SPL crossover (§3.3) operates within one population on one problem distribution. The cross-app operator is a new, additive operator that runs across populations and produces an app-agnostic child that gets stored in the abstract library `L_C`, not in any `S_a`.

```
CrossAppCrossover(A_src, {S_{a_k}}, category C, π_ref):

    # ─── Collect the current per-app champions' LAYER2 for category C ───
    Ψ_C = []
    for a in A_src:
        F_a* = argmax_{F ∈ S_a} μ_F              # current best FSM for this app
        L2_a = F_a*.LAYER2[C]                     # generic part for category C only
        Ψ_C.append( (a, L2_a, evidence=per-task reward trace for a) )

    # ─── Build cross-app crossover reflection prompt (for frozen π_ref) ───
    Ψ_cr_cross = """
    You are synthesizing an APP-AGNOSTIC strategy for task category '{C}'.
    Below are {|A_src|} generic strategies, each produced by EvoFSM-RL on a
    different app, for this same category.

    {for each (a, L2_a, evidence) in Ψ_C:}
        ─── Source app: {a} ───
        Generic strategy:
        {L2_a}
        Evidence of what worked:
        {evidence}

    Produce a single UNIFIED LAYER2 block for category '{C}' that:
    1. keeps only claims that held across MULTIPLE source apps
    2. marks as "app-dependent variant" any claim that held in only one source
    3. explicitly lists failure modes that appeared in more than one source
    4. contains NO mention of any specific app name or widget name
    5. is structured as a layered-FSM LAYER2 block (abstract_steps,
       failure_modes, verification_checklist)
    """

    F_C_abstract ~ π_ref( Ψ_cr_cross )

    return F_C_abstract
```

The returned `F_C_abstract` is never pushed into any single-app `S_a` — it lives in `L_C` and is only used at deployment-time hand-off (§8.4). Pushing it back into `S_a` would dilute app-specific signal and break the single-app evolutionary pressure.

### 8.4 Deployment-Time Hand-off (Near vs Far Conditions)

Once Phase 1 has produced the single pretrained artifact `(π^pre_θ, {S_a}, {L_C})`, every target app `a_tgt` enters Phase 2 via the same hand-off routine. The routine branches on whether `a_tgt`'s category `C_tgt` was represented in the source pool; this branching is what defines the near-transfer vs far-transfer evaluation conditions. The branch is the *only* per-condition difference in the entire pipeline — Phase 3 (TTA) and Phase 4 (eval) are identical.

```
Deploy(a_tgt, π^pre_θ, {S_a}, {L_C}, A_src, frozen π_ref):

  1. Exploration: run 5–10 rollouts of π^pre_θ on a_tgt
                  → build F_{a_tgt}.LAYER1_skeleton
  2. Category probe: estimate C_tgt ∈ categories(a_tgt)
  3. Hand-off:
       if C_tgt ∈ categories(A_src):            # ── near-transfer condition
           a_near = argmin_{a ∈ A_src, C_tgt ∈ categories(a)} dist(a_tgt, a)
           L2_src = S_{a_near}.champion.LAYER2[C_tgt]
           F_{a_tgt}_init = LAYER2_translate(L2_src, a_tgt, π_ref)
       else:                                    # ── far-transfer condition
           F_C_abstract = most_recent_highest_rated( L_{C_tgt} )
           F_{a_tgt}_init = LAYER2_ground(F_C_abstract, a_tgt, π_ref)
  4. Seed S^init_{a_tgt} from mutations of F_{a_tgt}_init
  5. Split T_{a_tgt} ∩ C_tgt → T_adapt (70%) / T_eval (30%)

Phase 3 (TTA) and Phase 4 (eval): identical to §8.0.
```

The two conditions are **evaluated on disjoint pools of target apps**, never on the same app: a given `a_tgt` is assigned to exactly one condition by construction of the source pool. The paper's main result is therefore two adaptation curves per baseline — one per panel — reported side-by-side, not two numbers for the same target.

**Optional hand-off ablations (one adaptation curve each, on the same target pool of a single condition).** These are small sanity checks around the hand-off step itself; they do not spawn new training runs.

| Ablation | Hand-off step (Phase 2) | Purpose |
|----------|--------------------------|---------|
| H-none | skip step 3; use static `F^0_{a_tgt}` only | lower bound (= B1 under this condition) |
| H-weights-only | skip step 3 LAYER2 injection; seed S from mutations of `F^0_{a_tgt}` | isolate contribution of the pretrained LoRA |
| H-single-source (near only) | use a random in-category source `a` instead of argmin | test sensitivity to source choice |
| H-stale-L_C (far only) | use oldest L_C entry instead of most-recent | test `L_C` staleness claim (see §8.6) |
| H-full (default) | as above | the reported number |

H-none / H-weights-only correspond to B1 / B2 from the baseline table (§8) and so are already reported; the remaining rows are only run if main results warrant it.

### 8.5 Why this requires both a shared LoRA and cross-app crossover

It is worth stating plainly: neither mechanism is sufficient alone, under either condition.

A shared LoRA with only per-app (non-cross) crossover would learn procedural knowledge but leave the declarative knowledge fragmented across `K` disjoint LAYER2 blocks, each still anchored in its source app's evidence. In the near-transfer condition this is survivable (pick the in-category source and translate); in the far-transfer condition there is no in-category source to pick and the declarative knowledge needed to cold-start `a_tgt` does not exist.

Cross-app crossover without a shared LoRA would produce a nice abstract LAYER2 block but no aligned procedural weights, so at deployment time π_θ is a zero-shot base model that has never been trained to *execute* anything in category `C_tgt`. The abstract block gives good *planning* but the policy cannot follow it.

The combination is what the far-transfer condition actually rests on, and it is also what lifts the near-transfer condition above single-source transfer. This is the part of the algorithm that most directly distinguishes our work from a straight application of E-SPL.

### 8.6 Risks of the Transfer Story (Things That Can Break)

1. **Layered reflection brittleness.** The two-layer discipline relies entirely on π_ref correctly separating generic from app-specific content. If the frozen model is loose about this, LAYER2 content will silently leak app-specific strings and transfer will degrade quietly. Mitigation: lint LAYER2 diffs for app-name and resource-id substrings; reject diffs that fail the lint.

2. **Visual negative transfer (near-transfer condition).** If the nearest source and the target look visually very different (dark mode vs light, icon-style vs text-style), the shared LoRA can make the policy *worse* on the target than the zero-shot base. This showed up repeatedly in our pilot. Report B2 (weights only) separately per condition and be prepared for it to be below zero-shot on individual near-transfer targets.

3. **Multi-source interference hurting far-transfer.** Gradients from different source apps may cancel on behaviors that are genuinely app-specific but were accidentally exercised during category-C rollouts, producing a shared LoRA that is *worse* than any single-source LoRA on any one app. This most visibly hurts the far-transfer condition, because far-transfer has no in-category source to fall back on. Mitigation: keep `|A_src|` small (3–4) and monitor per-app validation throughout joint training; if any source app regresses past a threshold, fall back to smaller `|A_src|`.

4. **L_C staleness.** The abstract library L_C is written during training and read at transfer time. If joint training runs a long time, earlier `F_C_abstract` entries become stale. We use the most recent highest-rated entry; ablating "most recent vs highest rated" is a cheap sanity check.

5. **Cross-app crossover cadence.** Running it every iteration is wasteful (shared LoRA hasn't moved enough). Running it only at the end collapses to a one-shot synthesis. A geometric schedule (every 2^n iterations) is a reasonable default.

---

## 9. Suggested Hyperparameters (Adapted from Paper)

| Parameter | E-SPL (Paper) | Our Setting (GUI) | Rationale |
|-----------|--------------|-------------------|-----------|
| M (FSMs per iter) | Not specified (likely 4-8) | 2-4 | Expensive rollouts |
| N (rollouts per FSM per task) | ~8-16 | 1-3 | Expensive rollouts |
| LoRA rank | 32 | 16-32 | Qwen3-VL-8B is smaller than DeepSeek-v3.1 |
| β (KL coeff) | 0 (with LoRA) | 0 | Same reasoning |
| λ (optimism) | Not specified | 1.0-2.0 | Encourage exploring new FSMs |
| T (temperature) | Not specified | 1.0 | Standard softmax |
| Δσ (child uncertainty) | Not specified | 1.0-2.0 | Higher → more exploration of children |
| (μ_0, σ_0) | Standard TrueSkill (25, 25/3) | (25, 8.33) | Standard |
| p_crossover | Not specified | 0.3-0.5 | Lower initially, ablate |
| K (sliding window) | Not specified | 10-20 | Keep recent FSMs |
| Iterations | 15-30 (in paper) | 30-100 | More needed due to fewer rollouts/iter |

---

## 10. Open Questions for Team Discussion

1. **LoRA rank:** E-SPL uses rank 32 on much larger models. For 8B Qwen3-VL, rank 16 may suffice. Needs ablation.

2. **N (rollouts per FSM per task):** With expensive emulator runs, N=1 is tempting but gives noisy V_i estimates. N=2-3 is a minimum for meaningful TrueSkill comparisons. Can we parallelize across emulators?

3. **Mutation-only vs. Mutation+Crossover:** Paper shows mixed results. For GUI FSMs, crossover might be MORE useful than for math (structured knowledge is easier to recombine). Worth testing.

4. **Reference model for self-reflection:** Paper is clear: use frozen π_ref. But for us, should the mutation/crossover reflection also see the task trajectory screenshots? If so, we may need a multimodal π_ref (frozen Qwen3-VL-8B).

5. **FSM size control:** E-SPL prompts can grow without bound. FSMs need bounded complexity. Options: max states/transitions, complexity penalty in reward, periodic FSM compression via LLM.

6. **Per-app vs. multi-app training:** E-SPL trains on a single problem distribution. Should we train one LoRA per app, or one shared LoRA across apps (with per-app FSMs)? Shared LoRA = better cross-app transfer but potential interference.

7. **Task sampling:** E-SPL samples one problem per iteration. For GUI agents with fewer total tasks per app, should we cycle through tasks systematically or sample randomly with replacement?

8. **The "agentic search" analogy:** The paper's agentic search experiment (gpt-oss-120b on NQ/HotpotQA) is closest to our setting. There, RL contributes much more than evolution (44.2% vs 34.0%). Does this predict that weight updates will be especially important for GUI agents too?

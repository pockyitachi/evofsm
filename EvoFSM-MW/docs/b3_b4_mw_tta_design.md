# B3 / B4 on MobileWorld — TTA experiment design

**Date**: 2026-06-11 · **Status**: ACTIVE (user-approved design)
**Supersedes**: the 2026-06-08 "TTA on MW = symbolic-only, no LoRA channel" decision
(`mobileworld_split.md` adapt-protocol section). That restriction is retired: B4 keeps
its original definition from `EvoFSM-RL/plan/algorithm_design.md` §8 — joint FSM
evolution **and** weight adaptation. The 51/110 task-disjoint split itself is unchanged
and stays authoritative in `configs/mobileworld_splits.yaml`.

**Goal.** Test-time adaptation on the MW adapt split at BOTH levels simultaneously:
- **prompt level** — per-app FSM / per-category L_C evolution (declarative knowledge);
- **weight level** — LoRA GRPO on the same rollouts (procedural knowledge);
so the deployed model gets stronger on the MW domain, and the paper gets its
B3 vs B4 adaptation curves (the E-SPL joint-optimization claim, GUI edition).

---

## 1. Definitions

| | FSM/L_C | weights | rollout backend |
|---|---|---|---|
| **B3** | evolves (per-app FSM + per-category L_C) | frozen at π^pre | eval harness vs frozen vLLM server |
| **B4** | evolves (same controller) | **shared LoRA, GRPO** (Path C: update every step over that step's M×N trajs) | verl (on-policy) |

`B4 − B3` at matched iteration budget = the weight-channel contribution.
Optional consistency check: B4 Path-A (LoRA frozen) must reproduce B3 within noise.

## 2. Weight topology — **one shared LoRA** (decided 2026-06-11)

Strict per-app weights (the AW Paradigm-A winner) die on MW's eval composition:
**65/110 (59%) eval tasks are multi-app** (Tier-A 27/27, Tier-B 36/58, Tier-C 2/25),
and only 33/110 are single-app-on-a-head-app — per-app adapters would cover 30% of
eval and be invisible exactly where the compositional headline (Tier-A) lives.
A single shared LoRA covers all 110 unambiguously, eats all 51 adapt tasks
(thickest data), and has design precedent: Phase-1 §8.2 is exactly
"shared LoRA + per-app FSM populations". Narrative: *weights adapt to the
deployment domain; symbolic knowledge adapts at app/category granularity.*
(Per-app weights remain available as an AW-side ablation if reviewers ask.)

**Initialization**: merge the Phase-1 π^pre LoRA into the base →
"MW base" = π^pre exactly; mount a fresh adapter r32/α64 (language-only,
`exclude_modules=.*visual.*`). verl's `ref_in_actor` (adapter off ⇒ base)
then anchors KL to π^pre *exactly*, as algorithm_design §662 requires.
lr 2e-5, KL β=0.001 low_var_kl (Phase-1-relaunch values).

**Symbolic side**: FSM per app (15 apps in adapt), L_C per category, root population
seeded from the **B2′ configuration** (app Layer-2 + category L_C — the strongest
static variant, 9.2±1.6 in the 25-run study). Evolution edits Layer-2 / L_C only
(layer2_only + app-name lint), per the two-layer discipline.

## 3. The adapt loop (one iteration)

```
task t ~ schedule(51)                       # §4
select M=2 variants from S_app(t) by optimistic softmax  p_i ∝ exp((μ_i+λσ_i)/T), window K=15
for each variant i: N_t rollouts (adaptive N, §4), fresh MW container each, temp 1.0
V_i = mean success  → TrueSkill tournament update
GRPO: each (variant) row is its own uid group → within-group advantage  (B4 only)
mutation: best variant if V_best>0 or every 3rd iteration —
   reflection + layer2-only diff via claude-opus-4-8 (frozen π_ref role), lint, append child
```

Evolution and RL **share the same trajectories** (E-SPL efficiency rule — never roll
separately for fitness). MW has no seeds: repeats of the same fixed task differ via
temperature sampling, the changing variant, and (B4) the moving weights; the
memorize-vs-generalize question is settled by the task-disjoint 110-task eval.

## 4. Budget & schedule (user-approved 2026-06-11)

- **Total ≤ ~1,200–1,500 MW rollouts ≈ 3 days at 8 concurrent containers.**
- **Round 1** — uniform pass: 51 iterations (one per adapt task), M=2 × N=4
  → 408 rollouts. Purpose: locate the boundary set (tasks with ≥1 mixed group).
- **Round 2+** — boundary-focused: tasks that ever produced a mixed group get
  N=8; never-mixed dead tasks get N=2 or are skipped; ~100 further iterations
  weighted toward boundary tasks (task weights ∝ recency-decayed mixed-rate).
- **B3 control**: same iteration schedule and budget, no weight updates.
- Rationale: weight channel volume is ~1/15 of Phase-1 — this is light
  specialization (weld the boundary tasks), not pretraining. N=4 ⇒ P(mixed)≈28%
  at p=0.08; N=8 ⇒ ≈49%.

## 5. Snapshots & evaluation

- Per-step: population JSON + TrueSkill state persisted; LoRA ckpt every 25 steps
  (keep all; adapters ~350 MB).
- **Snapshots at k ∈ {0, ⌈⅓⌉, ⌈⅔⌉, final} of the iteration budget.
  Every snapshot is evaluated on the FULL 110-task eval split** (user decision —
  no subsampling), frozen: champion guidance per app/category + that snapshot's
  weights, fresh containers, same harness/protocol as the B-series variance study
  (concurrency 5, ~5–6 h per snapshot).
- Deliverable: SR_eval(k) curves for B3 and B4 (+ B1 flat reference 8.2±0.8 and
  B2′ static 9.2±1.6 from the variance study), overall + per-tier.
- k=0 note: B4's k=0 = π^pre + B2′ root guidance — this is also the "π^pre vs base"
  cross-benchmark readout (compare against pi350mai's π^pre_350@B1 when available).

## 6. Infrastructure (the MW–SkyRL bridge)

New code, all under our repos; AW training path untouched:

1. **`mobile_world` task family in skyrl-agent** — `MWContainerManager`
   (fresh `mobile_world:reset` container per rollout, `--network mwnet`,
   per-rollout port allocation, destroy after judge — MW containers never reset
   app state, so **no reuse, ever**) + `MWTask` (load task by name from the
   MobileWorld runtime, initialize, step `mobile_use` actions, judge success).
   Registered alongside `android_world` in the agent/task registries.
2. **`MWAdaptDataset`** (verl `custom_cls`) — rows `{task, variant_slot∈{A,B}, apps,
   tier}`; one uid per row ⇒ GRPO groups = variants, for free.
3. **Evolution controller** — a backend-agnostic module (population, TrueSkill,
   optimistic-softmax selection, Opus-4.8 mutation with layer2 lint, persistence,
   wandb logging). For **B4** it hooks into verl via the *existing*
   `train_dataset.on_batch_end(batch)` hook (`verl_trainer.py:834`): read this
   step's per-uid rewards → TrueSkill → select next variants → write the next
   step's guidance files. Round-2 task weighting uses the curriculum-sampler hook
   (`verl_trainer.py:818`) or a segmented relaunch (resume_mode=auto). For **B3**
   the same controller drives the EvoFSM-MW eval harness against a frozen vLLM
   serve of π^pre (pi350-style) — no verl.
4. **Variant guidance injection** — per-(app, slot) guidance JSON files in the run
   dir; the rollout agent resolves guidance through the same injection path as
   B2-family evals, keyed by the row's `variant_slot`. Prompt format identical to
   the B-series harness (`prompt.py` + `action_parse.py`) so B3/B4/eval are
   byte-comparable at the prompt level.
5. **wandb** (user: log everything): project `skyagent-android`,
   experiments `mw-b4-tta-*` / `mw-b3-tta-*`; controller logs per-step variant
   ratings, mutation events, mixed-group rate, boundary-set size into the same run.

## 7. Risks & guards

- **Gradient sparsity** (success concentrated in few tasks): adaptive N + boundary
  focus; if B4−B3 still small, report honestly as "weight channel contributes
  little in low-data TTA" — a finding, not a failure.
- **Instance memorization** (no seeds): KL anchor to π^pre + small lr; disjoint
  110-task eval is the verdict.
- **Host pressure**: 8 MW containers will coexist with Phase-1's 16 AW emulators —
  watch RAM before raising concurrency.
- **π^pre gating** (user): real B3/B4 runs start only after Phase-1 reaches
  ≥ step 200; smoke uses base + fresh LoRA (pipeline validation only).
- **Smoke** (GPU7, shared with the pi350 vLLM tenant — `GPU_MEM_UTIL≈0.55`):
  1–2 adapt tasks × 2 iterations, M=2 N=2, one GRPO update + one Opus mutation +
  snapshot save + wandb. Code/pipeline pass only.

## 7b. Smoke findings (2026-06-11, pipeline GREEN)

2 tasks × 2 iterations × (M=2 × N=2) on GPU7 (shared with the MAI-UI vLLM
tenant, `GPU_MEM_UTIL=0.45` — fsdp2 without offload holds fp32 master params,
~32 GB): full chain green — MW pool boot (~7 min), rollouts (~10-12
turns/episode), GRPO update + adapter ckpt per step, TTA hook (TrueSkill +
stub mutation + next-step guidance) closed the loop; exit 0.

- **Known quirk**: verl's `fit()` returns on the last step *before* the
  `on_batch_end` hook → the FINAL iteration gets no TrueSkill/mutation update.
  Negligible (1 of ~150 iterations; populations/ckpts complete). B3's own
  driver loop has no such gap.
- **Injected-guidance size is real**: avg_input_tokens jumped 43.9k → 93.8k
  when full-library guidance kicked in. Before the real run, cap per-app
  Layer-2 blocks (top-K categories) or apply B2'''-style entry retrieval (§8).
- Batch plumbing detail: verl pops `instance` out of the train batch — the
  controller reconstructs slots from uid groups (requires `data.shuffle=False`,
  sequential sampler; both are in the launcher).

## 8. Deferred / out of scope

Per-app-weights ablation (AW side); B2'''-style entry retrieval on the root
guidance (kept off to limit variables); success-replay auxiliary loss; Path-B
(per-step LoRA update) ablation.

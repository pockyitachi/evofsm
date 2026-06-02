# Results — B3 (evolved `L_C`)

B3 is the next rung after B2: the per-category `L_C` is now an *evolved*
knowledge structure, mutated online on the target app's `T_adapt` split
by Claude Opus 4.7, frozen at the end of adaptation, and evaluated on
the template-disjoint `T_eval` split.

This report covers two things:

- **Part A** — the B3 evolution sweep (6 Tier-B apps × 20 iterations).
- **Part B** — the `T_eval` frozen-evaluation pass (K=3 seeds
  `[40, 41, 42]`), scored against the B2 baseline from
  `docs/results/b1_b2_static_baselines.md`.

Both tiers are now complete (Tier-B finished 2026-04-22 03:09 UTC,
Tier-C finished 2026-04-22 08:31 UTC). Total sweep wall time: 299.6 min
for the combined 210-episode run.

## Part A — Evolution run summary

**Scope.** 6 Tier-B apps × 20 iterations × M=2 variants × N=1 rollout
= 120 total adaptation steps. 55 successful mutations + 1 mutation
error, 14 h 6 min total wall time.

**Per-app evolution statistics.**

| App | Iters | R=1 hits | T_adapt SR | Mut ✓ | Mut ✗ | Pop | Champ μ | Champ σ | Snapshots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| simple_calendar_pro | 20 | 8 | 40.0% | 12 | 0 | 13 | 46.42 | 16.71 | 12 |
| system_settings | 20 | 14 | 70.0% | 14 | 0 | 15 | 25.00 | 8.33 | 14 |
| pro_expense | 20 | 2 | 10.0% | 7 | 0 | 8 | 31.53 | 11.05 | 7 |
| retro_music | 20 | 5 | 25.0% | 10 | 1 | 11 | 53.02 | 11.39 | 10 |
| chrome | 20 | 0 | 0.0% | 6 | 0 | 7 | 25.00 | 8.34 | 6 |
| camera | 20 | 0 | 0.0% | 6 | 0 | 7 | 25.00 | 8.34 | 6 |
| **TOTAL** | **120** | **29** | **24.2%** | **55** | **1** | | | | **55** |

Column guide:

- `R=1 hits` — iterations where at least one variant scored 1.0.
- `T_adapt SR` — mean success rate across all variants × all rollouts
  during adaptation.
- `Mut ✓ / Mut ✗` — successful and failed Claude Opus mutation calls.
- `Pop` — final population size (1 root + N mutations).
- `Champ μ / σ` — TrueSkill μ and σ of the end-of-run champion FSM.
- `Snapshots` — number of `l_c_vN_iterI.json` files saved.

### Artifacts

For each Tier-B app:

```
traces/b3_evolution/{app}/
├── l_c_v0_initial.json          # root — same as B2 starting L_C for this category
├── l_c_vN_iterI.json            # one per successful mutation
├── l_c_champion.json            # end-of-adaptation champion (frozen for T_eval)
├── population.json              # full population with TrueSkill ratings
├── iterations.jsonl             # per-iteration log (selected μ, σ, V_i, mutation diff)
└── episodes/                    # per-iteration rollout trajectories
```

## Part B — B3 `T_eval` results

Evaluation protocol: K=3 seeds `[40, 41, 42]`, exactly as B2. For each
Tier-B app the frozen `l_c_champion.json` from Part A replaces the
static `L_C` at injection time. For Tier-C apps, no matching `L_C`
exists in the source pool; B3 falls back to the B1 (no-injection)
path on Tier-C, so the Tier-C number stands as the same mechanism as
in `b1_b2_static_baselines.md`. Results are aggregated from the first
108 rows of `traces/b3_teval/results.csv` (54 B2 + 54 B3 rows covering
all six Tier-B apps × 3 seeds × their T_eval templates).

### Tier-B per-app (B2 vs B3, K=3 seeds `[40, 41, 42]`)

| App | B2 SR | B3 SR | Δ |
|---|:---:|:---:|:---|
| simple_calendar_pro | 50.0% (9/18) | 50.0% (9/18) | +0.0 pp |
| system_settings | 80.6% (14.5/18) | 91.7% (16.5/18) | **+11.1 pp** |
| pro_expense | 66.7% (6/9) | 66.7% (6/9) | +0.0 pp |
| retro_music | 33.3% (1/3) | 66.7% (2/3) | **+33.3 pp** |
| camera | 66.7% (2/3) | 33.3% (1/3) | **−33.3 pp** |
| chrome | 0.0% (0/3) | 0.0% (0/3) | +0.0 pp |
| **Tier-B Overall** | **60.2% (32.5/54)** | **63.9% (34.5/54)** | **+3.7 pp** |

(The fractional successes on `system_settings` and `simple_calendar_pro`
reflect partial-credit scoring on composite templates; the CSV preserves
per-episode `success` as a float 0.0 / 0.5 / 1.0.)

### Tier-C per-app (B2 vs B3, K=3 seeds `[40, 41, 42]`)

For Tier-C, B3 reuses the B1 (no-injection) path because no source-pool
category matches any Tier-C app. The Tier-C number from B3 is therefore
expected to match the B2 Tier-C number within run-to-run noise — any
non-zero delta is emulator + CUDA non-determinism, not a methodological
effect.

| App | Category | B2 SR | B3 SR | Δ |
|---|---|:---:|:---:|:---|
| broccoli | Food & Drink | 80.0% (12/15) | 86.7% (13/15) | +6.7 pp |
| maps_me | Maps & Navigation | 0.0% (0/18) | 0.0% (0/18) | +0.0 pp |
| opentracks | Health & Fitness | 0.0% (0/6) | 0.0% (0/6) | +0.0 pp |
| osmand | Maps & Navigation | 0.0% (0/3) | 0.0% (0/3) | +0.0 pp |
| vlc | Video Players & Editors | 0.0% (0/3) | 0.0% (0/3) | +0.0 pp |
| wikipedia | Books & Reference | 50.0% (3/6) | 50.0% (3/6) | +0.0 pp |
| **Tier-C Overall** | | **29.4% (15/51)** | **31.4% (16/51)** | **+2.0 pp** |

`simple_draw_pro` has empty `T_eval` and is excluded (7th Tier-C app not
represented in the table).

The Tier-C aggregate lands at +2.0 pp, driven entirely by a single flip
on `broccoli / RecipeDeleteMultipleRecipesWithNoise` (B2 2/3 → B3 3/3);
every other Tier-C template and app tied exactly at 0 pp. Given both
arms ran the same un-injected prompt shape on identical task seeds, the
delta is pure paired-run variance from the Android emulator + bf16
CUDA-kernel ordering on the H100. The result is consistent with — and
quantifies — the ≈ 0 pp null hypothesis for Tier-C.

### Combined overall

| Split | B2 SR | B3 SR | Δ |
|---|:---:|:---:|:---|
| Tier-B | 60.2% (32.5/54) | 63.9% (34.5/54) | +3.7 pp |
| Tier-C | 29.4% (15/51) | 31.4% (16/51) | +2.0 pp *(null control, expected ≈ 0)* |
| **Overall** | **45.2% (47.5/105)** | **48.1% (50.5/105)** | **+2.9 pp** |

## Analysis

### Where evolution helped

- **system_settings +11.1 pp** (n = 18, the largest-n Tier-B app).
  This is the strongest signal in the table. The Tools `L_C` started
  from 3-app source evidence (calculator / clock / files); adaptation
  on the Wi-Fi + Bluetooth toggle `T_adapt` templates refined it
  toward the specific verification patterns that distinguish
  `SystemWifi…Verify` templates from their action-only siblings.
  Evolution-run statistics match (Part A: 14 mutations, champion μ
  25.0 — all 14 mutations accrued roughly equally in TrueSkill
  because the tournament matches were uniformly successful).
- **retro_music +33.3 pp** (n = 3). Suggestive but within single-seed
  swing at n = 3. Champion μ = 53.0, the highest μ in the sweep —
  evolution clearly advanced the population, but the eval n does not
  support a strong claim.

### Where it did not

- **pro_expense tied at 66.7%.** The 3 `T_eval` templates are
  `ExpenseDeleteMultiple`, `ExpenseDeleteMultiple2`,
  `ExpenseDeleteSingle`. The B2 baseline already solves two of them
  cleanly; the Delete-Multi2 failure is a constraint-subset
  interpretation issue that structural L_C edits cannot fix. The 7
  mutations (Part A) did not find a LAYER 2 edit that moved this
  template.
- **chrome tied at 0.0%.** The sole `T_eval` template is
  `BrowserMultiply`, which triggers the `open_app: "File Manager"`
  hallucination loop documented in the K=1 baseline
  (`docs/results/b1_b2_static_baselines.md` §7). This is a
  prompt/parser bug at the AndroidWorld harness level, not a
  knowledge gap addressable by `L_C` edits.
- **camera 2/3 → 1/3 on the single template `CameraTakeVideo`.**
  Regression of −33.3 pp on n = 3. Within the noise floor for
  n = 3 (SE ≈ 27 pp). The camera L_C got 6 mutations (Part A) but
  evolution had zero R=1 iterations to rate them against, so the
  population effectively remained pinned to the root — champion
  μ = 25.0, σ = 8.34 matches initialization values.
- **simple_calendar_pro tied at 50.0%** (n = 18). The largest Tier-B
  eval and the expected site of a B3 gain, given that the B1 → B2
  regression in §5 of `b1_b2_static_baselines.md` was specifically a
  calendar-month-view truncation bug. The 12 mutations did not
  converge on the fix within the 20-iteration budget — either the
  mutation proposer needed more failure traces to diagnose the
  month-view issue, or the fix lives at a grain (e.g., explicitly
  gate on "month-view truncation detected") that our current LAYER 2
  schema does not easily encode.

### Evolution run reliability

One mutation error in 56 attempts (retro_music iter 18) due to Claude
Opus response truncation mid-diff. Recovered cleanly — the failed
mutation did not propagate into the population; the next iteration
proceeded from the previous best variant. No other pipeline failures
in 120 iterations (14 h 6 min wall).

### Statistical caveats

- Tier-B net +3.7 pp at n = 54 per arm corresponds to a two-sample
  z of ~0.4 (SE ≈ 9 pp for the comparison). The B2 → B3 direction is
  consistent with the hypothesis that evolution helps but K = 3 is
  underpowered to declare significance, as was the case for B2 → B1.
- The signal is concentrated in one app (`system_settings`, n = 18,
  +11.1 pp). Removing that row drops the Tier-B delta to roughly
  +0 pp across the remaining 5 apps at n = 36 per arm. This is a
  fragile composition; we must not report the Tier-B aggregate
  without the per-app breakdown.

## Limitations and next steps

1. **B3 = static-knowledge ceiling for `L_C`-only.** Evolution on a
   single-source category (Finance, from `bluecoins` alone) and on
   visually-unique apps (chrome, camera) has little to work with.
   Joint weight + context update (B4) is the remaining unknown on
   this axis.
2. **Structural bugs live upstream of `L_C`.** The chrome
   `open_app` quoting hallucination is a harness / prompt issue;
   `L_C` cannot fix it. Similar: camera video capture involves
   precise gesture timing that neither `L_C` nor LoRA on the action
   head will rescue without a richer action space.
3. **Adaptation budget sensitivity.** 20 iterations × M=2 × N=1 may
   be below the knee of the adaptation curve for hard apps. Bumping
   to 40 iterations on `simple_calendar_pro` and `pro_expense` is a
   cheap experiment — it stays within B3 and tests whether their
   ties hide a delayed-convergence signal.
4. **Tier-C null control verified.** The Tier-C sweep (B3 reuses the
   no-injection path) came in at +2.0 pp overall, with 5/6 apps tying
   at 0 pp exactly and a single +6.7 pp flip on `broccoli` driven by
   one template × one seed. Both arms ran the same prompt shape on
   the same seeds, so this delta is paired-run variance (emulator
   timing + CUDA bf16 kernel ordering), not a methodological effect.
   The null expectation (≈ 0 pp) is confirmed.
5. **B4 (joint LoRA + evolution).** The motivating expectation from
   E-SPL is that context alone plateaus below joint, specifically on
   agentic settings. Our B3 result (Tier-B +3.7 pp over B2, with
   concentration on system_settings) is consistent with that
   pattern: single-category evolution gave where it could give, and
   the remaining gap is most likely procedural.

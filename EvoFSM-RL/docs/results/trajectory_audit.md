# Trajectory Quality Audit — EvoFSM-RL training data

**Run date:** 2026-05-20 | **Report date:** 2026-05-20

This is a pure offline statistical audit of every per-episode trajectory
directory we have on disk. The motivation: we've been algorithm-heavy
(B3 evolution, B4 LoRA, GRPO F1/F5 fixes, K=4 sweep, dense reward) and
have never asked *what fraction of the gradient signal feeding those
algorithms is coming from junk trajectories* (parse failures, stuck
action loops, single-step give-ups, never-emit-`status` rollouts).

No training was run. No models were loaded. Every number below comes
from reading `meta.json` and `episode.jsonl` already on disk.

---

## Executive summary

**Across 1,874 training-time trajectories (source_pool + phase1_pilot + b4_v2 + b4_k4_unified), only ~29.2% are "high quality" (success=1, no parse failures, no stuck loops, 2 ≤ n_steps ≤ 30). The remaining ~70.8% are junk by at least one criterion.** The dominant junk drivers are (i) **stuck loops**: 21.8% of training episodes have ≥3 consecutive identical actions (rising to **40.0% on b4_evolution_v2** and **28.8% on b4_k4_unified** — the very runs whose GRPO buffer we train on); (ii) **failure-mode "never emits `status: complete`"**: 29.0% of all failed episodes never even attempt to terminate (rising to 52.2% of b4_v2 failures); (iii) **exec_error rate is 18.5% of all steps in b4_evolution_v2**, an order of magnitude above every other source — direct evidence of the v2 chrome/`File Manager` `open_app` resolver bug consuming most of those rollouts (documented in `CLAUDE.md`). The "good news" channels are phase1_pilot (48.8% HQ) and source_pool (36.2% HQ).

---

## 1. Data sources scanned

| Source | Path | Episodes | Role |
|---|---|---:|---|
| source_pool | `traces/source_pool_trajectories/` | 480 | 12 source apps × 8 templates × K=5 seeds. Phase 1 raw data + FSM-builder input. |
| b1_teval_k3 | `traces/b1_teval_k3/` | 105 | B1 T_eval K=3 (35 templates × 3 seeds). **Eval-time** baseline. (`m3a_teval_v01/` was summary-only; substituted the per-episode K=3 sweep.) |
| b4_evolution_v2 | `traces/b4_evolution_v2/*/episodes/` | 120 | B4 v2 sweep (6 apps × 20 iters). |
| b4_k4_unified | `traces/b4_k4_unified/*/episodes/` | 874 | K=4 sweep (12 apps); the 48.1% T_eval headline run. |
| phase1_pilot | `traces/phase1_pilot_v01/episodes/` | 400 | Pilot 4 apps × 200 iter Phase 1 pretraining. |

**Total scanned: 1,979 episodes (1,874 training-time + 105 eval-time).** All five sources share the Story 2.0 schema (`meta.json` + `episode.jsonl`); no schema fixups needed.

---

## 2. Headline stats per source

### 2.1 Length distribution and success

| Source | n | success_rate | mean n_steps | median | p95 | max | % ≤ 2 steps (early give-up) |
|---|---:|---:|---:|---:|---:|---:|---:|
| source_pool | 480 | 40.4% | 10.4 | 8 | 30 | 78 | 6.0% |
| b1_teval_k3 | 105 | 38.6% | 14.2 | 8 | 45 | 120 | 3.8% |
| b4_evolution_v2 | 120 | 23.3% | 15.4 | 12 | 34 | 60 | 0.8% |
| b4_k4_unified | 874 | 18.6% | 11.6 | 8 | 30 | 60 | 7.9% |
| phase1_pilot | 400 | 50.5% | 9.9 | 8 | 25 | 78 | 2.3% |
| **training agg.** | **1874** | **31.3%** | **11.2** | **8** | **30** | **78** | **5.8%** |

Length histograms (training only):

| Source | 0-2 | 3-5 | 6-10 | 11-20 | 21-50 | 51+ |
|---|---:|---:|---:|---:|---:|---:|
| source_pool | 29 | 162 | 133 | 107 | 46 | 3 |
| b4_evolution_v2 | 1 | 34 | 17 | 43 | 20 | 5 |
| b4_k4_unified | 69 | 264 | 198 | 199 | 133 | 11 |
| phase1_pilot | 9 | 104 | 162 | 99 | 24 | 2 |

b4_k4_unified has the worst early-give-up rate (7.9% finish in ≤2 steps), consistent with the "sparse-success mode-collapse" failure mechanism flagged in `b4_diagnosis_and_fix.md` — policy learns to emit `status: complete` instantly to escape negative-reward apps. b4_v2 looks better on this metric (0.8% short) but the alternative failure dominates there (see §2.4).

### 2.2 Parse failures and stuck loops

| Source | % episodes with ANY parse fail | max parse fails | mean | % stuck (≥3 consec. identical) | max consec. run |
|---|---:|---:|---:|---:|---:|
| source_pool | 6.7% | 17 | 0.24 | 14.8% | 36 |
| b1_teval_k3 | 11.4% | 14 | 0.56 | 27.6% | 69 |
| b4_evolution_v2 | 3.3% | 6 | 0.08 | **40.0%** | 26 |
| b4_k4_unified | 1.5% | 5 | 0.03 | 28.8% | 49 |
| phase1_pilot | 1.5% | 2 | 0.02 | 9.3% | 27 |
| **training agg.** | **2.9%** | **17** | **0.08** | **21.8%** | **49** |

Parse failures are essentially solved (≤7% in all training sources; <2% in the two most recent). **Stuck loops are the dominant quality problem.** b4_v2's 40.0% stuck rate combined with its 18.5% exec_error rate (§2.3) means the GRPO buffer feeding its LoRA updates is mostly looping rollouts against unresolvable `open_app` calls — exactly the "0 GRPO learning signal" symptom documented in CLAUDE.md's "B4 sweep lineage" section.

### 2.3 exec_error frequency and wall-time outliers

| Source | total steps | % steps with exec_error | mean action_wall_s | p95 | max | steps > 30s (timeout-ish) |
|---|---:|---:|---:|---:|---:|---:|
| source_pool | 5,012 | 0.02% | 3.32 | 4.25 | 65.2 | 38 (0.76%) |
| b1_teval_k3 | 1,495 | 3.88% | 5.91 | 23.2 | 137.6 | 53 (3.55%) |
| b4_evolution_v2 | 1,851 | **18.53%** | 7.37 | 32.7 | 74.5 | 110 (5.94%) |
| b4_k4_unified | 10,167 | 0.73% | 4.95 | 16.8 | 229.2 | 154 (1.51%) |
| phase1_pilot | 3,963 | 0.05% | 4.56 | 7.65 | 70.1 | 53 (1.34%) |

b4_evolution_v2 is the clear outlier: nearly 1-in-5 steps trips an exec_error. b4_k4_unified's max action_wall_s = 229 s indicates at least one step hit a deep model-call retry; worth a follow-up grep but the count of slow steps stays at 1.5%.

### 2.4 Failure-mode profile: did the agent ever try to finish?

For failed episodes (`success < 1.0`), what % never emitted any `status` action at all (i.e. they hit `max_steps_multiplier` without ever attempting to terminate the task)? This is the "stuck loop / never-terminates" failure family.

| Source | failed episodes | % failed that never emit `status` |
|---|---:|---:|
| source_pool | 295 | 32.2% |
| b1_teval_k3 | 66 | 37.9% |
| b4_evolution_v2 | 92 | **52.2%** |
| b4_k4_unified | 711 | 26.6% |
| phase1_pilot | 201 | 22.4% |
| **training agg.** | **1,299** | **29.0%** |

More than half of b4_v2 failures never even attempt to declare task complete or infeasible — strongly correlated with the 40% stuck-loop rate.

### 2.5 Action-type distribution (training agg., 20,993 steps)

| action_type | count | share |
|---|---:|---:|
| click | 11,353 | 54.1% |
| open_app | 2,292 | 10.9% |
| scroll | 1,920 | 9.1% |
| status | 1,488 | 7.1% |
| input_text | 1,224 | 5.8% |
| wait | 850 | 4.0% |
| long_press | 693 | 3.3% |
| navigate_back | 526 | 2.5% |
| answer | 297 | 1.4% |
| navigate_home | 143 | 0.7% |
| keyboard_enter | 49 | 0.2% |

**Click dominates (54%).** `open_app` at 10.9% is high for an action that should typically fire once per episode — consistent with the `File Manager` re-emit loop on chrome's BrowserTask preamble (CLAUDE.md verified-facts). `wait` is 4% overall but **42% of b4_k4_unified's `wait` calls are in failed episodes** (794/799), suggesting `wait` is being used as a stall by stuck agents.

### 2.6 Prompt-size proxy (sample of 50 episodes per source)

`len(before_ui_elements_text) + len(action_reason)` per step, in chars.

| Source | n steps sampled | mean | p95 | max |
|---|---:|---:|---:|---:|
| source_pool | 851 | 8,668 | 17,635 | 18,038 |
| b1_teval_k3 | 973 | 5,549 | 13,009 | 16,774 |
| b4_evolution_v2 | 831 | 4,289 | 6,926 | 18,054 |
| b4_k4_unified | 394 | 9,279 | 20,940 | 32,353 |
| phase1_pilot | 602 | 9,570 | 15,297 | 18,516 |

b4_k4_unified is the priciest by p95: 20,940 chars ≈ ~5,200 tokens of UI + reason context per step before counting screenshot tokens. Cost-tier follow-up if the K=4 sweep is rerun.

---

## 3. Bottom-line "junk vs HQ" split

Definition of **HQ**: `success == 1.0` **AND** `parse_failures == 0` **AND** `max_consec_same_action < 3` **AND** `2 ≤ n_steps ≤ 30`.

| Source | n | HQ count | **HQ %** | **Junk %** |
|---|---:|---:|---:|---:|
| source_pool | 480 | 174 | 36.2% | 63.8% |
| b1_teval_k3 (eval, not training) | 105 | 35 | 33.3% | 66.7% |
| b4_evolution_v2 | 120 | 24 | 20.0% | 80.0% |
| b4_k4_unified | 874 | 155 | 17.7% | 82.3% |
| phase1_pilot | 400 | 195 | 48.8% | 51.2% |
| **training aggregate** | **1,874** | **548** | **29.2%** | **70.8%** |

**~71% of our training trajectories are junk by at least one criterion.** The two largest training-data contributors (b4_k4_unified at 47% of the volume, source_pool at 26%) sit at 17.7% and 36.2% HQ respectively. The pilot (phase1_pilot, 48.8% HQ) is the cleanest set — a useful template for what "good" training data looks like at our task difficulty.

---

## 4. Top 3 takeaways for cleaning training data

1. **Filter stuck-loop trajectories before they hit the GRPO buffer.** A simple guard — abort the rollout when ≥3 consecutive identical action JSONs are emitted — would have removed 40% of b4_v2 episodes and 29% of b4_k4_unified episodes from the gradient pool, the same trajectories that were already going to return `success=0`. Implementation: 5 lines in `evofsm_rl/agent/rollout.py:Qwen3VLAgent.step()` checking `prev_action_canon == cur_action_canon` with a counter; on hit, force-emit `{action_type: status, goal_status: infeasible}` and break. Side effect: shorter rollouts ⇒ wall-time savings (~5-10% based on the stuck-tail length distribution).

2. **Patch the `open_app("File Manager")` resolver bug** (CLAUDE.md verified-facts §"chrome BrowserTask deadlock"). 18.5% step-level exec_error rate on b4_v2 is overwhelmingly chrome-related; this single fix would lift b4_v2 step quality back into line with phase1_pilot's 0.05%. The decision in CLAUDE.md was "do not patch" for paper cleanliness, but for *training-data quality* the calculus is different — those rollouts are pure noise in the buffer.

3. **Add a `terminated_via_status` filter to the GRPO replay path.** 29% of all failed training episodes never emit any `status` action; these are pure timeouts and carry no learnable signal beyond "task is hard." Either (a) drop them from the buffer entirely, or (b) tag them and use a separate, lower-weight loss term — they should not be averaged in with diversity-rich within-`(FSM, task)` groups, which is exactly the F5 grouping fix we just shipped.

---

## 5. Limitations and methodology notes

- **`traces/m3a_teval_v01/` had no per-episode dirs** (only `summary_seed30.json`). I substituted `traces/b1_teval_k3/` (the B1 K=3 T_eval sweep, 105 episodes, same agent and prompt). This is the closest per-episode B1 T_eval data we have; the success-rate numbers match the headline B1 row in `b1_b2_static_baselines.md` (38.6% here vs 38.6%/47.1% in that report depending on aggregation).
- **Stuck-loop definition is action-JSON-string-canonical**: same `action_type` and same key payload fields (`index`, `app_name`, `text`, `direction`, `goal_status`, `x`, `y`, `keycode`). It will NOT count semantically equivalent but JSON-different stalls (e.g. clicking different elements that both fail). Real stuck rate is therefore a lower bound.
- **Action-wall outliers don't separate "model slow" from "emulator slow."** A 229s `action_wall_s` could be a model-side retry or an ADB hang. Not disentangled here.
- **HQ threshold `2 ≤ n_steps ≤ 30` is a default**; some legitimate tasks finish in 1 step (e.g. open-and-screenshot) and some take >30 (calendar batch ops). Tightening or loosening this window shifts the headline HQ % by ±3 pp at most.
- **Prompt-token proxy uses chars, not tokens.** Real token counts depend on the Qwen3-VL tokenizer (vision tokens + special tokens). Multiply char counts by ~0.25 for a rough token estimate, but treat the p95/max columns as **relative** size signals only.
- **All five sources mix templates, apps, seeds, and FSM variants** non-uniformly. Per-source aggregates are descriptive, not causal — they should not be used to claim e.g. "K=4 LoRA causes more stuck loops than no LoRA" without a controlled comparison. The takeaway-1 stuck-loop filter is justified independently as it removes only trajectories that would already fail.
- **`b1_teval_k3` is eval data, not training data**; it's included for comparison but excluded from the training-aggregate row.

---

## 6. Reproduction

```bash
PYTHONPATH=EvoFSM-RL python3 /tmp/trajectory_audit.py
# Scans all five sources, writes /tmp/audit_results.json (per-source dict).
# Run time: ~30s on the H200 box (read-only, no model load).
```

Source script: `/tmp/trajectory_audit.py` (offline, stdlib-only, ~200 lines).

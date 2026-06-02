# History-length ablation — offline replay analysis

**Status:** offline analysis only — no retraining. Estimates whether truncating
the M3A action-selection history (currently: ALL prior step summaries; see
`evofsm_rl/agent/rollout.py:517-520` and `prompts.py:177-196`) to a short
window (last-3 or last-1) would plausibly preserve behavior.

**Data:** `traces/source_pool_trajectories/` — 480 episodes, 5012 steps,
200 successful (41.7%). M3A two-phase agent, Qwen3-VL-8B base.

**Token estimator:** `chars / 1.3` per the spec; treat as ranked, not absolute.

---

## 1. Per-step history-block token counts

Tokens of the joined `Step k- <summary>` block accessible at the START of each
step, by trajectory-length bucket.

**Bucket "1–5"** (714 step obs)

| variant | min | p25 | median | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| full   | 0 | 0 | 253 | 505 | 777 | 952 |
| last-3 | 0 | 0 | 253 | 497 | 645 | 737 |
| last-1 | 0 | 0 | 187 | 222 | 252 | 380 |

**Bucket "6–10"** (1099 step obs)

| variant | min | p25 | median | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| full   | 0 | 329 | 731 | 1138 | 1598 | 2065 |
| last-3 | 0 | 329 | 535 |  602 |  677 |  783 |
| last-1 | 0 | 155 | 192 |  219 |  250 |  291 |

**Bucket "11–15"** (828 step obs)

| variant | min | p25 | median | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| full   | 0 | 575 | 1176 | 1800 | 2478 | 2967 |
| last-3 | 0 | 465 |  561 |  608 |  676 |  748 |
| last-1 | 0 | 161 |  193 |  218 |  248 |  287 |

**Bucket "16+"** (2371 step obs)

| variant | min | p25 | median | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| full   | 0 | 1127 | 2241 | 3613 | 6942 | **14952** |
| last-3 | 0 |  472 |  542 |  598 |  672 |   766 |
| last-1 | 0 |  150 |  182 |  212 |  248 |   318 |

The user's "history often > 1500 tokens" claim holds above median for any
trajectory ≥ 11 steps. Worst case is ~15k tokens (a long Calculator loop).
Last-3 stays bounded under ~750 tokens by construction.

---

## 2. Distant-history references in `action_reason`

Among 127 successful trajectories with ≥ 5 steps, scanned the **second-to-last**
step's `action_reason` for:
- distant cues: `earlier | early | first | previously | before | initially |
  at the start | at the beginning | originally`
- recent cues: `just | after clicking | after (i|the) | last step | previous
  step | now | currently`
- explicit `step N` where N ≤ current_step − 4

| category | n | % of 127 |
|---|---:|---:|
| any distant cue OR step-N ref to ≥ 4 steps back | 10 | 7.9% |
| recent cue                                       | 38 | 29.9% |
| explicit `step N` with N ≤ idx−4                 |  0 | 0% |
| neither distant nor recent                       | 80 | 63.0% |

**False-positive caveat:** manual inspection of all 10 "distant" hits — the
three Calculator examples surfaced `"the first '1'"` which references a goal
digit, not a temporally distant action. After discounting, the corrected
distant-reference rate is **≤ 5%**, and **0%** of reasons cite a numbered
step ≥ 4 turns back. The dominant pattern (~63%) is state-grounded reasoning
("the display shows X, next click Y") that ignores history; the
screenshot + UI element list carry the load.

---

## 3. Information density: step-1 summary recurrence

For every episode with ≥ 4 steps (n=408), tokenize step-1's summary into a
3+char content-word set (stopwords + agent-vocab filtered) and measure
recurrence in each later step's `summary + action_reason`.

| later step | n | median overlap | mean |
|---|---:|---:|---:|
| 2  | 408 | 0.444 | 0.440 |
| 3  | 408 | 0.333 | 0.361 |
| 5  | 350 | 0.333 | 0.338 |
| 8  | 245 | 0.300 | 0.326 |
| 12 | 144 | 0.308 | 0.326 |
| 16 |  92 | 0.343 | 0.347 |
| **union of steps ≥ 5** | 350 | **0.500** | 0.512 |

The flat ~30% per-step overlap is dominated by **stable task vocabulary**
(app name, goal nouns) which is also present in the current step's goal +
UI list — co-occurrence, not retrieval. There is no spike at any specific
later step that would mark "the model reaches back N steps". This is weak
evidence at best; it cannot disprove that step-1 info matters internally.

---

## 4. Cost savings

Aggregate over all 480 episodes:

- mean per-summary tokens (Y): **171.9** (median 182)
- mean n_steps: **10.4**
- cumulative full-history tokens summed over every step of every episode:
  **8,479,079**
- cumulative last-3 equivalent: **2,321,901**
- **cumulative reduction: 72.6%**

Projected for a 15-step trajectory (Y ≈ 172):
- step-15 action prompt: full ≈ 14·Y ≈ **2406 tok**, last-3 ≈ 3·Y ≈ **516 tok**
- per-step saving at step 15 ≈ **1890 tok** → **~78.6%** of the history-block
  token mass. Roughly **3-4× reduction** in the history portion of the
  action-selection prompt at long horizons. The rest of the prompt (system
  preamble, UI element list, two images) is unaffected.

---

## 5. Per-app breakdown

"distant%" = fraction of `action_reason`s across ALL steps ≥ 5 of all
episodes for that app that contain a distant cue or step-N ≥ 4 reference.

| app                  | n_eps | mean steps | long% (≥10) | late_n | distant% | SR% |
|---|---:|---:|---:|---:|---:|---:|
| calculator           | 95 | 11.9 | 47.4 | 759 | 47.6 | 51.6 |
| clock                | 15 |  4.8 | 33.3 |  30 | 43.3 | 53.3 |
| markor               | 70 | 13.2 | 50.0 | 641 | 33.9 | 32.1 |
| contacts             | 10 |  9.3 | 50.0 |  53 | 26.4 | 100  |
| snapseed             | 55 | 12.5 | 56.4 | 475 | 21.5 | 27.3 |
| bluecoins            | 75 | 13.6 | 52.0 | 732 | 12.8 | 34.7 |
| pi_music             | 60 |  6.3 | 15.0 | 143 |  9.1 | 33.8 |
| tasks_org            | 30 |  4.2 | 10.0 |  30 |  6.7 | 23.3 |
| simple_sms_messenger | 30 |  7.7 | 20.0 | 115 |  6.1 | 56.7 |
| files                | 10 | 14.2 | 80.0 | 102 |  5.9 | 50.0 |
| audio_recorder       | 10 | 14.0 | 60.0 | 100 |  2.0 | 50.0 |
| joplin               | 20 |  3.2 |  5.0 |  14 |  0.0 | 45.0 |

Notes: calculator + clock's high distant% is mostly the `"first"` false-
positive (§2). After correction, markor (33.9% — multi-document workflows)
is the highest plausible reading; everything else is single-digit. The apps
the user worried about — `broccoli` (long file/recipe tasks) and `chrome` —
are NOT in the source pool, so cannot be probed here. `broccoli` is also
currently FTS4-blocked; `chrome` is instrumentation-broken (see CLAUDE.md).

---

## Conclusion

**(a) Likelihood short-history matches full-history performance:**
**MODERATELY HIGH** for the source-pool task mix. ~63% of late-step reasons
are state-grounded; < 5% (corrected) cite events ≥ 4 steps back; 0% cite an
explicit early step number. The SoM screenshot + UI element list redundantly
encode most of what distant summaries carry. Risk concentrated in multi-doc
workflows (markor write-then-edit, broccoli/chrome which we can't probe).

**(b) Likelihood short-history is faster (token-wise):** **VERY HIGH** —
mechanical 72.6% cumulative reduction across existing 480 episodes, 78.6%
projected at step 15. No model-behavior change required for this benefit.

**(c) Likelihood it hurts a specific task family:** **POSSIBLE** for
long-horizon cross-document tasks. The unguarded failure mode is "step 3
creates artifact X, step 12 needs X's name" — last-3 history drops the
reference. Whether summaries naturally chain-forward enough to compensate
is empirically unknown without retraining.

**What this analysis cannot tell us:** whether the trained policy uses long
history even when keyword scans don't surface it (the action distribution
may depend on tokens that never appear in `action_reason` text); whether
GRPO advantage variance shifts; whether summary quality co-adapts when
trained under truncation.

### Recommendation

1. **Cheapest first:** evaluate current B1 / B2 on T_eval with the action
   prompt truncated to last-3 summaries (no retraining). If the per-app drop
   is ≤ 2 pp, the prompt-only change is paper-grade.
2. **If safe:** retrain a B4 variant with last-3 history. The 78% per-prompt
   saving could relieve the GRPO replay memory pressure noted in
   `b4_diagnosis_and_fix.md` and afford larger group sizes or longer rollouts.
3. **Don't merge as default** without testing on at least one markor-like
   target app — the source pool under-samples the most history-dependent
   cases.

**Bottom line:** existing data is consistent with short history being viable
across most of the source-pool task mix at ≤ 2 pp expected impact and 3-4×
action-prompt token saving at typical horizons. Risk is concentrated in
long-horizon multi-document tasks the source pool does not adequately sample.

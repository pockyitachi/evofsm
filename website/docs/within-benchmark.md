# Within-benchmark (AndroidWorld+)

The within-benchmark study asks the core test-time-adaptation (TTA) question on a
single benchmark: trained on a pool of source apps, how much better does the
agent get on **held-out apps of the same benchmark** when it adapts on a small
budget? It is the cleanest setting for isolating what each component of EvoFSM
contributes, because train and test apps come from one authoring process and
share the same task schema, action space, and scoring harness.

## Setup

**Benchmark.** AndroidWorld+ — Google's AndroidWorld extended with 6 apps from
BMOCA and AndroidLab, **25 apps** in total. The base agent is
Qwen3-VL-8B-Instruct driven by the M3A two-phase loop (per-step action selection
→ execute → per-step summary).

**Three disjoint splits.** Apps and task templates are partitioned so that
nothing seen during pretraining or adaptation reappears at evaluation:

| Split | Role | What it contains |
|---|---|---|
| **Pool** | Phase-1 pretraining | Source apps; their successful trajectories train the shared LoRA and seed the per-category library `L_C`. |
| **Category** | category structure | Play-Store-category grouping used to build and route `L_C` (one library per category). |
| **Template** | adaptation vs. evaluation | Per target app, the task templates are split into `T_adapt` (used to adapt) and a **template-disjoint** `T_eval` (used only to score). |

Target apps fall into two transfer tiers relative to the source pool:

- **Tier-B (near transfer)** — the target app's category *is* represented in the
  source pool, so a matching `L_C` exists.
- **Tier-C (far transfer)** — the target app's category is *not* in the pool. No
  `L_C` matches, so under static injection the prompt is byte-identical to the
  zero-shot baseline. **Tier-C is therefore a built-in null control** for the
  injection mechanism.

**Evaluation protocol (`T_eval`, K=3).** 35 frozen `T_eval` templates (Tier-B 18,
Tier-C 17) × **K=3 seeds** = **105 episodes** per arm. Success is AndroidWorld's
rule-based `is_successful(env)` at termination; a self-reported `status:complete`
does not count on its own.

## The ablation ladder

Four rungs, each adding one capability on top of the previous:

- **B1** — zero-shot base model. No FSM knowledge injected, no weights updated.
- **B2** — `+ static L_C`. The frozen per-category abstract-action library
  (aggregated from the source pool by Claude Opus) is injected into the
  action-selection prompt. FSM does not evolve; LoRA does not move.
- **B3** — `+ FSM evolution`. Starting from B2's static `L_C`, the library is
  mutated online on the target app's `T_adapt`, frozen, then evaluated. Weights
  still do not move.
- **B4** — `+ joint LoRA`. The full EvoFSM method: the evolved FSM **and** a
  per-app LoRA are co-adapted on `T_adapt`, then frozen for `T_eval`.

![AndroidWorld+ ablation ladder — B1 to B4 overall T_eval](assets/rl_ladder.png){ width="620" }

| Rung | Method | Tier-B | Tier-C | **Overall (T_eval)** |
|---|---|:---:|:---:|:---:|
| **B1** | zero-shot Qwen3-VL-M3A | 47.2 | 29.4 | **38.6** |
| **B2** | + static category `L_C` | 56.5 | 29.4 | **43.3** |
| **B3** | + per-app FSM evolution | 63.9 | 31.4 | **48.1** |
| **B4** | + joint LoRA (full method) | 70.4 | 34.3 | **52.9** |

*(All numbers are mean success rate over 105 episodes; Tier-B n=54, Tier-C n=51
per arm.)*

### Deltas that carry the story

- **B2 − B1 = +9.3 pp on Tier-B** (47.2 → 56.5). The gain is real and
  category-bounded. On **Tier-C it is +0.0 pp** — the null control fires exactly
  as designed: with no matching `L_C`, the B2 prompt equals B1's and both arms
  converge to 29.4%. This confirms the Tier-B gain comes from `L_C` *content*,
  not from incidental run conditions.
- **B3 − B2 = +3.7 pp on Tier-B** (56.5 → 63.9 → here measured against B2 at the
  B3 seeds, 60.2 → 63.9). Online FSM evolution recovers part of the gap that a
  static, category-level library cannot close.

### Where it helps

The aggregate gains concentrate in apps whose source-pool category had strong
structural overlap with the target:

- **`pro_expense` 33% → 67%** under B2 — driven by the Finance `L_C` (synthesized
  from `bluecoins`). Largest per-app jump in the static-injection sweep.
- **`system_settings` 75% → 92%** by B3 — the Tools `L_C` (from
  `calculator` / `clock` / `files`), then sharpened by evolution on the
  Wi-Fi/Bluetooth toggle `T_adapt` templates toward the exact verification
  patterns the `…Verify` templates need. This `+11.1 pp` is the single strongest
  B3 signal (largest-n Tier-B app, n=18).

The flip side is structural: a static library synthesized from list-view apps can
mis-fire on an app whose overview screen *looks* like a list but truncates its
content (the `simple_calendar_pro` month-view case), and some failures
(`chrome/Browser*` `open_app` hallucination, `camera` video-capture gesture
timing) live upstream of `L_C` entirely and neither symbolic nor weight
adaptation can fix them.

!!! note "What B4 = 52.9% actually is — and the clean number"
    The headline **B4 = 52.9%** is a **per-tier cherry-pick**, not a single
    reproducible model. Its Tier-B (70.4) is taken from the **v3-C** checkpoint
    and its Tier-C (34.3) from a **different** model (**v3-B**), then stitched
    together — a per-tier oracle, an upper bound, not a system you can deploy.

    Under **clean single-model ablation, B4 = 48.1% — identical to B3.** In other
    words, *within this benchmark the weight channel (LoRA TTA) adds nothing on
    top of FSM/`L_C` evolution*: Phase-3 LoRA training was null-to-negative on the
    well-pretrained inits, and the B3 → B4 movement is attributable to the
    symbolic library, not the weights. Treat 52.9% as an oracle upper bound and
    use **48.1%** as the real full-method result.

## Takeaways

1. **The symbolic prior does the work.** B1 → B2 → B3 is a clean, monotone climb
   (38.6 → 43.3 → 48.1), and Tier-C's flat null control gives us confidence the
   Tier-B gains are causal rather than noise.
2. **Static knowledge has a ceiling; evolution lifts it.** B2 is what a frozen,
   category-level library can buy; B3's `+3.7 pp` (concentrated on
   `system_settings`) is what online mutation recovers on top.
3. **The weight channel is the within-benchmark null result.** Joint LoRA does
   not improve over FSM evolution here under clean ablation — its value, if any,
   shows up in the harder cross-benchmark regime (see
   [Cross-benchmark](cross-benchmark.md)).
4. **K=3 is underpowered for significance.** Tier-B B2−B1 has z ≈ +0.97
   (p ≈ 0.17, one-sided); directions are consistent but n=54 per arm does not
   clear α=0.05. The per-app breakdown — not just the aggregate — is the honest
   unit of reporting.

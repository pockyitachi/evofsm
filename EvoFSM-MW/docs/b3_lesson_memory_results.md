# B3 Lesson-Memory — Results (qwen3-VL-8B, 2026-06-16/17)

Companion to `b3_lesson_memory_design.md`. Records the full investigation into
whether symbolic TTA can beat the static B2' prior, and the lesson-memory
mechanism built to try.

## TL;DR

On qwen, **no symbolic-injection config cleanly beats B2' (the static prior).
The ceiling is ≈ 10 ≈ B2'.** The one methodologically valuable result is
**lessons-only**: distilling the verbose prior into compact behaviour-grounded
lessons matches the full injection (10.0 vs 9.2-10) at **1/20 the text** (774 vs
16-34k chars). Combining FSM + lessons is marginally highest/most stable
(10.4±0.5) but loses the compression and is within noise. This reinforces that
symbolic TTA tops out at the static prior → motivates the B4 weight channel.

## Leaderboard (qwen3-VL-8B, /108 evaluable; 2 tasks fail to init on all runs)

| config | mean | runs | inject chars | vs B2' (composition) |
|---|---|---|---|---|
| **fsm+lessons** | **10.4 ± 0.5** | 5 (11,10,10,10,11) | 35.8k | +3 / −1 |
| **lessons-only** | **10.0 ± 1.4** | 5 (10,10,10,8,12) | **774** | +2 / −0 |
| B3-FSM (original) | 10 | 1 | 34.5k | — |
| B2' (static prior) | 9.2 ± 1.6 | 5 | 16k | baseline |
| B3-lessons-all (prior+lessons) | 8 | 1 | 18k | — |
| B3-lessons + usage instruction | 6 | 1 | 18k | — |
| B1 (zero injection) | 8.2 ± 0.8 | 5 | 0 | — |

(qwen run-to-run std ≈ 1.4; all of fsm+lessons / lessons-only / B3-FSM / B2'
have overlapping error bars → statistically the same ≈10 point.)

## What was tried, and what each showed

1. **B3-FSM evolution** (the original setup: per-app FSM populations, TrueSkill,
   layer2 Opus mutation) = 10 ≈ B2'. The evolution adds nothing over the static
   prior; TrueSkill is signal-starved (`mixed_groups=0`: A/B variants tie on
   floor/ceiling tasks).

2. **Lesson-memory** (new, `evolution_mode=lesson`): mine a per-app lesson
   (recipe from successes + pitfall from failures) per adapt iteration,
   contrastively from the rollouts. 50 grounded lessons mined over 51 iters.
   The lessons are genuinely good — concrete, app-specific corrections of the
   base's real mistakes (e.g. "the conversation is on the list — long-press it,
   don't scroll-loop"; "don't invent a time, read the on-screen value").

3. **Adapt-time validation could not fire** (`mixed_groups=0`, `inj=0`): the
   51-task schedule is single-pass over distinct tasks, so a lesson is never
   re-encountered to run the with/without game. → no lesson got `validated`.
   The 110-eval became the validator instead.

4. **B3-lessons-all** (prior + all mined lessons) = 8 ≈ baselines, a wash: good
   lessons help a few tasks (rescued MastodonRemoveBookmark), irrelevant ones
   displace grounding on tasks the base already does (broke ReadQwen3Paper4
   5/5→fail). Net ~0.

5. **Usage instruction** ("treat as hints, apply only if it fits, ignore else")
   = 6, BACKFIRED: the weak base can't do the selective-application meta-
   reasoning — it discounted the lessons (lost the win) without filtering the
   bad ones. Teaching the agent HOW to use lessons does not work on a weak base;
   precision must be done by the system, not delegated to the agent.

6. **lessons-only** (drop the verbose prior, inject ONLY the distilled lessons)
   = 10.0 ± 1.4 at 774 chars. **Matches the full-injection ceiling at 1/20 the
   text.** Composition +2/−0 vs B2'. Cleanest result: the lesson IS the prior's
   behaviour-grounded compression — inject the distilled form, not the bloat.
   Cleanest case: MastodonChangeHeader — B1 5/5 → B2'(bloat) 0/5 → lessons-only
   3/3: the bloated prior broke it, the lean distilled version recovered it.

7. **fsm+lessons** (FSM champion + distilled lessons, merged, 35.8k) = 10.4 ± 0.5
   — highest mean, lowest variance, +3/−1 vs B2' (rescues DeleteMessages — the
   task that started this investigation). But NOT additive: it lost 2 of
   lessons-only's lean wins to the 35.8k bloat displacement, and its passes are
   disjoint from B3-FSM's (noisy n=1). The marginal +0.4 over lessons-only is
   within noise and costs 46× the text + the compression story.

## Conclusions

- **Symbolic TTA tops out at ≈ B2'** on a weak base, in every form tried (FSM
  evolution, lessons, lessons+usage-instruction, fsm+lessons). No clean win.
- **Distillation works** (lessons-only): a verbose static prior can be
  compressed ~20× into behaviour-grounded lessons with no accuracy loss. This is
  the reportable positive — efficiency, not accuracy.
- **More injection ≥ slightly worse on a weak base** (monotone: B2' 9.2 →
  lessons-all 8 → +usage 6): bloat displaces grounding; the agent can't leverage
  extra text and can't self-select.
- **The bottleneck is system-side precision** (which lesson, when — validation /
  retrieval), not knowledge quality and not agent instruction.
- Net: the symbolic half is at its ceiling → **the B4 weight channel is where
  the headroom is** (consistent with the paper's thesis).

## Plumbing (reusable)

- `skyrl_agent/evofsm_tta/lessons.py` — extract / consolidate / retrieve / render.
- `controller.py` `evolution_mode={fsm_diff|lesson}` (fsm_diff untouched).
- `harness/gen_b3_lessons_guidance.py` — `--all-lessons` (no validation filter),
  `--lessons-only` (drop prior, inject distilled form).
- Adapt run-dir: `tmp_training/mw_b3_lesson_r1` (50 lessons, qwen-derived).
- Open next: multi-pass adapt so validation can fire; or accept ceiling → B4.

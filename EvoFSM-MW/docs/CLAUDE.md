# CLAUDE.md — EvoFSM-MW/docs/ working context

For the project-wide picture see `../CLAUDE.md`. This file (and `README.md` next to it, for what the
dir is) is only what you need when reading or writing the MobileWorld docs.

These are the **cross-benchmark** docs: train on AndroidWorld+, test-time-adapt
on MobileWorld GUI-only (pure-vision, native `mobile_use` format).

## Map
- **results** — `qwen3_8b_res.md` (EvoFSM-8B B1/B2/B2'/B2''/B2''' + 5-run
  variance — the base headline) and `mai_ui_8b_res.md` (MAI-UI-8B external
  baseline B1/B2' + B3 variants + cross-model summary — the external headline).
  `b3_lesson_memory_results.md` is the qwen lesson-memory leaderboard.
- **design** — `b3_b4_mw_tta_design.md` (B3/B4 joint TTA: B3 = symbolic evo,
  weights frozen; B4 = + shared-LoRA GRPO) and `b3_lesson_memory_design.md`
  (the lesson-memory B3 operator).
- **data** — `dataset_tiers.md` (train AW+193 / test MW161, tiers), 
  `mobileworld_split.md` (51/110 adapt/eval split), `lc_injection_multiapp.md`
  (3-tier per-app injection). The three data docs are written in Chinese.

## Key facts (the verdict)
- **Headline result, both models: symbolic TTA tops out at ≈ the static prior.**
  B2 (static) and B3 (evolution) both plateau there: qwen B3 10 ≈ B2' 9.2;
  MAI B3 25 ≈ B2' 26.4. The B4 **weight channel** is where the remaining
  headroom is — this is the paper's point, don't soften it.
- **Best static config = B2'** = app-Layer-2 + category L_C. It seeds B3/B4 and
  is the variant transferred to MAI-UI (`strongest_variant.txt`).
- **Cross-benchmark Layer-1 injection is strictly harmful** — empirically, every
  run. Inject Layer-2 / L_C only.
- **MobileWorld containers do NOT reset app state between tasks** — a used
  container changes initial conditions, so containers are created **fresh per run
  and never reused**. Every B-series number assumes this.
- Numbers are `/110` eval (denominator fixed); 2 tasks
  (`CheckMeetingEventAskUserTask`, `ThanksgivingPrepTask`) fail to init on every
  run from the same upstream harness bug (model-independent) → effectively `/108`.

## Source of truth
- The machine-readable split is `../configs/mobileworld_splits.yaml v1.0-MW`
  (+ `gen_mobileworld_splits.py`), not any prose doc. `configs/teval_tasklist.txt`
  is the 110-task eval list actually run.
- `b3_b4_mw_tta_design.md` **supersedes** `mobileworld_split.md` §1's
  "symbolic-only, no LoRA channel" decision — B4 keeps its joint definition. The
  51/110 split in `mobileworld_split.md` itself is still authoritative.
- In `qwen3_8b_res.md`, §5's 5-run **means** supersede §3's single-run numbers
  (§3 kept as the run-1 / mechanism record).

## Don't
- Don't quote a single-run number as the result where a 5-run mean exists — use
  the means (e.g. B2' = **9.2±1.6**, not the run-1 10), and respect the error
  bars: fsm+lessons / lessons-only / B3-FSM / B2' all overlap at ≈10 on qwen.
- Don't read the static prior as a *win*: it is roughly baseline-neutral
  cross-benchmark, and on MAI-UI the qwen guidance helps little because MAI
  already has the read-before-acting discipline internally.
- Don't invent tiers/splits/counts — trace to `dataset_tiers.md` /
  `../configs/mobileworld_splits.yaml`. Tiers are **category-only**
  (seen / novel Play-Store category), never "unseen app".
- Don't create docs outside this folder; new MW docs go here.

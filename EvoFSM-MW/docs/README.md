# docs/ — the MobileWorld paper trail

The prose behind EvoFSM's **cross-benchmark** story: train on AndroidWorld+,
test-time-adapt on **MobileWorld** GUI-only (pure-vision). Everything here is the
experiment reports behind every MW number, the B3/B4 TTA designs, and the
dataset/split/injection specs the MW side derives from. For the project as a
whole, see `../../EvoFSM-RL/CLAUDE.md`.

## Layout

| File | Group | Role |
|---|---|---|
| `qwen3_8b_res.md` | results | **Headline.** EvoFSM-8B (Qwen3-VL-8B) B-series — B1 zero-shot → B2/B2'/B2'' static injection + the 5-run variance study (+ B2''') and the π^pre-350 check. |
| `mai_ui_8b_res.md` | results | **Headline.** MAI-UI-8B external (near-ceiling) baseline — B1 + B2' injection-transfer, the B3 variants, and the cross-model summary. |
| `b3_lesson_memory_results.md` | results | The B3 lesson-memory operator on qwen — leaderboard + what each config showed. |
| `b3_b4_mw_tta_design.md` | design | The B3/B4 **joint** TTA design — definitions, the one-shared-LoRA weight topology, init from π^pre. |
| `b3_lesson_memory_design.md` | design | The B3 lesson-memory operator — what evolves (per-app recipe+pitfall lessons) and how TrueSkill is made decidable. |
| `dataset_tiers.md` | data | Train AW+193 / test MW161, the 12 seen categories, Tier-B/C/A definitions, and the MW→AW same-category L_C index. |
| `mobileworld_split.md` | data | The task-disjoint adapt/eval (51/110) split — rationale + deterministic rules. |
| `lc_injection_multiapp.md` | data | The 3-tier per-app injection resolution (app FSM → category L_C → bootstrap) for multi-app tasks. |

## Where do I look for X

- **The numbers / which config wins?** → `qwen3_8b_res.md` (base) + `mai_ui_8b_res.md` (external), then `b3_lesson_memory_results.md` for the symbolic-TTA ceiling.
- **How is B3 vs B4 run / what does the weight channel add?** → `b3_b4_mw_tta_design.md`.
- **What apps/tasks, how split into tiers, and why?** → `dataset_tiers.md` (tiers) + `mobileworld_split.md` (adapt/eval split).
- **How does symbolic injection resolve for a multi-app task?** → `lc_injection_multiapp.md`.
- **A high-level "what is this project?"** → `../../EvoFSM-RL/CLAUDE.md` (no top-level CLAUDE in EvoFSM-MW).

See `CLAUDE.md` in this directory for AI working context.

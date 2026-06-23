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

## Ablation results at a glance

Verified numbers so you don't have to open each doc. `/110` eval; `±` is the 5-run std.

**B2 — static-config sweep (EvoFSM-8B / Qwen3-VL-8B)** — source: [`qwen3_8b_res.md`](qwen3_8b_res.md) §5

| Config | What's injected | Score (5-run, /110) |
|---|---|---|
| B1 | nothing (zero-shot) | 8.2 ± 0.8 |
| B2 | full app FSM + category `L_C` | 7.0 ± 0.7 |
| **B2′** | **app Layer-2 + `L_C`** (no Layer-1) | **9.2 ± 1.6** |
| B2″ | category `L_C` only | 8.4 ± 1.1 |
| B2‴ | B2′ + entry retrieval | 8.6 ± 1.7 |

**B3 — symbolic-TTA variants, both models** — sources: [`mai_ui_8b_res.md`](mai_ui_8b_res.md) §3, [`b3_lesson_memory_results.md`](b3_lesson_memory_results.md)

| Variant | EvoFSM-8B (/110) | MAI-UI-8B (/110) |
|---|---|---|
| B2′ baseline | 9.2 | 26.4 |
| FSM-evo | 10 | 27 |
| lessons-only | 10.0 ± 1.4 | 29.0 ± 4.5 |
| fsm + lessons | 10.4 ± 0.5 | 25.6 ± 3.0 |

**What this says (honest framing):**
- Best static config is **B2′** (app-Layer-2 + `L_C`, no Layer-1). Adding Layer-1 (full app FSM, B2) is **strictly harmful cross-benchmark** — it drops below the zero-shot baseline on every run.
- Symbolic TTA (B3) **tops out at ≈ B2′** on both models — `lessons-only` is the nominal best but sits within run-noise of B2′ / FSM-evo / fsm+lessons. Its real win is **efficiency** (~1/20 the prior's text), not accuracy. The remaining headroom is the B4 weight channel, not more symbols.

See `CLAUDE.md` in this directory for AI working context.

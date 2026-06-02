# EvoFSM-RL

Test-time-adaptive GUI agent that combines per-app FSM evolution with policy-level RL/LoRA fine-tuning, evaluated on `android_world_plus`.

This subproject lives inside the `android_world_plus` fork because it imports the benchmark's task registry, evaluators, and emulator harness directly. Treat the parent repo as the runtime environment; everything project-specific (design docs, tickets, splits, code) goes under `EvoFSM-RL/`.

---

## Layout

```
EvoFSM-RL/
├── README.md                    ← this file
├── CLAUDE.md                    ← project context for Claude (terminology, conventions, status)
├── docs/                        ← design memos
│   ├── taxonomy.md                 — Play-Store-category-based app taxonomy
│   ├── app_task_inventory.md       — per-app/per-task source-of-truth inventory (194 tasks / 25 apps)
│   ├── splits_protocol.md          — T_adapt vs T_eval definitions
│   └── citations.md                — citations supporting the split protocol
├── plan/                        ← roadmap
│   ├── project_plan.md             — overall project plan
│   ├── linear_tickets.md           — Linear ticket breakdown (Epics, Stories, Tasks)
│   └── algorithm_design.md         — EvoFSM algorithm design memo
├── presentations/               ← stakeholder-facing slides (.docx)
│   ├── algorithm_design_zh.docx
│   └── project_plan_zh.docx
├── configs/                     ← machine-readable configuration
│   ├── splits.yaml                 — K=1 baseline split (apps + templates)
│   └── multi_seed_config.yaml      — K=5 / K=3 multi-seed overlay
└── evofsm_rl/                   ← Python package (code goes here)
    └── __init__.py
```

---

## Key concepts

- **Source pool** (12 apps / 96 templates) → Phase 1 multi-source pretraining of per-app L1 FSMs and per-category `L_C` library.
- **Tier-B** (6 apps / 50 templates) → near-transfer: target apps whose Play Store category IS represented in the source pool.
- **Tier-C** (7 apps / 46 templates) → far-transfer: target apps in 6 Play Store categories absent from the source pool.
- **`T_adapt` / `T_eval`** → within each target app, templates are partitioned (template-disjoint) into an adaptation set and an evaluation set. TTA loop runs on `T_adapt`; frozen-checkpoint eval runs on `T_eval`. See `docs/splits_protocol.md`.

Total: **25 active primary-task apps / 194 task templates** (192 app-attributable + 2 generic). All counts grep-verified against the registry as of 2026-04-14 — see `docs/app_task_inventory.md`.

---

## Status (2026-04-15)

- ✅ **Epic 0 — Benchmark Design & Project Setup — COMPLETE.**
  - 0.1 — `evofsm_rl/taxonomy.py` + `configs/task_categories.csv` + `docs/taxonomy.md`
  - 0.2 — `evofsm_rl/splits.py` (loader over `configs/splits.yaml`)
  - 0.3 — Template-disjoint T_adapt/T_eval baked into `splits.yaml`; loader + tests
  - 12/12 unit tests passing (`tests/test_taxonomy_splits.py`)
- ⏳ Epic 1 — Infrastructure — next up

### Quick start
```bash
cd EvoFSM-RL/
PYTHONPATH=. python3 tests/test_taxonomy_splits.py        # run sanity tests
PYTHONPATH=. python3 -m scripts.generate_task_categories_csv   # regenerate CSV
```

# artifacts/ — static FSM / L_C knowledge layer

The symbolic prior the agent injects before acting: per-app two-layer FSMs
(`static_fsms_*`) and their per-Play-Store-category Layer-2 aggregation
(`L_C_*`). Small (~2.2 MB), versioned, hand-built — not training output.

Each FSM JSON is `{version, app, layer1, layer2, metadata}`; each L_C JSON is
`{category, layer2.categories[...]}` (abstract-action blocks: precondition,
abstract_steps, failure_modes, verification_checklist).

| Directory | Role | Status |
|---|---|---|
| `static_fsms/` | Original 12-app FSMs (`.json` + `.txt` pairs, plus `backup_pre_patch/`); the seed layer B2/B3 shipped with. | **superseded** by v2 |
| `static_fsms_om/` | 19 FSMs synthesized from OpenMobile chain-of-thought traces; the bulk source for the v2 rebuild. | intermediate |
| `static_fsms_b4/` | 2 FSMs (`maps_me`, `wikipedia`) recovered from b4 a11y trajectories; the gap-fillers folded into v2. | intermediate |
| `static_fsms_v2/` | **Current** 25-app FSM layer (19 from OM + 2 from b4 + 4 reused source-pool). The version the paper draft uses. | **active** |
| `L_C/` | Original 6-category Layer-2 library aggregated from `static_fsms/`. | **superseded** by v2 |
| `L_C_v2/` | **Current** 12-category Layer-2 library aggregated from `static_fsms_v2/`. The version the paper draft uses. | **active** |
| `L_C_v3/` | 12-category exclude-system variant (system-app categories trimmed); newer experiment. | experimental |

`_cleanup_summary.json` is a leftover Layer-2 linter log (per-app
transferability-violation rounds), not a knowledge file.

The PRM / value-head checkpoints that used to live here were moved to
`../archive/ppo_prm/`.

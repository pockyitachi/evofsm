# configs/ — MobileWorld cross-benchmark split config

The MobileWorld (MW) side of EvoFSM's cross-benchmark TTA setup: train on
AndroidWorld+ (193), test on **MobileWorld** GUI-only (161). This dir holds the
test-domain split — which MW tasks are `T_adapt` (TTA fitness set) vs `T_eval`
(frozen), and each task's tier — plus the deterministic generator that produces it.
No model / hyperparam configs live here; those stay on the EvoFSM-RL side.

| File | Role |
|---|---|
| `mobileworld_splits.yaml` | **Source of truth for the MW split.** Per-task `adapt`/`eval` assignment + per-task tier, over the 161 GUI-only tasks. Task-disjoint (a task is wholly in adapt or eval). Carries `meta.version` (`v1.0-MW`), `meta.totals` (161 / adapt 51 / eval 110) and `meta.per_tier`. Hand-written by the generator for layout control — **do not hand-edit**. |
| `gen_mobileworld_splits.py` | The deterministic generator for the yaml. Walks the MobileWorld task definitions, drops `agent-mcp` tasks (161 of 201 remain GUI-only), assigns tiers by Play-category, and splits Tier-B/C ~40% to adapt. No RNG — sorted by class name, so reruns are byte-stable. |
| `teval_tasklist.txt` | The **110-task eval list** (the `eval` records of the yaml), flattened for harness consumption. **Gotcha: it is ONE line, COMMA-separated** (no trailing newline, not one-task-per-line) — readers must `split(',')`. |

## Tiers (CATEGORY-level)

Tier is assigned by the Play-category membership of a task's apps, not the app
identity (6 MW system apps share packages, so unseen-*app* tiering was dropped):

- **Tier-B** — all apps fall in categories seen in the AW+ source pool. 58 eval.
- **Tier-C** — novel categories only (Social: Mastodon / Mattermost; Shopping:
  Taodian). 25 eval.
- **Tier-A** — mixed multi-app (seen + novel together); the compositional
  headline — **all go to eval**, none to adapt. 27 eval.

Totals: 51 adapt / 110 eval of the 161 GUI-only tasks.

## Source of truth

`mobileworld_splits.yaml` is authoritative for split membership and tiers;
`teval_tasklist.txt` is derived from its `eval` records. To change the split,
edit the **rules in `gen_mobileworld_splits.py`**, rerun it, and **bump
`meta.version`** — never hand-edit the yaml or the txt. Reproducibility (and a
clean cross-benchmark claim) depends on the generator staying the single source.

See `CLAUDE.md` here for working context, and `../README.md` for the EvoFSM-MW
picture (`docs/mobileworld_split.md` for the split rationale).

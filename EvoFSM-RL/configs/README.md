# configs/ — run, split & model configs

Declarative inputs for the EvoFSM-RL pipeline: which apps/templates land in
which split, how many seeds to run, and which base weights the policy loads.
No code — these are read by `evofsm_rl/splits.py`, `evofsm_rl/model/loader.py`,
and the `../scripts/` entry points.

| File / dir | Role |
|---|---|
| `splits.yaml` | **Source of truth for splits.** K=1 baseline: source pool (12 apps / 96 templates) + Tier-B (6 apps, near-transfer) + Tier-C (7 apps, far-transfer), each target app pre-split into `T_adapt` / `T_eval` templates. Carries `meta.version` (currently `v1.1-K1`). |
| `multi_seed_config.yaml` | K=5 (adapt/pretrain) and K=3 (eval) seed-count overlay applied **on top of** `splits.yaml`. Defines `task_random_seeds` and per-tier episode budgets. Does not redefine membership. |
| `model.yaml` | Source of truth for the base model: HF repo id, revision SHA, dtype, max_pixels (Mac vs A100 role split). Nothing else should hardcode model name / revision / dtype. |
| `model_fingerprint.lock.json` | Auto-generated fingerprint (architecture, param count, resolved SHA, vocab size). The loader writes it once, then fails loud on mismatch. Don't hand-edit. |
| `task_categories.csv` | One row per task: `task_name → app, tier, play_category, task_type`. Backs `evofsm_rl/taxonomy.py`. |
| `source_pool_96.txt` | Pretraining task list: 96 source-pool templates (`Template:source`), derived from `splits.yaml > source_pool`. |
| `baseline_50task.txt` | Original source-pool 50-task sweep list (`Template:tier`), used for the `m3a_50task_v01` baseline. |
| `t_eval/teval_v01.txt` | Held-out eval task list: 35 templates (18 Tier-B + 18 minus degenerate = 17 Tier-C), derived from `splits.yaml` held-out blocks. The paper Table 1 zero-shot row runs this. |

**Source of truth:** `splits.yaml` is authoritative for split membership;
everything else (`multi_seed_config.yaml`, `source_pool_96.txt`,
`t_eval/teval_v01.txt`, the counts in `task_categories.csv`) is derived from it.
`model.yaml` is authoritative for the base model. Never edit `splits.yaml`
without bumping `meta.version` and recording the change in `meta.revision_notes`
— downstream loaders and reproducibility depend on it.

See `CLAUDE.md` here for working context, and `../CLAUDE.md` for the
project-wide picture.

# artifacts/ — MW injection inputs + eval run transcripts

Scratch+products directory for the MobileWorld (MW) cross-benchmark B-series
evals. It holds two very different kinds of thing, and **the actual eval results
are not in here** — see the warning at the bottom.

This is ~4.2 GB and mostly disposable. Don't treat it as a clean knowledge dir
like `../../EvoFSM-RL/artifacts/`.

## What's durable (keep) — the injection guidance

Frozen prompt-injection bundles, one per static/evolved variant, baked by
`../harness/gen_b2_guidance.py` / `gen_b3_guidance.py` from the
`static_fsms_v2` + `L_C_v2` symbolic layer. Each is `{meta, tasks}`: `meta`
records `mode` / `fsm_dir` / `lc_dir` / per-tier coverage `stats`; `tasks` maps
each of the 161 MW templates to its `tier`, `apps`, and the injected `text`
(or empty). These are the *inputs* a run was driven with.

| File | Variant |
|---|---|
| `b2_guidance.json` | B2 — full FSM (Layer-1 + Layer-2). Net-harmful cross-benchmark; kept for the ablation. |
| `b2p_guidance.json` | **B2' — app-Layer-2 + `L_C`, no Layer-1. The strongest static variant** (see `strongest_variant.txt` = `b2p`). |
| `b2pp_guidance.json`, `b2ppp_guidance.json` | B2'' / B2''' — category-only and retrieval-trimmed static variants. |
| `b2p_v3_guidance.json` | B2' rebaked against the `L_C_v3` exclude-system layer. |
| `b3*_guidance.json` | B3 FSM-evolution champions and lesson/how-to variants (`b3champ`, `b3fsm_lesson`, `b3lesson*`, `b3mai_*`). |
| `task_goals.json` | MW template → natural-language goal (108 entries); plain lookup. |
| `strongest_variant.txt` | One line: which variant won (`b2p`). |

## What's disposable (residue) — run transcripts, NOT results

- **`*.log` (~1.9 GB, ~90 files)** — stdout transcripts of the unattended eval
  chains (`*_stdout` / `*_driver` / `*_chain` / `*_envrun` / `*_console` /
  `*_smoke`, plus `vllm_*` server logs). Useful for debugging a run that already
  happened; not the source of truth for any score. Several are 100+ MB.
- **`*_eval_logs/` (~2.7 GB)** — per-run log subdirs. `rep_eval_logs/` (1.7 GB)
  is the B-series variance study; `pi200/250/300_eval_logs/` and
  `mai_pre100/175_eval_logs/` are the π^pre LoRA-checkpoint eval runs.
- **`watch_*.sh` / `run_*.sh`** — the babysitter/launcher shell scripts these
  chains were driven by. Operational, not products.

## ⚠️ Where the real eval results actually live

**Not here.** The per-task pass/fail used to compute every B-series number in
`../docs/` lives in **`../../MobileWorld/traj_logs/<run>/<Task>/result.txt`**
(each file is `score: 1.0|0.0` + a `reason:`). The `.log` files in this
directory are stdout transcripts of those runs — convenient to grep, but **not**
the success-count source of truth. To recount a number, aggregate the
`result.txt` files under `MobileWorld/traj_logs/`, not the logs here.

The narrative results and final tables are written up in `../docs/`
(`qwen3_8b_res.md`, `mai_ui_8b_res.md`, …).

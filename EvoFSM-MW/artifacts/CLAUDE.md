# CLAUDE.md — artifacts/ working context

See `../README.md` for what this directory is, and `../CLAUDE.md` for the
project-wide picture.
This file is only what you need when touching *this* directory.

This dir = **MW eval injection inputs + run transcripts**, ~4.2 GB, mostly
disposable. It is **not** a clean knowledge layer — that's
`../../EvoFSM-RL/artifacts/` (the `static_fsms_*` / `L_C_*` source).

## Map of the contents (by kind, not by file)
- **Guidance JSONs (`*_guidance.json`) — durable.** Frozen `{meta, tasks}`
  prompt-injection bundles, one per variant, baked by `../harness/gen_b2_*` /
  `gen_b3_*.py` from the v2 symbolic layer. `meta.mode` + `meta.stats` tell you
  what was injected and the per-tier coverage; `tasks[<Template>]` carries the
  injected `text`/`tier`/`apps`. These are the *inputs* a run consumed.
  - `b2p_guidance.json` = **B2'** = app-Layer-2 + `L_C`, no Layer-1 = the
    strongest static variant (`strongest_variant.txt` → `b2p`). Default to it.
  - `b2_guidance.json` = full FSM incl. Layer-1 — kept only for the ablation;
    Layer-1 is empirically net-harmful cross-benchmark.
  - `b3*_guidance.json` = B3 FSM-evolution champions / lessons.
- **`task_goals.json` — durable.** Template → goal string (108). Pure lookup.
- **`*.log` (~1.9 GB) — disposable run transcripts.** stdout of unattended eval
  chains (`*_stdout/_driver/_chain/_envrun/_console/_smoke`, `vllm_*`). Grep to
  debug a past run; **not a results file**.
- **`*_eval_logs/` (~2.7 GB) — disposable.** Per-run log subdirs
  (`rep_eval_logs/` = variance study 1.7 GB; `pi*/mai_pre*_eval_logs/` = π^pre
  checkpoint runs).
- **`watch_*.sh` / `run_*.sh` — operational.** The babysitter/launcher scripts.

## Don't
- **Don't read scores from the `.log` files here.** The per-task pass/fail
  behind every `../docs/` number lives in
  **`../../MobileWorld/traj_logs/<run>/<Task>/result.txt`** (`score: 1.0|0.0`).
  To recount, aggregate those `result.txt` files — the logs here are transcripts.
- **Don't treat guidance JSONs as the symbolic prior.** They're frozen *bakes*;
  the editable source of truth is `../../EvoFSM-RL/artifacts/static_fsms_v2` +
  `L_C_v2`. To change injection, regenerate via `../harness/gen_b2_guidance.py`,
  don't hand-edit the JSON.
- **Don't quote `b2_guidance` (full FSM) as the static result** — B2' (`b2p`)
  is the reported static config; B2's Layer-1 hurts cross-benchmark.
- **Don't delete `*_guidance.json`, `task_goals.json`, or
  `strongest_variant.txt`** when reclaiming space — those are the durable
  products. The `.log` files and `*_eval_logs/` dirs are the safe ~4 GB to prune.
- **Don't reuse an MW run/container across evals** — MobileWorld containers do
  not reset app state; every run must be fresh (these logs are from one-shot
  chains for exactly that reason).

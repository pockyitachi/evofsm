# traces/ — trajectory & evaluation outputs

All rollout trajectories, per-episode dumps, and evaluation summaries.
**Gitignored — large, regenerable data, not source.** Each subdirectory is one
run or sweep.

## Naming
- `b1/b2/b3/b4_*` — baseline arms (B1 zero-shot … B4 joint)
- `*_teval` — held-out `T_eval` evaluations (the reportable numbers)
- `phase1_*` — Phase-1 shared-LoRA pretraining
- `m3a_*` — early M3A baselines
- `*_smoke` — quick smoke runs

Abandoned-route and giant init-ablation-sweep trajectories have been moved out
to `../archive/*/*.zip` to reclaim disk.

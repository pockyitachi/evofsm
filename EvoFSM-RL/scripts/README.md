# scripts/ — CLI entry points

Executable command-line drivers for the EvoFSM-RL pipeline: FSM/`L_C`
synthesis, the B1–B4 baseline progression, single-task eval probes, and the
offline analysis / plotting helpers. These are *entry points*, not importable
modules — the reusable logic lives in `../evofsm_rl/`.

## FSM synthesis

| Script | Role |
|---|---|
| `build_all_fsms.py` | Batch-synthesize a per-app static FSM (`F^0_a`) over the source-pool apps via the Anthropic API; lints Layer-2 + writes `artifacts/static_fsms/{app}.json` + `.txt`. |
| `build_L_C.py` | Aggregate the per-app FSMs into per-Play-Store-category `L_C` (Layer-2 library) and lint for leakage → `artifacts/L_C/{slug}.json`. |

## Baselines: eval & evolution sweeps

| Script | Role |
|---|---|
| `run_b2_eval.py` | B2 — static `L_C` injected into the agent prompt, eval on `T_eval` (Tier-B injected, Tier-C falls back to B1). No training. |
| `run_b3_evolution.py` | B3 — evolve `L_C` (Layer-2 only) on one app's `T_adapt`; writes `population.json` + `l_c_champion.json`. |
| `run_b3_teval.py` | B3 — frozen eval comparing `b2_static` vs `b3_evolved` `L_C` on `T_eval`. |
| `run_b4_evolution.py` | B4 — B3 evolution + GRPO LoRA fine-tuning of the policy on one app's `T_adapt`. |
| `run_b4_teval.py` | B4 — frozen joint eval (LoRA adapter + evolved `L_C`) on `T_eval`. Paper headline B4 row. |
| `run_b4_sweep.sh` | Driver: B4 evolution over the 6 Tier-B apps × 20 iters → `traces/b4_evolution_v2/`. |
| `run_b4_k4_dense_sweep.sh` | Driver: K=4 + `--use-dense-reward` ablation, limited to the 3 dense-capable apps. |
| `run_phase1_pretraining.py` | Phase 1 — shared-LoRA GRPO pretraining over a source pool (static FSMs, no mutation) → `π^pre_θ`. |

## Basic eval probes

| Script | Role |
|---|---|
| `run_rollout.py` | Run a single AndroidWorld task with the base agent; print the rule-based `is_successful` score. |
| `baseline_10task.py` | N-task difficulty-tiered sweep with per-episode persistence flags + a `summary.json`. |
| `coord_probe.py` | Coordinate-calibration probe — confirm the model emits original-resolution pixel coords (not resized-canvas). |
| `model_smoke.py` | Story 1.2 forward-pass smoke test — load checkpoint, check fingerprint lock, emit logits. |

## Analysis (underscore-prefixed, no emulator/model)

| Script | Role |
|---|---|
| `_aggregate_b1_vs_b2_k3.py` | Aggregate B1 vs B2 K=3 summaries → side-by-side success comparison with stderr. |
| `_fsm_quality_digest.py` | Per-app digest of FSM structure + trajectory success stats for the quality audit. |
| `_probe_lint_violations.py` | Run `lint_layer2` over every static FSM and print PASS/FAIL + violations. |
| `_relint_L_C.py` | Re-lint already-written `L_C` files against their source FSMs (no recompute). |

## Environment setup (legacy — superseded by the AWAvd2 snapshot)

| Script | Role |
|---|---|
| `bootstrap_avd.sh` | Create a fresh `Pixel_6_API_33` AVD (arm64 / x86_64). |
| `install_plus_apks.sh` | Sideload the 6 Plus-repo APKs onto a running AVD. |

## Misc

| Script | Role |
|---|---|
| `clean_layer2.py` | Send Layer-2 blocks that fail lint to Claude for a generalized rewrite (Layer-1 untouched). |
| `plot_convergence.py` | Convergence figure for B3/B4 evolution logs (success, GRPO loss, champion TrueSkill `mu`). Also called as a library at the end of the evolution scripts. |
| `patch_fsm_story22.py` | One-shot fix for the bluecoins / clock / pi_music FSMs from the quality memo. |
| `generate_task_categories_csv.py` | Build `configs/task_categories.csv` from `splits.yaml` + task metadata. |

## Usage

Run from the **repo root** with `PYTHONPATH` set:

```bash
cd /shared/linqiang/evofsm_project && source .venv/bin/activate
PYTHONPATH=android_world_plus:EvoFSM-RL python EvoFSM-RL/scripts/<script>.py --help
```

The eval/probe scripts need a booted AWAvd2 emulator (see `../CLAUDE.md`
"Canonical AVD") and take `--console-port/--grpc-port/--adb-path`; most have a
`--dry-run` (mock rollouts) or `--synthetic` path for offline iteration. The
FSM-synthesis and `clean_layer2.py` scripts call the Anthropic API and need
`ANTHROPIC_API_KEY`. Each script's docstring is the authoritative usage block.

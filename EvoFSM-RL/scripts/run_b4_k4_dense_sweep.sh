#!/usr/bin/env bash
# K=4 + dense reward ablation sweep.
#
# Mirrors the existing K=4 sweep (traces/b4_k4_{A,B,C,D}/) byte-for-byte
# on hyperparameters; the only difference is ``--use-dense-reward``.
# Limits to the 3 apps whose multi-row templates actually exercise the
# dense path (pro_expense, simple_calendar_pro, broccoli). For the other
# 9 apps in the K=4 sweep, dense ≡ binary by construction (their tasks
# don't override get_dense_reward), so re-running them would burn GPU
# for an identical result. T_eval (run_b4_teval.py) stays binary per
# the CLAUDE.md design rule.
#
# Config pinned from traces/b4_k4_A/pro_expense_log.txt:
#   N=4 rollouts, M=2 select, 20 iter, update_every=3, checkpoint_every=5,
#   lora_lr=3e-4, min_n_active=3, kl_beta=0.05,
#   init_lora=phase1_pilot_v01/lora_checkpoints/final (π^pre_θ),
#   ref_lora=same (KL anchor on Phase 1 pilot).

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

# Activate venv (script is invoked via nohup so PATH may not have it)
source "${PROJECT_DIR}/.venv/bin/activate"

OUTPUT_ROOT="EvoFSM-RL/traces/b4_k4_dense"
mkdir -p "${OUTPUT_ROOT}"

# Multi-row apps only. Other 9 apps in K=4 sweep see no change under dense.
# Tier-B: pro_expense, simple_calendar_pro — use source-pool L_C as init.
# Tier-C: broccoli — bootstrap LAYER 2 from target trajectories
#         (same flag the K=4=48.1% sweep used; see traces/b4_k4_B/broccoli_log.txt).
APPS=(pro_expense simple_calendar_pro broccoli)

PILOT_LORA="EvoFSM-RL/traces/phase1_pilot_v01/lora_checkpoints/final"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SWEEP_LOG="${OUTPUT_ROOT}/sweep.log"
echo "K=4 + dense sweep starting at $(date -u)" | tee -a "${SWEEP_LOG}"
echo "Output root: ${OUTPUT_ROOT}" | tee -a "${SWEEP_LOG}"
echo "Apps: ${APPS[*]}" | tee -a "${SWEEP_LOG}"
echo "Pilot LoRA (init + ref): ${PILOT_LORA}" | tee -a "${SWEEP_LOG}"

for app in "${APPS[@]}"; do
  echo "" | tee -a "${SWEEP_LOG}"
  echo "==========================================" | tee -a "${SWEEP_LOG}"
  echo "Starting B4 K=4 + dense: ${app}  $(date -u)" | tee -a "${SWEEP_LOG}"
  echo "==========================================" | tee -a "${SWEEP_LOG}"

  APP_LOG="${OUTPUT_ROOT}/${app}_log.txt"
  APP_OUT="${OUTPUT_ROOT}/${app}"

  # Tier-C (broccoli) needs --enable-bootstrap-fsm; Tier-B apps don't.
  if [ "${app}" = "broccoli" ]; then
    EXTRA_FLAGS=(--enable-bootstrap-fsm)
  else
    EXTRA_FLAGS=()
  fi

  set +e
  PYTHONPATH=android_world_plus:EvoFSM-RL \
    python EvoFSM-RL/scripts/run_b4_evolution.py \
      --app "${app}" \
      --n-iterations 20 \
      --n-rollouts 4 \
      --m-select 2 \
      --lora-rank 16 --lora-lr 3e-4 \
      --lora-update-every 3 \
      --lora-checkpoint-every 5 \
      --min-n-active 3 \
      --kl-beta 0.05 \
      --kl-log-ratio-clip 10.0 \
      --init-lora-from "${PILOT_LORA}" \
      --ref-lora-from "${PILOT_LORA}" \
      --use-dense-reward \
      --console-port 5712 --grpc-port 8712 \
      --adb-path "$(pwd)/android-sdk/platform-tools/adb" \
      --output-dir "${APP_OUT}" \
      "${EXTRA_FLAGS[@]}" \
      2>&1 \
      | grep -v ANTHROPIC_API_KEY \
      | tee "${APP_LOG}"
  EXIT_CODE=${PIPESTATUS[0]}
  set -e

  if [ "${EXIT_CODE}" -ne 0 ]; then
    echo "ERROR: ${app} failed with exit code ${EXIT_CODE}" | tee -a "${SWEEP_LOG}"
  fi

  echo "Finished ${app} at $(date -u), exit=${EXIT_CODE}" | tee -a "${SWEEP_LOG}"
done

echo "" | tee -a "${SWEEP_LOG}"
echo "K=4 + dense sweep complete at $(date -u)" | tee -a "${SWEEP_LOG}"

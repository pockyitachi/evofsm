#!/usr/bin/env bash
# B4 evolution sweep driver — 6 Tier-B apps × 20 iterations each.
#
# Design choices motivated by the 2026-04-23 first-pass sweep post-mortem:
#
#   * ``set -e -o pipefail`` — earlier driver used plain ``|``, which
#     swallows the Python exit code and returns ``tee``'s 0. Four apps
#     OOM'd mid-run and the for-loop moved on reporting ``exit=0``. We
#     now capture the real Python exit via ``PIPESTATUS[0]`` and log it.
#   * Per-app log file — split from the combined sweep log so each app
#     is triage-able in isolation. The combined log still exists via tee.
#   * ``CUDA_VISIBLE_DEVICES=2`` — our assigned GPU (see CLAUDE.md).
#   * ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` — reduces
#     allocator fragmentation across the many forward passes per run.
#   * ``-u`` disabled on purpose — we tolerate unset ANTHROPIC_API_KEY
#     just to surface the error from python itself with a clearer
#     message than bash "unbound variable".

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

# Output directory — write to b4_evolution_v2 so the previous run's
# partial artifacts under b4_evolution/ remain available for diff.
OUTPUT_ROOT="EvoFSM-RL/traces/b4_evolution_v2"
mkdir -p "${OUTPUT_ROOT}"

APPS=(pro_expense simple_calendar_pro system_settings retro_music camera chrome)

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SWEEP_LOG="${OUTPUT_ROOT}/sweep.log"
echo "B4 sweep v2 starting at $(date -u)" | tee -a "${SWEEP_LOG}"
echo "Output root: ${OUTPUT_ROOT}" | tee -a "${SWEEP_LOG}"

for app in "${APPS[@]}"; do
  echo "" | tee -a "${SWEEP_LOG}"
  echo "==========================================" | tee -a "${SWEEP_LOG}"
  echo "Starting B4 evolution: ${app}  $(date -u)" | tee -a "${SWEEP_LOG}"
  echo "==========================================" | tee -a "${SWEEP_LOG}"

  APP_LOG="${OUTPUT_ROOT}/${app}_log.txt"
  APP_OUT="${OUTPUT_ROOT}/${app}"

  set +e
  PYTHONPATH=android_world_plus:EvoFSM-RL \
    python EvoFSM-RL/scripts/run_b4_evolution.py \
      --app "${app}" \
      --console-port 5710 --grpc-port 8710 \
      --n-iterations 20 \
      --lora-rank 16 --lora-lr 1e-4 \
      --lora-update-every 5 \
      --lora-checkpoint-every 5 \
      --output-dir "${APP_OUT}" \
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
echo "B4 sweep v2 complete at $(date -u)" | tee -a "${SWEEP_LOG}"

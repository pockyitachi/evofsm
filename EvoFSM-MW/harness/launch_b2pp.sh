#!/bin/bash
# B2'' eval launch — original EvoFSM-RL B2 recipe, 1:1. App tier disabled:
# EVERY Tier-B app resolves via its Play-category L_C (Settings→tools.json
# etc.); Tier-C empty. Replicates the +9.3pp AW recipe cross-benchmark.
# Guidance: artifacts/b2pp_guidance.json (gen_b2_guidance.py --mode category-only)
# Fresh lq_b2pp containers: backend 6830-32, viewer 7890-92, vnc 5830-32, adb 5740-42.
set -e

if pgrep -f "bin/mw eval" >/dev/null; then
  echo "another mw eval is still running — aborting B2'' launch."
  exit 1
fi

exec 9>/tmp/evofsm_b2pp_eval.lock
if ! flock -n 9; then
  echo "Another B2'' eval already running — aborting."
  exit 1
fi

cd /shared/linqiang/evofsm_project/MobileWorld

export EVOFSM_B2_GUIDANCE=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts/b2pp_guidance.json

.venv/bin/mw eval \
  --agent_type /shared/linqiang/evofsm_project/EvoFSM-MW/harness/qwen3vl_b2_agent.py \
  --task "$(tr -d '\n' < /shared/linqiang/evofsm_project/EvoFSM-MW/configs/teval_tasklist.txt)" \
  --max_round 50 \
  --model_name Qwen3-VL-8B-Instruct \
  --llm_base_url http://localhost:8001/v1 \
  --step_wait_time 3 \
  --max-concurrency 3 \
  --env-name-prefix lq_b2pp \
  --env-image mobile_world:reset \
  --log_file_root traj_logs/b2pp_qwen3vl8b \
  --enable_mcp \
  --enable_user_interaction

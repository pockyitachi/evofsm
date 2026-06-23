#!/bin/bash
# B2' eval launch — B2 minus Layer-1. The 6 shared-package system apps get ONLY
# their FSM's own Layer-2 ("app workflow knowledge", ~6k chars); other Tier-B
# apps unchanged (category L_C); Tier-C empty. Minimal-diff ablation vs B2:
# the B2−B2' delta isolates cross-benchmark Layer-1 harm.
# Guidance: artifacts/b2p_guidance.json (gen_b2_guidance.py --mode app-l2)
# Fresh lq_b2p containers: backend 6820-22, viewer 7880-82, vnc 5820-22, adb 5730-32.
set -e

if pgrep -f "bin/mw eval" >/dev/null; then
  echo "another mw eval is still running — aborting B2' launch."
  exit 1
fi

exec 9>/tmp/evofsm_b2p_eval.lock
if ! flock -n 9; then
  echo "Another B2' eval already running — aborting."
  exit 1
fi

cd /shared/linqiang/evofsm_project/MobileWorld

export EVOFSM_B2_GUIDANCE=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts/b2p_guidance.json

.venv/bin/mw eval \
  --agent_type /shared/linqiang/evofsm_project/EvoFSM-MW/harness/qwen3vl_b2_agent.py \
  --task "$(tr -d '\n' < /shared/linqiang/evofsm_project/EvoFSM-MW/configs/teval_tasklist.txt)" \
  --max_round 50 \
  --model_name Qwen3-VL-8B-Instruct \
  --llm_base_url http://localhost:8001/v1 \
  --step_wait_time 3 \
  --max-concurrency 3 \
  --env-name-prefix lq_b2p \
  --env-image mobile_world:reset \
  --log_file_root traj_logs/b2p_qwen3vl8b \
  --enable_mcp \
  --enable_user_interaction

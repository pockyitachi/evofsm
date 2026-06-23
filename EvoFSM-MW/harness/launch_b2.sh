#!/bin/bash
# B2 eval launch — mirrors the B1 command exactly except: agent file, log dir,
# EVOFSM_B2_GUIDANCE, and --env-name-prefix lq_b2.
#
# B2 runs on FRESH lq_b2 containers, NEVER on the used lq_b1 ones: MobileWorld
# containers do NOT reset app state between tasks, so a container that already
# ran B1's 110 episodes carries dirty app state — reusing it would give B2
# different initial conditions than B1 had. Fresh-per-run = same protocol B1
# itself used. lq_b2 ports are offset so lq_b1 can stay up for inspection:
#   backend 6810-6812, viewer 7870-7872, vnc 5810-5812, adb 5720-5722
#   (adb 5720 also avoids 5710=AWAvd2 training emulator)
# Bring-up (normally done by auto_chain_b2.sh):
#   cd /shared/linqiang/evofsm_project/MobileWorld
#   .venv/bin/mw env run --count 3 --name-prefix lq_b2 --image mobile_world:reset \
#       --backend-start-port 6810 --viewer-start-port 7870 --vnc-start-port 5810 \
#       --adb-start-port 5720 --launch-interval 20 \
#       --env-file /shared/linqiang/MobileWorld/.env
# Task list: configs/teval_tasklist.txt = the 110 GUI-only T-eval (eval-split)
# tasks — identical to B1's list; the adapt split (51) is NOT run.
set -e

# Guard 1 — never overlap with the B1 eval (shares the vLLM; serial by design).
if pgrep -af "bin/mw eval" | grep -q "b1_qwen3vl8b"; then
  echo "B1 eval is still running — aborting B2 launch."
  exit 1
fi

# Guard 2 — single-instance lock. Several watchers/sessions may try to start
# B2; only the first passes. fd 9 is inherited by mw eval, so the lock stays
# held for the eval's whole lifetime.
exec 9>/tmp/evofsm_b2_eval.lock
if ! flock -n 9; then
  echo "Another B2 eval already running (/tmp/evofsm_b2_eval.lock held) — aborting."
  exit 1
fi

cd /shared/linqiang/evofsm_project/MobileWorld

export EVOFSM_B2_GUIDANCE=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts/b2_guidance.json

.venv/bin/mw eval \
  --agent_type /shared/linqiang/evofsm_project/EvoFSM-MW/harness/qwen3vl_b2_agent.py \
  --task "$(tr -d '\n' < /shared/linqiang/evofsm_project/EvoFSM-MW/configs/teval_tasklist.txt)" \
  --max_round 50 \
  --model_name Qwen3-VL-8B-Instruct \
  --llm_base_url http://localhost:8001/v1 \
  --step_wait_time 3 \
  --max-concurrency 3 \
  --env-name-prefix lq_b2 \
  --env-image mobile_world:reset \
  --log_file_root traj_logs/b2_qwen3vl8b \
  --enable_mcp \
  --enable_user_interaction

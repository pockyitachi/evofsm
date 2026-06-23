#!/bin/bash
set -u
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
MW=/shared/linqiang/evofsm_project/MobileWorld
PROJ=/shared/linqiang/evofsm_project; HARNESS=$PROJ/EvoFSM-MW/harness
ENVFILE=/shared/linqiang/MobileWorld/.env; GUID=$ART/b3fsm_lesson_guidance.json
note(){ echo "[fsm-redo $(date '+%F %T')] $*" | tee -a "$ART/tonight_chain.log"; }
for c in $(docker ps -a --format '{{.Names}}'|grep '^lq_b3fl'); do docker rm -f "$c">/dev/null 2>&1;done
note "redo rep (=rep1 replacement): starting 5 fresh lq_b3fl containers"
(cd "$MW" && .venv/bin/mw env run --count 5 --name-prefix lq_b3fl --image mobile_world:reset \
  --backend-start-port 7020 --viewer-start-port 7970 --vnc-start-port 5930 --adb-start-port 5820 \
  --launch-interval 20 --env-file "$ENVFILE") >"$ART/tonight_envrun.log" 2>&1 || { note "env run failed"; exit 1; }
dl=$(( $(date +%s)+1800 )); ok=0
while :; do h=$(docker ps --filter name=lq_b3fl_ --format '{{.Status}}'|grep -c '(healthy)'); [ "$h" -ge 5 ]&&{ ok=1;break;}; [ "$(date +%s)" -ge "$dl" ]&&break; sleep 20; done
[ "$ok" = 1 ]||{ note "containers not ready"; exit 1; }
note "5/5 healthy — launching eval"
cd "$MW"
EVOFSM_B2_GUIDANCE="$GUID" .venv/bin/mw eval --agent_type "$HARNESS/qwen3vl_b2_agent.py" \
  --task "$(tr -d '\n' < $PROJ/EvoFSM-MW/configs/teval_tasklist.txt)" \
  --max_round 50 --model_name Qwen3-VL-8B-Instruct --llm_base_url http://localhost:8001/v1 \
  --step_wait_time 3 --max-concurrency 5 --env-name-prefix lq_b3fl --env-image mobile_world:reset \
  --log_file_root traj_logs/b3fsm_lesson_qwen3vl8b_r6 --enable_mcp --enable_user_interaction \
  >"$ART/tonight_eval_b3fsm_lesson_qwen3vl8b_r6.log" 2>&1
d=$MW/traj_logs/b3fsm_lesson_qwen3vl8b_r6
n=$(ls "$d"/*/result.txt 2>/dev/null|wc -l); s=$(grep -l "score: 1.0" "$d"/*/result.txt 2>/dev/null|wc -l)
for c in $(docker ps -a --format '{{.Names}}'|grep '^lq_b3fl'); do docker rm -f "$c">/dev/null 2>&1;done
note "FSM-REDO DONE: success $s / $n  → fsm+lessons 5th valid rep"

#!/bin/bash
# Waits for the B3-lesson adapt (51 iters) to COMPLETE, then:
#   1. render eval guidance = B2'(v2) prior + VALIDATED lessons -> b3lesson_guidance.json
#   2. eval on the 110-task split with the qwen B2 agent + that guidance
#      (qwen3-VL-8B @ :8001), vs B2'(qwen) 10 / B1 ~9.
# Fresh 5 containers; tmux b3lessonevalchain. PAUSES if adapt PAUSED.
set -u
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
LOG=$ART/b3lesson_eval_chain.log
MW=/shared/linqiang/evofsm_project/MobileWorld
PROJ=/shared/linqiang/evofsm_project
SKY=$PROJ/SkyRL-AndroidWorld/skyrl-agent
HARNESS=$PROJ/EvoFSM-MW/harness
ENVFILE=/shared/linqiang/MobileWorld/.env
GUID=$ART/b3lesson_guidance.json
note() { echo "[b3lesson-eval $(date '+%F %T')] $*" | tee -a "$LOG"; }
clean() { for c in $(docker ps -a --format '{{.Names}}' | grep '^lq_b3lese'); do docker rm -f "$c" >/dev/null 2>&1; done; }

note "ARMED: waiting for B3-lesson adapt to COMPLETE"
while :; do
  grep -q "B3LESSON ROUND1 COMPLETE" "$ART/b3lesson_chain.log" 2>/dev/null && break
  grep -q "PAUSED" "$ART/b3lesson_chain.log" 2>/dev/null && { note "adapt PAUSED — NOT starting eval"; exit 1; }
  sleep 180
done
note "adapt complete — rendering B3 lesson eval guidance"
PYTHONPATH="$PROJ/EvoFSM-RL:$SKY" "$SKY/.venv/bin/python" "$HARNESS/gen_b3_lessons_guidance.py" \
  --run-dir "$SKY/tmp_training/mw_b3_lesson_r1" --out "$GUID" --all-lessons 2>&1 | tail -3 | tee -a "$LOG"
[ -s "$GUID" ] || { note "guidance render failed — PAUSED"; exit 1; }

curl -s --max-time 5 http://localhost:8001/v1/models | grep -q "Qwen3-VL" || { note "qwen vLLM :8001 down — PAUSED"; exit 1; }
note "starting 5 lq_b3lese containers (backend 6990-6994)"
(cd "$MW" && .venv/bin/mw env run --count 5 --name-prefix lq_b3lese --image mobile_world:reset \
   --backend-start-port 6990 --viewer-start-port 7940 --vnc-start-port 5900 \
   --adb-start-port 5790 --launch-interval 20 --env-file "$ENVFILE") >"$ART/b3lesson_eval_envrun.log" 2>&1 \
  || { note "env run failed — PAUSED"; exit 1; }
deadline=$(( $(date +%s) + 1800 ))
while :; do
  h=$(docker ps --filter name=lq_b3lese_ --format '{{.Status}}' | grep -c '(healthy)')
  [ "$h" -ge 5 ] && { note "5/5 healthy"; break; }
  [ "$(date +%s)" -ge "$deadline" ] && { note "containers not ready — PAUSED"; clean; exit 1; }
  sleep 20
done

note "launching B3-lesson eval (110 tasks, qwen B2 agent + prior+lessons, :8001)"
cd "$MW"
EVOFSM_B2_GUIDANCE="$GUID" .venv/bin/mw eval \
  --agent_type "$HARNESS/qwen3vl_b2_agent.py" \
  --task "$(tr -d '\n' < $PROJ/EvoFSM-MW/configs/teval_tasklist.txt)" \
  --max_round 50 --model_name Qwen3-VL-8B-Instruct --llm_base_url http://localhost:8001/v1 \
  --step_wait_time 3 --max-concurrency 5 \
  --env-name-prefix lq_b3lese --env-image mobile_world:reset \
  --log_file_root traj_logs/b3lesson_qwen3vl8b --enable_mcp --enable_user_interaction \
  >"$ART/b3lesson_eval_stdout.log" 2>&1
d=$MW/traj_logs/b3lesson_qwen3vl8b
n=$(ls "$d"/*/result.txt 2>/dev/null | wc -l); s=$(grep -l "score: 1.0" "$d"/*/result.txt 2>/dev/null | wc -l)
clean
note "B3LESSON EVAL DONE: $n/110, success $s  (vs B2'(qwen) 10 / B1 ~9)"

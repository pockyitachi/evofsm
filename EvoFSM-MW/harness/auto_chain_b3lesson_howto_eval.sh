#!/bin/bash
# Re-eval B3 lessons with a STRONGER "how to use lessons" preamble (selective,
# screenshot-first, ignore-if-not-fitting). Same lessons as B3-all; only the
# usage instruction differs. Tests whether teaching the agent HOW to consume
# lessons removes the over-application losses (e.g. ReadQwen3Paper4 5/5->fail).
# vs B3-all 8 / B2'(qwen) 9.2 / B1 8.2.
set -u
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
LOG=$ART/b3lesson_howto_eval_chain.log
MW=/shared/linqiang/evofsm_project/MobileWorld
PROJ=/shared/linqiang/evofsm_project
HARNESS=$PROJ/EvoFSM-MW/harness
ENVFILE=/shared/linqiang/MobileWorld/.env
GUID=$ART/b3lesson_howto_guidance.json
note() { echo "[b3lesson-howto $(date '+%F %T')] $*" | tee -a "$LOG"; }
clean() { for c in $(docker ps -a --format '{{.Names}}' | grep '^lq_b3lhow'); do docker rm -f "$c" >/dev/null 2>&1; done; }

[ -s "$GUID" ] || { note "howto guidance missing — ABORT"; exit 1; }
curl -s --max-time 5 http://localhost:8001/v1/models | grep -q "Qwen3-VL" || { note "qwen vLLM :8001 down — ABORT"; exit 1; }
note "starting 5 lq_b3lhow containers (backend 7000-7004)"
(cd "$MW" && .venv/bin/mw env run --count 5 --name-prefix lq_b3lhow --image mobile_world:reset \
   --backend-start-port 7000 --viewer-start-port 7950 --vnc-start-port 5910 \
   --adb-start-port 5800 --launch-interval 20 --env-file "$ENVFILE") >"$ART/b3lesson_howto_envrun.log" 2>&1 \
  || { note "env run failed — ABORT"; exit 1; }
deadline=$(( $(date +%s) + 1800 ))
while :; do
  h=$(docker ps --filter name=lq_b3lhow_ --format '{{.Status}}' | grep -c '(healthy)')
  [ "$h" -ge 5 ] && { note "5/5 healthy"; break; }
  [ "$(date +%s)" -ge "$deadline" ] && { note "containers not ready — ABORT"; clean; exit 1; }
  sleep 20
done
note "launching B3-lesson-howto eval (110 tasks, qwen B2 agent + lessons + usage instruction)"
cd "$MW"
EVOFSM_B2_GUIDANCE="$GUID" .venv/bin/mw eval \
  --agent_type "$HARNESS/qwen3vl_b2_agent.py" \
  --task "$(tr -d '\n' < $PROJ/EvoFSM-MW/configs/teval_tasklist.txt)" \
  --max_round 50 --model_name Qwen3-VL-8B-Instruct --llm_base_url http://localhost:8001/v1 \
  --step_wait_time 3 --max-concurrency 5 \
  --env-name-prefix lq_b3lhow --env-image mobile_world:reset \
  --log_file_root traj_logs/b3lesson_howto_qwen3vl8b --enable_mcp --enable_user_interaction \
  >"$ART/b3lesson_howto_eval_stdout.log" 2>&1
d=$MW/traj_logs/b3lesson_howto_qwen3vl8b
n=$(ls "$d"/*/result.txt 2>/dev/null | wc -l); s=$(grep -l "score: 1.0" "$d"/*/result.txt 2>/dev/null | wc -l)
clean
note "B3LESSON-HOWTO EVAL DONE: $n/110, success $s  (vs B3-all 8 / B2' 9.2 / B1 8.2)"

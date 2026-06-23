#!/bin/bash
# B3 lessons-ONLY x3: inject only the adapt-distilled lessons (drop the verbose
# B2' prior). The lesson IS the prior's behaviour-grounded compression, so we
# inject the distilled form, not the bloat (median ~770 chars vs ~16k). 3 reps,
# fresh containers each rep, report mean vs B2'(qwen) 9.2 / B1 8.2 / B3-FSM 10.
set -u
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
LOG=$ART/b3lesson_only_eval_chain.log
MW=/shared/linqiang/evofsm_project/MobileWorld
PROJ=/shared/linqiang/evofsm_project
SKY=$PROJ/SkyRL-AndroidWorld/skyrl-agent
HARNESS=$PROJ/EvoFSM-MW/harness
ENVFILE=/shared/linqiang/MobileWorld/.env
GUID=$ART/b3lesson_only_guidance.json
note() { echo "[b3lesson-only $(date '+%F %T')] $*" | tee -a "$LOG"; }
clean() { for c in $(docker ps -a --format '{{.Names}}' | grep '^lq_b3lo'); do docker rm -f "$c" >/dev/null 2>&1; done; }

note "render lessons-only guidance"
PYTHONPATH="$PROJ/EvoFSM-RL:$SKY" "$SKY/.venv/bin/python" "$HARNESS/gen_b3_lessons_guidance.py" \
  --run-dir "$SKY/tmp_training/mw_b3_lesson_r1" --out "$GUID" --all-lessons --lessons-only 2>&1 | tail -2 | tee -a "$LOG"
[ -s "$GUID" ] || { note "guidance render failed — ABORT"; exit 1; }
curl -s --max-time 5 http://localhost:8001/v1/models | grep -q "Qwen3-VL" || { note "qwen vLLM :8001 down — ABORT"; exit 1; }

declare -a RES
for r in 1 2 3; do
  note "=== rep $r/3: starting 5 fresh lq_b3lo containers (backend 7010-7014) ==="
  clean
  (cd "$MW" && .venv/bin/mw env run --count 5 --name-prefix lq_b3lo --image mobile_world:reset \
     --backend-start-port 7010 --viewer-start-port 7960 --vnc-start-port 5920 \
     --adb-start-port 5810 --launch-interval 20 --env-file "$ENVFILE") >"$ART/b3lesson_only_envrun_r$r.log" 2>&1 \
    || { note "rep $r env run failed — skip"; RES[$r]="FAIL"; continue; }
  deadline=$(( $(date +%s) + 1800 )); ok=0
  while :; do
    h=$(docker ps --filter name=lq_b3lo_ --format '{{.Status}}' | grep -c '(healthy)')
    [ "$h" -ge 5 ] && { ok=1; break; }
    [ "$(date +%s)" -ge "$deadline" ] && break
    sleep 20
  done
  [ "$ok" = 1 ] || { note "rep $r containers not ready — skip"; RES[$r]="FAIL"; clean; continue; }
  note "rep $r: 5/5 healthy — launching eval"
  cd "$MW"
  EVOFSM_B2_GUIDANCE="$GUID" .venv/bin/mw eval \
    --agent_type "$HARNESS/qwen3vl_b2_agent.py" \
    --task "$(tr -d '\n' < $PROJ/EvoFSM-MW/configs/teval_tasklist.txt)" \
    --max_round 50 --model_name Qwen3-VL-8B-Instruct --llm_base_url http://localhost:8001/v1 \
    --step_wait_time 3 --max-concurrency 5 \
    --env-name-prefix lq_b3lo --env-image mobile_world:reset \
    --log_file_root "traj_logs/b3lesson_only_qwen3vl8b_r$r" --enable_mcp --enable_user_interaction \
    >"$ART/b3lesson_only_eval_stdout_r$r.log" 2>&1
  d=$MW/traj_logs/b3lesson_only_qwen3vl8b_r$r
  n=$(ls "$d"/*/result.txt 2>/dev/null | wc -l); s=$(grep -l "score: 1.0" "$d"/*/result.txt 2>/dev/null | wc -l)
  RES[$r]="$s/$n"
  note "rep $r DONE: success $s / $n"
  clean
done
note "B3LESSON-ONLY x3 DONE: r1=${RES[1]:-?} r2=${RES[2]:-?} r3=${RES[3]:-?}  (vs B2' 9.2 / B1 8.2 / B3-FSM 10)"

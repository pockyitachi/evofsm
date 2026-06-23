#!/bin/bash
# Tonight: (1) lessons-only reps 4-5 (top up to 5-run mean), then
#          (2) fsm+lessons reps 1-5 (B3-FSM champion + distilled lessons — tests
#              whether the two complementary knowledge sets combine past 10).
# 7 eval reps total, SEQUENTIAL, fresh containers each, qwen3-VL-8B @ :8001.
set -u
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
LOG=$ART/tonight_chain.log
MW=/shared/linqiang/evofsm_project/MobileWorld
PROJ=/shared/linqiang/evofsm_project
HARNESS=$PROJ/EvoFSM-MW/harness
ENVFILE=/shared/linqiang/MobileWorld/.env
note() { echo "[tonight $(date '+%F %T')] $*" | tee -a "$LOG"; }

# eval_rep <guidance> <prefix> <bport> <vport> <nport> <aport> <trajroot> <rep>
eval_rep() {
  local guid=$1 prefix=$2 bp=$3 vp=$4 np=$5 ap=$6 troot=$7 rep=$8
  for c in $(docker ps -a --format '{{.Names}}' | grep "^$prefix"); do docker rm -f "$c" >/dev/null 2>&1; done
  curl -s --max-time 5 http://localhost:8001/v1/models | grep -q Qwen3-VL || { note "qwen :8001 down — ABORT chain"; exit 1; }
  note "$troot rep $rep: starting 5 fresh $prefix containers (backend $bp)"
  (cd "$MW" && .venv/bin/mw env run --count 5 --name-prefix "$prefix" --image mobile_world:reset \
     --backend-start-port "$bp" --viewer-start-port "$vp" --vnc-start-port "$np" \
     --adb-start-port "$ap" --launch-interval 20 --env-file "$ENVFILE") >"$ART/tonight_envrun.log" 2>&1 \
    || { note "$troot rep $rep env run failed — skip"; return; }
  local dl=$(( $(date +%s) + 1800 )) ok=0
  while :; do
    h=$(docker ps --filter name=${prefix}_ --format '{{.Status}}' | grep -c '(healthy)')
    [ "$h" -ge 5 ] && { ok=1; break; }
    [ "$(date +%s)" -ge "$dl" ] && break
    sleep 20
  done
  [ "$ok" = 1 ] || { note "$troot rep $rep containers not ready — skip"; for c in $(docker ps -a --format '{{.Names}}'|grep "^$prefix"); do docker rm -f "$c">/dev/null 2>&1; done; return; }
  note "$troot rep $rep: 5/5 healthy — launching eval"
  cd "$MW"
  EVOFSM_B2_GUIDANCE="$guid" .venv/bin/mw eval --agent_type "$HARNESS/qwen3vl_b2_agent.py" \
    --task "$(tr -d '\n' < $PROJ/EvoFSM-MW/configs/teval_tasklist.txt)" \
    --max_round 50 --model_name Qwen3-VL-8B-Instruct --llm_base_url http://localhost:8001/v1 \
    --step_wait_time 3 --max-concurrency 5 --env-name-prefix "$prefix" --env-image mobile_world:reset \
    --log_file_root "traj_logs/${troot}_r${rep}" --enable_mcp --enable_user_interaction \
    >"$ART/tonight_eval_${troot}_r${rep}.log" 2>&1
  local d=$MW/traj_logs/${troot}_r${rep}
  local n=$(ls "$d"/*/result.txt 2>/dev/null|wc -l) s=$(grep -l "score: 1.0" "$d"/*/result.txt 2>/dev/null|wc -l)
  note "$troot rep $rep DONE: success $s / $n"
  for c in $(docker ps -a --format '{{.Names}}'|grep "^$prefix"); do docker rm -f "$c">/dev/null 2>&1; done
}

note "ARMED: lessons-only r4-5 + fsm+lessons r1-5 (7 reps, sequential)"
# Part 1 — lessons-only top-up (already have r1-r3 = 10/10/10)
for r in 4 5; do
  eval_rep "$ART/b3lesson_only_guidance.json" lq_b3lo 7010 7960 5920 5810 b3lesson_only_qwen3vl8b "$r"
done
# Part 2 — fsm + lessons-only merge
for r in 1 2 3 4 5; do
  eval_rep "$ART/b3fsm_lesson_guidance.json" lq_b3fl 7020 7970 5930 5820 b3fsm_lesson_qwen3vl8b "$r"
done
note "TONIGHT DONE — lessons-only r4-5 + fsm+lessons r1-5 complete"

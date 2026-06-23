#!/bin/bash
# MAI lesson eval: lessons-only x5 + fsm+lessons x5 on MAI-UI-8B @ :8001 (GPU7).
# Mirrors auto_chain_b3mai_eval.sh (MAI agent, guidance via EVOFSM_B2_GUIDANCE)
# but loops 2 guidance files x 5 reps, fresh containers per rep. Resumable: a rep
# dir with >=105 results is skipped. Kills only linqiang-owned lq_mle containers.
# Compare: MAI B1 26.2 / B2' 26.4 (both x5 means) / B3-FSM 27 (n=1).
# tmux b3mailessoneval.
set -u
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
LOG=$ART/b3mai_lesson_eval_chain.log
MW=/shared/linqiang/evofsm_project/MobileWorld
PROJ=/shared/linqiang/evofsm_project
HARNESS=$PROJ/EvoFSM-MW/harness
ENVFILE=/shared/linqiang/MobileWorld/.env
PREFIX=lq_mle
note() { echo "[mai-lesson-eval $(date '+%F %T')] $*" | tee -a "$LOG"; }
count() { ls "$MW/traj_logs/$1"/*/result.txt 2>/dev/null | wc -l; }
succ()  { grep -l "score: 1.0" "$MW/traj_logs/$1"/*/result.txt 2>/dev/null | wc -l; }
api_has(){ curl -s --max-time 5 http://localhost:8001/v1/models 2>/dev/null | grep -q "MAI-UI-8B"; }
clean() { for c in $(docker ps -a --format '{{.Names}}' | grep "^${PREFIX}"); do docker rm -f "$c" >/dev/null 2>&1; done; }

bring_up() {
  note "  starting 5 ${PREFIX} containers (backend 6980-6984)"
  (cd "$MW" && .venv/bin/mw env run --count 5 --name-prefix "$PREFIX" --image mobile_world:reset \
     --backend-start-port 6980 --viewer-start-port 7930 --vnc-start-port 5890 \
     --adb-start-port 5780 --launch-interval 20 --env-file "$ENVFILE") >"$ART/b3mai_lesson_eval_envrun.log" 2>&1 \
    || { note "  env run FAILED"; return 1; }
  local deadline h
  deadline=$(( $(date +%s) + 1800 ))
  while :; do
    h=$(docker ps --filter "name=${PREFIX}_" --format '{{.Status}}' | grep -c '(healthy)')
    [ "$h" -ge 5 ] && { note "  5/5 healthy"; return 0; }
    [ "$(date +%s)" -ge "$deadline" ] && { note "  containers not ready in 30min"; return 1; }
    sleep 20
  done
}

# run_one <guidance> <troot> <rep>
run_one() {
  local guid=$1 troot=$2 rep=$3 dir="${2}_r${3}"
  api_has || { note "  MAI vLLM :8001 down before ${dir}"; return 1; }
  clean; sleep 10
  bring_up || return 1
  note "  eval ${dir} -> traj_logs/${dir}"
  cd "$MW"
  EVOFSM_B2_GUIDANCE="$guid" .venv/bin/mw eval \
    --agent_type "$HARNESS/mai_ui_b2_agent.py" \
    --task "$(tr -d '\n' < $PROJ/EvoFSM-MW/configs/teval_tasklist.txt)" \
    --max_round 50 --model_name MAI-UI-8B --llm_base_url http://localhost:8001/v1 \
    --step_wait_time 3 --max-concurrency 5 \
    --env-name-prefix "$PREFIX" --env-image mobile_world:reset \
    --log_file_root "traj_logs/${dir}" --enable_mcp --enable_user_interaction \
    >"$ART/b3mai_lesson_eval_stdout_${dir}.log" 2>&1
  note "  ${dir} done: $(count "$dir")/110, success $(succ "$dir")"
  [ "$(count "$dir")" -ge 105 ]
}

note "ARMED: MAI lessons-only x5 + fsm+lessons x5 (MAI-UI-8B @ :8001, ${PREFIX} 6980-6984)"
api_has || { note "MAI vLLM :8001 not serving — abort"; exit 1; }
[ -s "$ART/b3mai_lesson_only_guidance.json" ] || { note "lessons-only guidance missing — abort"; exit 1; }
[ -s "$ART/b3mai_fsm_lesson_guidance.json" ]  || { note "fsm+lessons guidance missing — abort"; exit 1; }

# (guidance, troot) pairs
run_config() {  # $1=guidance $2=troot
  local guid=$1 troot=$2 r dir
  for r in 1 2 3 4 5; do
    dir="${troot}_r${r}"
    if [ "$(count "$dir")" -ge 105 ]; then
      note "=== ${dir} already complete ($(count "$dir")/110, success $(succ "$dir")) — skip ==="
      continue
    fi
    note "=== run ${dir} ==="
    if ! run_one "$guid" "$troot" "$r"; then
      note "  ${dir} FAILED — retry once"
      mv "$MW/traj_logs/$dir" "$MW/traj_logs/${dir}_failed1" 2>/dev/null
      run_one "$guid" "$troot" "$r" || { note "  ${dir} failed TWICE — PAUSED"; exit 1; }
    fi
  done
}

run_config "$ART/b3mai_lesson_only_guidance.json" b3mai_lessononly
run_config "$ART/b3mai_fsm_lesson_guidance.json"  b3mai_fsmlesson
clean
lo=$(for r in 1 2 3 4 5; do printf '%s ' "$(succ b3mai_lessononly_r$r)"; done)
fl=$(for r in 1 2 3 4 5; do printf '%s ' "$(succ b3mai_fsmlesson_r$r)"; done)
note "MAI LESSON EVAL COMPLETE — lessons-only: $lo | fsm+lessons: $fl  (vs MAI B1 26.2 / B2' 26.4 / B3-FSM 27)"

#!/bin/bash
# Off-chain B1 eval of the MAI-UI-8B Phase-1 LoRA step-175 (pi^pre^MAI), merged
# into HF and served as MAI-UI-8B on GPU7 :8001. Stock MAI agent, ZERO injection
# (= B1), x5 reps on MobileWorld. Mirrors eval_pi200_b1.sh. Resumable (rep with
# >=105 results skipped). Fresh containers per rep. Kills only linqiang-owned
# lq_e175 containers. Compare: MAI base B1 26.2 / B2' 26.4.
set -u
MW=/shared/linqiang/evofsm_project/MobileWorld
TRAJ=$MW/traj_logs
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
PROJ=/shared/linqiang/evofsm_project
ENVFILE=/shared/linqiang/MobileWorld/.env
ELOG=$ART/mai_pre175_eval_logs
LOG=$ART/mai_pre175_b1.log
BASE_URL=http://localhost:8001/v1
PREFIX=lq_e175
mkdir -p "$ELOG"

note()  { echo "[maipre175 $(date '+%F %T')] $*" | tee -a "$LOG"; }
count() { ls "$TRAJ/$1"/*/result.txt 2>/dev/null | wc -l; }
succ()  { grep -l "score: 1.0" "$TRAJ/$1"/*/result.txt 2>/dev/null | wc -l; }
api_has(){ curl -s --max-time 5 "$BASE_URL/models" 2>/dev/null | grep -q "MAI-UI-8B"; }

teardown() {
  for c in $(docker ps -a --format '{{.Names}}' | grep "^${PREFIX}_"); do
    docker rm -f "$c" >/dev/null 2>&1 && note "  removed $c"
  done
}

bring_up() {
  note "  starting 5 ${PREFIX} containers (backend 6910-6914)"
  (cd "$MW" && .venv/bin/mw env run --count 5 --name-prefix "$PREFIX" --image mobile_world:reset \
      --backend-start-port 6910 --viewer-start-port 7945 --vnc-start-port 5905 \
      --adb-start-port 5795 --launch-interval 20 --env-file "$ENVFILE") \
      >"$ELOG/envrun_last.log" 2>&1 || { note "  mw env run FAILED"; return 1; }
  local deadline h
  deadline=$(( $(date +%s) + 1800 ))
  while :; do
    h=$(docker ps --filter "name=${PREFIX}_" --format '{{.Status}}' | grep -c '(healthy)')
    [ "$h" -ge 5 ] && { note "  ${PREFIX}: 5/5 healthy"; return 0; }
    [ "$(date +%s)" -ge "$deadline" ] && { note "  ${PREFIX} not ready in 30min"; return 1; }
    sleep 20
  done
}

run_one() {  # $1=rep
  local rep=$1 dir="maib1pre175_qwen3vl8b_r${rep}"
  api_has || { note "  step-175 vLLM (MAI-UI-8B) down before r${rep}"; return 1; }
  teardown; sleep 10
  bring_up || return 1
  note "  eval maib1pre175_r${rep} -> traj_logs/$dir"
  cd "$MW"
  .venv/bin/mw eval --agent_type mai_ui_agent \
    --task "$(tr -d '\n' < $PROJ/EvoFSM-MW/configs/teval_tasklist.txt)" \
    --max_round 50 --model_name MAI-UI-8B --llm_base_url "$BASE_URL" \
    --step_wait_time 3 --max-concurrency 5 \
    --env-name-prefix "$PREFIX" --env-image mobile_world:reset \
    --log_file_root "traj_logs/$dir" --enable_mcp --enable_user_interaction \
    >"$ELOG/maib1pre175_r${rep}.log" 2>&1
  note "  maib1pre175_r${rep} done: $(count "$dir")/110, success $(succ "$dir")"
  [ "$(count "$dir")" -ge 105 ]
}

note "ARMED: MAI step-175 (pi^pre) B1 x5 on GPU7:8001, containers ${PREFIX} 6910-6914"
api_has || { note "step-175 MAI not serving on :8001 — abort"; exit 1; }

for rep in 1 2 3 4 5; do
  dir="maib1pre175_qwen3vl8b_r${rep}"
  if [ "$(count "$dir")" -ge 105 ]; then
    note "=== r${rep} already complete ($(count "$dir")/110, success $(succ "$dir")) — skip ==="
    continue
  fi
  note "=== run r${rep} ==="
  if ! run_one "$rep"; then
    note "  r${rep} FAILED — retry once"
    mv "$TRAJ/$dir" "$TRAJ/${dir}_failed1" 2>/dev/null
    run_one "$rep" || { note "  r${rep} failed TWICE — PAUSED"; exit 1; }
  fi
done

teardown
note "MAI PRE-175 B1 COMPLETE — per-rep success: $(for r in 1 2 3 4 5; do printf 'r%s=%s ' "$r" "$(succ maib1pre175_qwen3vl8b_r$r)"; done) (vs MAI base B1 26.2 / B2' 26.4)"

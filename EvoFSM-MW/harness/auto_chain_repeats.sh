#!/bin/bash
# 5x variance study master chain — 16 runs, rep-major order:
#   r2: B1 B2 B2' B2''   r3: ...   r4: ...   r5: ...
# Every run gets 5 FRESH lq_rep containers (backend 6840-44, viewer 7900-04,
# vnc 5860-64, adb 5750-54); containers torn down between runs and at the end.
# A run that yields <105 results is retried ONCE (failed dir moved aside,
# fresh containers); a second failure pauses the chain.
# Runs in tmux session `brepeats`. Eval stdout: artifacts/rep_eval_logs/.
set -u
MW=/shared/linqiang/evofsm_project/MobileWorld
TRAJ=$MW/traj_logs
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
LOG=$ART/repeats_chain.log
HARNESS=/shared/linqiang/evofsm_project/EvoFSM-MW/harness
ENVFILE=/shared/linqiang/MobileWorld/.env
ELOG=$ART/rep_eval_logs
mkdir -p "$ELOG"

note() { echo "[rep-chain $(date '+%F %T')] $*" | tee -a "$LOG"; }
count() { ls "$TRAJ/$1"/*/result.txt 2>/dev/null | wc -l; }
succ()  { grep -l "score: 1.0" "$TRAJ/$1"/*/result.txt 2>/dev/null | wc -l; }
vllm_ok() { curl -s --max-time 5 http://localhost:8001/v1/models | grep -q "Qwen3-VL"; }

teardown() {
  for c in $(docker ps -a --format '{{.Names}}' | grep "^lq_rep_"); do
    docker rm -f "$c" >/dev/null 2>&1 && note "  removed $c"
  done
}

bring_up() {
  note "  starting 5 lq_rep containers (backend 6840-6844)"
  (cd "$MW" && .venv/bin/mw env run --count 5 --name-prefix lq_rep --image mobile_world:reset \
      --backend-start-port 6840 --viewer-start-port 7900 --vnc-start-port 5860 \
      --adb-start-port 5750 --launch-interval 20 --env-file "$ENVFILE") \
      >"$ELOG/envrun_last.log" 2>&1 \
    || { note "  mw env run FAILED (see rep_eval_logs/envrun_last.log)"; return 1; }
  local deadline=$(( $(date +%s) + 1800 )) h ok p
  while :; do
    h=$(docker ps --filter "name=lq_rep_" --format '{{.Status}}' | grep -c '(healthy)')
    ok=0
    for p in 6840 6841 6842 6843 6844; do
      curl -sf --max-time 3 "http://localhost:$p/health" >/dev/null 2>&1 && ok=$((ok+1))
    done
    if [ "$h" -ge 5 ] && [ "$ok" -ge 5 ]; then note "  lq_rep: 5/5 healthy"; return 0; fi
    if [ "$(date +%s)" -ge "$deadline" ]; then note "  lq_rep not ready after 30 min (healthy=$h ok=$ok)"; return 1; fi
    sleep 20
  done
}

run_one() {  # $1=config $2=rep ; returns 0 on >=105 results
  local cfg=$1 rep=$2 dir
  dir="${cfg}_qwen3vl8b_r${rep}"
  teardown
  sleep 10
  bring_up || return 1
  note "  eval ${cfg}_r${rep} -> traj_logs/$dir"
  bash "$HARNESS/launch_rep.sh" "$cfg" "$rep" >"$ELOG/${cfg}_r${rep}.log" 2>&1
  local n s
  n=$(count "$dir"); s=$(succ "$dir")
  note "  ${cfg}_r${rep} done: $n/110, success $s"
  [ "$n" -ge 105 ]
}

note "ARMED: 16 runs, rep-major r2..r5 x [b1 b2 b2p b2pp], 5 containers/run, concurrency 5"
vllm_ok || { note "vLLM :8001 down at start — PAUSED"; exit 1; }

for rep in 2 3 4 5; do
  for cfg in b1 b2 b2p b2pp; do
    vllm_ok || { note "vLLM :8001 down before ${cfg}_r${rep} — PAUSED"; exit 1; }
    note "=== run ${cfg}_r${rep} (round $rep) ==="
    if ! run_one "$cfg" "$rep"; then
      note "  ${cfg}_r${rep} FAILED — moving dir aside, retrying once with fresh containers"
      mv "$TRAJ/${cfg}_qwen3vl8b_r${rep}" "$TRAJ/${cfg}_qwen3vl8b_r${rep}_failed1" 2>/dev/null
      if ! run_one "$cfg" "$rep"; then
        note "  ${cfg}_r${rep} failed TWICE — chain PAUSED. Inspect rep_eval_logs/${cfg}_r${rep}.log"
        exit 1
      fi
    fi
  done
  note "=== round $rep complete: B1=$(succ b1_qwen3vl8b_r$rep) B2=$(succ b2_qwen3vl8b_r$rep) B2p=$(succ b2p_qwen3vl8b_r$rep) B2pp=$(succ b2pp_qwen3vl8b_r$rep) ==="
done

teardown
note "REPEATS COMPLETE — all 16 runs done; lq_rep containers removed."

#!/bin/bash
# B2''' tail chain — waits for the 16-run variance chain to COMPLETE, then
# runs b2ppp (entry-level retrieval guidance) 5 times: r1..r5, fresh lq_rep
# containers per run, retry-once policy. Runs in tmux session b2pppchain.
# If the variance chain PAUSES or dies without completing, this chain logs
# and exits WITHOUT running (human decides).
set -u
MW=/shared/linqiang/evofsm_project/MobileWorld
TRAJ=$MW/traj_logs
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
REPLOG=$ART/repeats_chain.log
LOG=$ART/b2ppp_chain.log
HARNESS=/shared/linqiang/evofsm_project/EvoFSM-MW/harness
ENVFILE=/shared/linqiang/MobileWorld/.env
ELOG=$ART/rep_eval_logs
mkdir -p "$ELOG"

note() { echo "[b2ppp-chain $(date '+%F %T')] $*" | tee -a "$LOG"; }
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
    || { note "  mw env run FAILED"; return 1; }
  local deadline=$(( $(date +%s) + 1800 )) h ok p
  while :; do
    h=$(docker ps --filter "name=lq_rep_" --format '{{.Status}}' | grep -c '(healthy)')
    ok=0
    for p in 6840 6841 6842 6843 6844; do
      curl -sf --max-time 3 "http://localhost:$p/health" >/dev/null 2>&1 && ok=$((ok+1))
    done
    if [ "$h" -ge 5 ] && [ "$ok" -ge 5 ]; then note "  lq_rep: 5/5 healthy"; return 0; fi
    if [ "$(date +%s)" -ge "$deadline" ]; then note "  lq_rep not ready after 30 min"; return 1; fi
    sleep 20
  done
}

run_one() {  # $1=rep
  local rep=$1 dir="b2ppp_qwen3vl8b_r${rep}"
  teardown
  sleep 10
  bring_up || return 1
  note "  eval b2ppp_r${rep} -> traj_logs/$dir"
  bash "$HARNESS/launch_rep.sh" b2ppp "$rep" >"$ELOG/b2ppp_r${rep}.log" 2>&1
  local n s
  n=$(count "$dir"); s=$(succ "$dir")
  note "  b2ppp_r${rep} done: $n/110, success $s"
  [ "$n" -ge 105 ]
}

note "ARMED: waiting for the 16-run variance chain to complete, then b2ppp x5"
while :; do
  grep -q "REPEATS COMPLETE" "$REPLOG" 2>/dev/null && break
  if grep -q "PAUSED" "$REPLOG" 2>/dev/null; then
    note "variance chain PAUSED — NOT starting b2ppp. Resolve and rerun this chain."
    exit 1
  fi
  if ! pgrep -f "auto_chain_repeats.s[h]" >/dev/null 2>&1; then
    sleep 120  # grace, then re-check the terminal lines
    grep -q "REPEATS COMPLETE" "$REPLOG" 2>/dev/null && break
    note "variance chain process gone without COMPLETE — NOT starting b2ppp."
    exit 1
  fi
  sleep 120
done
note "variance chain complete — starting b2ppp x5"

for rep in 1 2 3 4 5; do
  vllm_ok || { note "vLLM down before b2ppp_r${rep} — PAUSED"; exit 1; }
  note "=== run b2ppp_r${rep} ==="
  if ! run_one "$rep"; then
    note "  b2ppp_r${rep} FAILED — retrying once"
    mv "$TRAJ/b2ppp_qwen3vl8b_r${rep}" "$TRAJ/b2ppp_qwen3vl8b_r${rep}_failed1" 2>/dev/null
    if ! run_one "$rep"; then
      note "  b2ppp_r${rep} failed TWICE — chain PAUSED."
      exit 1
    fi
  fi
done

teardown
note "B2PPP COMPLETE — 5 runs: $(for r in 1 2 3 4 5; do printf 'r%s=%s ' "$r" "$(succ b2ppp_qwen3vl8b_r$r)"; done)"

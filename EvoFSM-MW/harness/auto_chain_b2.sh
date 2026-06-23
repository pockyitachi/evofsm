#!/bin/bash
# B1→B2 auto-chain watcher v3 (fresh-container edition). Runs in tmux session
# b2eval so it survives SSH drops and does NOT depend on any Claude session.
#   1. wait for the B1 `mw eval` process (PID baked in below) to exit
#   2. instant gates: B1 results >= 105/110, vLLM :8001 healthy
#   3. wait-gate: >= 2/3 (target 3/3) FRESH lq_b2 backends healthy on
#      6810-6812, up to 30 min (containers are brought up in advance; this
#      tolerates them still booting when B1 ends)
#   4. settle 30s, then exec launch_b2.sh (flock'd — double launch is harmless)
# lq_b1 containers are left untouched (different ports); tear down manually
# with `docker rm -f lq_b1_0 lq_b1_1 lq_b1_2` when no longer needed.
set -u
B1_PID=30475
RES=/shared/linqiang/evofsm_project/MobileWorld/traj_logs/b1_qwen3vl8b
LOG=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts/b2_autochain.log
HARNESS=/shared/linqiang/evofsm_project/EvoFSM-MW/harness
B2_PORTS="6810 6811 6812"

note() { echo "[autochain $(date '+%F %T')] $*" | tee -a "$LOG"; }

count_healthy() {
  local h=0 port
  for port in $B2_PORTS; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://localhost:$port/health" || true)" = "200" ] && h=$((h + 1))
  done
  echo "$h"
}

# sanity: is the baked-in PID really the B1 eval? (guards against PID reuse)
if kill -0 "$B1_PID" 2>/dev/null && ! tr '\0' ' ' < "/proc/$B1_PID/cmdline" | grep -q "b1_qwen3vl8b"; then
  note "PID $B1_PID is not the B1 eval anymore — aborting, no launch"
  exit 1
fi

note "v3 armed: fresh lq_b2 backends on $B2_PORTS; watching B1 (pid $B1_PID), $(ls "$RES"/*/result.txt 2>/dev/null | wc -l)/110 done"
while kill -0 "$B1_PID" 2>/dev/null; do sleep 60; done

n=$(ls "$RES"/*/result.txt 2>/dev/null | wc -l)
s=$(grep -l "score: 1.0" "$RES"/*/result.txt 2>/dev/null | wc -l)
note "B1 process exited. results: $n/110, success: $s"

if [ "$n" -lt 105 ]; then
  note "B1 looks incomplete (<105) — NOT launching B2. Resume B1 or launch manually."
  exit 1
fi
if ! curl -s --max-time 5 http://localhost:8001/v1/models | grep -q "Qwen3-VL"; then
  note "vLLM on :8001 not healthy — NOT launching B2."
  exit 1
fi

deadline=$(( $(date +%s) + 1800 ))
while :; do
  h=$(count_healthy)
  [ "$h" -ge 3 ] && break
  if [ "$(date +%s)" -ge "$deadline" ]; then
    if [ "$h" -ge 2 ]; then
      note "deadline hit with $h/3 lq_b2 backends healthy — proceeding degraded"
      break
    fi
    note "only $h/3 lq_b2 backends healthy after 30 min — NOT launching B2."
    exit 1
  fi
  note "waiting for lq_b2 backends: $h/3 healthy..."
  sleep 60
done

note "gates passed ($(count_healthy)/3 lq_b2 backends), settling 30s..."
sleep 30
note "launching B2 on fresh lq_b2 (logs: this tmux pane + traj_logs/b2_qwen3vl8b)"
exec bash "$HARNESS/launch_b2.sh"

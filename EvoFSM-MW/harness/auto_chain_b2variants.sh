#!/bin/bash
# Overnight master chain: B2 → B2' → B2''. Runs in tmux session b2variants,
# independent of SSH / Claude sessions. Every eval gets FRESH containers
# (MobileWorld containers do NOT reset app state — never reuse across runs).
#   B2'  (b2p):  app Layer-2 only      backend 6820-22, adb 5730-32
#   B2'' (b2pp): category L_C only     backend 6830-32, adb 5740-42
# Eval stdout goes to artifacts/b2{p,pp}_eval_stdout.log; this log stays terse.
set -u
B2_PID=261237
MW=/shared/linqiang/evofsm_project/MobileWorld
TRAJ=$MW/traj_logs
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
LOG=$ART/b2_variants_chain.log
HARNESS=/shared/linqiang/evofsm_project/EvoFSM-MW/harness
ENVFILE=/shared/linqiang/MobileWorld/.env

note() { echo "[chain $(date '+%F %T')] $*" | tee -a "$LOG"; }
count() { ls "$TRAJ/$1"/*/result.txt 2>/dev/null | wc -l; }
succ()  { grep -l "score: 1.0" "$TRAJ/$1"/*/result.txt 2>/dev/null | wc -l; }
vllm_ok() { curl -s --max-time 5 http://localhost:8001/v1/models | grep -q "Qwen3-VL"; }

teardown() {  # $1 = exact container prefix incl. trailing underscore
  for c in $(docker ps -a --format '{{.Names}}' | grep "^$1"); do
    docker rm -f "$c" >/dev/null 2>&1 && note "  removed $c"
  done
}

bring_up() {  # $1=name-prefix $2=backend $3=viewer $4=vnc $5=adb
  note "starting 3 $1 containers (backend $2-$(($2+2)), adb $5-$(($5+2)))"
  (cd "$MW" && .venv/bin/mw env run --count 3 --name-prefix "$1" --image mobile_world:reset \
      --backend-start-port "$2" --viewer-start-port "$3" --vnc-start-port "$4" \
      --adb-start-port "$5" --launch-interval 20 --env-file "$ENVFILE") \
      >"$ART/${1}_envrun.log" 2>&1 \
    || { note "mw env run FAILED for $1 (see ${1}_envrun.log)"; return 1; }
  local deadline=$(( $(date +%s) + 1800 )) h ok p
  while :; do
    h=$(docker ps --filter "name=${1}_" --format '{{.Status}}' | grep -c '(healthy)')
    ok=0
    for p in "$2" "$(($2+1))" "$(($2+2))"; do
      curl -sf --max-time 3 "http://localhost:$p/health" >/dev/null 2>&1 && ok=$((ok+1))
    done
    if [ "$h" -ge 3 ] && [ "$ok" -ge 3 ]; then note "$1: 3/3 healthy + /health 200"; return 0; fi
    if [ "$(date +%s)" -ge "$deadline" ]; then note "$1 not ready after 30 min (healthy=$h ok=$ok)"; return 1; fi
    sleep 20
  done
}

# ── Stage 0: wait for the running B2 to finish ──
if kill -0 "$B2_PID" 2>/dev/null && ! tr '\0' ' ' < "/proc/$B2_PID/cmdline" | grep -q "b2_qwen3vl8b"; then
  note "PID $B2_PID is not the B2 eval — aborting chain"; exit 1
fi
note "armed: watching B2 (pid $B2_PID), $(count b2_qwen3vl8b)/110 done; plan = B2' (app-L2) then B2'' (category-L_C), fresh containers each"
while kill -0 "$B2_PID" 2>/dev/null; do sleep 60; done
note "B2 exited: $(count b2_qwen3vl8b)/110, success $(succ b2_qwen3vl8b)"
if [ "$(count b2_qwen3vl8b)" -lt 105 ]; then note "B2 incomplete (<105) — aborting chain"; exit 1; fi
vllm_ok || { note "vLLM :8001 down — aborting chain"; exit 1; }

# ── Stage 1: B2' (app Layer-2 only) ──
teardown lq_b2_
sleep 15
bring_up lq_b2p 6820 7880 5820 5730 || { note "B2' env failed — aborting chain"; exit 1; }
note "launching B2' (app Layer-2 only) -> traj_logs/b2p_qwen3vl8b"
bash "$HARNESS/launch_b2p.sh" >"$ART/b2p_eval_stdout.log" 2>&1
note "B2' finished: $(count b2p_qwen3vl8b)/110, success $(succ b2p_qwen3vl8b)"
if [ "$(count b2p_qwen3vl8b)" -lt 105 ]; then note "B2' incomplete (<105) — NOT starting B2''"; exit 1; fi
vllm_ok || { note "vLLM :8001 down after B2' — aborting chain"; exit 1; }

# ── Stage 2: B2'' (category L_C only, original recipe) ──
teardown lq_b2p_
sleep 15
bring_up lq_b2pp 6830 7890 5830 5740 || { note "B2'' env failed — aborting chain"; exit 1; }
note "launching B2'' (category L_C only) -> traj_logs/b2pp_qwen3vl8b"
bash "$HARNESS/launch_b2pp.sh" >"$ART/b2pp_eval_stdout.log" 2>&1
note "B2'' finished: $(count b2pp_qwen3vl8b)/110, success $(succ b2pp_qwen3vl8b)"

note "CHAIN COMPLETE — success counts: B1=9 B2=$(succ b2_qwen3vl8b) B2'=$(succ b2p_qwen3vl8b) B2''=$(succ b2pp_qwen3vl8b) (each /110). lq_b2pp containers left up for inspection."

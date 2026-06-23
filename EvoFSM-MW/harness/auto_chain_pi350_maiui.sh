#!/bin/bash
# Final tail chain v2 (resumable). Stages:
#   P — π^pre step-350 vLLM on GPU0 :8001 (idempotent) → b1pi350 × 5
#   M — MAI-UI-8B vLLM on GPU7 :8002 (pre-deployed 2026-06-11; launched here
#       only if missing) → maib1 × 5 → maib2v × 5 (winner: strongest_variant.txt)
# v2 deltas: per-rep RESUME (skip dirs with ≥105 results; wait if the eval
# flock is held by an orphan run), per-stage base_url, no GPU0 swap for MAI.
# Every run: fresh lq_rep containers (MobileWorld has no app-state reset).
# Kill discipline: only linqiang-owned processes, verified via /proc.
set -u
MW=/shared/linqiang/evofsm_project/MobileWorld
TRAJ=$MW/traj_logs
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
LOG=$ART/pi350_maiui_chain.log
HARNESS=/shared/linqiang/evofsm_project/EvoFSM-MW/harness
ENVFILE=/shared/linqiang/MobileWorld/.env
ELOG=$ART/rep_eval_logs
VPY=/shared/linqiang/evofsm_project/SkyRL-AndroidWorld/skyrl-agent/.venv/bin/python
PI350=/shared/linqiang/models/pi_pre_350_hf
mkdir -p "$ELOG"

note() { echo "[pi350mai $(date '+%F %T')] $*" | tee -a "$LOG"; }
count() { ls "$TRAJ/$1"/*/result.txt 2>/dev/null | wc -l; }
succ()  { grep -l "score: 1.0" "$TRAJ/$1"/*/result.txt 2>/dev/null | wc -l; }
api_has() { curl -s --max-time 5 "$1/models" 2>/dev/null | grep -q "$2"; }

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
  local deadline h ok p
  deadline=$(( $(date +%s) + 1800 ))
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

kill_our_8001() {
  local pid pgid
  pid=$(ss -tlnp 2>/dev/null | grep ":8001 " | grep -oP 'pid=\K[0-9]+' | head -1)
  if [ -z "$pid" ]; then note "  :8001 has no listener"; return 0; fi
  if [ "$(stat -c %U /proc/$pid 2>/dev/null)" != "linqiang" ]; then
    note "  :8001 owned by $(stat -c %U /proc/$pid 2>/dev/null) — NOT ours, refusing to kill"; return 1
  fi
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
  note "  killing our vLLM on :8001 (pid $pid)"
  kill -TERM -- "-$pgid" 2>/dev/null || kill -TERM "$pid"
  local deadline
  deadline=$(( $(date +%s) + 120 ))
  while ss -tln 2>/dev/null | grep -q ":8001 "; do
    [ "$(date +%s)" -ge "$deadline" ] && { kill -KILL -- "-$pgid" 2>/dev/null; sleep 5; break; }
    sleep 3
  done
  sleep 5
  note "  :8001 freed"
}

launch_vllm() {  # $1=model $2=served_name $3=extra_env $4=gpu $5=port
  local free
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$4")
  if [ "$free" -lt 40000 ]; then
    note "  GPU$4 only ${free}MiB free (<40G) — cannot launch $2"; return 1
  fi
  note "  launching vLLM: $2 (gpu$4 :$5, util 0.30)"
  env $3 CUDA_VISIBLE_DEVICES="$4" nohup "$VPY" -m vllm.entrypoints.openai.api_server \
      --model "$1" --served-model-name "$2" --dtype bfloat16 \
      --gpu-memory-utilization 0.30 --max-model-len 32768 \
      --port "$5" --host 0.0.0.0 >"$ART/vllm_$2_gpu$4.log" 2>&1 &
  disown
  local deadline
  deadline=$(( $(date +%s) + 1500 ))
  while :; do
    api_has "http://localhost:$5/v1" "$2" && { note "  vLLM $2 healthy"; return 0; }
    [ "$(date +%s)" -ge "$deadline" ] && { note "  vLLM $2 not healthy after 25 min"; return 1; }
    sleep 15
  done
}

lock_free() { ( flock -n 9 ) 9>/tmp/evofsm_rep_eval.lock 2>/dev/null; }

run_one() {  # $1=config $2=rep $3=served_name $4=base_url
  local cfg=$1 rep=$2 dir
  dir="${cfg}_qwen3vl8b_r${rep}"
  api_has "$4" "$3" || { note "  vLLM $3 down before ${cfg}_r${rep}"; return 1; }
  teardown
  sleep 10
  bring_up || return 1
  note "  eval ${cfg}_r${rep} -> traj_logs/$dir"
  bash "$HARNESS/launch_rep.sh" "$cfg" "$rep" >"$ELOG/${cfg}_r${rep}.log" 2>&1
  note "  ${cfg}_r${rep} done: $(count "$dir")/110, success $(succ "$dir")"
  [ "$(count "$dir")" -ge 105 ]
}

run_five() {  # $1=config $2=served_name $3=base_url
  local rep dir
  for rep in 1 2 3 4 5; do
    dir="$1_qwen3vl8b_r${rep}"
    until lock_free; do
      note "  eval lock held (orphan run?) — waiting before $1_r${rep}"
      sleep 120
    done
    if [ "$(count "$dir")" -ge 105 ]; then
      note "=== $1_r${rep} already complete ($(count "$dir")/110, success $(succ "$dir")) — skip ==="
      continue
    fi
    note "=== run $1_r${rep} ==="
    if ! run_one "$1" "$rep" "$2" "$3"; then
      note "  $1_r${rep} FAILED — retrying once"
      mv "$TRAJ/$dir" "$TRAJ/${dir}_failed1" 2>/dev/null
      if ! run_one "$1" "$rep" "$2" "$3"; then
        note "  $1_r${rep} failed TWICE — chain PAUSED."
        return 1
      fi
    fi
  done
}

# ── Stage 0: require the B2''' chain to have completed ──
grep -q "B2PPP COMPLETE" "$ART/b2ppp_chain.log" 2>/dev/null \
  || { note "b2ppp not complete — NOT starting"; exit 1; }
note "ARMED v2: stage P (pi-pre-350 @gpu0:8001) -> stage M (MAI-UI-8B @gpu7:8002)"

# ── Stage P: π^pre step-350 under the B1 setting ──
P_DONE=1
for r in 1 2 3 4 5; do [ "$(count b1pi350_qwen3vl8b_r$r)" -ge 105 ] || P_DONE=0; done
if [ "$P_DONE" = 1 ]; then
  note "stage P already complete (5/5 runs) — skipping, no pi350 vLLM needed"
else
  if api_has http://localhost:8001/v1 "pi-pre-350"; then
    note "pi-pre-350 already serving on :8001 — reusing"
  else
    kill_our_8001 || { note "PAUSED (kill refused)"; exit 1; }
    launch_vllm "$PI350" "pi-pre-350" "" 0 8001 || { note "PAUSED (pi350 vLLM)"; exit 1; }
  fi
  run_five b1pi350 "pi-pre-350" http://localhost:8001/v1 || exit 1
fi
note "stage P complete: $(for r in 1 2 3 4 5; do printf 'r%s=%s ' "$r" "$(succ b1pi350_qwen3vl8b_r$r)"; done)"

# ── Stage M: MAI-UI-8B on GPU7 :8002 (pre-deployed; no GPU0 swap) ──
if api_has http://localhost:8002/v1 "MAI-UI-8B"; then
  note "MAI-UI-8B already serving on gpu7:8002 — reusing"
else
  launch_vllm "Tongyi-MAI/MAI-UI-8B" "MAI-UI-8B" "HF_HOME=/shared/linqiang/hf_home HF_HUB_OFFLINE=1" 7 8002 \
    || { note "PAUSED (MAI vLLM)"; exit 1; }
fi
run_five maib1 "MAI-UI-8B" http://localhost:8002/v1 || exit 1
note "maib1 complete: $(for r in 1 2 3 4 5; do printf 'r%s=%s ' "$r" "$(succ maib1_qwen3vl8b_r$r)"; done)"

if [ ! -s "$ART/strongest_variant.txt" ]; then
  note "strongest_variant.txt missing — PAUSED before maib2v."
  exit 1
fi
note "winner variant: $(cat "$ART/strongest_variant.txt")"
run_five maib2v "MAI-UI-8B" http://localhost:8002/v1 || exit 1
note "maib2v complete: $(for r in 1 2 3 4 5; do printf 'r%s=%s ' "$r" "$(succ maib2v_qwen3vl8b_r$r)"; done)"

teardown
note "PI350MAI COMPLETE — pi350(B1)x5 + MAI-UI b1 x5 + MAI-UI $(cat "$ART/strongest_variant.txt") x5 done. vLLMs left up (gpu0 pi-pre-350, gpu7 MAI-UI-8B)."

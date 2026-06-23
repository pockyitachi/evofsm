#!/bin/bash
# B3 tail chain — waits for PI350MAI COMPLETE (all MAI-UI runs done), then:
#   1. kill OUR MAI vLLM on gpu7:8002  -> GPU7 fully freed (user decision)
#   2. ensure base Qwen3-VL-8B-Instruct vLLM on gpu0:8001 (B3 frozen policy =
#      BASE, user decision 06-12 — same base as the B1/B2' reference lines)
#   3. run scripts/run_b3_mw_tta.py round-1 (51 iters, M=2 x N=4)
#      with --recycle-after 1: FRESH CONTAINER PER EPISODE (MW never resets
#      app state; same-task repeats make reuse extra poisonous; the manager
#      default of 25 episodes/container is explicitly overridden here).
# Container facts verified 2026-06-12: manager boots with --rm --privileged
# --network mwnet, 4 disjoint port bands from port_base (6940..+3x100),
# pre-checks ports, destroys pool in cleanup().
set -u
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
LOG=$ART/b3_chain.log
SKY=/shared/linqiang/evofsm_project/SkyRL-AndroidWorld/skyrl-agent
RUN_DIR=tmp_training/mw_b3_base_r1
BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct
SERVED=Qwen3-VL-8B-Instruct
VPY=$SKY/.venv/bin/python

note() { echo "[b3-chain $(date '+%F %T')] $*" | tee -a "$LOG"; }
api_has() { curl -s --max-time 5 "$1/models" 2>/dev/null | grep -q "$2"; }

kill_our_port() {  # $1=port — only linqiang-owned listeners
  local pid pgid
  pid=$(ss -tlnp 2>/dev/null | grep ":$1 " | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -z "$pid" ] && { note "  :$1 has no listener"; return 0; }
  if [ "$(stat -c %U /proc/$pid 2>/dev/null)" != "linqiang" ]; then
    note "  :$1 owned by $(stat -c %U /proc/$pid 2>/dev/null) — NOT ours, refusing"; return 1
  fi
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
  note "  killing our vLLM on :$1 (pid $pid)"
  kill -TERM -- "-$pgid" 2>/dev/null || kill -TERM "$pid"
  local deadline; deadline=$(( $(date +%s) + 120 ))
  while ss -tln 2>/dev/null | grep -q ":$1 "; do
    [ "$(date +%s)" -ge "$deadline" ] && { kill -KILL -- "-$pgid" 2>/dev/null; sleep 5; break; }
    sleep 3
  done
  note "  :$1 freed"
}

# ── Stage 0: wait for the MAI chain to finish everything ──
note "ARMED: waiting for PI350MAI COMPLETE, then GPU7 free -> base vLLM gpu0 -> B3 round-1"
while :; do
  grep -q "PI350MAI COMPLETE" "$ART/pi350_maiui_chain.log" 2>/dev/null && break
  if grep -qE "PAUSED" "$ART/pi350_maiui_chain.log" 2>/dev/null; then
    note "MAI chain PAUSED — NOT starting B3. Resolve and rerun."
    exit 1
  fi
  sleep 180
done
note "MAI complete — starting B3 stage"

# ── Gates ──
KEY=$(grep "^ANTHROPIC_API_KEY=" "$SKY/.env" | cut -d= -f2)
[ -z "$KEY" ] && { note "ANTHROPIC_API_KEY missing in skyrl-agent/.env — PAUSED (refuse stub run)"; exit 1; }
export ANTHROPIC_API_KEY="$KEY"
[ -s "$SKY/$RUN_DIR/schedule.json" ] || { note "schedule.json missing — PAUSED"; exit 1; }
busy=""
for p in $(seq 6940 6947) $(seq 7040 7047) $(seq 7140 7147) $(seq 7240 7247); do
  ss -tln 2>/dev/null | grep -q ":$p " && busy="$busy $p"
done
[ -n "$busy" ] && { note "B3 ports busy:$busy — PAUSED (pick new --port-base)"; exit 1; }

# ── Stage 1: free GPU7 (kill our MAI instance) ──
kill_our_port 8002 || { note "PAUSED (gpu7 kill refused)"; exit 1; }

# ── Stage 2: base vLLM on gpu0:8001 (idempotent) ──
if api_has http://localhost:8001/v1 "$SERVED"; then
  note "base vLLM already on :8001 — reusing"
else
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
  [ "$free" -lt 40000 ] && { note "GPU0 only ${free}MiB free — PAUSED"; exit 1; }
  note "launching base vLLM on gpu0:8001 (util 0.28)"
  env HF_HOME=/shared/linqiang/hf_home HF_HUB_CACHE=/shared/huggingface/hub HF_HUB_OFFLINE=1 \
      CUDA_VISIBLE_DEVICES=0 nohup "$VPY" -m vllm.entrypoints.openai.api_server \
      --model "$BASE_MODEL" --served-model-name "$SERVED" --dtype bfloat16 \
      --gpu-memory-utilization 0.24 --max-model-len 32768 \
      --port 8001 --host 0.0.0.0 >"$ART/vllm_base_gpu0_b3.log" 2>&1 &
  disown
  deadline=$(( $(date +%s) + 1500 ))
  while ! api_has http://localhost:8001/v1 "$SERVED"; do
    [ "$(date +%s)" -ge "$deadline" ] && { note "base vLLM not healthy after 25 min — PAUSED"; exit 1; }
    sleep 15
  done
  note "base vLLM healthy"
fi

# ── Stage 3: B3 round-1 (crash-resume tolerant) ──
# The driver has no per-episode container fault tolerance: under host overload
# a recycled container can fail to boot, and one unreachable container kills
# the whole driver (rc=1). We auto-resume from state.json (iteration +
# populations + mutations all persisted), cleaning leftover containers each
# time, until iter reaches TOTAL or it's genuinely stuck. pool 3 (was 5) to
# ease boot pressure on the shared host (load ~240, 16 sibling env61 emulators).
cd "$SKY"
TOTAL_ITERS=51
clean_b3() { for c in $(docker ps -a --format '{{.Names}}' | grep "^lq_b3tta"); do docker rm -f "$c" >/dev/null 2>&1; done; }
get_iter() { python3 -c "import json;print(json.load(open('$RUN_DIR/state.json'))['iteration'])" 2>/dev/null || echo 0; }
prev_iter=-1; stuck=0
for attempt in $(seq 1 12); do
  it=$(get_iter)
  [ "$it" -ge "$TOTAL_ITERS" ] && break
  note "B3 driver start (attempt $attempt, from iter $it): pool 3, M2xN4, recycle-after 1"
  "$VPY" scripts/run_b3_mw_tta.py \
    --run-dir "$RUN_DIR" \
    --server http://localhost:8001/v1 --model "$SERVED" \
    --pool-size 3 --n 4 --max-steps 30 \
    --recycle-after 1 \
    --name-prefix lq_b3tta --port-base 6940 \
    --exp-name mw-b3-tta-base-r1 >>"$ART/b3_driver.log" 2>&1
  rc=$?
  clean_b3
  cur=$(get_iter)
  note "driver stopped rc=$rc at iter $cur"
  [ "$cur" -ge "$TOTAL_ITERS" ] && break
  if [ "$cur" -le "$prev_iter" ]; then stuck=$((stuck + 1)); else stuck=0; fi
  if [ "$stuck" -ge 3 ]; then note "stuck at iter $cur (3 no-progress resumes) — PAUSED"; exit 1; fi
  prev_iter=$cur
  sleep 30
done
final=$(get_iter)
if [ "$final" -ge "$TOTAL_ITERS" ]; then
  note "B3 ROUND1 COMPLETE — populations/champions in $SKY/$RUN_DIR; base vLLM left on gpu0:8001"
else
  note "B3 round-1 incomplete at iter $final after 12 attempts — PAUSED"; exit 1
fi

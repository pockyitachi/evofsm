#!/bin/bash
# MAI lesson-memory full run (51 iters, evolution_mode=lesson, --base mai) on
# MAI-UI-8B @ :8001 (GPU7). Mines MAI's OWN lessons. Crash-resume tolerant
# (<=15 attempts, 3 no-progress -> PAUSE). tmux b3mailessonchain.
set -u
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
LOG=$ART/b3mailesson_chain.log
SKY=/shared/linqiang/evofsm_project/SkyRL-AndroidWorld/skyrl-agent
RUN_DIR=tmp_training/mw_b3_mai_lesson_r1
VPY=$SKY/.venv/bin/python
note() { echo "[b3mailesson-chain $(date '+%F %T')] $*" | tee -a "$LOG"; }
api_has() { curl -s --max-time 5 http://localhost:8001/v1/models 2>/dev/null | grep -q "MAI-UI-8B"; }
get_iter() { python3 -c "import json;print(json.load(open('$SKY/$RUN_DIR/state.json'))['iteration'])" 2>/dev/null || echo 0; }
clean() { for c in $(docker ps -a --format '{{.Names}}' | grep '^lq_mail'); do docker rm -f "$c" >/dev/null 2>&1; done; }

api_has || { note "MAI vLLM :8001 down — PAUSED"; exit 1; }
[ -s "$SKY/$RUN_DIR/schedule.json" ] || { note "schedule.json missing — PAUSED"; exit 1; }
KEY=$(grep '^ANTHROPIC_API_KEY=' /shared/linqiang/evofsm_project/.env | cut -d= -f2)
[ -z "$KEY" ] && { note "ANTHROPIC_API_KEY missing — PAUSED"; exit 1; }
export ANTHROPIC_API_KEY="$KEY"
export EVOFSM_TTA_EVOLUTION_MODE=lesson
export WANDB_MODE=disabled

note "ARMED: MAI lesson full 51 iters (mode=lesson, MAI-UI-8B@:8001, base=mai, pool 2, recycle-1, crash-resume)"
cd "$SKY"
TOTAL=51; prev=-1; stuck=0
for attempt in $(seq 1 15); do
  it=$(get_iter); [ "$it" -ge "$TOTAL" ] && break
  api_has || { note "MAI vLLM down before iter $it — PAUSED"; exit 1; }
  note "driver start (attempt $attempt, from iter $it)"
  "$VPY" scripts/run_b3_mw_tta.py --base mai \
    --run-dir "$RUN_DIR" --server http://localhost:8001/v1 --model MAI-UI-8B \
    --pool-size 2 --n 4 --max-steps 30 --recycle-after 1 \
    --name-prefix lq_mail --port-base 6960 --exp-name mw-b3-mai-lesson-r1 >>"$ART/b3mailesson_driver.log" 2>&1
  clean; cur=$(get_iter); note "driver stopped at iter $cur"
  [ "$cur" -ge "$TOTAL" ] && break
  if [ "$cur" -le "$prev" ]; then stuck=$((stuck + 1)); else stuck=0; fi
  [ "$stuck" -ge 3 ] && { note "stuck at iter $cur (3 no-progress) — PAUSED"; exit 1; }
  prev=$cur; sleep 30
done
if [ "$(get_iter)" -ge "$TOTAL" ]; then note "MAILESSON ROUND1 COMPLETE — lessons in $SKY/$RUN_DIR/lessons"; else note "incomplete at $(get_iter) — PAUSED"; exit 1; fi

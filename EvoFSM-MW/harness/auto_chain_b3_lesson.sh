#!/bin/bash
# B3 lesson-memory full run (51 iters, evolution_mode=lesson) on qwen3-VL-8B.
# Design: EvoFSM-MW/docs/b3_lesson_memory_design.md. Crash-resume tolerant
# (host overload can kill the driver — auto-resume from state.json, <=15
# attempts, 3 no-progress -> PAUSE). Runs in tmux b3lessonchain.
set -u
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
LOG=$ART/b3lesson_chain.log
SKY=/shared/linqiang/evofsm_project/SkyRL-AndroidWorld/skyrl-agent
RUN_DIR=tmp_training/mw_b3_lesson_r1
VPY=$SKY/.venv/bin/python
note() { echo "[b3lesson-chain $(date '+%F %T')] $*" | tee -a "$LOG"; }
api_has() { curl -s --max-time 5 http://localhost:8001/v1/models 2>/dev/null | grep -q "Qwen3-VL"; }
get_iter() { python3 -c "import json;print(json.load(open('$SKY/$RUN_DIR/state.json'))['iteration'])" 2>/dev/null || echo 0; }
clean() { for c in $(docker ps -a --format '{{.Names}}' | grep '^lq_b3les'); do docker rm -f "$c" >/dev/null 2>&1; done; }

api_has || { note "qwen vLLM :8001 down — PAUSED"; exit 1; }
[ -s "$SKY/$RUN_DIR/schedule.json" ] || { note "schedule.json missing — PAUSED"; exit 1; }
KEY=$(grep '^ANTHROPIC_API_KEY=' /shared/linqiang/evofsm_project/.env | cut -d= -f2)
[ -z "$KEY" ] && { note "ANTHROPIC_API_KEY missing — PAUSED"; exit 1; }
export ANTHROPIC_API_KEY="$KEY"
export EVOFSM_TTA_EVOLUTION_MODE=lesson
export WANDB_MODE=disabled

note "ARMED: B3 lesson full 51 iters (mode=lesson, qwen3-VL-8B@:8001, base=evofsm, pool 2, recycle-1, crash-resume)"
cd "$SKY"
TOTAL=51; prev=-1; stuck=0
for attempt in $(seq 1 15); do
  it=$(get_iter); [ "$it" -ge "$TOTAL" ] && break
  api_has || { note "qwen vLLM down before iter $it — PAUSED"; exit 1; }
  note "driver start (attempt $attempt, from iter $it)"
  "$VPY" scripts/run_b3_mw_tta.py --base evofsm \
    --run-dir "$RUN_DIR" --server http://localhost:8001/v1 --model Qwen3-VL-8B-Instruct \
    --pool-size 2 --n 4 --max-steps 30 --recycle-after 1 \
    --name-prefix lq_b3les --port-base 6940 --exp-name mw-b3-lesson-r1 >>"$ART/b3lesson_driver.log" 2>&1
  clean; cur=$(get_iter); note "driver stopped at iter $cur"
  [ "$cur" -ge "$TOTAL" ] && break
  if [ "$cur" -le "$prev" ]; then stuck=$((stuck + 1)); else stuck=0; fi
  [ "$stuck" -ge 3 ] && { note "stuck at iter $cur (3 no-progress) — PAUSED"; exit 1; }
  prev=$cur; sleep 30
done
if [ "$(get_iter)" -ge "$TOTAL" ]; then note "B3LESSON ROUND1 COMPLETE — lessons in $SKY/$RUN_DIR/lessons"; else note "incomplete at $(get_iter) — PAUSED"; exit 1; fi

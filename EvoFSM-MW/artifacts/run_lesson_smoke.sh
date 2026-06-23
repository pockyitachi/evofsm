#!/bin/bash
cd /shared/linqiang/evofsm_project/SkyRL-AndroidWorld/skyrl-agent
export EVOFSM_TTA_EVOLUTION_MODE=lesson
export ANTHROPIC_API_KEY="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"
export WANDB_MODE=disabled
.venv/bin/python scripts/run_b3_mw_tta.py --base evofsm \
  --run-dir tmp_training/mw_b3_lesson_smoke --server http://localhost:8001/v1 --model Qwen3-VL-8B-Instruct \
  --pool-size 2 --n 4 --max-steps 30 --recycle-after 1 \
  --name-prefix lq_lsmoke --port-base 6940 --iterations 4 --exp-name mw-lesson-smoke \
  >>/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts/lesson_smoke_driver.log 2>&1
echo "[lesson-smoke $(date '+%F %T')] driver exited rc=$? at iter $(python3 -c "import json;print(json.load(open('tmp_training/mw_b3_lesson_smoke/state.json'))['iteration'])" 2>/dev/null)" >> /shared/linqiang/evofsm_project/EvoFSM-MW/artifacts/lesson_smoke_driver.log

#!/bin/bash
cd /shared/linqiang/evofsm_project/SkyRL-AndroidWorld/skyrl-agent
export EVOFSM_TTA_EVOLUTION_MODE=lesson ANTHROPIC_API_KEY="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" WANDB_MODE=disabled
.venv/bin/python scripts/run_b3_mw_tta.py --base mai \
  --run-dir tmp_training/mw_b3_mai_lesson_smoke --server http://localhost:8001/v1 --model MAI-UI-8B \
  --pool-size 2 --n 4 --max-steps 30 --recycle-after 1 \
  --name-prefix lq_mails --port-base 6960 --iterations 3 --exp-name mw-mai-lesson-smoke \
  >>/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts/mai_lesson_smoke.log 2>&1
echo "[mai-lesson-smoke $(date '+%F %T')] driver exited rc=$? iter=$(python3 -c "import json;print(json.load(open('tmp_training/mw_b3_mai_lesson_smoke/state.json'))['iteration'])" 2>/dev/null)" >> /shared/linqiang/evofsm_project/EvoFSM-MW/artifacts/mai_lesson_smoke.log

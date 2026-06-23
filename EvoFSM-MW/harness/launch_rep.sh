#!/bin/bash
# Parametrized single-run launcher for the 5x variance study.
# Usage: launch_rep.sh <config> <rep>   e.g. launch_rep.sh b2p 3
#   config: b1 | b2 | b2p | b2pp
#   rep:    2..5  (run dir suffix _rN)
# Always runs on the lq_rep container set (5 fresh containers, backend
# 6840-6844) brought up by auto_chain_repeats.sh, max-concurrency 5.
set -e
CONFIG=$1
REP=$2
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
HARNESS=/shared/linqiang/evofsm_project/EvoFSM-MW/harness

MODEL_NAME=Qwen3-VL-8B-Instruct
BASE_URL=http://localhost:8001/v1
ENV_PREFIX=lq_rep   # default container set; overridable for off-chain evals
case "$CONFIG" in
  b1)    AGENT=qwen3vl; GUIDANCE="" ;;
  # π^pre LoRA step-N checkpoint under the B1 setting (no injection); served on
  # GPU6 :8003 (pi-pre-300). Off-chain eval -> own container prefix to avoid the
  # variance-study lq_rep set. Override via PI_PORT / PI_PREFIX / PI_MODEL env.
  b1pi300|b1pi250|b1pi275|b1pi225|b1pi200)
           AGENT=qwen3vl; GUIDANCE=""; MODEL_NAME=${PI_MODEL:?set PI_MODEL}
           BASE_URL=http://localhost:${PI_PORT:-8003}/v1; ENV_PREFIX=${PI_PREFIX:?set PI_PREFIX} ;;
  b2)    AGENT=$HARNESS/qwen3vl_b2_agent.py; GUIDANCE=$ART/b2_guidance.json ;;
  b2p)   AGENT=$HARNESS/qwen3vl_b2_agent.py; GUIDANCE=$ART/b2p_guidance.json ;;
  b2pp)  AGENT=$HARNESS/qwen3vl_b2_agent.py; GUIDANCE=$ART/b2pp_guidance.json ;;
  b2ppp) AGENT=$HARNESS/qwen3vl_b2_agent.py; GUIDANCE=$ART/b2ppp_guidance.json ;;
  # B3 round-1 evolved champions (snapshot eval) — base model, :8001
  b3champ) AGENT=$HARNESS/qwen3vl_b2_agent.py; GUIDANCE=$ART/b3champ_guidance.json ;;
  # π^pre full-FT step-350 checkpoint under the B1 setting (no injection)
  b1pi350) AGENT=qwen3vl; GUIDANCE=""; MODEL_NAME=pi-pre-350 ;;
  # MAI-UI-8B external baseline: stock agent (B1 setting); served on GPU7 :8002
  maib1)   AGENT=mai_ui_agent; GUIDANCE=""; MODEL_NAME=MAI-UI-8B; BASE_URL=http://localhost:8002/v1 ;;
  # MAI-UI-8B + strongest B2-variant guidance (winner in strongest_variant.txt)
  maib2v)
    AGENT=$HARNESS/mai_ui_b2_agent.py
    WINNER=$(cat "$ART/strongest_variant.txt" 2>/dev/null)
    [ -z "$WINNER" ] && { echo "strongest_variant.txt missing — cannot launch maib2v"; exit 2; }
    GUIDANCE=$ART/${WINNER}_guidance.json
    [ -f "$GUIDANCE" ] || { echo "guidance not found: $GUIDANCE"; exit 2; }
    MODEL_NAME=MAI-UI-8B
    BASE_URL=http://localhost:8002/v1
    ;;
  *) echo "unknown config: $CONFIG"; exit 2 ;;
esac

if pgrep -f "bin/mw eval" >/dev/null; then
  echo "another mw eval is still running — aborting ${CONFIG}_r${REP}."
  exit 1
fi
exec 9>/tmp/evofsm_rep_eval.lock
if ! flock -n 9; then
  echo "rep lock held — aborting ${CONFIG}_r${REP}."
  exit 1
fi

cd /shared/linqiang/evofsm_project/MobileWorld
[ -n "$GUIDANCE" ] && export EVOFSM_B2_GUIDANCE=$GUIDANCE

.venv/bin/mw eval \
  --agent_type "$AGENT" \
  --task "$(tr -d '\n' < /shared/linqiang/evofsm_project/EvoFSM-MW/configs/teval_tasklist.txt)" \
  --max_round 50 \
  --model_name "$MODEL_NAME" \
  --llm_base_url "$BASE_URL" \
  --step_wait_time 3 \
  --max-concurrency 5 \
  --env-name-prefix "$ENV_PREFIX" \
  --env-image mobile_world:reset \
  --log_file_root "traj_logs/${CONFIG}_qwen3vl8b_r${REP}" \
  --enable_mcp \
  --enable_user_interaction

#!/bin/bash
# Waits for the MAI-B3 adapt run (51 iters) to COMPLETE, then:
#   1. render its evolved champions -> b3mai_champ_guidance.json
#   2. eval on the 110-task split with the MAI agent + those champions
#      (MAI-UI-8B @ :8001), vs MAI-B1 (26.2) / MAI-B2' (26.4).
# Fresh 5 containers; tmux b3maievalchain. PAUSES if adapt PAUSED.
set -u
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
LOG=$ART/b3mai_eval_chain.log
MW=/shared/linqiang/evofsm_project/MobileWorld
SKY=/shared/linqiang/evofsm_project/SkyRL-AndroidWorld/skyrl-agent
HARNESS=/shared/linqiang/evofsm_project/EvoFSM-MW/harness
ENVFILE=/shared/linqiang/MobileWorld/.env
GUID=$ART/b3mai_champ_guidance.json
note() { echo "[b3mai-eval $(date '+%F %T')] $*" | tee -a "$LOG"; }
clean() { for c in $(docker ps -a --format '{{.Names}}' | grep '^lq_b3me'); do docker rm -f "$c" >/dev/null 2>&1; done; }

note "ARMED: waiting for MAI-B3 adapt to COMPLETE"
while :; do
  grep -q "B3MAI ROUND1 COMPLETE" "$ART/b3mai_chain.log" 2>/dev/null && break
  grep -q "PAUSED" "$ART/b3mai_chain.log" 2>/dev/null && { note "adapt PAUSED — NOT starting eval"; exit 1; }
  sleep 180
done
note "adapt complete — rendering MAI-B3 champions"
cd "$SKY"   # gen_b3_guidance.py resolves --run-dir + FSM/L_C paths relative to SKY
"$SKY/.venv/bin/python" "$HARNESS/gen_b3_guidance.py" \
  --run-dir tmp_training/mw_b3_mai_r1 --out "$GUID" 2>&1 | tail -2 | tee -a "$LOG"
[ -s "$GUID" ] || { note "champion guidance render failed — PAUSED"; exit 1; }

curl -s --max-time 5 http://localhost:8001/v1/models | grep -q "MAI-UI-8B" || { note "MAI vLLM :8001 down — PAUSED"; exit 1; }
note "starting 5 lq_b3me containers (backend 6970-6974)"
(cd "$MW" && .venv/bin/mw env run --count 5 --name-prefix lq_b3me --image mobile_world:reset \
   --backend-start-port 6970 --viewer-start-port 7920 --vnc-start-port 5880 \
   --adb-start-port 5770 --launch-interval 20 --env-file "$ENVFILE") >"$ART/b3mai_eval_envrun.log" 2>&1 \
  || { note "env run failed — PAUSED"; exit 1; }
deadline=$(( $(date +%s) + 1800 ))
while :; do
  h=$(docker ps --filter name=lq_b3me_ --format '{{.Status}}' | grep -c '(healthy)')
  [ "$h" -ge 5 ] && { note "5/5 healthy"; break; }
  [ "$(date +%s)" -ge "$deadline" ] && { note "containers not ready — PAUSED"; exit 1; }
  sleep 20
done

note "launching MAI-B3 eval (110 tasks, MAI agent + evolved champions, :8001)"
cd "$MW"
EVOFSM_B2_GUIDANCE="$GUID" .venv/bin/mw eval \
  --agent_type "$HARNESS/mai_ui_b2_agent.py" \
  --task "$(tr -d '\n' < /shared/linqiang/evofsm_project/EvoFSM-MW/configs/teval_tasklist.txt)" \
  --max_round 50 --model_name MAI-UI-8B --llm_base_url http://localhost:8001/v1 \
  --step_wait_time 3 --max-concurrency 5 \
  --env-name-prefix lq_b3me --env-image mobile_world:reset \
  --log_file_root traj_logs/b3mai_eval_qwen3vl8b --enable_mcp --enable_user_interaction \
  >"$ART/b3mai_eval_stdout.log" 2>&1
d=$MW/traj_logs/b3mai_eval_qwen3vl8b
n=$(ls "$d"/*/result.txt 2>/dev/null | wc -l); s=$(grep -l "score: 1.0" "$d"/*/result.txt 2>/dev/null | wc -l)
clean
note "B3MAI EVAL DONE: $n/110, success $s  (vs MAI-B1 26.2 / MAI-B2' 26.4)"

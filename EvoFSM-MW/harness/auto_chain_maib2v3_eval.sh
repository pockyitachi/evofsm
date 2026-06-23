#!/bin/bash
# MAI-UI B2' eval using the TIGHTENED L_C_v3 guidance (system-app pollution
# removed). Same agent/model/mode as maib2v (B2', app-l2) — only the injected
# L_C differs (v2 bloated 161 cats → v3 124, communication 20→8). Isolates the
# "L_C content" variable. Fresh 5 containers; vs MAI-B1 26.2 / MAI-B2'(v2) 26.4.
set -u
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
LOG=$ART/maib2v3_eval_chain.log
MW=/shared/linqiang/evofsm_project/MobileWorld
HARNESS=/shared/linqiang/evofsm_project/EvoFSM-MW/harness
ENVFILE=/shared/linqiang/MobileWorld/.env
GUID=$ART/b2p_v3_guidance.json
note() { echo "[maib2v3 $(date '+%F %T')] $*" | tee -a "$LOG"; }
clean() { for c in $(docker ps -a --format '{{.Names}}' | grep '^lq_b2v3'); do docker rm -f "$c" >/dev/null 2>&1; done; }

[ -s "$GUID" ] || { note "b2p_v3 guidance missing — ABORT"; exit 1; }
curl -s --max-time 5 http://localhost:8001/v1/models | grep -q "MAI-UI-8B" || { note "MAI vLLM :8001 down — ABORT"; exit 1; }

note "starting 5 lq_b2v3 containers (backend 6980-6984)"
(cd "$MW" && .venv/bin/mw env run --count 5 --name-prefix lq_b2v3 --image mobile_world:reset \
   --backend-start-port 6980 --viewer-start-port 7930 --vnc-start-port 5890 \
   --adb-start-port 5780 --launch-interval 20 --env-file "$ENVFILE") >"$ART/maib2v3_envrun.log" 2>&1 \
  || { note "env run failed — ABORT"; exit 1; }
deadline=$(( $(date +%s) + 1800 ))
while :; do
  h=$(docker ps --filter name=lq_b2v3_ --format '{{.Status}}' | grep -c '(healthy)')
  [ "$h" -ge 5 ] && { note "5/5 healthy"; break; }
  [ "$(date +%s)" -ge "$deadline" ] && { note "containers not ready — ABORT"; clean; exit 1; }
  sleep 20
done

note "launching MAI-B2'(v3) eval (110 tasks, app-l2 + L_C_v3, :8001)"
cd "$MW"
EVOFSM_B2_GUIDANCE="$GUID" .venv/bin/mw eval \
  --agent_type "$HARNESS/mai_ui_b2_agent.py" \
  --task "$(tr -d '\n' < /shared/linqiang/evofsm_project/EvoFSM-MW/configs/teval_tasklist.txt)" \
  --max_round 50 --model_name MAI-UI-8B --llm_base_url http://localhost:8001/v1 \
  --step_wait_time 3 --max-concurrency 5 \
  --env-name-prefix lq_b2v3 --env-image mobile_world:reset \
  --log_file_root traj_logs/maib2v3 --enable_mcp --enable_user_interaction \
  >"$ART/maib2v3_eval_stdout.log" 2>&1
d=$MW/traj_logs/maib2v3
n=$(ls "$d"/*/result.txt 2>/dev/null | wc -l); s=$(grep -l "score: 1.0" "$d"/*/result.txt 2>/dev/null | wc -l)
clean
note "MAIB2'v3 EVAL DONE: $n/110, success $s  (vs MAI-B1 26.2 / MAI-B2'(v2) 26.4)"

#!/bin/bash
# B3 snapshot eval (1 run): wait for the 3 lq_rep containers to be healthy,
# then run the 110-task eval with B3-evolved-champion guidance on the base
# model (:8001). Compares B3 vs B1(8.2)/B2'(9.2). tmux session b3eval.
set -u
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
LOG=$ART/b3eval_run.log
note() { echo "[b3eval $(date '+%F %T')] $*" | tee -a "$LOG"; }

note "waiting for 3 lq_rep containers healthy + /health 200"
deadline=$(( $(date +%s) + 1800 ))
while :; do
  h=$(docker ps --filter name=lq_rep_ --format '{{.Status}}' | grep -c '(healthy)')
  ok=0; for p in 6840 6841 6842; do curl -sf --max-time 3 "http://localhost:$p/health" >/dev/null 2>&1 && ok=$((ok+1)); done
  [ "$h" -ge 3 ] && [ "$ok" -ge 3 ] && { note "3/3 healthy"; break; }
  [ "$(date +%s)" -ge "$deadline" ] && { note "containers not ready after 30min — ABORT"; exit 1; }
  sleep 15
done

note "launching B3 eval (base :8001, b3champ guidance, concurrency 3)"
bash /shared/linqiang/evofsm_project/EvoFSM-MW/harness/launch_rep.sh b3champ 1 >"$ART/b3eval_stdout.log" 2>&1
rc=$?
d=/shared/linqiang/evofsm_project/MobileWorld/traj_logs/b3champ_qwen3vl8b_r1
n=$(ls "$d"/*/result.txt 2>/dev/null | wc -l); s=$(grep -l "score: 1.0" "$d"/*/result.txt 2>/dev/null | wc -l)
if [ "$rc" -ne 0 ] && [ "$n" -lt 105 ]; then
  note "B3 EVAL FAILED (launch rc=$rc, only $n/110) — see b3eval_stdout.log"; exit 1
fi
note "B3 EVAL DONE: $n/110, success $s"

#!/bin/bash
# Off-chain eval: pi^pre LoRA step-200 under the B1 setting × 5 reps on MobileWorld.
# Mirror of eval_pi250_b1.sh but model=pi-pre-200 (GPU3 :8003, already serving),
# own container set lq_e200 (ports 6860-6864 — 250's set is torn down). Resumable:
# a rep dir with >=105 results is skipped. Fresh containers per rep. Kills only
# linqiang-owned containers. Compare: base 8.2 / pi250 7.8 / pi300 7.2 / pi350 6.8.
set -u
MW=/shared/linqiang/evofsm_project/MobileWorld
TRAJ=$MW/traj_logs
ART=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts
HARNESS=/shared/linqiang/evofsm_project/EvoFSM-MW/harness
ENVFILE=/shared/linqiang/MobileWorld/.env
ELOG=$ART/pi200_eval_logs
LOG=$ART/pi200_b1.log
BASE_URL=http://localhost:8003/v1
PREFIX=lq_e200
mkdir -p "$ELOG"

export PI_PORT=8003 PI_PREFIX=$PREFIX PI_MODEL=pi-pre-200

note()  { echo "[pi200 $(date '+%F %T')] $*" | tee -a "$LOG"; }
count() { ls "$TRAJ/$1"/*/result.txt 2>/dev/null | wc -l; }
succ()  { grep -l "score: 1.0" "$TRAJ/$1"/*/result.txt 2>/dev/null | wc -l; }
api_has(){ curl -s --max-time 5 "$BASE_URL/models" 2>/dev/null | grep -q "pi-pre-200"; }

teardown() {
  for c in $(docker ps -a --format '{{.Names}}' | grep "^${PREFIX}_"); do
    docker rm -f "$c" >/dev/null 2>&1 && note "  removed $c"
  done
}

bring_up() {
  note "  starting 5 ${PREFIX} containers (backend 6860-6864)"
  (cd "$MW" && .venv/bin/mw env run --count 5 --name-prefix "$PREFIX" --image mobile_world:reset \
      --backend-start-port 6860 --viewer-start-port 7920 --vnc-start-port 5880 \
      --adb-start-port 5770 --launch-interval 20 --env-file "$ENVFILE") \
      >"$ELOG/envrun_last.log" 2>&1 || { note "  mw env run FAILED"; return 1; }
  local deadline h ok p
  deadline=$(( $(date +%s) + 1800 ))
  while :; do
    h=$(docker ps --filter "name=${PREFIX}_" --format '{{.Status}}' | grep -c '(healthy)')
    ok=0
    for p in 6860 6861 6862 6863 6864; do
      curl -sf --max-time 3 "http://localhost:$p/health" >/dev/null 2>&1 && ok=$((ok+1))
    done
    [ "$h" -ge 5 ] && [ "$ok" -ge 5 ] && { note "  ${PREFIX}: 5/5 healthy"; return 0; }
    [ "$(date +%s)" -ge "$deadline" ] && { note "  ${PREFIX} not ready in 30min"; return 1; }
    sleep 20
  done
}

run_one() {  # $1=rep
  local rep=$1 dir="b1pi200_qwen3vl8b_r${rep}"
  api_has || { note "  pi-pre-200 vLLM down before r${rep}"; return 1; }
  teardown; sleep 10
  bring_up || return 1
  note "  eval b1pi200_r${rep} -> traj_logs/$dir"
  bash "$HARNESS/launch_rep.sh" b1pi200 "$rep" >"$ELOG/b1pi200_r${rep}.log" 2>&1
  note "  b1pi200_r${rep} done: $(count "$dir")/110, success $(succ "$dir")"
  [ "$(count "$dir")" -ge 105 ]
}

note "ARMED: pi-pre-200 (step200) B1 x5 on GPU3:8003, containers ${PREFIX} 6860-6864"
api_has || { note "pi-pre-200 not serving on :8003 — abort"; exit 1; }

for rep in 1 2 3 4 5; do
  dir="b1pi200_qwen3vl8b_r${rep}"
  if [ "$(count "$dir")" -ge 105 ]; then
    note "=== r${rep} already complete ($(count "$dir")/110, success $(succ "$dir")) — skip ==="
    continue
  fi
  note "=== run r${rep} ==="
  if ! run_one "$rep"; then
    note "  r${rep} FAILED — retry once"
    mv "$TRAJ/$dir" "$TRAJ/${dir}_failed1" 2>/dev/null
    run_one "$rep" || { note "  r${rep} failed TWICE — PAUSED"; exit 1; }
  fi
done

teardown
note "PI200 B1 COMPLETE — per-rep success: $(for r in 1 2 3 4 5; do printf 'r%s=%s ' "$r" "$(succ b1pi200_qwen3vl8b_r$r)"; done)"

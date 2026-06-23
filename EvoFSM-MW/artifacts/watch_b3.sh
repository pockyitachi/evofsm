#!/bin/bash
# Notify-only watcher for the B3 tail chain + driver. Terminal-state detection
# anchored to chain-log line count at START (base) so a stale PAUSED line can't
# false-trigger after relaunch. Driver progress is FILTERED: routine V=0 iters
# are silenced; only V>0 iters, mutations, episode failures, and errors surface.
CLOG=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts/b3_chain.log
DLOG=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts/b3_driver.log
base=$(wc -l < "$CLOG" 2>/dev/null || echo 0)
lastc=$base
lastd=$(wc -l < "$DLOG" 2>/dev/null || echo 0)   # skip backlog: start from current end
while :; do
  c=$(wc -l < "$CLOG" 2>/dev/null || echo "$lastc")
  if [ "$c" -gt "$lastc" ]; then tail -n +"$((lastc+1))" "$CLOG"; lastc=$c; fi
  if [ -f "$DLOG" ]; then
    d=$(wc -l < "$DLOG")
    if [ "$d" -gt "$lastd" ]; then
      tail -n +"$((lastd+1))" "$DLOG" | awk '
        /\[B3\] iter/ { if ($0 ~ /V_A=0\.000 V_B=0\.000/) next; print; next }
        /mutated|\[B3\] done|episode failed|Traceback|ERROR/ { print }
      ' | head -20
      lastd=$d
    fi
  fi
  after=$(tail -n +"$((base+1))" "$CLOG" 2>/dev/null)
  echo "$after" | grep -q "B3 ROUND1 COMPLETE" && exit 0
  if echo "$after" | grep -q "PAUSED"; then
    echo "B3 链 PAUSED — 看 tmux b3chain 与 b3_chain.log / b3_driver.log"; exit 0
  fi
  if ! pgrep -f "auto_chain_b3.s[h]" >/dev/null 2>&1; then
    sleep 30
    after=$(tail -n +"$((base+1))" "$CLOG" 2>/dev/null)
    if ! pgrep -f "auto_chain_b3.s[h]" >/dev/null 2>&1 \
       && ! echo "$after" | grep -qE "B3 ROUND1 COMPLETE|PAUSED"; then
      echo "⚠️ b3 链进程消失且无终态 — 疑似崩溃, 看 tmux b3chain"; exit 0
    fi
  fi
  sleep 90
done

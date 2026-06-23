#!/bin/bash
# Notify-only watcher for the b2ppp x5 tail chain. Relays chain-log lines,
# exits on COMPLETE / PAUSED / not-starting.
LOG=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts/b2ppp_chain.log
last=$(wc -l < "$LOG" 2>/dev/null || echo 0)
while :; do
  cur=$(wc -l < "$LOG" 2>/dev/null || echo "$last")
  if [ "$cur" -gt "$last" ]; then tail -n +"$((last+1))" "$LOG"; last=$cur; fi
  grep -q "B2PPP COMPLETE" "$LOG" 2>/dev/null && exit 0
  if grep -qE "PAUSED|NOT starting" "$LOG" 2>/dev/null; then
    echo "b2ppp 链停了 — 看 tmux b2pppchain 与 b2ppp_chain.log"
    exit 0
  fi
  sleep 60
done

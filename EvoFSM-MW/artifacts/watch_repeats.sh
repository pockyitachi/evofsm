#!/bin/bash
# Notify-only watcher for the 16-run variance chain (tmux brepeats).
# Relays chain-log lines; adds stall warnings; exits on COMPLETE/PAUSED.
LOG=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts/repeats_chain.log
TRAJ=/shared/linqiang/evofsm_project/MobileWorld/traj_logs

last=$(wc -l < "$LOG" 2>/dev/null || echo 0)
stall=0
while :; do
  cur=$(wc -l < "$LOG" 2>/dev/null || echo "$last")
  if [ "$cur" -gt "$last" ]; then tail -n +"$((last+1))" "$LOG"; last=$cur; fi
  grep -q "REPEATS COMPLETE" "$LOG" 2>/dev/null && exit 0
  if grep -q "PAUSED" "$LOG" 2>/dev/null; then
    echo "重复实验链已暂停 — 看 tmux brepeats 与 repeats_chain.log"
    exit 0
  fi
  # stall: eval alive but newest result.txt across rep dirs >35 min old
  if pgrep -f "bin/mw eva[l]" >/dev/null 2>&1; then
    newest=$(ls -t "$TRAJ"/*_qwen3vl8b_r[2-5]/*/result.txt 2>/dev/null | head -1)
    if [ -n "$newest" ]; then
      age=$(( $(date +%s) - $(stat -c %Y "$newest") ))
      if [ "$age" -gt 2100 ] && [ "$stall" -eq 0 ]; then
        echo "警告: 重复实验已 $((age/60)) 分钟无新结果 — 可能卡死"
        stall=1
      fi
      [ "$age" -le 2100 ] && stall=0
    fi
  fi
  sleep 60
done

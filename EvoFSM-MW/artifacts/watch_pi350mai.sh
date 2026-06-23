#!/bin/bash
# Notify-only watcher for the pi350+MAI-UI tail chain.
LOG=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts/pi350_maiui_chain.log
last=$(wc -l < "$LOG" 2>/dev/null || echo 0)
while :; do
  cur=$(wc -l < "$LOG" 2>/dev/null || echo "$last")
  if [ "$cur" -gt "$last" ]; then tail -n +"$((last+1))" "$LOG"; last=$cur; fi
  grep -q "PI350MAI COMPLETE" "$LOG" 2>/dev/null && exit 0
  if grep -qE "PAUSED|NOT starting" "$LOG" 2>/dev/null; then
    echo "pi350/MAI-UI 链停了 — 看 tmux pi350mai 与 pi350_maiui_chain.log"
    exit 0
  fi
  # 链进程消失但日志无终态 = 崩溃(如未捕获的 bash 错误打到 stderr)
  if ! pgrep -f "auto_chain_pi350_maiui.s[h]" >/dev/null 2>&1; then
    sleep 30
    if ! pgrep -f "auto_chain_pi350_maiui.s[h]" >/dev/null 2>&1 \
       && ! grep -qE "PI350MAI COMPLETE|PAUSED" "$LOG" 2>/dev/null; then
      echo "⚠️ pi350mai 链进程消失且无终态日志 — 疑似崩溃, 看 tmux pi350mai 的 stderr"
      exit 0
    fi
  fi
  sleep 90
done

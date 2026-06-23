#!/bin/bash
# Notify-only watcher for the B1→B2 chain (the chain itself runs in tmux
# b2eval, independent of any Claude session). Emits one line per event:
# autochain log lines, B2 progress milestones, completion, death, stalls.
LOG=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts/b2_autochain.log
MW=/shared/linqiang/evofsm_project/MobileWorld
B1RES=$MW/traj_logs/b1_qwen3vl8b
B2RES=$MW/traj_logs/b2_qwen3vl8b
B1_PID=30475

last=$(wc -l < "$LOG" 2>/dev/null || echo 0)
milestone=0 launched=0 dead=0 stall=0
while :; do
  cur=$(wc -l < "$LOG" 2>/dev/null || echo "$last")
  if [ "$cur" -gt "$last" ]; then tail -n +"$((last+1))" "$LOG"; last=$cur; fi
  grep -q "launching B2" "$LOG" 2>/dev/null && launched=1
  if grep -qE "NOT launching|aborting" "$LOG" 2>/dev/null; then
    echo "AUTOCHAIN 中止(见上面日志行) — 需人工接管 tmux b2eval"
    exit 0
  fi

  if [ "$launched" -eq 1 ]; then
    n=$(ls "$B2RES"/*/result.txt 2>/dev/null | wc -l)
    if [ "$n" -ge 110 ]; then
      s=$(grep -l "score: 1.0" "$B2RES"/*/result.txt 2>/dev/null | wc -l)
      echo "B2 全部完成: $n/110, 成功 $s"
      exit 0
    fi
    if [ "$n" -ge "$((milestone+25))" ]; then
      s=$(grep -l "score: 1.0" "$B2RES"/*/result.txt 2>/dev/null | wc -l)
      echo "B2 进度: $n/110, 成功 $s"
      milestone=$n
    fi
    if pgrep -f "qwen3vl_b2_agent" >/dev/null 2>&1; then
      dead=0
    else
      dead=$((dead+1))
      if [ "$dead" -ge 3 ]; then
        echo "B2 eval 进程消失但仅 $n/110 — 疑似挂掉, 查看 tmux b2eval"
        exit 0
      fi
    fi
  fi

  # stall detection: newest result.txt older than 35 min while eval alive
  act=""
  if [ "$launched" -eq 1 ]; then act=$B2RES
  elif kill -0 "$B1_PID" 2>/dev/null; then act=$B1RES; fi
  if [ -n "$act" ]; then
    newest=$(ls -t "$act"/*/result.txt 2>/dev/null | head -1)
    if [ -n "$newest" ]; then
      age=$(( $(date +%s) - $(stat -c %Y "$newest") ))
      if [ "$age" -gt 2100 ] && [ "$stall" -eq 0 ]; then
        echo "警告: $(basename "$act") 已 $((age/60)) 分钟无新结果 — 可能卡死"
        stall=1
      fi
      [ "$age" -le 2100 ] && stall=0
    fi
  fi
  sleep 60
done

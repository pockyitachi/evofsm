#!/bin/bash
# Notify-only watcher for the B2→B2'→B2'' overnight chain (chain itself runs
# in tmux b2variants). Emits: every chain-log line, per-variant progress
# milestones, stall warnings. Exits on CHAIN COMPLETE or abort.
LOG=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts/b2_variants_chain.log
TRAJ=/shared/linqiang/evofsm_project/MobileWorld/traj_logs

last=$(wc -l < "$LOG" 2>/dev/null || echo 0)
mp=0 mpp=0 stall=0
while :; do
  cur=$(wc -l < "$LOG" 2>/dev/null || echo "$last")
  if [ "$cur" -gt "$last" ]; then tail -n +"$((last+1))" "$LOG"; last=$cur; fi
  if grep -q "CHAIN COMPLETE" "$LOG" 2>/dev/null; then exit 0; fi
  if grep -qE "aborting chain|NOT starting" "$LOG" 2>/dev/null; then
    echo "变体链中止 — 查看 tmux b2variants 与 $LOG"
    exit 0
  fi

  d="$TRAJ/b2p_qwen3vl8b"
  if [ -d "$d" ]; then
    n=$(ls "$d"/*/result.txt 2>/dev/null | wc -l)
    if [ "$n" -ge $((mp+25)) ] && [ "$n" -lt 110 ]; then
      echo "B2' 进度: $n/110, 成功 $(grep -l 'score: 1.0' "$d"/*/result.txt 2>/dev/null | wc -l)"
      mp=$n
    fi
  fi
  d="$TRAJ/b2pp_qwen3vl8b"
  if [ -d "$d" ]; then
    n=$(ls "$d"/*/result.txt 2>/dev/null | wc -l)
    if [ "$n" -ge $((mpp+25)) ] && [ "$n" -lt 110 ]; then
      echo "B2'' 进度: $n/110, 成功 $(grep -l 'score: 1.0' "$d"/*/result.txt 2>/dev/null | wc -l)"
      mpp=$n
    fi
  fi

  # stall: an eval is alive but its newest result is >35 min old
  act=""
  if pgrep -f "qwen3vl_b2_agent" >/dev/null 2>&1; then
    if [ -d "$TRAJ/b2pp_qwen3vl8b" ]; then act="$TRAJ/b2pp_qwen3vl8b"
    elif [ -d "$TRAJ/b2p_qwen3vl8b" ]; then act="$TRAJ/b2p_qwen3vl8b"
    else act="$TRAJ/b2_qwen3vl8b"; fi
  fi
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

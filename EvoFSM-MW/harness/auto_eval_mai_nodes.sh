#!/bin/bash
# 自动评估 MAI π^pre 节点 300/400: 等ckpt出现 -> merge -> 找空卡部署vLLM -> B1×5
# -> 记录磁盘最终成功数 -> 清理. 到400评估完后停训练. tmux: mainodeeval
set -u
P=/shared/linqiang/evofsm_project; SKY=$P/SkyRL-AndroidWorld
CK=$SKY/tmp_training/ckpts/skyagent-android-evofsm-phase1-maiui8b-lora32
MW=$P/MobileWorld; ENVFILE=/shared/linqiang/MobileWorld/.env
LOG=$P/EvoFSM-MW/artifacts/mai_node_eval.log
HARNESS=$P/EvoFSM-MW/harness
note(){ echo "[node-eval $(date '+%F %T')] $*" | tee -a "$LOG"; }
have_ckpt(){ [ -f "$CK/global_step_$1/actor/lora_adapter/adapter_model.safetensors" ]; }
free_gpu(){ for g in 0 3 4 5 7; do u=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|tr -d ' '); [ $((143771-u)) -ge 42000 ] && { echo $g; return; }; done; }

eval_step(){  # $1=STEP
  local S=$1 HF=/shared/linqiang/models/mai_pre_${S}_hf gpu vp
  note "=== 节点 $S: merge ==="
  HF_HOME=/shared/linqiang/hf_home "$SKY/skyrl-agent/.venv/bin/python" "$SKY/skyrl-agent/scripts/merge_lora_to_hf.py" \
    --base Tongyi-MAI/MAI-UI-8B --adapter $CK/global_step_$S/actor/lora_adapter \
    --ckpt-hf $CK/global_step_$S/actor/huggingface --out $HF >>"$LOG" 2>&1
  [ -f $HF/model.safetensors.index.json ] || { note "节点$S merge失败"; return 1; }
  gpu=""; for i in $(seq 1 60); do gpu=$(free_gpu); [ -n "$gpu" ] && break; sleep 60; done
  [ -z "$gpu" ] && { note "节点$S 没找到eval空卡"; return 1; }
  note "节点$S: 部署vLLM到GPU$gpu:8001"
  CUDA_VISIBLE_DEVICES=$gpu HF_HOME=/shared/linqiang/hf_home nohup "$SKY/skyrl-agent/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --model $HF --served-model-name MAI-UI-8B --dtype bfloat16 --gpu-memory-utilization 0.24 \
    --max-model-len 32768 --port 8001 --host 0.0.0.0 >/shared/linqiang/tmp/vllm_node_$S.log 2>&1 &
  local vpid=$!
  for i in $(seq 1 40); do curl -s --max-time 5 http://localhost:8001/v1/models 2>/dev/null|grep -q MAI-UI-8B && break; sleep 6; done
  curl -s --max-time 5 http://localhost:8001/v1/models 2>/dev/null|grep -q MAI-UI-8B || { note "节点$S vLLM起不来"; kill $vpid 2>/dev/null; return 1; }
  local PRE=lq_n$S
  for r in 1 2 3 4 5; do
    for c in $(docker ps -aq --filter name=$PRE 2>/dev/null); do docker rm -f $c >/dev/null 2>&1; done
    (cd $MW && .venv/bin/mw env run --count 5 --name-prefix $PRE --image mobile_world:reset \
       --backend-start-port 6900 --viewer-start-port 7940 --vnc-start-port 5900 --adb-start-port 5790 \
       --launch-interval 20 --env-file $ENVFILE) >>"$LOG" 2>&1
    local dl=$(( $(date +%s)+1800 )); while :; do [ $(docker ps --filter name=${PRE}_ --format '{{.Status}}'|grep -c healthy) -ge 5 ] && break; [ $(date +%s) -ge $dl ] && break; sleep 20; done
    note "节点$S r$r: 跑eval"
    (cd $MW && .venv/bin/mw eval --agent_type mai_ui_agent \
      --task "$(tr -d '\n' < $P/EvoFSM-MW/configs/teval_tasklist.txt)" --max_round 50 \
      --model_name MAI-UI-8B --llm_base_url http://localhost:8001/v1 --step_wait_time 3 --max-concurrency 5 \
      --env-name-prefix $PRE --env-image mobile_world:reset \
      --log_file_root traj_logs/maib1pre${S}_qwen3vl8b_r$r --enable_mcp --enable_user_interaction) >>"$LOG" 2>&1
    local d=$MW/traj_logs/maib1pre${S}_qwen3vl8b_r$r
    note "节点$S r$r: $(grep -l 'score: 1.0' $d/*/result.txt 2>/dev/null|wc -l)/$(ls $d/*/result.txt 2>/dev/null|wc -l) 成功"
    for c in $(docker ps -aq --filter name=$PRE 2>/dev/null); do docker rm -f $c >/dev/null 2>&1; done
  done
  kill $vpid 2>/dev/null; sleep 3; pkill -9 -f "mai_pre_${S}_hf" 2>/dev/null
  local res=""; for r in 1 2 3 4 5; do d=$MW/traj_logs/maib1pre${S}_qwen3vl8b_r$r; res="$res $(grep -l 'score: 1.0' $d/*/result.txt 2>/dev/null|wc -l)"; done
  note "★ 节点$S 完成: 逐rep成功(磁盘/110) =$res"
}

note "ARMED: 等 ckpt 300/400, 自动评估; 到400评估完停训练"
for S in 300 400; do
  note "等待 ckpt $S ..."
  while ! have_ckpt $S; do sleep 180; done
  note "ckpt $S 出现, 开始评估"
  eval_step $S || note "节点$S 评估出错, 继续"
done
note "停训练 (到400评估完)"
tmux kill-session -t mailoraresume 2>/dev/null
for pat in verl_android_wandb_evofsm_phase1_lora verl_main_ppo "uv run.*verl" "ray::" raylet gcs_server; do
  for pid in $(pgrep -f "$pat" 2>/dev/null); do [ "$(ps -o user= -p $pid 2>/dev/null)" = linqiang ] && kill -9 $pid 2>/dev/null; done
done
note "ALL DONE: 300/400 评估完, 训练已停"

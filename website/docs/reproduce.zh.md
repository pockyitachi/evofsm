# 复现

两项研究，两套测评流程。**基准内**阶梯在预烘焙的 Android 模拟器上运行；**跨基准**阶梯用 vLLM 提供骨干模型，并驱动全新的 MobileWorld 容器。两者共享同一套符号核心（FSM / `L_C`），经 `PYTHONPATH` 从 `EvoFSM-RL/evofsm_rl/` 导入。

!!! note "完整 flag 在子项目里"
    下面的命令是可复现的骨架。每个脚本的 flag 都比示例多——传 `--help`，并阅读各子项目的 `README.md` / `harness/README.md`（以及每个脚本顶部的 docstring）获取确切、权威的接线。

## 基准内（AndroidWorld+）

### 1 — 环境

Python 3.12 venv；把基准和包都放进 path。

```bash
cd /shared/linqiang/evofsm_project && source .venv/bin/activate
export PYTHONPATH=android_world_plus:EvoFSM-RL
```

### 2 — 启动预烘焙模拟器

单个预烘焙 AVD——**`AWAvd2`，快照 `apps_ready_dec2025`**——已装好全部 app。以**只读**方式启动，确保快照永不被改。**不要**跑 `emulator_setup=True` / `bootstrap_avd.sh`；快照里已有这些 app。

```bash
ANDROID_AVD_HOME=$PWD/android-sdk/avd ANDROID_SDK_ROOT=$PWD/android-sdk \
  $PWD/android-sdk/emulator/emulator -avd AWAvd2 -port 5710 -grpc 8710 \
  -snapshot apps_ready_dec2025 -no-snapshot-save -read-only \
  -no-window -no-audio -no-boot-anim &

# verify (prints 1 ~30s after boot):
adb -s emulator-5710 wait-for-device shell getprop sys.boot_completed
```

不同的 `-port`/`-grpc` 对可以让多个实例共存。

### 3 — 跑消融（B1 → B4）

每一档恰好加一个机制。每个脚本都接受 `--console-port 5710 --grpc-port 8710`；传 `--help` 看完整 flag 集。

```bash
# B1 — zero-shot baseline on T_eval (M3A, Qwen3-VL-8B)
python EvoFSM-RL/scripts/baseline_10task.py        --console-port 5710 --grpc-port 8710 ...

# B2 — + static category L_C injection
python EvoFSM-RL/scripts/run_b2_eval.py            --console-port 5710 --grpc-port 8710 ...

# B3 — evolve L_C on T_adapt (per Tier-B app), then frozen eval on T_eval
python EvoFSM-RL/scripts/run_b3_evolution.py       --app pro_expense ...
python EvoFSM-RL/scripts/run_b3_teval.py           --console-port 5710 --grpc-port 8710 ...

# B4 — joint LoRA + FSM (full method): Phase-1 pretrain, then per-app adapt + eval
python EvoFSM-RL/scripts/run_phase1_pretraining.py ...
python EvoFSM-RL/scripts/run_b4_evolution.py       --app pro_expense ...
python EvoFSM-RL/scripts/run_b4_teval.py           --console-port 5710 --grpc-port 8710 ...
```

FSM / `L_C` 知识层已预构建在 `artifacts/` 下。要重新生成，设置 `ANTHROPIC_API_KEY` 并运行 `scripts/build_all_fsms.py` + `scripts/build_L_C.py`。

## 跨基准（MobileWorld）

本研究复用符号核心，但换上纯视觉的 Qwen3-VL `mobile_use` 测评流程和 MobileWorld 基准。agent 在 MobileWorld 自己的 venv 里运行（`mw eval ...`）；guidance 生成器在项目根的主 venv 里运行，带 `PYTHONPATH=EvoFSM-RL`。

### 1 — 提供骨干模型并启动全新容器

在 `localhost:8001` 用 vLLM 提供骨干模型，然后为本次运行拉起**全新**的 MobileWorld 容器。

```bash
# backbone served on localhost:8001 (Qwen3-VL-8B / MAI-UI-8B)
# image: mobile_world:reset   ·   network: mwnet
```

!!! warning "绝不复用容器"
    MobileWorld 容器在任务之间**不**重置 app 状态，所以跑过一次 eval 的容器带着脏状态。每次运行都用 `mwnet` 网络上的全新容器池（host 的默认 docker bridge 已不在）。只拆你自己创建的容器。启动时的 MCP-401 噪音对 GUI-only 的 110 任务清单无害。

### 2 — 构建注入 guidance

B2 系列 guidance 已预构建在 `artifacts/` 下。要重新生成最强静态配置（**B2′**，app 级 Layer-2 + category `L_C`，无 Layer-1）：

```bash
python harness/gen_b2_guidance.py --mode app-l2     # -> artifacts/b2p_guidance.json
```

!!! note "跨基准下 Layer-1 严格有害"
    源环境的 Layer-1 状态描述会挤掉接地行为，所以最强静态配置是 `--mode app-l2`（只用 Layer-2）。不要把 Layer-1 加回来。其他模式：`--mode full`（B2，完整 app FSM）·`--mode category-only`（B2″，category `L_C`）。

### 3 — 在 110 任务的 `T_eval` 上评测

在 110 任务 eval 清单上运行 `mw eval`。B1 = 原版 agent；B2′ = 文件路径 agent（`harness/qwen3vl_b2_agent.py`），通过 `EVOFSM_B2_GUIDANCE` 指向 guidance JSON。

```bash
mw eval ... --max_round 50 --max-concurrency 3 --enable_mcp --enable_user_interaction
```

Eval 轨迹落在 `MobileWorld/traj_logs/<run>/<task>/result.txt`——`score: 1.0` 即通过，分母为 110。各臂的确切命令和容器端口段在 `EvoFSM-MW/harness/README.md` 与 `docs/qwen3_8b_res.md` 的 setup 块里。

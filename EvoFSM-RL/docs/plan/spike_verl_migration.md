# Spike plan — 评估把 EvoFSM-RL 的 online RL 迁到 verl / SkyRL

**日期**：2026-06-01
**作者**：Claude + linqiang
**状态**：proposed（spike，未启动）
**时长上限**：2–3 工作日（time-boxed，到点必须给 go/no-go）

---

## 0. 为什么做这个 spike

我们的 GRPO + LoRA 训练（Phase 1 产 π^pre、Phase 3 per-app TTA）是**自写的 ~500 行框架**。两个驱动力：

1. **正确性无背书**。`evofsm_rl/rl/grpo.py` 的 `grpo_step` loss/梯度数学**没有任何数值 ground-truth 测试**（`test_grpo.py` 只测了 `compute_reward` / `compute_advantages` 两个纯函数 + 一个返回全 0 的假 model 的字段管道）。审计还发现 3 个偏差/隐患：
   - 实现的是 **REINFORCE + group-mean baseline**，不是 paper/method.tex 写的 PPO-clip GRPO（没有 importance ratio ρ、没有 clip）。*在严格 on-policy 单步更新下 ρ≡1，对实际跑的东西数学上是对的*，但代码与论文公式不符。
   - **梯度 forward 时 dropout 开着**（`grpo.py:412` `model.train()` 为开 gradient-checkpointing，连带打开 LoRA dropout=0.05）→ 给梯度注入噪声。
   - **advantage 没除 std**（标准 GRPO 是 `(r-mean)/(std+eps)`）；**KL 在 sequence 级算**（数值脆，靠 `clip=10` 补丁），非 per-token。
2. **emulator-bound 瓶颈 + 算力**。rollout 占 ~70% wall-time。Phase 1 是算力大头（v3-B：12 app × 200 iter × K=2 = **4800 traj**；pilot 4800 同量级），且 v2/v3 历史上训飞过两次。并发 emulator rollout 是唯一能救 wall-time 的杠杆。

**候选**：[SkyRL-AndroidWorld](https://github.com/Guliisgreat/SkyRL-AndriodWorld)（Guliisgreat fork of NovaSky-AI/SkyRL，Apache-2.0，0 star，个人 fork）。底层训练后端是 **verl**（`skyrl_agent.integrations.verl.verl_main_ppo`），verl 是业界检验最充分的 RLHF 框架——正好回答"正确性"担忧。

---

## 1. 这个 fork 已经做好了什么 vs 我们仍要建什么

### ✅ 现成可复用（这是最大价值，也是我们自建要数周的部分）
- **AndroidWorld ↔ SkyRL 的 env 桥**：每个 emulator 包成 Docker + ADB + FastAPI server，`/step`（截图）/ `/step_adb`（a11y tree）。
- **并发 emulator 池 + broker**：16+ 容器并行，broker 持久池"survives script restarts"。**= parked 的"并发 rollout 2-3x 提速"，预先做好。**
- **多模态截图 observation** 支持（base64 → pixel_values）。
- **verl 后端**：vetted PPO/GRPO，含内置 KL（见 §3）。

### ✅ 环境可行性（2026-06-01 已在 H200 box 验证）
- `docker 20.10.24` + **nvidia runtime** 在；`linqiang` 同时在 `kvm(109)` 和 `docker(999)` 组；`/dev/kvm` 可用。
- 这台机器**已经有人在跑 OSWorld 的 Docker GUI 容器**（happysixd/osworld-docker）→ Dockerized GUI-agent emulator 在本机被验证可行。
- 16 emulator 是 CPU/KVM/RAM-bound，**不占 GPU**，所以 2-GPU 限制不影响并发数；GPU 只用于 trainer 侧（vLLM + FSDP）。

### ❌ 仍要我们自己做 / 风险
- **base model 不匹配**：fork 示例配置是 **Qwen2-VL-7B**，我们要 **Qwen3-VL-8B**。可插（vLLM/verl 支持）但**没人在它的多模态路径上验过 Qwen3-VL**；我们的"每轮 2 张图 M3A packing"和它不一定一致 → 多模态模板风险。
- **LoRA 没提**：fork 大概率 full-FT。B4 是 rank-16 LoRA + 锚 π^pre。verl 支持 LoRA，但要我们加/验。
- **EvoFSM 符号那一半（Opus 演化 L_C + FSM population + TrueSkill）不在里面**，且在 generator-driven loop 里硬塞别扭。
  → **解耦方案**：SkyRL/verl 只管内层 RL/LoRA；L_C/FSM/Opus 演化保留在我们外层 loop，在训练轮次之间把当前 L_C 注入 env server 拼的 prompt。HTTP env 接口让这种解耦更干净。
- **成熟度**：0 star 个人 fork、无公开结果、churning Dockerfile（v8/v9/2026/tier4）→ **不是白送正确性**，我们仍需自验。

---

## 2. Milestones（Phase-1-first）

> **核心决策（本次更新）：第一个迁的不是某个 Phase 3 app，而是 Phase 1。**
> 理由：(1) Phase 1 是**纯 RL**，loop 里**无 Opus/L_C 演化**（FSM 冻结，`fsm_variant_id="static_{app}"`）→ verl 的零-glue 完美 fit，避开上面那条最难的符号集成；(2) Phase 1 的 **KL ref = base**（`--anchor-to-base`），正好是 **verl 默认行为**，连 §3 那个自定义 ref 都省了；(3) Phase 1 是**算力大头 + 历史最不稳**（训飞过两次）→ 并发 rollout 收益最大。迁完拿到一个稳的、并行的、vetted 的 π^pre，可直接喂回现有 Phase 3。

### M0 — 环境与冒烟（~0.5 天）
- clone fork，build 它的 Android Docker 镜像（`Dockerfile.full_adb_agent`）。
- **关键对照**：看它的镜像装的是哪套 AVD/apps，能否复用我们 `AWAvd2` snapshot 的 14 个 app（尤其 6 个 Plus app + audio_recorder 的 records 目录）。若它只有 vanilla AW app → 评估补 Plus app 的成本。
- 起 **2–4 个**容器（不上 16），跑一个它自带的 e2e demo，确认 `/step` 截图链路通。
- **Go/no-go**：能起容器 + demo 通 → 继续；起不来或 apps 缺太多 → 停，回写为什么。

#### M0 执行记录（2026-06-01）— ✅ build + smoke 通过，但发现 task 集需补

**镜像**：`androidworld:evofsm`（86GB，~63GB 与 base `androidworld:2026plusswipe` 共享层，新增 ~22GB）。
做法：**没从 fork clone 直接 build**（缺 `.android`/`RL4AndroidWorld` 大资产，走不通）；改为**复用现成 base + 烤入我们的 AVD**。零代码 patch，全靠 env-var 切换。emulator/系统镜像一致（build 35.3.11.0 / android-33 x86_64）→ 快照可 resume。

**smoke（私有 ADB 端口 5599 + 独立端口，`-gpu off`，对别人零影响）全过**：
- `apps_ready_dec2025` 快照 resume → `Emulator verified ready`；`/health`→ready。
- `/reset task_id=85`（sequential 模式）→ `SimpleCalendarAddOneEvent` 真初始化（拿到任务目标 + ui_elements + 2400×1080 截图）；`/step` 返回 reward/terminated/obs。
- 两个运行配方坑：(1) **`SERVER_PORT` 必须 ≠ `EMULATOR_PORT`**（同值则 server bind 自撞 `address already in use`）；(2) 默认 `ENV_SAMPLE_MODE=random` 忽略 `task_id`，要确定性选 task 须 `=sequential`。

**关键发现（回答 M0 第 53 行那条"apps 够不够"）**：镜像的 task registry 是**上游 vanilla android_world = 116 个 task**，`skyrl_server/registry_ext.py` 用**显式 import 白名单 + 硬编码 `_V8_ANDROID_WORLD_ORDER`** 固定这 116 的顺序（JSONL 用 task_id 当下标依赖此序）。我们 `android_world_plus` 的 6 个扩展 app 的 **task 定义代码不在镜像里**。
- 裂口：AVD 快照**物理装了**扩展 app，但**"怎么算成功"的 reward 代码缺席** → server 只能吐 116。
- EvoFSM rollout **走 skyrl_server（trainer HTTP 调容器）** → 此限制**在关键路径上，必须补**。

**兼容性已逐文件核实（image=2026+swipe vs plus=2025+apps，真分叉但互补）**：
| 文件 | 结论 |
|---|---|
| `json_action.py` | image 独有像素 swipe（base 依赖）→ **保 image** |
| `task_eval.py` | plus +`get_dense_reward()`（opt-in 追加）→ port 入 image |
| `sqlite_validators.py` | plus +dense-reward 行计数（image 侧无实质改动）→ 取 plus 版 |
| `adb_utils.py` | plus +6 app activity 映射（保 image 的 typo 修复）→ 合并 |
| `sqlite_utils.py` | plus 的 `pysqlite3` fallback import（严格更稳）→ 采 plus |
| `file_utils.py`/`interface.py`/`sqlite_schema_utils.py` 等 | image 更新或仅版权差；扩展 app 不依赖 plus 差异（`TMP_LOCAL_LOCATION` 无人引用）→ **保 image** |

**最终修补方案（以 image 为底 port plus 扩展；不换包，护住 skyrl_server 既有依赖）**：
1. 新增 6 个模块 `task_evals/single/{bluecoins,maps_me,pimusic,snapseed,wikipedia,calculator}.py` + `env/setup_device/bmoca_apps.py`（去 Walmart 部分）。
2. 合并上表 4 个文件的 plus 追加。
3. 改 `registry_ext.py`：import 6 模块 + 把 **77 个**新 task 类**追加到 `_V8_ANDROID_WORLD_ORDER` 第 115 之后** → android family **116 → 193**，0–115 原序不动 → 现有 JSONL task_id 不破。
4. rebuild 镜像。
- **walmart 排除**：模块在但 plus registry.py 注册 0 个 task，未接好，本轮不 port。
- **为何不反向**（以 plus 为底只 port swipe）：skyrl_server 是对着 image 这版 android_world 建/验过的，换旧 plus(2025) 有未知 skyrl_server 不兼容；以 image 为底、port 已验证的有界扩展面，风险最低。
- 详见 memory `project_m0_smoke_verified_recipe`。

**✅ 已实施并验证（2026-06-01）**：patch 层落地为新 tag **`androidworld:evofsm-tasks193`**（`FROM androidworld:evofsm` + COPY 6 模块 + 合并 4 core + 改 registry_ext，patch 层仅 286KB）。构建上下文 `/shared/linqiang/evofsm_project/build_evofsm_tasks/`。验证全过：build 前 mount dry-run + build 后静态 = `/get_n_tasks` 193；运行时起容器 → task_id 85 仍 `SimpleCalendarAddOneEvent`（0–115 不变）、116 `BluecoinsQuerySpendingOnDate` init + `/step` 通。**M1 起容器须用 `:evofsm-tasks193`**；旧 `:evofsm`(116) 留作回滚，暂未 retag。

### M1 — Phase 1 GRPO 在 verl 上复现（~1.5 天）⭐ 主 milestone
- 目标：在 verl 上跑 **Phase 1 pilot 配置** = 4 app（bluecoins, markor, calculator, contacts）× 200 iter × K=2，**Qwen3-VL-8B + rank-16 LoRA(q_proj,v_proj)**，KL ref = **base**（verl 默认），`use_kl_loss=True` / `low_var_kl` / `kl_loss_coef` 从 0.001 起扫（见 §3）。对标产物 = `traces/phase1_pilot_v01/`（pilot 200/2，config 见 `phase1_config.json`）。
- 三个验证点（缺一不可）：
  1. **Qwen3-VL-8B 多模态走得通**（2 图/轮 packing 不报错、logprob 合理）。
  2. **LoRA + base-ref KL 能插**（不是 full-FT）。
  3. **rollout 吞吐**：实测 traj/hour vs 我们现在单 emulator 的 loop，量化加速比。
- **产出 π^pre 质量对比**：用得到的 LoRA 做一次 standalone T_eval（frozen + b4_k4_v3binit 的 L_C），和现有 `pi_pre_pilot_4_200_nokl`（standalone 42.4%）比。注意：(a) 我们要加 KL（pilot 原版 nokl）；(b) verl 的 std-norm advantage + per-token KL 使 β 尺度不同，**期望 π^pre 质量 ≥ pilot，甚至更稳**。
- **Go/no-go**：3 个验证点全过 + π^pre standalone 不低于现有 pilot（容忍 noise ±3pp）→ 进 M2；任一硬卡（多模态/LoRA 插不进/吞吐没提升）→ 停，记结论。

### M2 —（条件触发）Phase 3 可行性探针（~0.5 天）
- 仅当 M1 绿。验证 §3 的自定义部分：把 verl 的 `actor_rollout_ref.ref` 指向 **π^pre**（需先把 π^pre merge 进 ref checkpoint 或用 verl LoRA-ref）。
- 验证解耦方案：外层脚本在两次训练轮之间改 env server 拼 prompt 用的 L_C，确认 observation 真的变了。
- **不要**在 M2 跑完整 Phase 3 sweep；只验"ref=π^pre 能配 + L_C 可外部注入"两个机制点。

### M3 — 决策（~0.5 天）
- 汇总：可行性、加速比、π^pre 质量、Phase 3 机制点、剩余工程量估计。
- 输出三选一：**(A) 全迁**（Phase 1 + Phase 3 都上 verl）/ **(B) 只迁 Phase 1**（拿稳的 π^pre，Phase 3 留自写）/ **(C) 不迁**（回写为什么，转而只做"数值等价测试 + 修 3 个偏差"给自写框架背书）。

---

## 3. KL anchor 迁移要点（verl 已内置，但有两件事必须自己处理）

verl 的 GRPO **内置 KL-to-reference**，不用手写（我们 `grpo.py` 的手写 KL + `clip=10` 补丁会变死代码）：
- `actor_rollout_ref.actor.use_kl_loss=True`（GRPO 该设 True，KL 加在 loss 侧）
- `kl_loss_type=low_var_kl`（**= 我们手写的 Schulman k3**，但 verl 是 per-token、稳定版，更优）
- `kl_loss_coef` = 我们的 β（verl 默认 0.001）
- ref policy 由 `actor_rollout_ref.ref` 维护
- ⚠️ **KL 别加两次**：用 loss 侧 `use_kl_loss` 就要把 reward 侧 `kl_coef`/`kl_ctrl` 关掉（verl issue #265）

**必须自己处理的两件事：**
1. **ref 指向谁**：Phase 1 ref=**base** = verl 默认（M1 省事的原因）；Phase 3 ref=**π^pre**，需把 π^pre 指给 `actor_rollout_ref.ref`（M2 验）。
2. **β 重扫**：我们 Table 6 的 β=0.05 甜点是针对**有缺陷的自写实现**（sequence-level KL + 无 std-norm + dropout 开）调的；verl 的 per-token KL + std-norm advantage 在不同尺度，从 0.001 起扫，别照搬 0.05。

---

## 4. 决策准则（一句话）

- 目标若是"**确认自写 GRPO 对不对**" → spike 不是最便宜路径，直接做"数值等价测试 + 修 dropout/std-norm/KL 3 偏差"。
- 目标若是"**拿更快、可扩展、vetted 的 RL 后端 + 铺 cross-base/30B 的路**" → 本 spike 的 Phase-1-first 路线是当前最佳起点，**先 time-box 验证、用数据决策，不盲目 all-in**。

## 5. 相关
- 自写 GRPO：`evofsm_rl/rl/grpo.py`，Phase 1 入口 `scripts/run_phase1_pretraining.py`（online RL，与 Phase 3 同一套 loop）
- 结果与归因争议：`docs/results/experiments.md`（B4 干净 pipeline 48.1%；唯一 L_C-固定的 LoRA 消融 v3-B 为 −5.7pp）
- π^pre 命名/产物：`docs/results/experiments.md` Table 2；pilot config `traces/phase1_pilot_v01/phase1_config.json`

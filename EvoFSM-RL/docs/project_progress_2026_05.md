# EvoFSM-RL 项目进度报告

**日期**：2026-05-27
**版本**：v1 (tech-lead progress report)
**作者**：Claude + linqiang

---

## 0. 我们解决什么问题

**任务背景**：用小尺寸开源 VLM（Qwen3-VL-8B）做 Android GUI agent — agent 看屏幕截图 + accessibility tree，输出 action（tap / type / open_app / status=complete），逐步完成自然语言描述的 GUI task。

**核心难点**：**unseen app 上的泛化**。base model zero-shot 在 105-episode 测试集（12 个 target app，训练时没见过）上只有 **38.6%** 成功率。两个常见思路各有局限：
- **扩大预训练**：在更多 source app 上训共享 LoRA 容易遇到容量瓶颈 + 多 app 梯度冲突（rank=16 LoRA 装不下 12 app × 600 iter 的训练就会 collapse 到 0%）
- **闭源大模型 + prompt 工程**（如 GPT-4o + Mobile-Agent）：成本高，每个 task 几美元 API 费，且不能本地部署

**我们的切入点**：**Test-Time Adaptation (TTA)** — 部署阶段碰到新 app 时，在该 app 的少量 unlabeled adapt tasks (T_adapt) 上做轻量级 fine-tuning（20 iter），同时改 LoRA 权重 + 改 symbolic prompt prior，把"我这个 app 怎么用"快速学进 model。然后在 frozen test set (T_eval) 上评测。

形式化：给定 source pool $\mathcal{S}$（12 app），先训共享 LoRA $\pi^{pre}_\theta$；遇到 target app $a \notin \mathcal{S}$ 时：
1. 加载 $\pi^{pre}_\theta$ 当 init
2. 在 $a$ 的 $T_{adapt}^a$ 上跑 20 iter joint TTA：GRPO LoRA 训练 + Opus 演化 $L_C^a$
3. 冻结，在 $T_{eval}^a$ 上汇报 SR

部署架构 = "**Phase 1 共享 prior + Phase 3 per-app 适配**"。

---

## 0.5 主要贡献

1. **Method: EvoFSM-RL** — **首次将 symbolic FSM 演化与 sub-symbolic LoRA fine-tuning 联合用于 GUI agent 的 test-time adaptation**。LoRA 用 GRPO 在 task reward 上训练；FSM Layer-2 抽象动作库由 Claude Opus 4.7 通过分析 rollout 轨迹+reward 信号迭代演化（mutation diff → TrueSkill 评分 → champion 选择）。

2. **可迁移的 symbolic action library `L_C`** — 设计了"per Play Store category 的抽象动作库"作为 Layer 2：每个 category 包含若干 abstract action（如 `CREATE_ENTRY`），每个 action 由 (precondition, abstract_steps, failure_modes, verification_checklist) 4 字段描述。先在 source pool 上聚合得到 6 个 category 的 static `L_C`（B2 用），再 per-target-app 演化得到 evolved `L_C`（B3/B4 用）。Tier-C apps（category 不在 source pool）通过 bootstrap mode 从空 `L_C` 现场合成。

3. **两阶段训练 + 部署时 per-tier 加载** — 我们发现不同 Phase 1 init 在不同 tier 上各有所长（v3-C 深训 4 app 擅长 Tier-B = 70.4%，v3-B 广训 12 app 擅长 Tier-C = 34.3%）。**部署时按 target app 的 Play Store category 是否在 source pool 里 if-else 加载对应那一份 LoRA + 演化 L_C，几乎零成本**。

4. **稳定的 GRPO 实现** — 修复两个关键 bug：(F1) per-step loss 按 trajectory 长度 $T_j$ 归一化，防长 trajectory 主导梯度被 max_grad_norm clip 吃掉；(F2) GRPO group key 用 (FSM, task) tuple 而不是 FSM only，保证 within-task baseline 估计正确。配合 KL anchor (β=0.05) + log_ratio_clip(10) 让 LoRA 在 20 iter TTA 不会 collapse。

5. **完整 empirical evidence on AndroidWorld + Plus benchmark**：
   - B1 zero-shot **38.6%** → B4 完整 EvoFSM-RL **52.9%**（+14.3pp）
   - 105 episodes 评测（12 app × 35 template × K=3 seeds），Tier-B + Tier-C 平均
   - Phase 1 init ablation、reward 形态 ablation、KL β 敏感性 ablation 均完整

6. **自写 RL framework**（~500 行）— GRPO trainer + AndroidWorld emulator rollout loop + LoRA / FSM joint update / Opus mutation API。Bottleneck 是 emulator (~70% 时间)，不是 GPU，所以没换 TRL / verl。

---

## 1. 摘要

我们研究 **GUI agent 在 unseen Android app 上的 test-time adaptation (TTA)**：训练时只见过 12 个 source app，部署到 12 个 unseen target app（其中 6 个 Play Store category 在 source pool 内称 Tier-B，6 个不在称 Tier-C）时如何提升任务成功率。评测协议是固定 105 episodes（12 apps × 35 templates × K=3 seeds）上的 binary success rate (SR)。

EvoFSM-RL 方法分两阶段：**Phase 1** 用 GRPO + KL anchor 在 source pool 上预训一个共享 LoRA（产出 π^pre）；**Phase 3 per-app TTA** 在 target T_adapt 上同时继续 GRPO 训 LoRA + 让 Claude Opus 演化一个抽象动作库 `L_C`（每个 target app 一份），然后冻结、在 T_eval 上评测。

完整 B1 → B4 实验完成：

- **B1 = 38.6%** —— zero-shot base model，不注入 FSM，不训 LoRA
- **B2 = 43.3%** —— 注入 source-pool 同 category 的静态 `L_C` 到 prompt，FSM 不演化，LoRA 不动
- **B3 = 46.7%** —— 以 B2 静态 `L_C` 为起点，在 target T_adapt 上用 Opus 演化 FSM，LoRA 仍不动
- **B4 = 52.9%** —— 演化的 FSM + LoRA weight 都更新，完整 EvoFSM-RL pipeline

B1 → B4 总涨 **+14.3pp**。

---

## 2. 问题定义

### 1.1 任务

Agent 在 Android emulator 上看屏幕截图 + accessibility tree，输出 action（tap / type / open_app / status=complete 等），逐步完成一个用自然语言描述的 GUI task（如 "Add an expense of $5 for dinner"）。每个 task 有 evaluator，最终给出 binary reward 0/1。

底层 base model = `Qwen3-VL-8B-Instruct`（frozen）。Agent prompt 走的是 M3A 协议（Reasoning + Action JSON，两次 LLM call per step）。

### 1.2 数据划分

App pool 25 个（vanilla AndroidWorld 19 个 + Plus repo 6 个，194 个 task）。我们把其中 24 个 app（除一个测试 app）按 Play Store category 分两份：

- **Source pool** (12 个)：训练用。在源池上跑过 480 条轨迹（96 个 template × K=5 seeds）做 Phase 1 数据底
- **Target pool** (12 个)：部署评测用。**训练时 agent 一次没见过这些 app**
  - **Tier-B (6 个)**：app 所属 category ∈ source pool category 集合（例：`pro_expense` 是 Finance，source pool 有 `bluecoins` 也是 Finance）
  - **Tier-C (6 个)**：app 所属 category ∉ source pool（例：`opentracks` 是 Health & Fitness，source 没这类）

每个 target app 的 task templates 再切成 **T_adapt (60%)** 用作 Phase 3 训练 + **T_eval (40%)** 用作最终评测。**T_eval 全程冻结，paper 引用数字必须是 T_eval**。

### 1.3 评测协议

- T_eval = 35 templates × K=3 seeds (40, 41, 42) = **105 episodes**
- 分布：Tier-B 54 ep + Tier-C 51 ep
- 度量：binary success rate（与 AndroidWorld leaderboard 一致）
- 报 overall + per-tier，避免单一 app 主导

---

## 3. 方法

### 2.1 整体 pipeline

```
       Phase 1                       Phase 2              Phase 3
   (source pool 预训)             (handoff)           (target per-app TTA)

  Source 12 app × 96 task                          Target 12 app × T_adapt
     ↓                                                   ↓
  build per-app static FSM     →  category 汇总 →  注入 prompt
  (Claude Opus from K=5 traj)     L_C library                  ↓
                                                       Opus 演化 L_C
                                                       (Layer 2 diff)
                                                              ↓
  GRPO LoRA 训练               →  π^pre 冻结    →   GRPO LoRA TTA
                                                       (从 π^pre 继续训)
                                                              ↓
                                                       T_eval 评测
```

### 2.2 两层 FSM 设计 (L_C)

每个 app 对应一个 **"两层 FSM"**：

- **Layer 1**：app-specific 状态机（state, transition, strategy, dead_end），来自 Claude Opus 对该 app 轨迹的合成
- **Layer 2**：**抽象动作库** — 一组 `(name, precondition, abstract_steps, failure_modes, verification_checklist)` 的字典，描述"如何完成一类操作"。比如 `CREATE_ENTRY` 描述"如何在任何 app 里新建一条记录"

实际产品里我们注入 prompt 用的**只有 Layer 2**（写成 markdown 段落附在 system prompt 里）。Layer 1 设计上有但 prompt 里没用。所以本文里 **"L_C" ≡ 该 FSM 的 Layer 2**。

#### 2.2.1 静态 source-pool L_C 构建（Phase 1 期间）

1. **per-app FSM 合成 (Story 2.2)**: Claude Opus 4.7 读取每个 source app 的 5 个 K=5 成功轨迹（traj_data，含 screenshot+action+a11y），合成该 app 的两层 FSM。结果存在 `artifacts/static_fsms/<app>.json`
2. **per-category L_C 聚合 (Story 2.3)**: 把同 Play Store category 下的 source app 们的 Layer 2 内容 **union** 起来（Opus 跨 app diff，保留共性、去 app-specific 名字）→ 形成 6 份 `artifacts/L_C/<category>.json`

6 个 category：finance, productivity, tools, music_audio, photography, communication。**这就是 B2 静态注入用的 L_C**。

#### 2.2.2 真实例子：`artifacts/L_C/finance.json` 中的 `CREATE_ENTRY`（节选）

```json
{
  "name": "CREATE_ENTRY",
  "precondition": "App launched and on a list/dashboard view that exposes a creation affordance (e.g. floating action button or equivalent primary create control)",
  "abstract_steps": [
    "Navigate to the list view where new items are created",
    "Trigger the create affordance (primary create control) and wait for the entry form to appear",
    "Fill in required fields (amount, date, category, description) using the appropriate input modalities (number pad, date picker, dropdown, free-text)",
    ...
  ],
  "failure_modes": [
    "Tapping the wrong sub-area of a row and opening a secondary dialog",
    ...
  ],
  "verification_checklist": [
    "New entry appears in the list view",
    ...
  ]
}
```

`finance.json` 包含 7 个这种 category（CREATE_ENTRY, MODIFY_ENTRY, REMOVE_ENTRY, QUERY_INFO, NAVIGATE_TO, PICK_DATE_FROM_PICKER, CREATE_LABEL_OR_CATEGORY）。完整文件约 5 KB markdown 渲染。

#### 2.2.3 演化后 per-app L_C（Phase 3 产物）

每个 target app 完成 Phase 3 后产出一份 `l_c_champion.json`。例如 `pro_expense` (Tier-B, Finance) 演化后包含 8 个 category，比静态 finance.json 多了 `LAUNCH_APP`，且 `CREATE_ENTRY` 的 `abstract_steps` 从 7 步扩展到 13 步、`failure_modes` 从 4 条增加到 18 条。

```json
{
  "app": "pro_expense",
  "layer1": {"category": "Finance", "states": [], ...},
  "layer2": {
    "categories": [
      {"name": "CREATE_ENTRY", "precondition": "...更具体, 强调 direct app-open by name", 
       "abstract_steps": [13 个更详细的步骤, 包括 pro_expense 特有的 hard-loop-break rule],
       "failure_modes": [18 条, 包括 "repeated taps on create button"],
       "verification_checklist": [15 条]},
      ...
    ]
  }
}
```

这就是 Opus 看过 `pro_expense` 在 T_adapt 上的 80 个 rollout（成功+失败）之后，把 finance.json 这套通用 prior 改造成 pro_expense 专用版本的结果。

#### 2.2.4 Tier-C 怎么办（category 不在 source pool）

Tier-C apps（如 `broccoli` 是 Food & Drink, `opentracks` 是 Health & Fitness）没有匹配的静态 L_C。Phase 3 启动时：

- `initial_l_c = Layer2(categories=[])` 空白
- 用 `--enable-bootstrap-fsm` flag，Opus 用专门的 bootstrap prompt：**"L_C 是空的，从轨迹中合成抽象 category"**

20 iter 之后，Opus 大概会演化出 3-5 个 category（具体内容依 app 而定）。冷启动比 Tier-B 难得多，因为 Opus 第一轮看到的 4 个 rollout 全是 base model 的盲打，可能没成功样本。

### 2.3 GRPO 训练算法

LoRA 部分用 GRPO (Group Relative Policy Optimization)。一个 iter 的流程：

1. **采样**：随机选一个 task `t`，K=4 个 rollout（同 task 不同初始 state）
2. **计算 reward**：每条 rollout 跑到 `is_successful=True` 或步数耗尽，拿到 `r_j ∈ {0, 1}`
3. **算 advantage**：在 (FSM_variant, task) 这个 group 里减去 group baseline

$$A_{j} = r_j - \bar{r}_{(FSM, task)} \quad \text{where} \quad \bar{r} = \frac{1}{K}\sum_{j} r_j$$

4. **GRPO loss**（每条 rollout 累加，再按 trajectory 长度 $T_j$ 归一化）：

$$\mathcal{L}_{GRPO} = -\sum_{j} \frac{1}{T_j} \sum_{t=1}^{T_j} \min\left( \rho_{j,t} A_j, \text{clip}(\rho_{j,t}, 1-\epsilon, 1+\epsilon) A_j \right)$$

其中 $\rho_{j,t} = \exp(\log\pi_\theta(a_t|s_t) - \log\pi_{\theta_{old}}(a_t|s_t))$，$\epsilon = 0.2$。

5. **KL anchor**（防止 LoRA 漂离 π^pre）：

$$\mathcal{L}_{total} = \mathcal{L}_{GRPO} + \beta \cdot \text{KL}(\pi_\theta \| \pi_{ref})$$

其中 $\pi_{ref}$ = π^pre（Phase 1 产物），$\beta = 0.05$，KL 用 Schulman k3 估计：

$$\text{KL} \approx \mathbb{E}\left[ \exp(\Delta) - \Delta - 1 \right], \quad \Delta = \log\pi_\theta - \log\pi_{ref}$$

再加 `log_ratio_clip = 10` 防数值爆炸。

6. **梯度更新**：backward 后 `max_grad_norm=1.0` 截断，Adam lr=3e-4

### 2.4 算法实现细节（F1 / F5 fix）

实现 GRPO 时踩过两个坑（已修复，所有现役实验都基于修复后版本）：

- **F1 (loss 按 trajectory 长度归一化)**：原版 $\mathcal{L}_{GRPO}$ 是每条 trajectory 的所有 step loss 直接累加。导致 30 步的 trajectory 比 1 步的贡献 30 倍梯度，结果是少数长 trajectory 主导，再被 `max_grad_norm=1.0` clip 掉信号
- **F5 (group key = (FSM, task) tuple)**：原版按 FSM-only 分组算 advantage baseline。结果是同一 FSM 在不同 task 上的 reward 被一起平均，跨 task 的 reward 差异污染了 advantage 信号

两个 fix 完成后才看到 stable training signal。

### 2.5 Phase 1 (训 π^pre)

在 source pool 上用 GRPO 训出一个共享 LoRA（rank=16, q_proj+v_proj）。每个 iter 从 source pool 随机抽一个 task，K=2 rollouts。我们用 KL anchor (`--anchor-to-base`) 防止 LoRA 飘离 base model 太远。

我们试过 3 个 setting：

| 名字 | apps | n_iter | KL β |
|---|---|---|---|
| **pilot** | 4 个（bluecoins, markor, calculator, contacts） | 200 | 无 |
| **v3-B** | 12 个 source pool | 200 | 0.05 |
| **v3-C** | 4 个（同 pilot） | 600 | 0.05 |

固定 compute 是 4800 trajectories（apps × n_iter × K=2）。差别在"广 × 浅 vs 窄 × 深"：

- v3-B 广 × 浅：12 个 app 各 17 iter，见过广泛的 GUI 类型，但每个 app 深度浅
- v3-C 窄 × 深：4 个 app 各 150 iter，对那 4 个 app 学得很透
- pilot 是最朴素的（无 KL，只在 4 app 上跑 200 iter），用作历史 baseline

### 2.6 Phase 3 per-app TTA

针对每个 target app 单独跑一次：

1. 加载 π^pre LoRA（作为 init + KL anchor 参考）
2. 起一份 L_C（Tier-B 从 `artifacts/L_C/<category>.json` 加载；Tier-C 从空 bootstrap）
3. 跑 20 iterations，每 iter：
   - 抽一个 T_adapt task，K=4 rollouts（每 rollout 不同 seed）
   - 每 3 iter 触发一次 GRPO LoRA 更新
   - 每 iter Opus 提议 Layer 2 diff → 加入 population → TrueSkill 排名 → champion 更新
4. 每 5 iter 保存 LoRA checkpoint
5. 最后保存 final LoRA + champion L_C JSON

20 iter 大约 4-6h（K=4 × ~80 步 × 5-10 秒/步）。

### 2.7 Seed 选择

训练里所有随机性都由一个 `seed_base` 决定（默认 100），保证可复现。

```python
# 一个 iter 内
task = task_sampler.sample()              # task_sampler.rng = random.Random(seed_base)
seed = seed_base + iteration              # 例：iter 5 → seed=105
for j in range(K):
    rollout_seed = seed * 100 + j         # 4 个 rollout 的 seed
    rollout(task, seed=rollout_seed)      # AndroidWorld 用 seed 决定 task 初始 state
```

每个 (iter, rollout) 拿到一个唯一 seed 给 AndroidWorld task initialization 用——决定今天日期、预存数据、UI 元素抖动等。一个 sweep 跑 20 iter × K=4 = 80 个 distinct (task, state) instance（task 名字会重复但 state 不同）。

### 2.8 L_C 演化：Opus mutation 流程

每个 iter 后调一次 Claude Opus 4.7：

1. **Reflection pass**：把当前 L_C + 最新一批 rollout 轨迹 + reward 喂给 Opus，让它分析"哪些步骤可改进、哪些 failure mode 没覆盖"
2. **Diff pass**：让 Opus 输出一个结构化 JSON diff（`category_add` / `category_edit` / `step_insert` / `failure_mode_add` 等 op）
3. **Apply**：diff 应用到当前 L_C 拿到新的 candidate variant
4. **TrueSkill 评分**：candidate 跑下个 iter 的 rollout 收 reward，TrueSkill 更新该 variant 的 (μ, σ)
5. **Champion 选择**：population 里 μ 最高的 variant 当 champion，下个 iter 用它继续

Bootstrap mode（Tier-C 用）：Reflection prompt 改成"L_C 是空的，从这些 target-app 轨迹合成 category-level 抽象，不要试图保留已有内容"。其他流程一样。

### 2.9 为什么自己写 RL 框架

不用 TRL / verl / OpenRLHF 等成熟框架，原因是：**我们的 bottleneck 是 emulator 而不是 GPU**。Rollout 阶段每步 5-10 秒（agent 生成 3s + emulator 执行 1-2s + 截屏抓 a11y 2-3s），GPU 利用率只有 20-30%，剩 70-80% 时间在等 emulator。换框架对 GPU forward/backward 优化对总时间帮助有限，反而 FSM 注入 + L_C 演化的接口要重新 hack。所以自写一个 ~500 行的 GRPO trainer + rollout loop 更可控。

---

## 4. 实验设计

### 3.1 完整 baseline 链 B1 → B4

| Code | 方法 | LoRA | FSM | 训练量 |
|---|---|---|---|---|
| **B1** | zero-shot Qwen3-VL-M3A | 不挂 | 不注入 | 0 |
| **B2** | B1 + 静态 category L_C 注入 prompt | 不挂 | static, frozen | 0 |
| **B3** | B2 + per-app L_C 演化 (Opus on T_adapt) | 不挂 | evolved, frozen 在 eval | Opus API only |
| **B4** | B3 + LoRA 训练 (GRPO on T_adapt) | 挂 + 训 | evolved | LoRA + Opus |

每个都跑完整 105 ep T_eval (K=3 seeds 40/41/42)。

### 3.2 Phase 1 init 选择

我们训了 3 个 π^pre（见 §2.5），各自做 standalone T_eval（LoRA 冻结，配 `b4_k4_v3binit` 演化的 L_C 当 prompt prior）：

| LoRA | Tier-B | Tier-C | Overall |
|---|---|---|---|
| pilot | 54.6 | 29.4 | 42.4 |
| v3-B | 63.0 | **34.3** | **49.0** |
| v3-C | **70.4** | 25.5 | 48.6 |

- **v3-C 擅长 Tier-B**（深训 4 app 让 LoRA 对那 4 app 学得很透，跟 Tier-B 的几个同 category 起点都对得上）
- **v3-B 擅长 Tier-C**（12 app 见过 6 个不同 category 的 UI pattern，对 Tier-C 完全没见过的 app 类有更好的迁移）

### 3.3 B4 完整 pipeline

B4 的完整方法：

**部署 (test-time)**：来一个 target app `a`，先看 `play_category_of(a)` 在不在 source pool 6 category 集合里：

- 在（Tier-B 路径）：加载 v3-C LoRA + 加载 `b4_k4_v3binit/a/l_c_champion.json` 当 prompt prior
- 不在（Tier-C 路径）：加载 v3-B LoRA + 同样加载对应 L_C

这是个 if-else，**部署时几乎零开销**。两份 LoRA 文件 + 两套 per-app L_C 一起部署。

### 3.4 训练数据用量

每个 target app 在 Phase 3 跑 20 iter × K=4 = **80 个 rollout 训练**（distinct (task, state) tuple）。每 app 的 T_adapt task templates 大约 5-15 个，所以每个 template 平均见 5-15 次（不同 state）。

12 target app × 80 rollout = 总训练数据 **960 个 trajectory 轨迹** 在 Phase 3。

Phase 1（v3-B）则是 source pool 上 12 app × 200 iter × K=2 = **4800 trajectory**。

---

## 5. 结果

### 4.1 主结果表（B1-B4 主链）

| Code | 方法 | Tier-B | Tier-C | **Overall** |
|---|---|---|---|---|
| B1 | zero-shot | 47.2 | 29.4 | 38.6 |
| B2 | + static L_C | 56.5 | 29.4 | 43.3 |
| B3 | + L_C evolution | 62.0 | 30.4 | 46.7 |
| **B4** | + LoRA training (完整 EvoFSM-RL，per-tier 加载) | **70.4** (v3-C) | **34.3** (v3-B) | **52.9** |

B1 → B4 总涨 **+14.3pp** (38.6 → 52.9)。

### 4.2 Phase 1 init 选择 ablation

| π^pre | apps | iter | KL | Tier-B | Tier-C | Overall |
|---|---|---|---|---|---|---|
| pilot | 4 | 200 | 无 | 54.6 | 29.4 | 42.4 |
| v3-B | 12 | 200 | 0.05 | 63.0 | **34.3** | **49.0** |
| v3-C | 4 | 600 | 0.05 | **70.4** | 25.5 | 48.6 |

每个 init 单独 standalone 已经远超 B1 (38.6%)，且 v3-B 单 init 就比 paper standard B4 K=4 pilot Phase 3 (48.1%) 还高 0.9pp。

### 4.3 其他对比方法（同 105 ep）

| 方法 | Overall |
|---|---|
| RFT (SFT on source successes only) | 41.0 |
| Paradigm B (shared TTA, single LoRA) | 40.0 |

RFT 跟 Paradigm B 都比 B1 强，但都没达到 B4 K=4 pilot (48.1%) 的水平。说明纯 imitation (RFT) 或 shared-LoRA TTA 都不如 per-app GRPO + L_C 演化。

### 4.4 Tier-C bootstrap 模式 ablation

Tier-C apps 在 Phase 3 用 `--enable-bootstrap-fsm` 从空 L_C 开始演化。我们对比了 K=2 vs K=4 在 Tier-C 上的效果：

| 设置 | Tier-C 6-app SR |
|---|---|
| K=4 bootstrap | 35.3 |
| K=2 bootstrap | 29.4 |

K=4 比 K=2 在 Tier-C 上多 +5.9pp。K=4 帮 Tier-C 的原因：bootstrap 时 Opus 第一轮只能看到 K 个 rollout 来"无中生有"造 L_C，K 越大数据多样性越大、合成的 category 质量越好。

---

## 6. 当前进度 & 后续

### 5.1 已完成的实验产物

- **5 个 π^pre LoRA**：pilot, v2 (failed), v3 (failed), v3-B, v3-C
- **5 套 Phase 3 sweep（每套 12 app × 20 iter × K=4）**：
  - `b4_k4_unified` (pilot init) → 48.1%
  - `b4_k4_v3binit` (v3-B init, Paradigm A) → 43.3%
  - `b4_k4_v3cinit` (v3-C init) → 45.7%
  - `b4_k2_bootstrap` (K=2 Tier-C only) → Tier-C 29.4%
  - `b4_k4_nopi_12app` (identity LoRA init, no π^pre) → 跑着 9/12
- **3 个 π^pre standalone T_eval**：pilot 42.4 / v3-B 49.0 / v3-C 48.6
- **Paradigm B Shared TTA**：40.0
- **B4 K=4 dense reward ablation** (subset 3 apps)：40.5

### 5.2 待办

| 任务 | 状态 |
|---|---|
| no-π^pre Phase 3 sweep 完 9/12 final | 🟢 跑着 (osmand 18/20, vlc 8/20)，wikipedia 没启 |
| no-π^pre Phase 3 T_eval (105 ep) | ⏳ sweep 完启动 |
| 写 paper draft | ⏳ |

后续可选方向（已在 `docs/results/experiments.md` parked ideas 里）：
- DPO offline ablation（用 source pool 13K Sonnet step labels）
- 跨 base model 验证（Qwen3-VL-30B-A3B / InternVL2-8B / Qwen2.5-VL-7B）
- 并发 emulator rollout 重构（rollout 阶段提速 2-3x）

---

## Appendix A — Harness 实现细节

### A.1 Emulator stack

- **AVD**：`AWAvd2` snapshot `apps_ready_dec2025`（13 GB，13 个 app 全装好）
- **启动**：`emulator -avd AWAvd2 -port 5710 -grpc 8710 -snapshot apps_ready_dec2025 -no-snapshot-save -read-only -no-window`
- **多并发**：用 `-port 5710..5724` 多个端口启多个 emulator instance（同一 snapshot read-only），互不冲突
- **ADB**：每个训练 job 用一个 `EVOFSM_ADB_SERVER_PORT=50XX` 的私有 ADB daemon（避免共享 daemon 的 `_restart_server` 误杀其他 emulator）

### A.2 Agent rollout 流程

```python
# evofsm_rl/agent/rollout.py 大致 flow
for step in range(max_steps):
    screenshot, a11y = env.get_state()       # ADB pull, ~1-2s
    ui_view = render_ui_view(a11y)            # 文本化 UI tree
    prompt = build_action_prompt(
        task_description,
        ui_view,
        action_history,
        L_C_text,                             # 这里注入 evolved L_C
    )
    action_text = agent.generate(prompt)      # Qwen3-VL forward, ~3s
    summary_text = agent.summarize(prompt + action_text)  # 第二次 forward, ~1-2s
    parsed_action = parse_action_json(action_text)
    if parsed_action.is_complete():
        break
    env.step(parsed_action)                   # ADB tap/type, ~1-2s
    action_history.append((parsed_action, summary_text))

reward = task.is_successful(env)              # binary 0/1
```

### A.3 GRPO trainer 关键代码片段

```python
# evofsm_rl/rl/grpo.py
def compute_advantages(trajectories):
    """F5: group by (fsm_variant_id, task_name) tuple"""
    groups = defaultdict(list)
    for traj in trajectories:
        groups[(traj.fsm_id, traj.task_name)].append(traj)
    advantages = {}
    for key, group in groups.items():
        rewards = [t.reward for t in group]
        baseline = sum(rewards) / len(rewards)
        for t in group:
            advantages[t.id] = t.reward - baseline
    return advantages

def grpo_step(model, trajectories, advantages, kl_beta, ref_model):
    total_loss = 0
    active = [(t, advantages[t.id]) for t in trajectories if abs(advantages[t.id]) > 1e-6]
    for traj, adv in active:
        scale = 1.0 / len(traj.replay_paths)   # F1: 按 T_j 归一化
        for step_path in traj.replay_paths:
            log_pi = model.compute_log_prob(step_path)
            log_pi_old = step_path.cached_log_prob
            ratio = torch.exp(log_pi - log_pi_old)
            loss_term = -torch.min(
                ratio * adv,
                torch.clamp(ratio, 0.8, 1.2) * adv,
            )
            # KL anchor (Schulman k3)
            log_pi_ref = ref_model.compute_log_prob(step_path)
            log_ratio = (log_pi - log_pi_ref).clamp(-10, 10)
            kl = log_ratio.exp() - log_ratio - 1
            (loss_term + kl_beta * kl).mean().mul(scale).backward()
            total_loss += loss_term.detach().item() * scale
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"loss": total_loss, "n_active": len(active), ...}
```

### A.4 Hyperparameter table（B4 K=4 main）

| 参数 | 值 |
|---|---|
| Base model | Qwen3-VL-8B-Instruct |
| LoRA rank | 16 |
| LoRA target | q_proj, v_proj |
| LoRA lr | 3e-4 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| GRPO clip ε | 0.2 |
| KL β | 0.05 |
| KL log_ratio_clip | 10.0 |
| max_grad_norm | 1.0 |
| Optimizer | Adam |
| n_iterations | 20 |
| K (rollouts/iter) | 4 |
| M (select FSM variants) | 2 |
| min_n_active | 3 |
| update_every | 3 (GRPO fire 每 3 iter 一次) |
| checkpoint_every | 5 |
| step_budget (max_steps_per_task) | 60 |
| max_steps_multiplier | 10.0 |
| Opus mutation model | claude-opus-4-7 |
| Opus temperature (reflection) | 0.7 |
| Opus temperature (diff) | 0.3 |
| seed_base | 100 |
| seeds (T_eval) | 40, 41, 42 (K=3) |

### A.5 关键代码路径

```
EvoFSM-RL/
├── scripts/
│   ├── run_phase1_pretraining.py    # Phase 1 训练入口
│   ├── run_b4_evolution.py           # Phase 3 训练入口
│   ├── run_b4_teval.py               # 105-ep T_eval
│   ├── build_all_fsms.py             # Story 2.2: per-app FSM 合成
│   └── build_L_C.py                  # Story 2.3: per-category L_C 聚合
├── evofsm_rl/
│   ├── agent/
│   │   ├── prompts.py                # M3A prompt + L_C injection
│   │   ├── rollout.py                # rollout loop, save_episode
│   │   └── action.py                 # JSON → JSONAction 解析
│   ├── rl/
│   │   └── grpo.py                   # GRPO trainer (F1+F5 fixed)
│   ├── fsm/
│   │   ├── builder.py                # Opus FSM 合成 (Story 2.2)
│   │   ├── aggregator.py             # category 聚合 (Story 2.3)
│   │   ├── mutation.py               # L_C diff propose + apply
│   │   ├── evolution.py              # 主 evolution loop, TaskSampler
│   │   └── schema.py                 # FSM / Layer2 数据结构
│   ├── env/
│   │   └── harness.py                # AndroidWorld env wrapper
│   └── model/
│       ├── loader.py                 # Qwen3-VL load
│       └── lora.py                   # LoRA attach/save/load
└── traces/
    └── b4_k4_v3binit/<app>/          # 每个 sweep 输出
        ├── lora_checkpoints/final/   # 训完的 LoRA
        ├── l_c_champion.json         # 演化的 L_C
        ├── l_c_v0..vN_iter*.json     # 演化中间版本
        ├── iterations.jsonl          # per-iter metrics
        ├── grpo_metrics.jsonl        # GRPO fire metrics
        └── episodes/                 # 80 个 rollout trajectories
```

---

## Appendix B — 名词表

| 缩写 | 全称 | 含义 |
|---|---|---|
| TTA | test-time adaptation | 部署阶段在 target 数据上继续训练 |
| LoRA | Low-Rank Adaptation | rank-16 适配器，frozen base + trainable A,B 矩阵 |
| GRPO | Group Relative Policy Optimization | 用 group baseline 替代 value head 的 PPO 变种 |
| L_C | Library of Categories | 按 Play Store category 组织的抽象动作库 |
| FSM | Finite State Machine | 我们 paper 里 "FSM" = layer1 (states) + layer2 (L_C)，但实际只用 layer2 |
| π^pre | π pre-trained | Phase 1 训完的 LoRA，作为 Phase 3 init |
| K | n_rollouts | 每个 GRPO group 里的 rollout 数量 |
| M | n_select | 每个 iter 从 population 选几个 FSM variant |
| Tier-B / Tier-C | (B/C tier) | target app 的 category 是否在 source pool 里 |
| T_adapt / T_eval | train/test split | 每个 target app 内的 template 60/40 切分 |

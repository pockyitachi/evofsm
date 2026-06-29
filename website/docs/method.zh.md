# 方法

EvoFSM 是面向移动 GUI 智能体的**测试时自适应（test-time adaptation, TTA）**方法。当
智能体被部署到此前未见过的 Android 应用上时，它会持续学习：既
**演化出该应用的一种结构化、人类可读的知识表示**，又
**在自身权重上微调一个 LoRA adapter**，二者都由智能体在该应用上自身 rollout 得到的
reward 驱动。这两路更新共享同一批轨迹——权重通道不额外消耗 rollout。

该设计将 **E-SPL**（Zhang 等，2026）的上下文+权重联合优化——它在 RL 训练一个共享
LoRA 的同时演化一段自由文本系统提示，用于单轮推理——迁移到了**多步、视觉、随机、
rollout 昂贵**的 GUI 场景。为让这种迁移成立，有两点改变：用一种*结构化*的知识格式
（两层 FSM）替代自由文本，并缩小每轮迭代的预算（M = 2 个变体，N = 1–2 次 rollout），
因为每个 Android episode 都要耗费 1–3 分钟的真实时间。

---

## 1. 两层 FSM

智能体"如何操作一个应用的知识"是一个两层**有限状态机（Finite State
Machine, FSM）**，渲染为文本后在每一步拼入动作选择提示——策略直接读取自身的知识。
这两层把*不能*迁移的与*能*迁移的分开。

!!! abstract "Layer 1——应用特定（不可迁移）"
    单个应用 UI 的有限状态图：状态、转移、视觉线索、
    元素位置、资源提示，以及具体的死胡同。每个字符串都指向一个
    具体的按钮、对话框或字段，因此这一层在该应用之外毫无意义。

    ```text
    APP: markor   CATEGORY: Productivity

    STATES:
      S1: FILE_LIST — Markor file browser
        visual_cues: ['red plus FAB bottom-right', "'Markor' title", 'list of files']
      S3: NEW_FILE_DIALOG — Dialog to create a new file/folder
        visual_cues: ["Name field pre-filled with 'my_note'", 'OK/CANCEL/FOLDER buttons']

    TRANSITIONS:
      S1 --click(fab_plus)--> S3   [post: dialog with default 'my_note' appears]
      S3 --click(OK)--> S8         [pre: name entered]  [post: new file opened in editor]
    ```

!!! abstract "Layer 2——类别通用（可迁移）"
    抽象的工作流模式，**不含任何应用特定字符串**，
    按 Play Store 类别索引。类别即任务形态原型——
    `ADD_ENTRY`、`QUERY_INFO`、`REMOVE_ENTRY`、`FILTER_LIST`、`TOGGLE_STATE`……
    每个块有四个字段：

    | 字段 | 含义 |
    |---|---|
    | `precondition` | 该原型何时适用 |
    | `abstract_steps` | 与应用无关的动作序列 |
    | `failure_modes` | 需提防的已知跨应用失败模式 |
    | `verification_checklist` | 终止前的后置条件检查 |

    ```json
    {
      "name": "TOGGLE_STATE",
      "precondition": "A binary control is visible and its state is observable",
      "abstract_steps": [
        "Locate the binary control (switch, checkbox, toggle)",
        "Read the current state from visual affordance (icon, label, color)",
        "If the target state differs from current, tap to flip",
        "Wait for the UI to settle and re-read the state from the same surface"
      ],
      "failure_modes": [
        "Tapping before the state is read, inverting an already-correct state",
        "Reading the state from a stale cached screen instead of the live one"
      ],
      "verification_checklist": [
        "Target-state indicator visible on the control itself",
        "No pending spinner / confirmation dialog blocking"
      ]
    }
    ```

!!! note "linter 是迁移的结构性保证"
    一个 linter（`lint_layer2`）会**拒绝任何含有应用名或 resource-id 子串的 Layer-2
    改动**。正是它阻止了从某个应用轨迹中获得的通用洞见被悄悄吸收进应用特定字符串
    而丢失。由于 Layer 2 已被清除一切应用特定内容，**它就是跨应用迁移的单元**——
    也是测试时演化唯一被允许编辑的对象。Layer 1 为其提供落地基础，但在自适应期间保持冻结。

### 类别库 `L_C`

同类别各应用的 Layer-2 块被聚合成单一的**按类别的库 `L_C`**。在预训练阶段，一个
reflector 模型（Claude Opus）对某类别下所有源应用的 Layer-2 块取并集，仅保留在
多个应用间都成立的论断，列举跨源失败模式，并剥除每一个应用特定字符串——由同一个
linter 校验。在部署时，`resolve_l_c_for_app(app)` 查找目标应用的 Play 类别并返回其
`L_C`；若该类别在预训练中从未出现（即远迁移条件），则返回 `None`。

```text
markor.L2  ┐
joplin.L2  ├──►  aggregate + lint  ──►  L_C / productivity.json   (e.g. QUERY_INFO,
tasks_org.L2 ┘                                                     ADD_ENTRY, … )
```

---

## 2. 联合自适应循环

**一个部署时的循环**在目标应用的自适应任务上联合自适应两路通道。两者消费*同一批* rollout。

```text
                 ┌──────────────  one iteration  ──────────────┐
  sample task t  │  select M=2 FSM variants (optimistic softmax)│
                 │  roll N rollouts per variant on t (same seed)│
                 │                                              │
   ┌─────────────┤  ┌─ PROMPT channel ─┐   ┌─ WEIGHT channel ─┐ │
   │ TrueSkill   │  │ frozen π_ref      │   │ GRPO step on the │ │
   │ rate & pick │  │ mutates winner's  │   │ shared LoRA from │ │
   │ best variant│  │ Layer-2 (FSM diff)│   │ the same rollouts│ │
   └─────────────┤  └───────────────────┘   └──────────────────┘ │
                 └─────────────────────────────────────────────┘
```

**提示通道——由冻结的参考 LLM 做 FSM 变异。**内存中保有一群 FSM
变体，每个带一个 **TrueSkill** 评分 `(μ, σ)`。rollout 之后，一个**冻结的参考策略
`π_ref`**（Claude Opus，*不是*可训练的智能体）读取近期轨迹窗口——成功与失败一并
读取——并提出一个被限制在 Layer 2 上的结构化 **FSM diff**。子代以父代的 `μ` 和被
`Δσ` 抬高的 `σ` 追加进群。

!!! tip "为何变异器要冻结"
    用正在 RL 更新的策略来兼做改动提议会引入漂移，从而降低改动质量；E-SPL 对此做过
    消融。EvoFSM 沿用该结论——撰写 FSM diff 的 reflector 全程保持固定。

**权重通道——组相对策略优化（GRPO）。**同一批
轨迹用于在基座 VLM 上做一次 LoRA 更新。GRPO 不用 critic；基线是**组内平均
reward**，分组以 **`(FSM_variant, task)` 元组**为键：

```text
A_{i,j} = r_{i,j} − V_i ,    V_i = mean reward over rollouts in group i
```

这正是**每个 `(FSM, task)` 单元需要 N ≥ 2 次 rollout** 的原因：单元素组没有同伴可减，
其 advantage 构造上恒为零。reward 是基于规则的 episode 成功度（来自 AndroidWorld
grader 的 `{0, 0.5, 1.0}`）外加一小项步数效率奖励；每条轨迹的贡献按其长度
（`1/T_j`）归一化，使一条 30 步和一条 1 步的 rollout 权重相等。梯度仅经
**LoRA**（rank 32，α 32，仅语言层 `q_proj`/`v_proj`）流动；基座权重保持冻结，由 LoRA
的隐式正则化替代显式 KL 惩罚。

**选择——由 TrueSkill 挑选 FSM 变体。**每轮迭代是一场锦标赛回合：通过**乐观
softmax** `p_i ∝ exp((μ_i + λσ_i) / T)`，从最新 K 个变体的滑动窗口中采样 M = 2 个
变体。`λσ_i` 项使选择偏向高不确定性（新引入）的变体，让它们在被淘汰前有机会积累
证据。回合结束后，一次贝叶斯 TrueSkill 更新根据这次成对结果调整评分；`μ` 最高的
变体即被变异的对象。

---

## 3. 两个阶段——同一个循环，跑两遍

完全相同的联合循环在两种尺度上运行。仅数据池和算力预算不同。

| | 阶段 1——源池预训练 | 阶段 2——目标应用部署 |
|---|---|---|
| **池** | 全部源应用（一个"训练集"） | 单个未见目标应用的 `T_adapt` |
| **LoRA** | 从头训练**共享的 `π^pre`** | 从 `π^pre` 续训，持续更新 |
| **FSM** | （最小范围：源 FSM 冻结；`L_C` 离线构建） | §2 的 Layer-2 演化实时运行 |
| **节奏** | **仅跑一次**，一次性固定成本 | 每个目标应用各跑一次，然后冻结 |

阶段 1 的产物——共享 adapter **`π^pre`**——被冻结，并在每次部署时复用为 LoRA 初始化。

!!! warning "为何部署不能从基座模型起步"
    一个 ≈ 20 次迭代 × M × N rollout 的部署预算，远低于从头训练 LoRA 所需的量——在
    该尺度上从随机 LoRA 起步看不到任何学习。阶段 1 → 阶段 2 的拆分，正是让联合 TTA 在
    部署时算力下可行的关键。阶段 2 之后，LoRA adapter 与 FSM 冠军**两者**都被冻结，
    并在留出的 `T_eval` 任务上评测一次。

---

## 4. B1 → B4 消融阶梯

EvoFSM 以一道阶梯来评测，每一级**恰好增加一个机制**，因此每个增量都隔离出该机制的
贡献。（此处给定义；数字见 [Within-benchmark](within-benchmark.md) 与
[Cross-benchmark](cross-benchmark.md) 研究。）

| 级 | 新增机制 | 该级是什么 |
|---|---|---|
| **B1** | — | **零样本。**基座 VLM 在标准的动作+摘要循环中运行，无 FSM，无 `L_C`。每个机制都必须越过的底线。 |
| **B2** | 静态类别迁移 | **静态 `L_C`。**将类别匹配的 `L_C` 注入动作选择提示并保持**冻结**——无演化，无权重更新。 |
| **B3** | 在线上下文演化 | **在线 `L_C` 演化。**以 `L_C` 播种的 FSM 成为受评种群的根；§2 的 Layer-2 演化循环在 `T_adapt` 上运行，随后冠军被冻结用于 `T_eval`。权重保持冻结。 |
| **B4** | 联合权重协同自适应 | **联合 LoRA + FSM。**在 B3 基础上加入 §2 的 GRPO 权重通道，从 `π^pre` 起步。两路通道协同自适应，然后双双冻结。 |

读取这些增量，可把测试时自适应的收益分解为三个可分离的部分：

- **B2 − B1**——*静态*类别知识迁移的价值。
- **B3 − B2**——在线*演化*这份可迁移知识的价值。
- **B4 − B3**——在已演化上下文之上做*权重协同自适应*的价值。

!!! info "远迁移是内置的零假设对照"
    类别从未在源池中出现过的目标应用会得到 `L_C = None`，因此在 B2/B3 下它们的提示
    与 B1 逐字节相同——那里的任何变动都是纯粹的 emulator/CUDA 非确定性。这使远迁移既
    是注入机制的零假设对照，又是 **B4 权重通道**必须独力承担全部迁移负载的场景，因为
    没有 `L_C` 可依靠。

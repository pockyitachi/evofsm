# 跨基准（MobileWorld）

EvoFSM 中最难的泛化测试：在一个基准上预训练，再部署到**另一套独立编写**的基准上。在 AndroidWorld+ 上学到的符号先验（`L_C` + 每个 app 的 FSM Layer-2）必须跨越环境边界迁移——面对全新的 app、全新的 harness、全新的 agent 格式——而唯一可依赖的对齐线索只有 Play Store 的类目。

## 设置

- **训练 → 测试** —— 在 **AndroidWorld+**（193 个任务）上预训练；在 **MobileWorld GUI-only** 上评估，这是一个 110 任务的评测集（`T_eval`）。
- **纯视觉** —— 没有 accessibility tree。Agent 只看到截图，并以 MobileWorld 原生的 `mobile_use` 格式行动。FSM/`L_C` 先验以文本形式拼接进 system prompt。
- **两个模型**覆盖能力区间：
    - **EvoFSM-8B**（Qwen3-VL-8B-Instruct）—— 在本基准上是个*弱底座*，zero-shot 约 8%。
    - **MAI-UI-8B** —— MobileWorld 自家的*第一方* GUI agent，是本基准原生的**接近上限**的外部基线（zero-shot 约 26%）。
- **方差** —— 每个配置都跑 **5×110 任务**；我们报告 `mean ± std`。容器**每轮全新创建、从不复用**（MobileWorld 任务间不重置 app 状态），所以并发只影响 wall-clock。
- **分母** —— 固定为 110。有两个任务（`CheckMeetingEventAskUserTask`、`ThanksgivingPrepTask`）因同一个上游 harness bug（与模型无关）在每轮都初始化失败，所以分数实际上是 108 满分。

这条阶梯与基准内研究一致：**B1** zero-shot → **B2** 静态符号先验 → **B3** 符号测试时演化 → **B4** 符号 + 权重联合适配。

## 两个模型上的 B1–B3

110 满分的成功数，5 轮 `mean ± std`。**B2′** 是最佳静态配置（app Layer-2 + 类目 `L_C`，无 Layer-1——见下方 findings）；B3 列是 **lessons-only** 符号演化变体，即可上报的正向结果。

![跨基准 —— 两个模型上符号 TTA 对比静态先验](assets/mw_models.png){ width="620" }

| Model | B1 (zero-shot) | B2′ (static prior) | B3 (lessons-only) |
|---|---|---|---|
| **EvoFSM-8B** (弱底座) | 8.2 ± 0.8 | 9.2 ± 1.6 | **10.0 ± 1.4** |
| **MAI-UI-8B** (接近上限) | 26.2 ± 2.4 | 26.4 ± 2.0 | **29.0 ± 4.5** |

MAI-UI 约为 EvoFSM-8B 底座的 3.2 倍——在它的主场基准上是个强得多的 GUI agent，在 Tier-A 多 app 组合任务上差距最大。两个 B3 数字在各自行里*名义上*最高，但考虑误差棒，与静态先验高度重叠。

## 关键发现

### 1. 静态注入的收益取决于模型能力

同一个 B2′ 先验帮到了弱模型，对强模型基本没有作用：

| Model | B2′ − B1 |
|---|---|
| EvoFSM-8B | **+1.0**（约 1σ） |
| MAI-UI-8B | **+0.2**（在噪声内） |

B2′ 的指导是从 **EvoFSM-8B** 的弱点中挖出来的——它缺的“先读后做 / 终止前先核验”的纪律。MAI-UI 内部已经具备这套工作流知识，所以注入没有增量，甚至会把它推离自己（更好的）策略。在 MAI 上，B2′ 下每一个可归因的大任务翻转都是*回退*，被零散的低于阈值的小增益抵消，总分持平：这是**过度规约**的失效模式。

### 2. 跨基准注入 Layer-1 严格有害

注入源环境 FSM 的 **Layer-1**（具体状态 / 转移 / 视觉线索）在*每一轮*都有害，所以最佳静态配置把它完全去掉：

- **B2′ = app Layer-2 + 类目 `L_C`** 是最佳静态配置。它是唯一高于基线的 EvoFSM-8B 配置，也是迁移到 MAI-UI 的变体。
- 这种伤害是**确定性、机制性**的，不是统计噪声。在两个 FlightMode 任务上，Layer-1 变体（B2）得 **0/5**，而所有 Layer-2 配置都得 **5/5**，在 25 轮中复现。轨迹分析显示，源环境的状态描述挤掉了模型自己 grounded 的例程，诱发不经截图核验就 `terminate(success)`。

> 原始 AndroidWorld+ 配方（类目 `L_C`，从不用 Layer-1）在主场环境净赚 **+9.3pp**；在这里它落到 ≈0。知识在 **app 粒度**（Layer-2）上比稀释到类目里更能挺过基准鸿沟，而 Layer-1 则完全挺不过去。

### 3. 符号 TTA（B3）的上限 ≈ 静态先验——两个模型皆然

符号测试时演化（FSM 演化和/或蒸馏出的 lessons，在不相交的 51 任务 `T_adapt` 划分上适配）并**没有**明显超过静态先验：

| EvoFSM-8B | /110 | | MAI-UI-8B | /110 |
|---|---|---|---|---|
| B2′ (static) | 9.2 ± 1.6 | | B2′ (static) | 26.4 ± 2.0 |
| B3 FSM-evo | ~10 | | B3 FSM-evo | 27 |
| B3 fsm+lessons | 10.4 ± 0.5 | | B3 fsm+lessons | 25.6 ± 3.0 |
| B3 lessons-only | 10.0 ± 1.4 | | B3 lessons-only | 29.0 ± 4.5 |

每个符号变体的质心都落在静态先验上（EvoFSM-8B 约 10，MAI-UI 约 26）。**lessons-only** 在两个模型上都是名义最优，但其分布（±1.4 / ±4.5）与先验重叠——在 n=5 下这是*名义更高、在噪声内*，不是显著胜出。它真正的优势是**效率**：蒸馏后的 lessons 以大约 **1/20–1/26 的注入 token** 达到与先验相当的效果（约 800 字符的 lesson 形式 vs 数千 token 的 FSM dump）。

> **为何会触顶。** 在单遍 51 任务的适配排程上，TrueSkill 选择处于信号匮乏状态（变体在地板/天花板任务上打平），所以“演化”坍缩成“挑一个固定 champion”——也就是另一种静态注入。符号这一半已到天花板；增长空间在**权重通道（B4）**。

## B4 —— 符号 + 权重联合适配

B4 是 B3 加上第二条通道：一个共享 LoRA 在 lessons 演化所用的**同一批** rollout 上通过 GRPO 适配（一个开关，`EVOFSM_TTA_EVOLUTION_MODE=lesson`）。双通道联合是本方法的卖点；在这种低数据（约 51 任务）的设定下，权重那一半作为诚实的局限来报告，而不是降级。

Base-direct（base Qwen3-VL-8B，无 π^pre），MW-110，×5 均值：

| config | mean /110 | symbolic form | weight channel |
|---|---|---|---|
| B1 (zero-shot) | 8.2 | — | — |
| B2′ (static prior) | 9.2 | full FSM L2 | — |
| B3 (symbolic evo) | ~10 | — | frozen |
| **B4 fsm-champion**（消融） | **8.2** | full FSM champion | 有害（把 10 拉低到 8.2） |
| **B4 lessons-only**（主结果） | **10.0** | distilled lessons | 中性（无拖累） |

对权重那一半的两种解读：

- **在完整 FSM 下，权重有害。** 固定符号 champion、扫 LoRA checkpoint，分数**单调下降**（10 → 9 → 9 → 8 → 8.2 ≈ B1）：约 6.8k token 的 FSM L2 dump 是个噪声训练信号，权重会对它过拟合。
- **在 lessons-only 下，权重中性。** 紧凑的 lessons-only 符号以约 **1/26** 的 token（约 264 vs 约 6847 prompt token）达到 **10.0 ≈ B3 天花板**——更小*且*更好。诚实的表述：lessons-only 让权重通道*无害*，但并没有让它*带来增量信号*。权重在哪种符号下训练，决定了联合是无害（lessons）还是有害（FSM）。

!!! note "B4 状态：进行中"
    上面的数字是 **base-direct** 版 B4（无 π^pre 初始化）：lessons-only = **10.0**，与 B3 持平。**π^pre 初始化**的联合（Phase-1 预训练 LoRA + lessons-only）**仍在评估中**——MAI-UI 的 π^pre 预训练正在跑，其 B1 曲线要到约 step 400 才开始抬头。在这个跨基准设定下，我们尚不主张权重通道带来增益；权重那一半的增长空间（更多适配数据、π^pre 初始化）属于未来工作。

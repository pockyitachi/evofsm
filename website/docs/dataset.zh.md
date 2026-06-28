# 数据集与划分

EvoFSM 在两个泛化层级上评测,每个层级有各自的数据集和划分协议。本页是两者的自包含参考;机器可读的权威来源是 `EvoFSM-RL/configs/splits.yaml`(基准内)和 `EvoFSM-MW/configs/mobileworld_splits.yaml`(跨基准)。

!!! note "两条迁移轴,一个共同思路"
    两项研究都测试**类别级迁移**——即预训练阶段学到的每类别抽象动作库 `L_C` 是否能帮助*同一 Google Play 类别*的某个 app。Tier 完全按 Play Store 类别定义,绝不按 app 身份。从不做"见过 / 新 app"这类断言。

---

## 基准内(AndroidWorld+)

**AndroidWorld+** 是一个 25-app、12-Play-类别的基准,在 Google 的 AndroidWorld(19 个活跃 app)基础上,加入从 BMOCA 导入的 6 个 app(Calculator、Snapseed、Wikipedia)以及 AndroidLab(Bluecoins、Maps.me、Pi Music)。元数据中的约 194 个模板里,**192 个可归属到 app**(另外两个是通用/复合的,从按 app 划分中排除)。类别直接取自每个 app 的 Play 列表——这是一套经外部验证的分类法,同时也定义了类别通用 `L_C` 的等价类。

划分分解为**三个独立、互不相交的层级**——app 所属的 pool、它锚定的类别迁移 tier,以及每个 app 内部模板互斥的 adapt/eval 划分。

### Level 1 — Pool(训练 app vs. 目标 app)

标准的 app 级训练/测试分离。**source pool** 用于 Phase-1 预训练(按 app 抽取静态 FSM、按类别合成 `L_C`);目标 app 在预训练中从不触及,仅在测试时适应阶段出现。

| Pool | 角色 | Apps | 模板数 |
|---|---|---:|---:|
| Source pool | Phase-1 预训练(FSM + `L_C`) | 12 | 96 |
| Tier-B(近) | 目标 — 类别在 source 中见过 | 6 | 50 |
| Tier-C(远) | 目标 — 类别为新 | 7 | 46 |

12 个 source-pool app 横跨 6 个 Play 类别(Productivity、Tools、Finance、Music & Audio、Photography、Communication)。这些正是测试时交接处存在匹配 `L_C` 的类别。

### Level 2 — 类别 tier(近迁移 vs. 远迁移)

目标 app 按其 Play 类别是否在 source pool 中出现来划分。分别测量近迁移和远迁移比合并成一个数更有信息量,也对应 Android Control 的 `unseen-app`(≈ Tier-B)与 `unseen-category`(≈ Tier-C)之分。

| Tier | 迁移 | 类别 | Apps | `T_adapt` | `T_eval` |
|---|---|---|---:|---:|---:|
| **Tier-B** | 近 — 类别见过,存在匹配 `L_C` | Productivity、Tools、Finance、Music & Audio、Photography、Communication | 6 | 32 | 18 |
| **Tier-C** | 远 — 类别为新,无匹配 `L_C` | Maps & Navigation、Food & Drink、Health & Fitness、Books & Reference、Video、Art & Design | 7 | 29 | 17 |

!!! info "Tier-C 是空对照"
    Tier-C 的 app 没有 source-pool 类别,因此 `resolve_l_c_for_app()` 返回 `None`,它们回退到 B1(无注入)路径。Tier-C 因此充当 `L_C` 注入机制的空对照。

    `simple_draw_pro`(Art & Design)只有一个模板,全部分配给 `T_adapt`;它的 `T_eval` 为空,在 Tier-C 汇总的 `T_eval` 分数中被排除。

### Level 3 — 模板划分(adapt vs. eval,在每个 app 内部)

在每个目标 app 内部,模板被划入两个**模板互斥**的集合——不仅是 seed 互斥,任务模板类本身就不同。这正是让 `T_eval` 测量适应泛化能力、而非在记住的模板上测量参数鲁棒性的关键,遵循元学习(MAML)的 support/query 协议,以及面向泛化的 RL 基准(Procgen)的 train-level/test-level 协议。

| 集合 | 用于 | 划分 | Seeds(K) |
|---|---|---|---|
| `T_adapt` | 在线适应(B3 FSM 演化,B4 联合 TTA) | 60%(确定性按字母序) | 5 — `[30, 31, 32, 33, 34]` |
| `T_eval` | 适应后单次冻结评测;头条数字 | 40% | 3 — `[40, 41, 42]` |

`T_eval` 的 seed 与 `T_adapt` 的 seed 互斥,因此即便某个模板在两个集合间共享参数值,也会落到不同的初始状态。划分规则在小 app 上平滑退化:`N = 2` 切成 1/1,`N = 1` 把唯一模板放进 `T_adapt`、`T_eval` 留空。

---

## 跨基准(MobileWorld)

跨基准研究在 **AndroidWorld+(193 个任务,全部为 source)** 上训练,在 **MobileWorld GUI-only(161 个任务)** 上测试时适应——后者是一个独立编写、纯视觉的基准。MobileWorld 仅用于评测:它的 161 个任务是硬编码的单实例,**无 seed(K = 1 固有)**,也没有内建的 adapt 划分,因此适应任务通过任务互斥划分*在 MobileWorld 内部*切出。

### Tier(类别级)

Tier 按任务所触及 app 的 Play 类别来分配,相对于 AndroidWorld+ 训练中见过的 12 个类别。唯一的**新**类别是 **Social**(Mastodon、Mattermost)和 **Shopping**(Taodian);其余所有 MobileWorld app 都落在见过的类别里。

| Tier | 定义 | 任务数 | `L_C` 注入 |
|---|---|---:|---|
| **Tier-B** | 类别见过 — 每个 app 的类别都在训练中见过 | 91 | 现成的类别 `L_C` |
| **Tier-C** | 类别为新 — 所有 app 都在新类别中(Social、Shopping) | 43 | Bootstrap `L_C`(来自目标 app 轨迹) |
| **Tier-A** | 混合 — 任务同时触及见过和新的类别 | 27 | 见过的现成 + 新的 bootstrap |
| **Total** | | **161** | |

每个 tier 由单 app 和跨 app 的情形组成:

| Tier | 组成 |
|---|---|
| Tier-B(91) | 36 单 app(全部见过)+ 55 跨 app(全部见过) |
| Tier-C(43) | 40 单 app(全部为新)+ 3 跨 app(全部为新) |
| Tier-A(27) | 跨 app 任务,把 1 个新 app 与 1–3 个见过的 app 混合 |

!!! warning "共享包名 ≠ 未见 app 的泛化"
    六个系统 app(Chrome、Contacts、Settings、Clock、Camera、Files)在 AndroidWorld+ 和 MobileWorld 之间共享同一个包名。项目**不**对它们做特殊处理——tier 只按类别。由于方法的贡献以*差值*报告(有 `L_C` − 无 `L_C`,策略固定),任何同 app 的优势都会抵消,不会偏置 `L_C` 的结论。Tier-B 的绝对准确率**不应**被解读为对未见 app 的泛化。

### Adapt / eval 划分(任务互斥)

划分是确定性的(按类名排序,无 RNG),`adapt_fraction = 0.40`。Tier-A **全部分配给 eval**——它是头条的"组合见过 + 新知识"集合,且没有单 app 任务可干净划分——这把整体 adapt 比例拉低到约 32%。多 app 任务可以进入 `T_adapt`:在按 app 的符号演化下,一个多 app 的 adapt 任务会同时演化它所触及每个 app 的 FSM/`L_C`,因此没有单 app 任务的 app 也能获得 adapt 覆盖,而任务级互斥防止泄漏。

| Tier | 任务数 | `T_adapt` | `T_eval` |
|---|---:|---:|---:|
| **Tier-B**(近) | 91 | 33 | 58 |
| **Tier-C**(远) | 43 | 18 | 25 |
| **Tier-A**(混合) | 27 | 0 | 27 |
| **Total** | **161** | **51** | **110** |

因为没有 seed,51 个 adapt 任务中的每一个都被复用为 B4 整个演化循环(约 20 次迭代 × K 次 rollout)中的适应度信号,所以 40% 的任务预算实际上构成数百次 rollout 经验,驱动符号演化。互斥的 110 任务 `T_eval`(包含多 app 组合)验证的是真正的泛化而非记忆。

!!! info "L_C 来源映射"
    注入**按 app**解析,无 tier 分支:任务所触及的每个 app,若其类别见过则取该类别的现成 `L_C`,否则取 bootstrap 的 `L_C`,之后按类别去重条目。见过的 MobileWorld app 映射到同类别的 AndroidWorld+ source app(例如 Mail/Messages/Contacts/Chrome → Communication;Files/Settings/Clock → Tools;Calendar → Productivity;Maps → Maps & Navigation;Camera/Gallery → Photography;Docreader → Books & Reference)。Mastodon、Mattermost 和 Taodian 无 source,使用 bootstrap `L_C`。

# MobileWorld 跨基准 TTA 切分（T_adapt / T_eval）

> 配套文件:`EvoFSM-MW/configs/mobileworld_splits.yaml`（逐任务归属）+ `configs/gen_mobileworld_splits.py`（可复现生成器）。
> tier 定义见 [`dataset_tiers.md`](dataset_tiers.md)。本文档记录**为什么这样切**和**切分规则**。

> **⚠️ SUPERSEDED(部分,2026-06-11)**:本文 §1 的「TTA 走符号-only、不跑 LoRA 通道」决策已被用户推翻
> ——B4 恢复算法设计文档的原始定义(FSM evo **+** weight evo 联合),权重通道 = 共享单 LoRA + GRPO。
> 现行设计见 **[`b3_b4_mw_tta_design.md`](b3_b4_mw_tta_design.md)**。
> 本文的 51/110 task-disjoint 切分本身仍然有效、继续沿用。

---

## 1. 背景与决策（2026-06-08）

**目标管线**:train(π^pre + static L_C)在 **AndroidWorld+ 193**;test 在 **MobileWorld 161 GUI-only**;EvoFSM 的 **TTA(B4)在测试域目标 app 上 adapt**,冻结评测。

**核心张力**:MobileWorld 是一个 eval-only benchmark——161 个**硬编码单实例任务,无 seed**(K=1 固有),不自带 adapt 划分,per-app 任务数也小。那 T_adapt 从哪来?

讨论排除的两条路:
- **❌ 用 AW+ 当 T_adapt**:AW+ 是源域,adapt 看不到任何 MW 目标域信号 → 等于重做预训练,"test-time"消失;且对 Tier-C(Social/Shopping,AW+ 无数据)完全是零,把 B4 远迁移卖点废掉。
- **❌ LoRA / online-RL 通道**:GRPO 需要每 prompt 多 rollout 且组内有 reward 方差。MW 无 seed → 同一固定实例反复跑 → 组内零方差(零梯度)+ 过拟合到实例。而且 clean ablation 已显示 Phase-3 LoRA null/negative。

**决策:保 TTA,在 MW 内部 task-disjoint 切 adapt/eval,TTA 走「符号-only」**(FSM/L_C bootstrap+evolve,跨基准暂不跑 LoRA 通道)。理由:
- 符号 evolution 由 Opus 从观察到的失败里提炼**可泛化的 FSM/L_C 模式**,几条 rollout 即可,**不需要 seed、不需要 GRPO 的 advantage 结构** → 绕开 MW 无-seed 的硬伤。
- 押注在方法真正被验证有效的那一半(L_C/FSM),而非 null 的 LoRA。

---

## 2. adapt 阶段的机制（为什么 40% 的任务数就够）

TTA / B4 是一个**迭代进化循环**(B4 sweep ~20 iters):维护 FSM/L_C 变体种群;每 iteration 把候选变体注入、在 T_adapt 上 rollout、用成功率当 fitness 给变体打分、Opus 看失败变异、选择存活。

→ **T_adapt 是「适应循环的 fitness 函数」,每个 adapt 任务被 rollout K×~N_iters 次,不是一次性数据点。** 所以 ~51 个不同 adapt 任务 = 数百次 rollout 经验在驱动进化,40% 的任务数足够。

**无-seed 的影响与缓解**:AW 里每任务的多次 rollout 来自不同 seed(实例有变化);MW 是同一固定实例的重复尝试(靠 policy 随机性)。对符号 evolution 影响小(提炼模式而非记忆轨迹),但 scenario 多样性低于 AW,FSM/L_C 有轻微「贴 adapt 实例」风险。缓解:① per-app 内选多样 adapt 任务;② eval 是 disjoint 的不同任务(且含多-app 组合),验出是否真泛化而非记忆。

---

## 3. 切分规则（确定性、可复现、task-disjoint）

由 `gen_mobileworld_splits.py` 生成,无 RNG(按 class 名排序):

1. **GUI-only** = `task_tags` 不含 `agent-mcp`(161 / 201)。
2. **eval tier** 按任务涉及 app 的 Play-category:全 seen → Tier-B;全 novel(Social/Shopping)→ Tier-C;混合 → Tier-A。
3. **Tier-A(全多-app、seen+novel)→ 全部 eval**:它是「组合 seen+novel 知识」的 headline,且 0 单-app 无法干净 per-app 切;其 novel 部分靠 Tier-C adapt 出的 FSM/L_C,seen 部分靠 static L_C。
4. **Tier-B / Tier-C → 按 app-combo(任务的 app 集合)分层**,每层按 class 名排序取前 `round(0.40 × n)` 进 adapt,其余 eval。
5. **多-app 任务可进 adapt**:对符号 per-app evolution,一个多-app adapt 任务同时演化它涉及的每个 app 的 FSM/L_C;只要 task-disjoint(任务整体 adapt 或 eval)就无泄露。这让连 0 单-app 的 Contacts/Gallery/Docreader 也获得 adapt 覆盖。

`adapt_fraction = 0.40`(项目 AW-内部协议是 60/40;此处取 40% adapt 以保住较大的 eval headline,Tier-A 全 eval 拉低总体 adapt 比例到 ~32%)。

---

## 4. 结果

| | 任务 | T_adapt | T_eval |
|---|---|---|---|
| **Tier-B**（近迁移,category 级) | 91 | 33 | 58 |
| **Tier-C**（远迁移,category 级) | 43 | 18 | 25 |
| **Tier-A**（混合) | 27 | 0 | 27 |
| **合计** | **161** | **51** | **110** |

**per-app adapt / eval 覆盖**(每个 app 都有 ≥1 adapt):

| category | app | adapt | eval |
|---|---|---|---|
| Communication | Mail | 12 | 31 |
| Communication | Messages | 8 | 18 |
| Communication | Contacts | 3 | 8 |
| Communication | Chrome | 3 | 7 |
| Productivity | Calendar | 6 | 21 |
| Tools | Files | 10 | 21 |
| Tools | Clock | 2 | 5 |
| Tools | Settings | 2 | 5 |
| Maps & Navigation | Maps | 4 | 5 |
| Photography | Gallery | 2 | 9 |
| Photography | Camera | 1 | 2 |
| Books & Reference | Docreader | 3 | 7 |
| **Social (novel)** | Mastodon | 11 | 30 |
| **Social (novel)** | Mattermost | 2 | 15 |
| **Shopping (novel)** | Taodian | 6 | 9 |

（per-app 数和 > 161,因多-app 任务对每个涉及 app 计一次。）

**已知薄弱点**:Mattermost(2)/Camera(1)/Clock(2)/Settings(2) 的 adapt 偏薄——这些 app 任务总数本就少。Tier-C 主力 Mastodon(11)/Taodian(6) adapt 充足。

---

## 5. 复现

```bash
python EvoFSM-MW/configs/gen_mobileworld_splits.py   # 重新生成 yaml(确定性)
```
改 `ADAPT_FRAC` 可调比例;改规则需同步本文档 + bump yaml `meta.version`。

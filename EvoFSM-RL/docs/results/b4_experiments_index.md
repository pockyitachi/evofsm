# B4 实验全索引

**日期**: 2026-05-19
**作者**: EvoFSM-RL Team
**目的**: 把所有跑过的 B4 相关实验(预训练 / 目标训练 / 评估 / smoke 验证 / 超参扫描)汇总成一个表,
每个实验都有 setup / 进展 / 发现 / 输出路径。Paper 写作时按图索骥。

---

## 0. 命名约定

| 缩写 | 全称 | 角色 |
|---|---|---|
| **Phase 1** | Source-pool pretraining | 在 12 个源池 app 上训练 shared LoRA → 产出 **π^pre** |
| **Phase 3** | Target-app fine-tune | 在 1 个目标 app 上从 π^pre 继续训练 → 产出 per-app (LoRA, L_C) |
| **B1** | zero-shot M3A baseline | 无 LoRA、无 L_C |
| **B2** | static L_C baseline | 无 LoRA、L_C 静态(无进化) |
| **B3** | FSM-only baseline | 无 LoRA、L_C 在线进化 |
| **B4** | joint LoRA + L_C | LoRA 训练 + L_C 进化(paper main 方法) |
| **K** | n_rollouts per (FSM, task) | GRPO 每组的样本数(K=2 默认,K=4 我们 ablation) |
| **β** | KL anchor 系数 | loss 里 KL 项的强度(0.05 默认) |
| **π^pre** | pretrained LoRA | Phase 1 产物,Phase 3 的起点 |
| **T_adapt / T_eval** | template-disjoint split | T_adapt 训练时用,T_eval 评估时用 |

---

## 1. Phase 1 实验(源池预训练 → π^pre)

| ID | 时期 | 关键 config | 状态 | 输出 | 发现 |
|---|---|---|---|---|---|
| **Phase 1 pilot** | 早期(~30 min) | 12 apps,~5-10 iter,β=0(无 KL anchor)| ✅ 完成 | `traces/phase1_pilot_v01/lora_checkpoints/final` | **π^pre_pilot** — Phase 3 v2 / Bootstrap / K=4 sweep 都拿它作起点。Tier-B 上 +7.4 pp 增益,Tier-C 上 -17.6 pp 回归,**净 overall 比 B1 低 4.8 pp** |
| **Phase 1 v2** | 5/13 20:10 → 5/15 19:19(~47h) | 12 apps,600 iter,**β=0** | ✅ 完成,但产物不能用 | `traces/phase1_full_v01/lora_checkpoints/final` | **π^pre_v2 灾难** — 600 iter 无 KL anchor 累积漂移。用在 B4-revert v2 跑出 5.7% overall(比 pilot 的 33.8% 还低 28 pp)。**证明 Phase 1 必须有 KL anchor** |
| **Phase 1 v3** | 5/18 03:37 → 5/19 ~17:00(~37h) | 12 apps,600 iter,**β=0.05,kl_log_ratio_clip=10,min_n_active=3,--anchor-to-base** | 🟡 **跑中**(iter 371/600,62%)| `traces/phase1_v3/lora_checkpoints/` | 中段 mean_kl=202 spike 但 clip 接住;后段 n_active=0 多说明源池任务接近收敛。**等 final/ 出来用 B4-revert v3 验证** |

---

## 2. Phase 3 / B4 训练实验(目标 app fine-tune → per-app LoRA + L_C)

| ID | 时期 | 关键 config | 状态 | 输出 | 发现 |
|---|---|---|---|---|---|
| **Phase 3 v1**(= **B4 raw v1**)| 4/22 | 6 apps × 20 iter,K=2,**β=0**,无 clip,无 reject | ✅ 完成,但灾难 | `traces/b4_evolution_v2/`(命名 v2 是因为它实际是修了 OOM 的 v2)| **Tier-B T_eval 15.7%(<< B3 63.9%)**。68-72% 的 trajectory 在 1-2 步 fire status:complete(mode-collapse)。GRPO fire 2 出现 grad_norm=2050 亿、mean_kl=160 亿的数值爆炸 |
| **Phase 3 v2** | 5/15 16:09 → 5/17 02:14(~34h)| 12 apps × 20 iter,K=2,**β=0.05,clip=10,min_n_active=3**,π^pre=pilot | ✅ 完成 | `traces/b4_phase3_v02/{app}/`(Tier-B 6 个 + Tier-C 6 个 option-a)| **修复成功**:Tier-B 60.2% vs B3 62%,无 mode-collapse。pro_expense Champion μ=49.27(K=2 首次突破)。Tier-C 用 `--allow-no-l-c` mutation OFF |
| **Bootstrap Tier-C sweep** | 5/16 06:46 → 5/17 ~05:00(~22h)| 6 Tier-C apps × 20 iter,K=2,β=0.05,**`--enable-bootstrap-fsm`**(option b)| ✅ 完成 | `traces/b4_bootstrap_tier_c/{app}/` + `traces/b4_bootstrap_pilot_opentracks/`(opentracks 单独 pilot)| Claude Opus 从空 LAYER 2 冷启动合成 L_C 内容质量高(broccoli 86.7%,vlc 0→33.3%,Tier-C 整体 33.3% vs B3 30.4%)。**这是 paper Tier-C 主线** |
| **K=4 sweep** | 5/18 06:01 → 5/19 ~10:00(~28h)| 12 apps × 20 iter,**K=4**(rollouts 翻倍),β=0.05,clip+reject,π^pre=pilot | 🟡 **11/12 完成**(chunk_C 还在跑 osmand + wikipedia)| `traces/b4_k4_{A,B,C,D}/{app}/` | **K=4 在 sparse-success app 有效**:simple_cal μ 25→57(K=2 stuck → K=4 突破),retro_music μ 34→54。**但 pro_expense μ 49→25(K=4 反而退步)**,K 不是单调好 |
| **β=0.03 sweep** | 5/19 03:48 → 5/19 ~19:00(~16h)| 12 apps × 20 iter,K=4,**β=0.03**(更小 KL),clip+reject,π^pre=pilot | 🟡 **刚启动**(6-way 并行) | `traces/b4_b003_{A...F}/{app}/` | 基于 β-smoke 数据:β=0.03 K=4 6-iter μ=38.82 vs β=0.05 同设置 μ=30.44。测 β=0.03 是否在 full 20-iter 也更好 |
| **Phase 3 v3** | (未跑) | 用 Phase 1 v3 final 作 π^pre + ref,重做 Phase 3 | ⏳ 等 Phase 1 v3 跑完 | 待 | paper 想测的"全套修复 + 更强 π^pre"是否能进一步提升 |

---

## 3. T_eval 实验(评估)

| ID | 时期 | 设置 | 状态 | 输出 | 主结果 |
|---|---|---|---|---|---|
| **B1 T_eval K=3** | 4 月 | zero-shot M3A,K=3 seeds | ✅ 完成 | `traces/b1_teval_k3/results.csv` | TB **47.2%** / TC **29.4%** / Overall **38.6%** |
| **B2 T_eval K=3** | 4 月 | 静态 L_C(无进化),无 LoRA | ✅ 完成 | `traces/b2_teval_k3/results.csv` | TB **56.5%** / TC 29.4% / Overall **43.3%** |
| **B3 T_eval** | 4 月 | FSM 进化(无 LoRA)| ✅ 完成 | `traces/b3_teval/results.csv` | TB **62.0%** / TC **30.4%** / Overall **46.7%**(**目前最强 baseline**)|
| **B4 raw v1 T_eval** | 5/14 | broken B4 v1 (LoRA 崩) | ✅ 完成 | `traces/b4_teval/results.csv` | TB 15.7% / TC 2.0% / Overall **9.0%**(灾难)|
| **(A) B4-revert v1** | 5/14 | **π^pre_pilot** + B4 v1 L_C(LoRA 还原)| ✅ 完成 | `traces/b4_revert_lora_teval/results.csv` | TB 54.6% / TC 11.8% / Overall **33.8%**(<B1,证明 LoRA 是元凶 + π^pre_pilot 单独伤 Tier-C)|
| **B4-revert v2** | 5/16 | **π^pre_v2** + B4 v1 L_C | ✅ 完成 | `traces/b4_revert_v2_teval/results.csv` | TB 11.1% / TC 0.0% / Overall **5.7%**(Phase 1 v2 无 KL 训坏了)|
| **Bootstrap T_eval (Tier-C only)** | 5/17 | Bootstrap 训出来的 6 Tier-C 跑 T_eval | ✅ 完成 | `traces/b4_bootstrap_tier_c_teval/results.csv` | TC 加权 **33.3%**(broccoli 86.7%,vlc 0→33%)|
| **Bootstrap pilot T_eval (opentracks)** | 5/17 | opentracks 单独 pilot 验证 | ✅ 完成 | `traces/b4_bootstrap_teval/results.csv` | opentracks 0/6 = 0%(但 B3 也 0%,opentracks 是 base 搞不定的 hard app) |
| **B4 v2 T_eval(paper main)** | 5/17 02:34 → 5/18 02:00 | unified(Tier-B 来自 Phase 3 v2 + Tier-C 来自 Bootstrap)| ✅ 完成 | `traces/b4_v2_teval/results.csv` | TB **60.2%** / TC **29.4%** / Overall **45.2%**(≈B3,无超越) |
| **K=4 T_eval** | (待跑) | K=4 sweep 产物的 12-app T_eval | ⏳ K=4 跑完后 | `traces/b4_k4_teval/` | 待数据 |
| **β=0.03 T_eval** | (待跑) | β=0.03 sweep 产物的 12-app T_eval | ⏳ β=0.03 sweep 跑完后 | `traces/b4_b003_teval/` | 待数据 |
| **B4-revert v3** | (待跑) | π^pre_v3 + B4 v1 L_C(评估 Phase 1 v3 单独贡献) | ⏳ Phase 1 v3 跑完后 | `traces/b4_revert_v3_teval/` | 待数据 |
| **B4 v3**(未跑) | — | Phase 3 v3 + Bootstrap v3 全套 | ⏳ Phase 1 v3 + 决定 β 后 | 待 | paper 终极配置 |

---

## 4. Smoke 验证(debugging / hyperparam validation)

| ID | 时期 | 目的 | 配置 | 状态 | 关键发现 |
|---|---|---|---|---|---|
| **F1 smoke** | 5/12 | 验证 F1 修复(loss /T_j 归一化) | simple_cal,6 iter,K=2 | ✅ | grad_norm 从 50-270 降到 O(1) |
| **F5 smoke** | 5/13 | 验证 F5 修复((FSM, task) 元组 grouping)| simple_cal,4 iter,K=2 | ✅ | advantage_std > 0,GRPO 信号存在 |
| **smoke v3 (β=0.02, k1)** | 5/15 02:00 | 验证 KL anchor 第一版(k1 estimator) | simple_cal,6 iter,K=2,β=0.02 | ✅ 但发现 k1 公式数学错误(mean_kl 漂到 -20) | k1 estimator 梯度方向不对,需要换 k3 |
| **smoke b05 (β=0.05, k1)** | 5/15 07:02 | 同上换 β=0.05 | simple_cal,6 iter,K=2,β=0.05 | ✅ fire 3 mean_R=0.668 突破 + Champion μ 31.72 | 但发现 k1 estimator fire 2 出现 mean_kl=16 亿数值爆炸 |
| **smoke_clip (β=0.02 + k3 + clip)** | 5/15 13:40 | 验证 k3 + log_ratio clip 修数值 | simple_cal,6 iter,K=2,β=0.02 + clip=10 | ✅ max kl=1.28(vs 无 clip 16 亿) | clip 工作正常;但 β=0.02 学习信号不够强 |
| **Phase 1 v3 smoke** | 5/18 02:46 → 04:08 | 验证 Phase 1 加 KL anchor 路径(`--anchor-to-base`) | 2 source apps,6 iter,K=2,β=0.05 | ✅ wiring 通,mean_kl 正确正值 | Phase 1 anchor 到 base policy(LoRA disabled)路径 work |
| **β=0.03 smoke** | 5/18 16:21 → 18:25 | β 扫描第 1 个 | simple_cal,6 iter,**K=4**,β=0.03 | ✅ μ=38.82 | β=0.03 K=4 早期表现优于 β=0.05 同设置(μ=30.44)|
| **β=0.07 smoke** | 5/18 16:22 → 17:13 | β 扫描第 2 个 | simple_cal,6 iter,K=4,β=0.07 | ✅ μ=25.00(stuck) | β=0.07 太紧,LoRA 不动,大部分 fire 被 reject |
| **β=0.10 smoke** | 5/18 16:22 → 18:37 | β 扫描第 3 个 | simple_cal,6 iter,K=4,β=0.10 | ✅ μ=25.00(stuck) | β=0.10 数值灾难:mean_kl 12000+,grad_norm 65000(clip 接住但 effective gradient 已乱)|

---

## 5. 关键 finding 汇总

### 5.1 v1 → v2 修复路径

| 问题(B4 raw v1) | 修复(B4 v2) | 证据 |
|---|---|---|
| GRPO 梯度被长 trajectory 主导 | **F1**: loss / T_j | smoke F1: grad_norm 50-270 → O(1) |
| advantage 跨 task 漂移 | **F5**: (FSM, task) 元组 grouping | smoke F5: adv_std 正常 |
| sparse signal → mode-collapse | **β=0.05 KL anchor** + clip + reject | smoke_clip + Phase 3 v2 全 12 app,Tier-B 15.7% → 60.2% |
| 数值上 exp(log_ratio) 爆炸 | **kl_log_ratio_clip=10** | β=0.10 smoke 看到 clip 接住数值灾难 |
| outlier-driven update 主导梯度 | **min_n_active=3** reject | pro_expense 5 个 fire 被 reject 救命 |

### 5.2 Phase 1 / Phase 3 / Bootstrap 的 β 角色不同

| 阶段 | 推荐 β | 理由 |
|---|---|---|
| Phase 1(600 iter,源池)| **β=0.05**(已用)| 长训练防 drift,LoRA 要"通用"。β=0(v2)崩了 |
| Phase 3(20 iter,目标 app)| **β=0.05** 或 **β=0.03**(待测)| 短训练,允许 app-specific 学习。β=0.03 smoke 显示更优 |
| Bootstrap(20 iter,Tier-C 空 L_C)| β=0.05(同 Phase 3)| 同 Phase 3 |

### 5.3 K rollouts(K=2 vs K=4)反直觉

| App | K=2 Champion μ | K=4 Champion μ | 变化 |
|---|---|---|---|
| simple_calendar_pro | 25.00(stuck)| **57.46** | ✅ +32 |
| retro_music | 34.03 | **54.01** | ✅ +20 |
| **pro_expense** | **49.27** | **25.00** | **❌ -24**(K=4 反而退步)|
| system_settings | 25.00 | 25.00 | 0 |
| broccoli (Tier-C bootstrap) | 32.26 | 33.15 | 持平 |

**K=4 不是单调好** —— 部分 app 提升,部分 app 回归。机制未完全理解。

### 5.4 当前 paper 主线

| 指标 | B4 v2 (current paper main) | 对比 |
|---|---|---|
| TB SR | 60.2% | B3 62.0%(略低) |
| TC SR | 29.4% | B3 30.4%(略低) |
| **Overall** | **45.2%** | B3 **46.7%**(≈持平,B3 略胜 1.5 pp)|

**B4 v2 ≈ B3 within noise** —— 当前的 paper 主结果不能 conclusively claim B4 超过 B3。

### 5.5 待验证的提升路径

| 假设 | 等数据 |
|---|---|
| **β=0.03 > β=0.05** | β=0.03 sweep T_eval(~5/19 晚)|
| **π^pre_v3 > π^pre_pilot** | Phase 1 v3 + B4-revert v3(~5/19 晚)|
| **K=4 net提升** | K=4 sweep T_eval(~5/19 下午)|
| **30B-A3B 突破 base 天花板** | 1 周以上工程 + 训练 |
| **dense reward 解锁 sparse-success app** | ~2-3 天工程 |

---

## 6. 当前并行实验状态(2026-05-19 03:48)

| 实验 | GPU | emulator | 进度 | ETA |
|---|---|---|---|---|
| K=4 sweep(chunk_C 余)| 5 | 5716 | osmand 4/20 + wikipedia 待 | ~6-7h |
| Phase 1 v3 | 6 | 5714 | iter 371/600 | ~14h |
| β=0.03 sweep chunk_A | 0 | 5710 | pro_expense | ~12h |
| β=0.03 sweep chunk_B | 1 | 5712 | retro_music | ~11h |
| β=0.03 sweep chunk_C | 2 | 5718 | vlc | ~12h |
| β=0.03 sweep chunk_D | 3 | 5720 | sim_cal + osmand | ~14h |
| β=0.03 sweep chunk_E | 4 | 5722 | chrome + maps_me | ~13h |
| β=0.03 sweep chunk_F | 7 | 5724 | broccoli + system + opentracks + wiki | ~16h |

**预期数据齐全的时间**:5/20 早晨 ~ 5/20 下午(取决于 T_eval 顺序)

---

## 附录:输出目录速查

```
traces/
├── phase1_pilot_v01/                # Phase 1 pilot (π^pre_pilot)
├── phase1_full_v01/                  # Phase 1 v2 (broken π^pre_v2)
├── phase1_v3/                        # Phase 1 v3 (KL-anchored,跑中)
├── b4_evolution_v2/                  # B4 raw v1 / Phase 3 v1 (broken)
├── b4_phase3_v01/                    # 5/13 Phase 3 sweep(产物在 B4 v1 用过,Tier-C 部分给 (A) revert 测试)
├── b4_phase3_v02/                    # **Phase 3 v2**(B4 v2 Tier-B 6 + Tier-C option-a 6)
├── b4_bootstrap_tier_c/              # **Bootstrap Tier-C 6 app**(option b)
├── b4_bootstrap_pilot_opentracks/    # opentracks bootstrap pilot
├── b4_unified/                       # symlink 合并(Tier-B 来自 Phase 3 v2 + Tier-C 来自 Bootstrap)
├── b4_k4_{A,B,C,D}/                  # K=4 sweep(K=2→K=4 ablation)
├── b4_b003_{A,B,C,D,E,F}/            # β=0.03 sweep(β=0.05→β=0.03 ablation)
├── b4_smoke_kl/                      # smoke v3 (β=0.02 k1 broken)
├── b4_smoke_kl_b05/                  # smoke b05 (β=0.05 k1 broken)
├── b4_smoke_kl_clip/                 # smoke_clip (β=0.02 + clip)
├── b4_smoke_beta_b003/               # β=0.03 smoke (K=4)
├── b4_smoke_beta_b007/               # β=0.07 smoke
├── b4_smoke_beta_b010/               # β=0.10 smoke
├── phase1_v3_smoke/                  # Phase 1 v3 pilot smoke
├── b1_teval_k3/                      # B1 T_eval K=3
├── b2_teval_k3/                      # B2 T_eval K=3
├── b3_teval/                         # B3 T_eval
├── b4_teval/                         # B4 raw v1 T_eval (broken 9%)
├── b4_revert_lora_teval/             # (A) B4-revert v1 (π^pre_pilot + B4 L_C)
├── b4_revert_v2_teval/               # B4-revert v2 (π^pre_v2 + B4 L_C)
├── b4_bootstrap_teval/               # Bootstrap T_eval pilot (opentracks only)
├── b4_bootstrap_tier_c_teval/        # Bootstrap T_eval (6 Tier-C apps)
└── b4_v2_teval/                      # **B4 v2 paper main** (45.2%)
```

---

## 一句话(给老板)

> 我们 paper 主结果 **B4 v2 overall 45.2% ≈ B3 46.7%**(无显著超越)。当前并行跑 4 个实验测能否突破:K=4 ablation(pro_expense 反例,但 sim_cal/retro_music 大幅提升)、β=0.03 ablation(基于 smoke 看更优)、Phase 1 v3 with KL anchor(可能产出更强 π^pre)。明天下午到晚上全部数据出齐,届时决定 paper 最终 framing(超 B3 / 持平 / 8B 天花板 finding)。

# 多 app L_C 注入设计（统一逐-app 解析）

> 在 EvoFSM-MW harness（pure-vision + mobile_use 新栈）里实现的 L_C 注入规格。
> 解决 MobileWorld 测试集 53%（85/161）是 cross-app 任务、且有 Tier-A 混合迁移的情况。
> 参考实现（旧 a11y 栈）：`EvoFSM-RL/evofsm_rl/fsm/injection.py`（单 app 解析）、
> `EvoFSM-RL/evofsm_rl/fsm/mutation.py:386`（bootstrap）、`EvoFSM-RL/evofsm_rl/agent/prompts.py:246`（渲染）。
> Tier 划分见 [`dataset_tiers.md`](dataset_tiers.md)。

---

## 0. 背景：为什么旧的单 app 注入不够

旧 EvoFSM-RL（AndroidWorld，大多单 app 任务）的 L_C 注入是单 app 假设的：
`resolve_l_c_for_app(app: str) → 单个 L_C string`，`_l_c_prompt_text: str | None`（单块）。

MobileWorld 测试集里 **53% 是 cross-app 任务**，且有 **Tier-A 混合迁移**（一个 task 同时含 seen-category 的 app 和 novel-category 的 app）。单 app 注入处理不了"一个 task 注入多块 L_C、且 novel 部分怎么办"。

---

## 1. 核心原则：不分 tier，逐 app 解析

**关键洞察**：不需要先判断 task 是 Tier-A/B/C 再用不同策略。只有一套逐-app 逻辑，三个 tier 是它的自然结果。

对 task 涉及的**每个 app**：
- 它的 category 有现成 L_C（seen，训练集覆盖）→ 用**现成 L_C**；
- 没有（novel）→ **bootstrap 一个 L_C**（从该 app 的探索轨迹引导，见 §4）。

然后**按 category 去重**：同 category 的多个 app 共用一块 L_C，不同 category 才多块。

```python
def resolve_l_c_for_task(task) -> str | None:
    # task.app_names: set[str]，来自 MobileWorld task 类的 app_names 字段
    blocks = {}  # category -> {"text": L_C_text, "apps": [app, ...], "source": "preset"|"bootstrap"}
    for app in task.app_names:
        cat = play_category_of(app)
        if cat in blocks:
            blocks[cat]["apps"].append(app)        # 同 category 去重，仅追加 app
            continue
        text = resolve_l_c_for_app(app)             # seen → 现成 L_C（artifacts/L_C/{slug}.json）
        source = "preset"
        if text is None:                            # novel → bootstrap
            text = bootstrap_l_c(app, cat)          # 从 app 的探索轨迹引导（见 §4）
            source = "bootstrap"
        blocks[cat] = {"text": text, "apps": [app], "source": source}
    if not blocks:
        return None
    return render_l_c_section(blocks)               # 见 §3
```

三个 tier 自动落位（代码里没有任何 `if tier`）：

| task 碰巧由哪些 cat 组成 | resolve 结果 | = tier |
|---|---|---|
| 全 seen category | 全现成 L_C | Tier-B |
| seen + novel 混合 | seen 现成 + novel bootstrap | Tier-A |
| 全 novel category | 全 bootstrap L_C | Tier-C |

---

## 2. 去重规则（按 category）

- **同 category 多 app**：注入**一块** L_C，标注服务的所有 app。
  例：`Mail + Messages`（都 Communication）→ 一块 Communication L_C，标「服务 Mail, Messages」。
- **不同 category**：每个 category 一块。
  例：`Calendar(Productivity) + Mail(Communication)` → 两块。
- novel 的 bootstrap 也按 category 去重：`Mastodon + Mattermost`（都 Social）→ 一块 bootstrap Social L_C。

---

## 3. 渲染：每块标明服务的 app

旧栈每块只标 `L_C CATEGORY: <name>`。多 app 任务里 agent 需要知道**哪块用于哪个 app**（操作 Calendar 查这块、操作 Mail 查那块），所以每块标签要带 app：

```
# Workflow knowledge (transferred from related apps)
<intro 一次>

## For [Calendar]  (category: Productivity, source: pretrained)
<Productivity L_C Layer-2 block>

## For [Mail, Messages]  (category: Communication, source: pretrained)
<Communication L_C Layer-2 block>

## For [Mastodon]  (category: Social, source: bootstrapped from this app's exploration)
<bootstrap Social L_C Layer-2 block>
```

- header / intro 出现**一次**；下面是**多块**，每块声明服务哪些 app + category + 来源（pretrained / bootstrapped）。
- 注入位置：mobile_use prompt 的 app_guidance 槽（旧栈是 `prompts.py` 的 `{l_c_section}`，新栈是 harness 的 `build_system_prompt`/`build_user_turn` 的对应槽）。

---

## 4. novel app 的 bootstrap L_C（统一处理 Tier-C 和 Tier-A 的 novel 部分）

机制已存在：`EvoFSM-RL/evofsm_rl/fsm/mutation.py:386 _build_bootstrap_l2_reflection_prompt(fsm, app, task_category, trajs_text)`。

当某 app 的 category 没有现成 L_C 时，**用该 app 的探索轨迹作为唯一知识源，从空 bootstrap 出一个 category 级 Layer-2 抽象库**（仍要求 app-agnostic，能泛化到同 category 其他 app）。

- 这正是 **Tier-C** 用的机制（`--enable-bootstrap-fsm`）。
- **Tier-A 的 novel app（Mastodon/Taodian/Mattermost）复用同一机制**——它们和 Tier-C 一样无匹配 category，不"裸跑"，而是 bootstrap 一个 L_C。

**依赖（注入时假设已就绪）**：bootstrap 需要该 novel app 的探索轨迹。这在 **TTA 的 Phase-2 cold-start**（对每个 target app 跑 5–10 条探索 rollout）准备。Tier-A 任务的 novel app 也要走这个探索 → bootstrap 流程；`resolve_l_c_for_task` 注入时直接取已 bootstrap 好的 L_C。

---

## 5. task → app_names

`task.app_names` 来自 MobileWorld task 类的 `app_names: set[str]` 字段（如 `{"Calendar", "Messages"}`），harness 在 reset/构造 prompt 时读取。app → category 映射用 `play_category_of(app)`（见 [`dataset_tiers.md`](dataset_tiers.md) §4 的 MW app→category 表；novel app 返回的 category 在训练集无匹配 → 触发 bootstrap 分支）。

---

## 6. 实现位置与 checklist

全部在 **EvoFSM-MW harness** 实现（旧 EvoFSM-RL 代码仅作参考）：

- [ ] `play_category_of(app)`：MW app → category（含 novel app 的 Social/Shopping）。
- [ ] `resolve_l_c_for_app(app)`：seen → 读 `EvoFSM-RL/artifacts/L_C/{slug}.json`；novel → None。
- [ ] `bootstrap_l_c(app, cat)`：取该 app 已 bootstrap 的 L_C（Phase-2 探索产出）。
- [ ] `resolve_l_c_for_task(task)`：§1 的逐-app 解析 + 按 category 去重。
- [ ] `render_l_c_section(blocks)`：§3 的多块渲染（每块标 app + category + 来源），注入 mobile_use prompt 的 app_guidance 槽。
- [ ] 接到现有 `AndroidEvoFSMAgent._resolve_app_guidance(task)` 的 TODO（当前返回 ""）。

---

## 7. 一句话

> **逐 app 解析（现成 or bootstrap）→ 按 category 去重 → 多块带 app 标签注入。** 不分 tier，Tier-A 的混合情况是这套逻辑的自然结果，每个 app 都有 L_C、无裸跑。

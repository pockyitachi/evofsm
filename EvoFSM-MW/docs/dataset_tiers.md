# EvoFSM 跨基准数据集与 Tier 划分

> 训练集 = AndroidWorld+ 193 任务（`androidworld:evofsm-tasks193`）。
> 测试集 = MobileWorld GUI-only 161 任务。
> 本文档定义：(1) 训练集的 category 分类，(2) 测试集的 5 类 / Tier 划分，(3) MW→AW 的同-category 索引（L_C 注入来源）。
>
> 数据来源：训练 category 取自 `EvoFSM-RL/configs/task_categories.csv`（192 app-attributable 行）；
> MW 任务 `app_names` / `task_tags` 取自 `MobileWorld/src/mobile_world/tasks/definitions/`（逐 class 枚举）。

---

## 1. 训练集：AndroidWorld+ 193 任务的 category 分类

12 个 Play-Store category。**这 12 类即「seen categories」**，决定测试集任务属于 Tier-B/C/A。

| play_category | apps (task 数) | 小计 |
|---|---|---|
| Productivity | joplin(4), markor(14), simple_calendar_pro(17), tasks_org(6) | 41 |
| Tools | calculator(19), clock(3), files(2), system_settings(15) | 39 |
| Finance | bluecoins(15), pro_expense(9) | 24 |
| Maps & Navigation | maps_me(15), osmand(3) | 18 |
| Music & Audio | audio_recorder(2), pi_music(12), retro_music(4) | 18 |
| Photography | camera(2), snapseed(11) | 13 |
| Food & Drink | broccoli(13) | 13 |
| Communication | chrome/Browser(3), contacts(2), simple_sms_messenger(6) | 11 |
| Books & Reference | wikipedia(6) | 6 |
| Health & Fitness | opentracks(6) | 6 |
| Video Players & Editors | vlc(2) | 2 |
| Art & Design | simple_draw_pro(1) | 1 |
| **合计** | | **192** |

> 注：CSV 覆盖 192 个 app-attributable 任务；AW+ 全集是 193 个 `task_id`（0–115 vanilla + 116–192 plus）。
> 训练阶段 193 个**全部是 source**（Phase-1 π^pre 预训练），CSV 里残留的 `source/tier_B/tier_C` 标注是 EvoFSM-RL **旧的 AW-内部** split，与本文档的跨基准 tier **无关**，仅 `play_category` 列被采用。

---

## 2. 测试集：MobileWorld 161 GUI-only 任务的 5 类分布

- GUI-only = `task_tags` 不含 `agent-mcp`（161 = 201 总数 − 40 MCP）。
- 每个任务的 `app_names` 集合决定它涉及哪些 app。
- **C-app（novel category，训练集没有）** = Mastodon / Mattermost（Social）、Taodian（Shopping）。
- 其余所有 app 都是 **B-app（seen category）**。
- ⚠️ **Tier 判据纯按 category（Play-Store 类别），不涉及 app 实例是否在训练中出现。** 「迁移」一律指 **category 级迁移**（同类知识能否帮一个该类的 app），不是 app 级；本文档不主张测试 app 是「未见过的 app」。「B-app / C-app」是「属于 seen / novel category 的 app」的简写，不是「seen / unseen app」。

| # | 情况 | 任务数 | → Tier |
|---|---|---|---|
| 1 | 单 app，app ∈ seen（B-app） | 36 | **Tier-B** |
| 2 | 单 app，app ∈ novel（C-app） | 40 | **Tier-C** |
| 3 | cross-app，涉及 app **全是 B-app** | 55 | **Tier-B** |
| 4 | cross-app，涉及 app **全是 C-app** | 3 | **Tier-C** |
| 5 | cross-app，涉及 app **B、C 混合** | 27 | **Tier-A** |
| | **合计** | **161** | |

### Tier 定义（本项目命名）

| Tier | = 情况 | 任务数 | 含义 | L_C 注入 |
|---|---|---|---|---|
| **Tier-B** | 1 + 3 | **91** | 近迁移（category 级）：所有涉及 app 的 category 都 seen | 全部现成 L_C |
| **Tier-C** | 2 + 4 | **43** | 远迁移（category 级）：所有涉及 app 的 category 都 novel | 全部 bootstrap L_C（从 target app 轨迹引导）|
| **Tier-A** | 5 | **27** | 混合迁移：同时含 seen + novel category 的 app | seen 现成 L_C + novel bootstrap L_C |

> **† 共享包名脚注（重要,避免被误读为 app 级泛化）。** 6 个系统 app 在训练集（AW+）与测试集（MobileWorld）两个 benchmark **共享同一包名**：Chrome (`com.android.chrome`)、Contacts (`com.google.android.contacts`)、Settings (`com.android.settings`)、Clock (`com.google.android.deskclock`)、Camera (`com.android.camera2`)、Files (`com.google.android.documentsui`)。本项目**不对其特殊处理**——tier 一律按 category 划分。说明:
> 1. 方法贡献以**差值**报告（有 L_C − 无 L_C，policy 固定）；这 6 个 app 的「同-app 优势」在 B1/B2 两侧都在,相减抵消,**不偏置 L_C 的 claim**。
> 2. 仅 Tier-B 的**绝对**准确率含此因素,**不应**解读为「对未见 app 的泛化」——本项目从不作此主张。
> 3. 包名核对来源:MW `runtime/utils/models.py:APP_DICT`，AW `env/adb_utils.py:_PATTERN_TO_ACTIVITY`。（Files 在 AW 另有 `com.simplemobiletools.filemanager.pro` 装于快照,open-app 映射的规范目标是 documentsui,与 MW 一致。）

> **注入逻辑统一为逐-app 解析（不分 tier）**：对 task 涉及的每个 app，其 category 有现成 L_C 就用现成、没有就 bootstrap 一个，再按 category 去重拼接。tier 只是事后描述 task 碰巧由哪些 cat 组成，注入代码无需 `if tier == ...` 分支——三个 tier 是同一套逻辑的自然结果。完整设计见 [`lc_injection_multiapp.md`](lc_injection_multiapp.md)。

---

## 3. 各情况任务明细

### Case 1 — 单 app / Tier-B（36）

| app | category | #任务 |
|---|---|---|
| Files | Tools | 7 |
| Mail | Communication | 6 |
| Settings | Tools | 6 |
| Maps | Maps & Navigation | 4 |
| Clock | Tools | 4 |
| Calendar | Productivity | 3 |
| Messages | Communication | 3 |
| Chrome | Communication | 2 |
| Camera | Photography | 1 |

### Case 2 — 单 app / Tier-C（40）

| app | category | #任务 |
|---|---|---|
| Mastodon | Social (novel) | 24 |
| Taodian | Shopping (novel) | 12 |
| Mattermost | Social (novel) | 4 |

### Case 3 — cross-app 全 B（55，Tier-B）

55 个任务的 app 集合全部落在 seen category 内（Files/Mail/Settings/Maps/Clock/Calendar/Messages/Chrome/Camera/Contacts/Gallery/Docreader 的两两/多重组合）。代表组合：Files+Mail、Calendar+Messages、Docreader+Files、Gallery+Mail、Camera+Gallery+Mail 等。

### Case 4 — cross-app 全 C（3，Tier-C）

| 任务 | app 集合 | 涉及 novel category |
|---|---|---|
| MastodonMattermostPostNoticeTask | Mastodon, Mattermost | Social + Social |
| MastodonMallPurchaseCommodityTask | Mastodon, Taodian | Social + Shopping |
| MastodonMallShareOrderTask | Mastodon, Taodian | Social + Shopping |

### Case 5 — cross-app 混合（27，Tier-A）

> 每个任务含 1 个 C-app + 1~3 个 B-app。括号内为 B-app（seen，可注入 L_C 的部分）。

**含 Mastodon（14）**
| 任务 | B-app（seen 部分） |
|---|---|
| MastodonCalendarMultiMemosTask | Calendar |
| MastodonChangeHeaderTask | Gallery |
| MastodonCreateMemoTask | Calendar |
| MastodonFollowTask | Contacts |
| MastodonInviteTask | Messages |
| MastodonMultiInviteTask | Messages |
| MastodonNewFilterTask | Files |
| MastodonPostEditedPhotoTask | Gallery |
| MastodonPostPollTask | Chrome |
| MastodonSavePhotosTask | Gallery |
| MastodonServerInfoReportTask | Mail |
| MastodonShareLocationTask | Gallery, Maps |
| MastodonSharePhotosAskUserTask | Gallery |
| MastodonUpdateContactsTask | Contacts |

**含 Mattermost（12）**
| 任务 | B-app（seen 部分） |
|---|---|
| LocalFileManagementTask | Files |
| MattermostCustomerFeedbackAnalysisTask | Calendar, Mail |
| MattermostDeadlineReconciliationTask | Calendar, Mail |
| MattermostEmailTask | Mail |
| MattermostIncidentEscalationTask | Calendar, Mail |
| MattermostProjectHandoverTask | Calendar |
| MattermostProjectStatusReportTask | Calendar, Mail |
| MattermostReadingGroupTask | Chrome |
| MattermostResourceConflictResolutionTask | Calendar, Mail |
| MattermostShiftCoverageTask | Calendar, Mail |
| MattermostTechnicalDebtTriageTask | Contacts, Messages |
| MattermostVisualInstructionResponseTask | Clock, Contacts |

**含 Taodian（1）**
| 任务 | B-app（seen 部分） |
|---|---|
| CartInfoNotificationTask | Messages |

---

## 4. 索引：MW app/category → 训练集（AW+）同-category source

L_C 是 **category 级**（一个 category 一份，`EvoFSM-RL/artifacts/L_C/{slug}.json`，由该 category 下所有 AW source app 的 FSM 合并而来）。
因此 MW 任务的 L_C 注入按下表的 category 解析：MW app → category → 该 category 在训练集里的 source app/task → 对应 L_C。

| MW app | MW category | seen? | 训练集同-category source apps（L_C 来源） |
|---|---|---|---|
| Mail | Communication | ✅ | chrome(Browser), contacts, simple_sms_messenger |
| Messages | Communication | ✅ | chrome(Browser), contacts, simple_sms_messenger |
| Contacts | Communication | ✅ | chrome(Browser), contacts, simple_sms_messenger |
| Chrome | Communication¹ | ✅ | chrome(Browser), contacts, simple_sms_messenger |
| Files | Tools | ✅ | calculator, clock, files, system_settings |
| Settings | Tools | ✅ | calculator, clock, files, system_settings |
| Clock | Tools | ✅ | calculator, clock, files, system_settings |
| Calendar | Productivity | ✅ | joplin, markor, simple_calendar_pro, tasks_org |
| Maps | Maps & Navigation | ✅ | maps_me, osmand |
| Camera | Photography | ✅ | camera, snapseed |
| Gallery | Photography | ✅ | camera, snapseed |
| Docreader | Books & Reference | ✅ | wikipedia |
| **Mastodon** | **Social** | ❌ novel | bootstrap L_C（从 Mastodon 轨迹引导，无 source）|
| **Mattermost** | **Social** | ❌ novel | bootstrap L_C（从 Mattermost 轨迹引导）|
| **Taodian** | **Shopping** | ❌ novel | bootstrap L_C（从 Taodian 轨迹引导）|

¹ Chrome 是浏览器，category 在 AW 里随 `chrome/Browser*` 归 Communication；亦可视作 Tools。

**对 Tier-A（case 5）的含义**：逐 app 解析——seen-app 用上表的现成 category L_C，novel-app（Mastodon/Mattermost/Taodian）用 bootstrap L_C（从该 app 轨迹引导），再按 category 去重拼接。所以 Tier-A 任务**每个 app 都有 L_C**（来源不同），不再有"裸跑"部分。详见 [`lc_injection_multiapp.md`](lc_injection_multiapp.md)。

---

## 5. 一句话汇总

```
训练 193 (AW+, 12 categories, 全 source)
  └─ T-eval → MobileWorld 161 GUI-only
       ├─ Tier-B  91  (case1 单app全seen 36 + case3 cross全seen 55)   → 全现成 L_C
       ├─ Tier-C  43  (case2 单app全novel 40 + case4 cross全novel 3)   → 全 bootstrap L_C
       └─ Tier-A  27  (case5 cross混合 seen+novel)                     → seen现成 + novel bootstrap
  注入逻辑不分 tier：逐 app 解析(现成/bootstrap) + 按 category 去重 → 见 lc_injection_multiapp.md
```

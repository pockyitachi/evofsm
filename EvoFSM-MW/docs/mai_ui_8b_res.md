# MAI-UI-8B on MobileWorld — external baseline + injection transfer

The paper's external-baseline row, plus a cross-model test of whether the
qwen-derived static guidance transfers to a stronger GUI agent. Companion to
`qwen3_8b_res.md` (the EvoFSM 8B B-series); same harness, same 110-task eval
split, same 5-run variance protocol.

Status: complete (2026-06-12→13, unattended chain `pi350mai`).

---

## 0. Setup (deltas vs `qwen3_8b_res.md` §0)

- **Model** — `Tongyi-MAI/MAI-UI-8B`, MobileWorld's own first-party GUI agent
  (architecture `Qwen3VLForConditionalGeneration`; served on a dedicated vLLM,
  GPU7 :8002, bf16). **Native to this benchmark** — it is the reference agent
  MobileWorld ships, so this is a near-ceiling external baseline, not a
  generic VLM.
- **Agent** — MW's stock `mai_ui_agent` (registry): `<thinking>/<tool_call>`
  format, multi-image history, `ask_user` + MCP actions. **temperature 0**
  (official default — near-deterministic decoding; run-to-run variance comes
  from emulator timing and the user-agent simulator only).
- **User-agent simulator** — `ask_user` actions are answered by **gpt-4.1**
  via the OpenAI API (`USER_AGENT_*` in `.env`). Required for the 25 AskUser
  tasks. Misconfigured (empty URL+key) until 2026-06-12; the maib1_r1 numbers
  below are from the fixed config.
- **Injection (maib2v)** — `mai_ui_b2_agent.py` subclasses the stock agent and
  overrides `system_prompt` to splice the guidance block before `## Note`
  (empty guidance ⇒ byte-identical to stock). Guidance = `b2p_guidance.json`
  = **B2′** (app Layer-2 + category L_C), the strongest static variant from
  the qwen variance study (`strongest_variant.txt`).
- Everything else identical to the qwen B-series: 110-task eval, fresh
  containers per run, score=1.0, 108/110 written (same two upstream-bug tasks,
  §4 note).

---

## 1. MAI-UI-8B B1 (stock, no injection)

### Results

5 runs: 25, 24, 30, 27, 25 → **26.2 ± 2.4 / 110 (23.8%)**.
Per-tier means: Tier-B 17.4 / 58 · Tier-C 5.4 / 25 · Tier-A 3.4 / 27.

### Analysis

- **~3.2× the EvoFSM 8B base** (qwen B1 8.2 ± 0.8). MAI-UI is a far stronger
  GUI agent on its home benchmark, as expected.
- The gap is widest on **Tier-A (multi-app composition)**: 3.4 vs qwen's 1.0
  (and qwen's best static config B2′ scored *0.0* there). Compositional,
  multi-app control is where the capability difference is largest.
- MAI passes, zero-shot, exactly the tasks qwen could only get via injection
  or not at all: `CheckEventTimeTask`, `ChromeSearchBeijingWeatherTask`,
  `TakeSelfieTask` — the read-before-acting / verify-before-terminate
  discipline that the qwen guidance was designed to supply is already internal
  to MAI.
- Run-to-run σ=2.4 (vs qwen's 0.8) despite temperature 0: the larger spread is
  the gpt-4.1 simulator + longer multi-app episodes, not policy stochasticity.

---

## 2. MAI-UI-8B + B2′ injection

### Results

5 runs: 25, 28, 25, 25, 29 → **26.4 ± 2.0 / 110 (24.0%)**.
Per-tier means: Tier-B 17.8 / Tier-C 5.6 / Tier-A 3.0.

Net vs stock: **+0.2 (within noise)**. Task-level flips (5-run counts,
|Δ| ≥ 3), maib1 → maib2v:

| Task | Δ | Mode | Tier |
|---|---|---|---|
| `SetAlarmTaskAskUser2` | 5 → 1 | injected | B (AskUser) |
| `ChromeSearchBeijingWeatherTask` | 4 → 1 | injected | B |
| `CartInfoNotificationTask` | 4 → 0 | injected | A |
| `AdjustFontIconMinimumTask` | 3 → 0 | injected | B |
| `SearchTopInfoAskUserTask` | 3 → 0 | injected | B (AskUser) |

### Analysis

- **Net-neutral overall, but every large attributable flip is a regression.**
  The five |Δ|≥3 movements all point the same way — injection *broke* tasks
  MAI was solving stock — and the flat total (26.2→26.4) means scattered
  sub-threshold gains (+1/+2 on other tasks) merely cancelled them.
  Injection is an **ineffective perturbation** on this model: no systematic
  lift, occasional damage to already-solved tasks.
- Caution on the evidence: 2 of the 5 regressions are AskUser tasks
  (`SetAlarm…`, `SearchTopInfo…`), which carry gpt-4.1 simulator variance.
  The 3 deterministic ones (`ChromeSearch`, `CartInfo`, `AdjustFontIcon`) are
  the cleaner signal — and `ChromeSearchBeijingWeatherTask` is the *same*
  over-procedural failure seen on qwen B2″ (§4.2 of the qwen doc): injected
  search/verify discipline makes the agent over-work a task it would otherwise
  shortcut.
- Mechanism: B2′ guidance was mined from **qwen-8B's** weaknesses. A model
  that already has that workflow knowledge gains nothing and is occasionally
  pushed off its own (better) policy — the **over-specification** failure
  mode, the mirror image of qwen's Layer-1 harm.

---

## 3. B3 — test-time evolution variants (MAI-UI-8B)

The B3 question, re-asked on the strong agent: does *symbolic* test-time
evolution (FSM evolution and/or distilled lessons, all adapted on the 51-task
`T_adapt` split) clear the static B2′ prior on MAI? Three configs, same harness
/ eval split / 5-run protocol as §§1–2. Mirror of the qwen B3 study
(`b3_lesson_memory_results.md`).

### Leaderboard (/110 eval; 108/110 written, same two upstream-bug tasks)

| config | mean ± std /110 | runs | Tier-B | Tier-C | Tier-A |
|---|---|---|---|---|---|
| **lessons-only** | **29.0 ± 4.5** | 5 (30,21,32,31,31) | 20.8 | 4.4 | 3.8 |
| FSM-evo | 27 | 1 | 18 | 6 | 3 |
| **fsm+lessons** | **25.6 ± 3.0** | 5 (30,25,27,24,22) | 18.4 | 4.2 | 3.0 |
| B2′ (static prior) | 26.4 ± 2.0 | 5 | 17.8 | 5.6 | 3.0 |
| B1 (stock) | 26.2 ± 2.4 | 5 | 17.4 | 5.4 | 3.4 |

(Tier means are sample means over the same runs; std is sample std. fsm+lessons,
B2′, B1 and FSM-evo all sit inside ±1σ of each other → statistically the same
≈26 point. lessons-only is nominally above but with the widest spread, see
below.)

### Analysis

- **Symbolic TTA does not clearly beat the static prior on MAI either.** The
  centre of mass for FSM-evo (27), fsm+lessons (25.6) and B2′ (26.4) is the same
  ~26 — exactly the §3-of-qwen story (`b3_lesson_memory_results.md`): symbolic
  evolution tops out *at* the static prior, on a strong base as on a weak one. As
  on qwen, TrueSkill is signal-starved on this single-pass 51-task schedule
  (variants tie on floor/ceiling tasks, no `mixed_groups`), so the "evolution"
  collapses to "pick a fixed champion" — i.e. another static injection.
- **`fsm+lessons` ≈ B1/B2′** (25.6 vs 26.2 / 26.4, ΔTotal −0.6 vs stock, well
  within the ±2–3σ run noise). The merged FSM-champion-plus-lessons block is
  net-neutral on a model that already has the workflow knowledge — the same
  over-specification wash seen for B2′ in §2: per-tier it is flat-to-slightly-
  down everywhere (Tier-C 4.2 vs 5.4/5.6 is the only consistent dip, the verbose
  block displacing MAI's own Mastodon/Taodian handling). No systematic lift.
- **`lessons-only` is nominally higher (29.0 vs ~26) but within noise.** It is
  the single config whose mean clears stock — and the lean distilled lessons (the
  ≈800-char form, not the bloat) being the *better* injection echoes qwen's
  cleanest positive (lessons-only matched the ceiling at 1/20 the text). **But
  the run-to-run std is 4.5** — the widest of any MAI config here — driven by one
  low run (21) against four in the 30–32 band; the 29.0 ± 4.5 interval overlaps
  B1 (26.2 ± 2.4) and B2′ (26.4 ± 2.0) heavily. At n=5 this is **not** a
  significant win; report it as *nominally higher, within noise*, not a result.
  Its only directional edge is Tier-B (20.8 vs 17.4/17.8), but that is exactly
  where the variance lives.
- **Net for the paper:** on the near-ceiling MAI agent, the symbolic TTA channel
  reproduces the qwen verdict — its ceiling is the static prior, and the one
  config that nominally exceeds it does so inside the noise floor. There is no
  capability-robust symbolic win on either model.

---

## 4. Cross-model summary

| Model | config | mean ± std /110 | Tier-B | Tier-C | Tier-A | Δ vs stock |
|---|---|---|---|---|---|---|
| EvoFSM 8B (qwen) | B1 stock | 8.2 ± 0.8 | 5.2 | 2.0 | 1.0 | — |
| EvoFSM 8B (qwen) | +B2′ | 9.2 ± 1.6 | 6.4 | 2.8 | 0.0 | **+1.0** |
| EvoFSM 8B (qwen) | B3 FSM-evo | 10 | — | — | — | +1.8 |
| EvoFSM 8B (qwen) | B3 lessons-only | 10.0 ± 1.4 | — | — | — | +1.8 |
| EvoFSM 8B (qwen) | B3 fsm+lessons | 10.4 ± 0.5 | — | — | — | +2.2 |
| MAI-UI-8B | B1 stock | 26.2 ± 2.4 | 17.4 | 5.4 | 3.4 | — |
| MAI-UI-8B | +B2′ | 26.4 ± 2.0 | 17.8 | 5.6 | 3.0 | **+0.2** |
| MAI-UI-8B | B3 FSM-evo | 27 | 18 | 6 | 3 | +0.8 |
| MAI-UI-8B | B3 lessons-only | 29.0 ± 4.5 | 20.8 | 4.4 | 3.8 | +2.8 (noise) |
| MAI-UI-8B | B3 fsm+lessons | 25.6 ± 3.0 | 18.4 | 4.2 | 3.0 | −0.6 |

**Static *and* symbolic-adaptive injection are both capability-bounded, and the
symbolic channel tops out at the static prior on both models.** On the weak
EvoFSM-8B base, B2′ is a small positive (+1.0, ~1σ) and the B3 symbolic configs
sit ~1 point above it at ≈10 — a low ceiling reached and held, never cleared
cleanly. On the near-ceiling MAI-UI-8B, B2′ is net-zero (+0.2), B3 fsm+lessons is
net-zero-to-slightly-down, and the one nominally-up config (lessons-only, +2.8)
carries a ±4.5 spread that overlaps stock — i.e. no significant symbolic win on
either model. This is the core argument for the *weight* channel (B4) over any
text-channel knowledge injection, static or evolved: hand-authored, distilled, or
TrueSkill-selected guidance all share the same low, capability-bounded ceiling
(≈ the static prior), whereas the B4 weight update is the only lever that can
move the deployed model's actual policy. **The symbolic half is at its ceiling;
the headroom is in B4.**

**Note — two chronic non-results (all models, all runs).**
`CheckMeetingEventAskUserTask` (upstream task-init bug: `'…Task' object has no
attribute 'today'`) and `ThanksgivingPrepTask` write no result in any of the
35 B-series + MAI runs; counted as failures throughout. Benchmark defects,
uniform across models — they do not affect any comparison.

**Note — AskUser parity.** qwen's B-series ran before the simulator fix, but
qwen almost never emits `ask_user` (every qwen run still wrote 108/110 — the
broken simulator crashed *no* qwen task), so the fix does not retro-bias the
qwen numbers. MAI *does* use `ask_user` and gets real gpt-4.1 replies — a
genuine tool-use capability difference, not an environment artifact.

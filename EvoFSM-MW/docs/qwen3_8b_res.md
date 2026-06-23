# Qwen3-VL-8B on MobileWorld — B-series results

Base model rows for the EvoFSM paper: zero-shot (B1), static symbolic injection
(B2) and its two ablations (B2', B2''). All runs share the same weights and
harness; they differ only in what is spliced into the system prompt.

Status: complete. Single runs 2026-06-09→10 (§1–§3); 5-run variance study +
B2''' 2026-06-10→11 (§5); π^pre step-350 under B1 setting 2026-06-11 (§6).
§3's single-run numbers are superseded by §5's means; §3 is kept as the
run-1 record and mechanism analysis. MAI-UI-8B results live in
`mai_ui_8b_res.md`.

---

## 0. Common setup

- **Model / serving** — `Qwen3-VL-8B-Instruct` on one shared vLLM server
  (`localhost:8001`), identical weights for every run.
- **Benchmark** — MobileWorld, GUI-only **eval split: 110 tasks**
  (`configs/teval_tasklist.txt` = the eval side of
  `mobileworld_splits.yaml v1.0-MW`; the 51-task adapt split is not run).
  Tier composition: **Tier-B 58 / Tier-C 25 / Tier-A 27**
  (category-seen / category-novel / mixed — `docs/dataset_tiers.md`).
- **Agent** — MobileWorld stock `Qwen3VLAgentMCP`
  (`MOBILE_QWEN3VL_PROMPT_WITH_ASK_USER`). B2-family runs load a file-path
  subclass that sets `self.app_guidance`; a 4-line MobileWorld patch splices it
  as a `# App guidance` section immediately before `# Response format`.
  Empty guidance ⇒ byte-identical prompt to B1.
- **Harness** — `mw eval`, `max_round 50`, `step_wait_time 3`,
  `max-concurrency 3`, `--enable_mcp --enable_user_interaction`.
  3 emulator containers per run (image `mobile_world:reset`, network `mwnet`),
  **created fresh for every run and never reused** — MobileWorld containers do
  not reset app state between tasks, so a used container changes initial
  conditions.
- **Scoring** — success = `result.txt` with `score: 1.0`; denominator is
  always 110. Tasks that crash in the harness and write no `result.txt` count
  as failures. Two tasks (`CheckMeetingEventAskUserTask`,
  `ThanksgivingPrepTask`) fail this way in every run (same harness error in B1
  and B2; model-independent).
- **Artifacts** — trajectories in `MobileWorld/traj_logs/{b1,b2,b2p,b2pp}_qwen3vl8b/`;
  injection JSONs in `EvoFSM-MW/artifacts/`; launch scripts in
  `EvoFSM-MW/harness/`.

---

## 1. B1 — zero-shot baseline

### 1.1 Setup

Stock agent, no injection. Run 2026-06-09 (tmux `b1eval`, containers
`lq_b1_{0,1,2}`).

### 1.2 Results

**9 / 110 (8.2%)**. 108/110 wrote `result.txt`.

| Tier | n | success |
|---|---|---|
| B | 58 | 5 |
| C | 25 | 3 |
| A | 27 | 1 |

Successes: `ChangeWallpaperTask`, `CheckEventTimeTask`,
`ChromeSearchBeijingWeatherTask`, `CloseFlightModeTask`, `OpenFlightModeTask`
(Tier-B); `CheckPuchasedItem`, `MastodonNewPostTask`, `MastodonReplyTask`
(Tier-C); `MastodonChangeHeaderTask` (Tier-A).

### 1.3 Analysis

- The Tier-B successes concentrate on shared-package system apps
  (Settings ×3-of-5, Chrome) and short flows (≤4 steps); successful episodes
  ground every action in the screenshot and verify before `terminate`.
- Tier-C is not uniformly harder at this floor: 3 Mastodon tasks pass
  zero-shot.
- B1 is both the floor and the control anchor: B2-family Tier-C prompts are
  byte-identical to B1, so Tier-C deltas in those runs measure sampling noise.

---

## 2. B2 — static symbolic injection (full app FSM + category L_C)

### 2.1 Setup

Injection artifact `b2_guidance.json`, rendered offline by
`harness/gen_b2_guidance.py --mode full` from the 2026-06-08 static layer:
`EvoFSM-RL/artifacts/static_fsms_v2/` (25 per-app FSMs) and
`EvoFSM-RL/artifacts/L_C_v2/` (12 category libraries).

Per-app resolution, per task (`docs/lc_injection_multiapp.md`):

1. **app tier** — app is one of the 6 shared-package system apps
   (Chrome, Contacts, Settings, Clock, Camera, Files) ⇒ inject its **full FSM
   text: Layer-1 (states / transitions / visual_cues / dead_ends) + Layer-2**.
2. **category tier** — otherwise, if the app's Play category has an L_C file ⇒
   inject the **category L_C (aggregated Layer-2)**, deduplicated per category
   across the task's apps.
3. **none** — novel category (Social / Shopping) ⇒ contributes nothing.

Each task is labeled by its strongest injected tier ("injection mode" in the
tables below): **app-FSM** (≥1 app-tier block), **category-only**
(category blocks only), **empty** (no injection ⇒ prompt = B1).
Run-list composition: app-FSM 45 / category-only 40 / empty 25.
Injected text size over the run list: median 23.8k chars, max 61.4k
(≈15.3k tokens; vLLM `max_model_len` 32768). App-tier blocks alone are
15.3k–25.0k chars each.

Run 2026-06-09 (tmux `b2eval`, fresh `lq_b2_{0,1,2}`, auto-chained after B1).

### 2.2 Results

**7 / 110 (6.4%)** — net **−2 vs B1**. 108/110 wrote `result.txt` (same two
missing as B1).

| Tier | n | B1 | B2 | Δ |
|---|---|---|---|---|
| B | 58 | 5 | 3 | −2 |
| C | 25 | 3 | 3 | 0 |
| A | 27 | 1 | 1 | 0 |

| Injection mode | n | B1 | B2 | Δ |
|---|---|---|---|---|
| app-FSM | 45 | 5 | 3 | **−2** |
| category-only | 40 | 1 | 1 | 0 |
| empty (control) | 25 | 3 | 3 | 0 |

Task-level divergences vs B1:

| Direction | Task | Mode | Injected blocks |
|---|---|---|---|
| lost | `OpenFlightModeTask` | app-FSM | Settings full FSM (25.0k chars) |
| lost | `CloseFlightModeTask` | app-FSM | Settings full FSM |
| lost | `CheckEventTimeTask` | app-FSM | Clock full FSM + Communication L_C (apps: Clock, Mail) |
| won | `TakeSelfieTask` | app-FSM | Camera full FSM |
| lost | `MastodonReplyTask` | empty | — (prompt = B1; sampling noise) |
| won | `SearchItemAndCheckoutTask` | empty | — (prompt = B1; sampling noise) |

### 2.3 Analysis

- **All net damage sits in the app-FSM mode** (−3 / +1). The category-only
  mode is null either way — but on a subset where B1 scores 1/40, so the
  headroom for visible gains is small. The control mode's ±1 churn under
  byte-identical prompts sets the noise scale: the aggregate −2 is at the edge
  of noise, but the mechanism below is not.
- **Mechanism (both FlightMode tasks, trajectories):**

  | B1 (success, 4 steps) | B2 (failure, 3 steps) |
  |---|---|
  | click Settings gear icon | swipe down for quick-settings — the path described by the injected Layer-1 state `S2: QUICK_SETTINGS` |
  | click "Network & internet" | click lands on the status bar (y≈27–51/999), not on a tile |
  | toggle "Airplane mode" switch | `terminate(success)` — "as indicated by the successful execution of the steps", **no screenshot verification** |
  | verify from screenshot → terminate | |

  The injected Layer-1 — written from source-environment observations, and
  explicitly headed `APP_SPECIFIC (non-transferable)` — displaces the model's
  own grounded routine and supplies enough workflow confidence to terminate
  without verifying. The eval oracle reports flight mode unchanged in both
  tasks.
- **Continuity with the AndroidWorld B2** (`EvoFSM-RL
  docs/results/b1_b2_static_baselines.md`): that run injected **category
  Layer-2 only** (never Layer-1) in the same environment the FSMs were learned
  from, and netted **Tier-B +9.3pp** — yet its §5 regression case
  (`SimpleCalendarEventsInTimeRange`, 66.7%→0%) shows the same failure shape:
  guidance-induced step truncation. Static injection has always carried this
  failure mode; matching environment and abstract-only content kept it
  outweighed there.
- `TakeSelfieTask` (Camera FSM, B1 ✗ → B2 ✓) shows app-level knowledge can
  also pay; the question is whether the transferable half (Layer-2) suffices.
- These observations motivate the two single-variable ablations below.

### 2.4 B2' — app Layer-2 only (remove Layer-1)

#### Setup

Identical to B2 except the app tier renders **only the FSM's own Layer-2**
(`--mode app-l2`, artifact `b2p_guidance.json`): e.g. the Settings block
shrinks 25,333 → 6,492 chars and contains no states / visual_cues. Category
and empty modes are unchanged, so **B2 − B2' isolates the effect of
cross-benchmark Layer-1**. Run-list sizes: median 17.8k, max 46.4k chars.
Run 2026-06-09→10 (chain `b2variants`, fresh `lq_b2p_{0,1,2}`,
backend 6820-22).

#### Results

**10 / 110 (9.1%)** — net **+1 vs B1, +3 vs B2**. 108/110 wrote `result.txt`
(same two missing).

| Tier | n | B1 | B2 | B2' |
|---|---|---|---|---|
| B | 58 | 5 | 3 | **7** |
| C | 25 | 3 | 3 | 3 |
| A | 27 | 1 | 1 | 0 |

| Injection mode | n | B1 | B2 | B2' |
|---|---|---|---|---|
| app-FSM → app Layer-2 | 45 | 5 | 3 | **6** |
| category-only | 40 | 1 | 1 | 1 |
| empty (control) | 25 | 3 | 3 | 3 |

Task-level divergences vs B1:

| Direction | Task | Mode | Note |
|---|---|---|---|
| won | `OpenFlightModeTask` | app | lost in B2 → **recovered** |
| won | `CloseFlightModeTask` | app | lost in B2 → **recovered** |
| won | `TakeSelfieTask` | app | B2's gain **kept** with Layer-2 only |
| won | `ReadQwen3PaperTask4` | app | ✗ in both B1 and B2 → ✓ (Files Layer-2 + Books & Reference L_C) |
| won | `DeleteEventAskUserTask` | category | Calendar → Productivity L_C |
| lost | `CheckEventTimeTask` | app | still lost (Clock Layer-2 + Communication L_C) |
| lost | `MastodonChangeHeaderTask` | category | prompt identical to B2 (which passed) ⇒ sampling |
| ±2 churn | `CheckPuchasedItem` −, `MastodonReplyTask` −, `MastodonRemoveBookmarkTask` +, `SearchItemAndCheckoutTask` + | empty | prompt identical to B1/B2 ⇒ sampling |

#### Analysis

- **The Layer-1 ablation is confirmed at task level.** Only app-mode prompts
  differ between B2 and B2'; that mode moves 3 → 6: both FlightMode tasks
  (the verified Layer-1 failure mechanism) recover, `TakeSelfieTask` survives
  — Layer-2 alone was sufficient for B2's only gain — and
  `ReadQwen3PaperTask4` flips ✗→✓ when the 25k-char Files FSM shrinks to its
  Layer-2. The single residual app-mode loss (`CheckEventTimeTask`) does not
  recover, so its damage source is not Layer-1 (candidates: Clock Layer-2,
  Communication L_C, or floor noise; B2'' adds a datapoint).
- **Noise scale.** For category-only and empty tasks the B2' prompt is
  byte-identical to B2 (and to B1 for empty), and those modes still flip ±1
  and ±2 — so the +1 overall vs B1 is within sampling churn. The signal is
  the paired app-mode flips, which track Layer-1 presence exactly, and
  Tier-B 5→7.
- Net reading: **cross-benchmark Layer-1 is strictly harmful (B2−B2' = −3 on
  changed prompts); transferable Layer-2 at app granularity is mildly
  positive** (+1 app-mode vs B1, +2 Tier-B vs B1).

### 2.5 B2'' — category L_C only (original EvoFSM-RL recipe)

#### Setup

App tier disabled entirely (`--mode category-only`, artifact
`b2pp_guidance.json`): every Tier-B/A app resolves through its Play-category
L_C — the 6 system apps included (Settings/Clock/Files → `tools.json` 21.8k
chars, Chrome/Contacts → `communication.json`, Camera → `photography.json`).
This is a 1:1 port of the EvoFSM-RL B2 recipe that scored Tier-B +9.3pp on
AndroidWorld+, now cross-benchmark and pure-vision. Run-list sizes: median
21.8k, max 42.1k chars. Run 2026-06-10 (chain `b2variants`, fresh
`lq_b2pp_{0,1,2}`, backend 6830-32).

#### Results

**8 / 110 (7.3%)** — net −1 vs B1, +1 vs B2, −2 vs B2'. 108/110 wrote
`result.txt` (same two missing).

| Tier | n | B1 | B2 | B2' | B2'' |
|---|---|---|---|---|---|
| B | 58 | 5 | 3 | 7 | 6 |
| C | 25 | 3 | 3 | 3 | 2 |
| A | 27 | 1 | 1 | 0 | 0 |

| Injection mode | n | B1 | B2 | B2' | B2'' |
|---|---|---|---|---|---|
| app (treatment varies per run) | 45 | 5 | 3 | 6 | 4 |
| category-only (same prompt in B2/B2'/B2'') | 40 | 1 | 1 | 1 | 2 |
| empty (same prompt in all four) | 25 | 3 | 3 | 3 | 2 |

Task-level divergences vs B1: lost `CheckEventTimeTask`,
`ChromeSearchBeijingWeatherTask` (app mode), `MastodonChangeHeaderTask`
(category), `CheckPuchasedItem`, `MastodonReplyTask` (empty); won
`TakeSelfieTask` (app), `DeleteEventAskUserTask`,
`ScheduleCoffeeTimeViaSmsTask` (category), `SearchItemAndCheckoutTask`
(empty).

#### Analysis

- **The original recipe does not reproduce its AndroidWorld gain
  cross-benchmark**: +9.3pp Tier-B there vs −1 overall / +1 Tier-B here —
  within sampling churn, i.e. roughly baseline-neutral.
- On the 45 app-mode tasks — the only tasks whose prompts differ across the
  B2 family — replacing app-specific knowledge with the category aggregate
  costs 2 vs B2' (6→4): `ChromeSearchBeijingWeatherTask` (B1 ✓, B2 ✓, B2' ✓,
  B2'' ✗ — Chrome-specific text replaced by the 20.7k-char Communication L_C)
  and `ReadQwen3PaperTask4` (only B2''s Files Layer-2 wins it). Both flight
  mode tasks stay recovered — confirming once more that B2's damage was
  Layer-1, not injection per se.
- `CheckEventTimeTask` is lost under **all three** injection treatments
  (B1 ✓ → B2/B2'/B2'' ✗). The common denominator across the three prompts is
  the Communication L_C block (Mail) — which is also implicated in the
  `ChromeSearchBeijingWeatherTask` loss above. The Communication L_C (largest
  L_C file, 20.7k chars) is the top audit candidate in the static layer.
- `TakeSelfieTask` passes under all three treatments (B1 ✗): the Camera
  knowledge helps in any form; the win does not require app granularity.

### 2.6 B2''' — entry-level retrieval (B2' + per-block top-3)

#### Setup

Implements §4.3 fix #1 on top of the frozen B2' artifact, single-variable:
same blocks, same rendering, same INTRO; per injected block, only the ≤3
entries a retriever judges relevant to the task instruction are kept (fixes
#2/#3 deliberately not applied). Retriever = the same Qwen3-VL-8B on :8001,
text-only, temperature 0, one call per (task, block); instructions extracted
from the B1 trajectories (`artifacts/task_goals.json`). Artifact
`b2ppp_guidance.json` (`gen_b2ppp_guidance.py`): 85 retrieved tasks, 0
parse fallbacks; 10 tasks emptied by retrieval (notably the
`ReadQwen3Paper*` / `CheckInvoice*` document-reading families — no entry
matched; they degrade to B1, including `ReadQwen3PaperTask4` which B2' had
won — a known retrieval miss kept for single-variable integrity). Injected
size over the run list: median 3,375 chars (B2': 17,826 — 5.3×
smaller), max 11,317. Example selections: `OpenFlightModeTask` →
[NAVIGATE_TO, TOGGLE_SETTING]; `CheckEventTimeTask` → Clock [QUERY_INFO,
CREATE_TIMED_ENTRY] + Communication [NAVIGATE_TO, QUERY_INFO].

Runs: 5× (r1–r5), fresh 5-container set per run, queued after the §5
variance study (chain: tmux `b2pppchain`).

#### Results

5 runs (fresh containers each): 7, 7, 9, 9, 11 → **8.6 ± 1.7** (B1: 8.2 ± 0.8).
Per-tier means: Tier-B 5.6 / Tier-C 2.6 / Tier-A 0.4.

#### Analysis

- Keeps every deterministic effect of the B2' family (§5): FlightMode pair
  5/5, `TakeSelfieTask` 5/5, `CheckEventTimeTask` 0/5 — the retrieved
  2–3-entry blocks preserve the mechanism wins at 1/5 of the tokens
  (median 3.4k vs 17.8k chars).
- **The one attributable regression is the known retrieval miss**:
  `ReadQwen3PaperTask4` drops 5/5 (B2') → 2/5. It is the only
  emptied-by-retrieval task that B2' was winning; the other 9 emptied tasks
  were 0/5 under B2' as well, so emptying them cost nothing. The library has
  no document-reading entry — a coverage gap, not a flaw in the retrieval
  idea.
- Net vs B2': −0.6 mean (within noise). Verdict: entry-level retrieval is
  a near-free 5× prompt compression; with one added document-reading entry
  (or a no-empty fallback to the full block) it would plausibly match or
  exceed B2'.

---

## 3. Cross-run summary

| Run | Injection | Overall /110 | Tier-B /58 | Tier-C /25 | Tier-A /27 | app-mode /45 |
|---|---|---|---|---|---|---|
| B1 | none | 9 | 5 | 3 | 1 | 5 |
| B2 | full app FSM + category L_C | 7 | 3 | 3 | 1 | 3 |
| **B2'** | **app Layer-2 + category L_C** | **10** | **7** | 3 | 0 | **6** |
| B2'' | category L_C only | 8 | 6 | 2 | 0 | 4 |

The last column is the clean treatment comparison: those 45 tasks are the only
ones whose prompts differ across the B2 family (category-only and empty
prompts are shared, so their cross-run flips — consistently ±1–2 — calibrate
sampling churn).

Reading:

1. **Cross-benchmark Layer-1 injection is strictly harmful.** On treated
   tasks: full FSM 3 vs app Layer-2 6. Mechanism verified at trajectory
   level (FlightMode pair): source-environment state descriptions displace
   grounded behavior and induce `terminate(success)` without verification.
   Both ablations that drop Layer-1 recover those tasks.
2. **App-specific Layer-2 is the best static treatment** (B2' — the only run
   above baseline: +1 overall, +2 Tier-B, +1 treated), but the margin is
   within sampling churn; only the paired task flips are individually
   attributable.
3. **The original EvoFSM-RL recipe (category L_C) does not transfer its
   AndroidWorld gain** (+9.3pp there → ≈0 here). Knowledge diluted across a
   category survives the benchmark gap worse than the same knowledge at app
   granularity.
4. **Static injection has a low ceiling on MobileWorld** at this model scale
   (8/110–10/110 around a 9/110 baseline). This is the gap the TTA loop (B4)
   targets: adapt the symbolic layer on the target benchmark instead of
   shipping source-environment knowledge verbatim — and per (1)–(2), its
   static prior should be the B2' configuration (Layer-2 / L_C only, never
   Layer-1).
5. Audit lead: `CheckEventTimeTask` fails under all three treatments and
   `ChromeSearchBeijingWeatherTask` fails exactly when Communication L_C
   replaces Chrome-specific text — the Communication L_C file (20.7k chars,
   largest) merits a content audit. → Done, see §4.

---

## 4. Communication L_C audit (2026-06-10)

Trigger: the two losses flagged in §3, reading point 5. Both tasks' guidance
contains the Communication L_C block.

### 4.1 Content

- **Provenance** — aggregated from 3 source apps: `chrome`, `contacts`,
  `simple_sms_messenger` (`metadata.apps`). 20 abstract-action entries,
  20.7k chars rendered.
- **Coverage gap: no email app in the source pool.** MobileWorld's
  Communication apps are Mail / Messages / Contacts / Chrome; Mail tasks
  therefore receive SMS/browser/contacts-shaped priors
  (`COMPOSE_AND_SEND_MESSAGE` is SMS-shaped; no entry covers reading a mail
  thread).
- **Leakage: clean.** No source-app names in the rendered text (names appear
  only in `metadata`); browser vocabulary in 3 lines
  (`FIND_TEXT_IN_RENDERED_PAGE` is explicitly browser-shaped).
- **Relevance dilution.** For any single task ~2–3 of the 20 entries apply;
  the rest is ballast (`DELETE_HISTORY_ENTRY`,
  `FIND_AND_REPLACE_IN_DOCUMENT`, `IMPORT_OR_EXPORT_DATA`, …).
- **Entry quality is reasonable** — `QUERY_INFO` even prescribes the read
  step ("4. Read the requested attribute/value …"). But its `failure_modes`
  do not include the failure actually observed below (leaving the screen
  without reading the value).

### 4.2 Behavioral evidence

- **`CheckEventTimeTask`** (B1 ✓ / B2 ✗ / B2' ✗ / B2'' ✗). B1: scroll inbox →
  open the mail → read "7:00 PM" → set the 6:00 PM alarm. b2p and b2pp
  (near-identical trajectories): search "Christmas party" → tap the result →
  leave Mail **without reading the time** → later anchor the alarm to
  current time + 1h (1:00 PM; oracle: "No alarm found at 6:00 PM"). The acted
  flow matches `SEARCH_ENTRY` + `QUERY_INFO` steps 1–3; step 4 (read the
  value) is dropped. **Selective absorption: the model takes the flow
  skeleton, not the verification discipline.**
- **`ChromeSearchBeijingWeatherTask`** (B1/B2/B2' ✓, B2'' ✗). The emulator is
  offline; B1 types the query, hits the no-internet page, and answers
  directly (oracle accepts 30). B2'' instead follows `SEARCH_ENTRY`
  discipline — re-types because "search not executed", then scrolls 9×
  "inspecting results" — and exhausts the round budget without answering
  ("Invalid answer: ''"). Guidance-induced procedural persistence removes the
  early-answer exit on a degenerate task. Chrome-specific text (B2/B2') does
  not induce this.

### 4.3 Verdict and fixes (ranked)

The library passes the name-leakage bar; the damage comes from the email
coverage gap plus whole-library injection.

1. **Entry-level retrieval at inject time** — inject only the 2–3 entries
   matching the task intent instead of the whole library. Natural extension
   of `lc_injection_multiapp.md`; applies equally to the B4 static prior.
2. Add the observed failure mode to `QUERY_INFO`: "navigating away before
   reading/recording the target value".
3. Add one discipline line to the injection INTRO: extract task-relevant
   facts from the current screen before navigating away.
4. Known gap (footnote, not fixable from our side): the AW+ source pool has
   no email app, so Communication→Mail is far-transfer in app-function terms
   even though the Play category matches.

---

## 5. Variance study — 5 runs per config

### 5.1 Protocol

Every config ran 5×110 tasks; run 1 = the §1–§3 runs (3 containers,
concurrency 3), runs 2–5 fresh (5 containers, concurrency 5; scoring is a
functional oracle, so concurrency affects wall-clock only). Every run got
freshly created containers (no app-state reset exists in MobileWorld
containers). B2''' ran r1–r5 under the 5-container protocol. Unattended
chains: tmux `brepeats` (16 runs, rep-major rounds so each config samples the
same time window) → `b2pppchain` (5 runs). The two chronic harness failures
(`CheckMeetingEventAskUserTask`, `ThanksgivingPrepTask`) wrote no result in
any of the 25 runs (counted as failures throughout).

### 5.2 Main table

| Config | Injection | 5 runs | mean ± std | Δ vs B1 | Tier-B /58 | Tier-C /25 | Tier-A /27 |
|---|---|---|---|---|---|---|---|
| B1 | none | 9, 8, 8, 7, 9 | 8.2 ± 0.8 | — | 5.2 | 2.0 | 1.0 |
| B2 | full app FSM + category L_C | 7, 8, 6, 7, 7 | **7.0 ± 0.7** | **−1.2** | 4.0 | 2.6 | 0.4 |
| **B2'** | app Layer-2 + category L_C | 10, 8, 10, 11, 7 | **9.2 ± 1.6** | **+1.0** | 6.4 | 2.8 | 0.0 |
| B2'' | category L_C only | 8, 8, 10, 9, 7 | 8.4 ± 1.1 | +0.2 | 6.2 | 2.0 | 0.2 |
| B2''' | B2' + entry retrieval | 7, 7, 9, 9, 11 | 8.6 ± 1.7 | +0.4 | 5.6 | 2.6 | 0.4 |

Noise calibration: Tier-C prompts are byte-identical across all five configs,
yet per-config Tier-C means span 2.0–2.8 — sampling alone moves a 25-task
slice by ±0.4. Overall-mean gaps of ≤1 are therefore not individually
interpretable; the per-task table below is.

### 5.3 Deterministic task flips (success count over 5 runs)

| Task | B1 | B2 | B2' | B2'' | B2''' | Reading |
|---|---|---|---|---|---|---|
| `OpenFlightModeTask` | 5 | **0** | 5 | 5 | 5 | Layer-1 harm: all-or-nothing, 25/25 runs consistent |
| `CloseFlightModeTask` | 5 | **0** | 5 | 5 | 5 | same |
| `CheckEventTimeTask` | 5 | **0** | **0** | **0** | **0** | lost under **every** injection form (0/20) — Communication L_C block implicated (§4) |
| `TakeSelfieTask` | **0** | 5 | 5 | 5 | 5 | injection gain: all-or-nothing, any form suffices |
| `ChromeSearchBeijingWeatherTask` | 5 | 3 | 3 | 4 | 4 | graded interference (offline-task early-answer exit, §4.2) |
| `ReadQwen3PaperTask4` | 1 | 4 | **5** | 2 | 2 | Files **Layer-2** specifically wins it; retrieval-emptied in B2''' → drops |

The four all-or-nothing tasks are deterministic under sampling (0 or 5 in
every config) — the injection effects are mechanistic, not statistical. They
also account for nearly the entire mean spread: B2−B1 = −2 deterministic
(2 FlightMode, CheckEventTime) + 1 (TakeSelfie) = −1 vs observed −1.2;
B2'−B1 = −1 + 1 + ~0.8 (ReadQwen3Paper4) ≈ +0.8 vs observed +1.0.

### 5.4 Conclusions (single-run §3 readings re-tested)

1. **Layer-1 injection is strictly harmful — upgraded to deterministic.**
   B2 is the only config below B1 (7.0 ± 0.7 vs 8.2 ± 0.8, the largest and
   most stable gap), and its FlightMode losses reproduce 0/5 vs 5/5
   everywhere else.
2. **B2' (app Layer-2 + category L_C) is the strongest static config**
   (9.2 ± 1.6) and is the MAI-UI injection setting
   (`strongest_variant.txt = b2p`). Its +1.0 overall is ~1σ, but it is
   composed of deterministic flips, not churn.
3. **The original category-only recipe stays ≈ neutral** (+0.2) —
   cross-benchmark, the AW +9.3pp does not transfer (confirmed with error
   bars).
4. **B2''' (entry retrieval) ≈ B2' at 1/5 the injected tokens**; its only
   systematic regression is the documented retrieval miss
   (`ReadQwen3PaperTask4`). Retrieval is the right default for the B4 prior;
   add a document-reading entry or a no-empty fallback before adopting.
5. **Static ceiling confirmed**: all five configs sit within 7.0–9.2 around
   a B1 of 8.2 — the TTA (B4) headroom argument stands.

---

## 6. π^pre (full-FT step 350) under the B1 setting

### 6.1 Setup

The Phase-1 full-FT GRPO checkpoint `global_step_350` (training diagnosed
flat: paired same-template gain −0.02, 60–75% zero-gradient steps), FSDP →
HF-merged at `/shared/linqiang/models/pi_pre_350_hf` (bf16; the merger
writes a stale `float32` in config.json — serve with `--dtype bfloat16`).
Served on GPU0 :8001 in place of the base vLLM (same flags). Eval = the B1
setting exactly: stock `qwen3vl` agent, no injection, same 110-task list,
5 runs, fresh containers each.

### 6.2 Results

5 runs: 8, 4, 6, 9, 7 → **6.8 ± 1.9** (base B1: 8.2 ± 0.8). Per-tier means
B1 → π^pre: Tier-B 5.2 → 4.6, Tier-C 2.0 → 1.2, Tier-A 1.0 → 1.0.

Task-level (5-run success counts, |Δ| ≥ 3): `CheckEventTimeTask` 5 → 1,
`ChromeSearchBeijingWeatherTask` 5 → 1. **No task improved by ≥3.**

### 6.3 Analysis

- 343 steps of full-FT GRPO produced **no transfer gain anywhere** and a
  mean −1.4 with doubled run-to-run variance (1.9 vs 0.8) — consistent with
  the training-side diagnosis (zero-advantage-dominated updates = noise
  injected into the policy, not learning).
- The two systematically degraded tasks are the same fragile
  read-and-act pair that prompt injection also damaged (§4.2/§5.3):
  fine-tuning noise and prompt interference erode the same thin
  margin — screen-reading discipline before acting.
- Verdict: this checkpoint is a slightly damaged base model. **Do not use
  it as the B4 prior**; π^pre should come from the LoRA restart (or default
  to the base model).

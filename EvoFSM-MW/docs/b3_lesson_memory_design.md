# B3 Lesson-Memory TTA — Design (α: TrueSkill retained)

Status: design, 2026-06-15. Supersedes nothing — adds a new B3 evolution operator
alongside the existing FSM-diff path (config switch, old path untouched).

## 0. TL;DR

B3 (test-time FSM evolution) currently **ties B2'** (static prior) on both bases:
qwen B3 10 ≈ B2' 10; MAI B3 25 ≈ B2' 26.4. Root cause: the evolution produces
nothing the static prior lacks, and its TrueSkill selection is signal-starved
(`mixed_groups=0`).

This design changes **what** evolves and **what the A/B game compares**, keeping
TrueSkill:

- Evolve a compact per-app **lesson table** (one lesson per task-type), each lesson
  = **recipe** (working path, from successes) + **pitfall** (mistake to avoid, from
  failures). Both mined contrastively from the same rollouts.
- TrueSkill is **retained** but plays a **decidable** game: slot A = prior, slot B =
  prior **+ a candidate lesson**, on boundary tasks. A≠B → ratings update → it ranks
  which lessons earn their place. (The old game compared two near-identical FSM
  variants → ties → no signal.)
- Inject **B2' prior + ≤2 retrieved lessons** matched to the task. B3's gain over B2'
  is exactly the value of the lessons — clean isolation.

Why this can beat B2' where FSM-diff couldn't: a lesson carries **two things the
generic prior structurally cannot have** — a concrete app-specific solution skeleton,
and the specific failure *this base* makes *on this app*.

## 1. Why B3 = B2' today (diagnosis, established empirically)

- **Evolution overlaps the prior.** `_reflect` asks "what workflow would fix the
  failure" → abstract category steps that duplicate the B2' L_C. Cleaning them up
  (bloat fix) yields ≈ B2'.
- **No selection signal.** TrueSkill needs decisive A≠B. We fed it two similar FSM
  variants on mostly floor/ceiling tasks (MAI: 39/51 iters V=0/0) → ties →
  `mixed_groups=0` → champions ≈ seed.
- **Thin per-app.** 51 tasks / ~15 apps ⇒ 2-3 mutations/app (Messages 2). Per-app
  evolution barely moves.
- **Regime is not the lever.** qwen has injection headroom (B2' 9.2 > B1 8.2) yet
  B3 still = B2' there. So "evolution > prior" is the binding gap, not base strength.

## 2. Mechanism

### 2.1 Lesson (the unit)
Per-app table; one lesson per task-type, keyed by `applies_when`:
```json
{ "applies_when": "deleting a specific item from a list",
  "recipe":  "open it on the list → long-press → trash → confirm",      // success side
  "pitfall": "don't scroll-search if visible; stop if screen unchanged", // failure side
  "evidence": ["iter12","iter38"], "hits": 2, "rating": {"mu":..,"sigma":..} }
```
Injected form (≤3 lines): `When <applies_when>: do <recipe>; avoid <pitfall>`. Either
side may be empty. **app-specific is allowed** — adapt and eval are the same app and
benchmark, so the "L1 is non-transferable across benchmarks" rule does not apply here.

### 2.2 Extraction (contrastive; replaces `_reflect` output)
A task's n rollouts usually split into successes and failures. Feed both:
```
Task: {task}  App: {app}
SUCCESS attempts (action skeletons): ...
FAILED  attempts (action skeletons): ...
Compare them. Output ONE lesson:
- recipe: the working path the successful runs took (omit if no success).
- pitfall: the single recurring mistake the failed runs made (omit if no failure).
- applies_when: the task-type/scenario this lesson is scoped to.
Rules: app-level behaviours OK ("conversation list", "overflow menu"); NO pixel
coords / resource IDs. No speculation beyond what the attempts show.
```
**Richest where there is contrast** (some success AND some failure) — i.e. boundary
tasks. Floor tasks → pitfall only, speculative (low trust). Ceiling → recipe only,
low value. This is exactly why boundary tasks matter (§5).

### 2.3 Consolidation (the 3 editing constraints, on the small table)
Before a new lesson enters, Opus reviews `new + current table`:
1. **modify/merge first** — if an existing lesson covers the same task-type, update
   it (refine recipe/pitfall, `hits++`), don't add.
2. **tag conflicts by scenario** — same `applies_when`, contradictory advice → split
   both by a finer `applies_when`.
3. **add last** — only if no existing lesson covers the intent.
Bounded by ~3-5 task-types/app, so the table stays small and this call is cheap.

### 2.4 Selection — TrueSkill retained, fed a decidable game
Population per app holds **lesson tables** (variants), not FSM variants.
- slot A = prior + table; slot B = prior + table **+ candidate lesson**.
- Run on a task whose `applies_when` matches the candidate. B>A ⇒ the lesson helps ⇒
  TrueSkill rating up, keep; B≤A ⇒ rating down, drop.
- On boundary tasks this is A≠B ⇒ `mixed_groups>0` ⇒ signal returns. The champion =
  the lesson table with the best-rated lessons.
- TrueSkill machinery unchanged; only the *contestants* (lesson tables) and the
  *game* (with/without a lesson) change.

### 2.5 Injection (retrieval, eval time)
For a task on app X: load X's champion table → match `applies_when` to the task goal
by **keyword/embedding** (not an LLM picker — avoids the B2''' retrieval cost) → take
top 1-2 → render ≤3 lines each → append **after the (tightened) B2' prior**. No match
⇒ inject nothing. So B3 injection ≈ B2' size + a few lines.

## 3. Relationship to the existing algorithm and B4

- **TrueSkill is kept** — we fix its input, not replace it. The "evolutionary
  selection" narrative stands; the contestants are lesson tables.
- **What departs:** the mutation operator (FSM-diff → lesson extraction) and the
  injected representation (L2 categories → prior + lessons). Framed honestly, B3
  becomes *evolutionary failure/success-grounded memory* rather than *evolutionary
  abstract-FSM*. This needs PI sign-off before it is the headline method.
- **B4:** shares the controller. If α stands, B4 = lesson-memory evolution + weight
  evolution. The weight half is unaffected (its signal is GRPO advantage). The old
  FSM-diff path remains available behind the config switch for any run that wants it.

## 4. Implementation (isolated; old path untouched)

- New module `skyrl_agent/evofsm_tta/lessons.py`: `extract_lesson`, `consolidate`,
  `retrieve`, lesson schema + table IO.
- `controller.py`: a config flag `evolution_mode = {fsm_diff | lesson}`. In `lesson`
  mode, the per-iteration hook calls extract→consolidate instead of `_mutate_from_texts`,
  and the A/B slots render prior±candidate-lesson. TrueSkill update unchanged.
- Eval injection: new renderer in `gen_b3_guidance.py` (or a sibling) that emits the
  prior + retrieved lessons. Reuses the L_C_v3 tightened prior.
- Nothing in the existing `fsm_diff` path is edited — pure addition behind the flag.

## 5. Experiment plan

- Base: **qwen3-VL-8B** (more sensitive than MAI; injection has headroom there).
- Run: 51-task adapt (lesson mode) → 110-task eval → number.
- Baselines on the same setup: **B1 (qwen) ~8-9, B2'(qwen) ~9-10**.
- **The number to beat is B2', not B1.**

## 6. Risks / open (must stay visible)

1. **Boundary thinness.** Lessons are only well-grounded on tasks with mixed
   success/failure. The fixed 51 may have few. Mitigation: extraction self-selects to
   contrastive tasks; if too thin after the run, add a B1-difficulty pass to curate a
   boundary-weighted adapt set. Not doing that pass up front — let the first run show
   whether it is needed.
2. **n=1 significance.** qwen numbers are 8-11, run-to-run std ~1.5. A single α run
   landing 1-2 above B2' is within noise. A clean claim needs ≥3 runs (paired B2').
   First run is directional only.
3. **Method identity** (see §3) — α reframes the evolution; align before it is the
   paper's headline.

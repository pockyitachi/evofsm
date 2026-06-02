"""Patch Story 2.2 FSMs — bluecoins / clock / pi_music.

Addresses the top-3 rework items from ``docs/fsm_quality_memo.md``:

  * bluecoins — add ``ADD_EXPENSE_WITH_DATE_AND_LABEL`` strategy to cover
    the 8 Add-* templates that are 0/5 SR with no end-to-end strategy.
  * clock — add ``ENTER_TIMER_DIGITS`` strategy encoding the shift-fill
    keypad semantics that the existing dead-end already diagnoses.
  * pi_music — drop the fully isolated ``folders_tab``; add return edges
    for the dangling ``sort_by_dialog_artists`` / ``track_context_menu``
    / ``playlist_detail_empty`` states; add ``artist_detail_view`` state
    with drill-in + back edges; add ``QUERY_ARTIST_ATTRIBUTE`` strategy
    for the 0/5 artist-lookup query templates.

Grounding: these strategies are *aspirational* — the target templates
have no successful trajectories in ``traces/source_pool_trajectories/``
(the 3 apps are all in the 0/5-SR bucket for their problem templates).
The strategy content is derived from (1) the inverse of the recorded
dead-end notes and (2) the app's UI conventions visible in failure
traces. Real validation comes at TTA time.

Flow: backup → load → mutate → validate (lint_layer2 + structural
checks) → write. No file is written unless the post-mutation FSM
round-trips and lints cleanly.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from evofsm_rl.fsm.schema import FSM, State, Strategy, Transition
from evofsm_rl.fsm.linter import lint_layer2

FSM_DIR = Path("EvoFSM-RL/artifacts/static_fsms")
BACKUP_DIR = FSM_DIR / "backup_pre_patch"


# ─────────────────────────────────────────────────────────────────────────
# Per-app patches
# ─────────────────────────────────────────────────────────────────────────


def patch_bluecoins(fsm: FSM) -> FSM:
    """Append ADD_EXPENSE_WITH_DATE_AND_LABEL end-to-end strategy.

    No new states/edges: transaction_editor → date_picker_dialog /
    label_select_dialog → transaction_editor → category_detail_view
    edges already exist; amount entry happens inline in the editor so
    no separate keypad state is needed.
    """
    fsm.layer1.strategies.append(Strategy(
        name="ADD_EXPENSE_WITH_DATE_AND_LABEL",
        preconditions=(
            "transactions_list reachable from main_dashboard; target amount, "
            "date, and label all known up front"
        ),
        steps=[
            "click the FAB (+) on transactions_list to open transaction_editor",
            "ensure the amount field is focused; if pre-filled, clear before typing (append-without-clear is the dominant fail mode)",
            "type the numeric amount via the inline keypad inside transaction_editor",
            "click the date row to open date_picker_dialog, pick the target day, click OK",
            "verify the date row in the editor now shows the selected date",
            "click the Labels row to open label_select_dialog",
            "if the target label is absent, use the 'Add Label...' input and click OK to create it; otherwise tap the existing entry",
            "click the green checkmark in the transaction_editor header to save",
            "after save, navigate up to transactions_list and confirm the new row is visible with correct amount, date, and label",
        ],
        success_signal=(
            "transactions_list shows a new transaction row whose amount, date, "
            "and label all match the inputs; the running total reflects the new entry"
        ),
        fallback=(
            "If the editor closes but transactions_list does not show the new row, "
            "do NOT emit status:complete — re-open via FAB, re-verify amount/date/label, "
            "then retry Save. If the row icon area was tapped by mistake opening "
            "icon_selector_dialog, dismiss via NONE/back and retry from the FAB."
        ),
    ))
    return fsm


def patch_clock(fsm: FSM) -> FSM:
    """Append ENTER_TIMER_DIGITS strategy grounded in the existing dead-end."""
    fsm.layer1.strategies.append(Strategy(
        name="ENTER_TIMER_DIGITS",
        preconditions="clock_main visible; target duration known as H:MM:SS",
        steps=[
            "from clock_main, click the Timer tab to enter timer_entry",
            "expand the target duration into a left-padded 6-digit string HHMMSS (e.g. 1h 23m 40s -> '012340')",
            "tap each digit in order left-to-right; the keypad uses odometer shift-fill semantics — the first tap lands in the least-significant position and shifts left as more digits arrive",
            "after all 6 digits, verify the display reads the target HH:MM:SS exactly (not a shifted variant like 00:12:34 when the goal was 01:23:40)",
            "if the display does not match, tap ⌫ to remove the trailing digit and re-align, or clear fully and re-enter from scratch",
            "click Start to begin the countdown",
        ],
        success_signal=(
            "the timer begins counting down from a value within 1 second of the target; "
            "the Start button is replaced by Pause/Cancel affordances"
        ),
        fallback=(
            "If the digit display never aligns to the target after two full re-entries, "
            "emit status:infeasible — do NOT invent indices or try to type directly into "
            "HH/MM/SS fields (they are not editable text, only via the shift-fill keypad)."
        ),
    ))
    return fsm


def patch_pi_music(fsm: FSM) -> FSM:
    """Drop folders_tab; add return edges; add artist_detail_view + ARTIST_LOOKUP."""
    # 1. Drop fully isolated folders_tab (no trajectory ever visited it;
    #    honest removal is preferable to fabricating edges).
    fsm.layer1.states = [s for s in fsm.layer1.states if s.id != "folders_tab"]
    fsm.layer1.transitions = [
        t for t in fsm.layer1.transitions
        if t.from_state != "folders_tab" and t.to_state != "folders_tab"
    ]

    # 2. New state for artist lookup target page
    fsm.layer1.states.append(State(
        id="artist_detail_view",
        description=(
            "Single-artist detail page showing the artist's tracks and, near "
            "the top, a summary of song/album counts and total duration"
        ),
        visual_cues=[
            "artist name rendered as the header/toolbar title",
            "list of track rows beneath the header, each with title and duration",
            "summary label near the top (e.g. 'N songs', 'M albums')",
        ],
        resource_hints=[],  # no stable rid captured from the failure traces
    ))

    # 3. Missing edges
    new_transitions = [
        # drill-in + back for the new state
        Transition(
            from_state="artists_tab",
            to_state="artist_detail_view",
            action="click(artist row)",
        ),
        Transition(
            from_state="artist_detail_view",
            to_state="artists_tab",
            action="click(Navigate up/Back)",
        ),
        # return edges for dangling dialogs/menus
        Transition(
            from_state="sort_by_dialog_artists",
            to_state="artists_tab",
            action="click(option) or click(CANCEL)",
        ),
        Transition(
            from_state="track_context_menu",
            to_state="tracks_tab",
            action="click(outside) or click(Navigate up)",
        ),
        Transition(
            from_state="playlist_detail_empty",
            to_state="my_music_playlists",
            action="click(Navigate up/Back)",
        ),
    ]
    fsm.layer1.transitions.extend(new_transitions)

    # 4. New strategy for artist-attribute queries (song/album count, duration)
    fsm.layer1.strategies.append(Strategy(
        name="QUERY_ARTIST_ATTRIBUTE",
        preconditions=(
            "my_music_playlists visible; target artist name known; "
            "attribute is one of: song count, album count, total duration"
        ),
        steps=[
            "click ARTISTS tab from my_music_playlists to reach artists_tab",
            "scroll the artist list, or use the Search icon, to locate the target artist by name",
            "click the artist row to drill into artist_detail_view",
            "read the queried attribute from the header/summary label at the top (e.g. 'N songs')",
            "if the attribute is visible, emit status:complete with the exact value as the answer (exactly once)",
            "if the library does not contain the target artist (the test library is ~15 tracks), emit status:infeasible instead of guessing",
        ],
        success_signal=(
            "artist_detail_view shows the target artist's name in the header and a "
            "numeric summary matching the queried attribute; status:complete is emitted "
            "exactly once with that number as the answer"
        ),
        fallback=(
            "If the detail view does not surface the attribute directly, fall back to "
            "counting the listed track rows manually. If the artist is absent from "
            "artists_tab, emit status:infeasible — Albums and Genres tabs do not group "
            "by artist so searching them will not recover the answer."
        ),
    ))

    return fsm


PATCHES = {
    "bluecoins": patch_bluecoins,
    "clock": patch_clock,
    "pi_music": patch_pi_music,
}


# ─────────────────────────────────────────────────────────────────────────
# Validation (post-mutation, before write)
# ─────────────────────────────────────────────────────────────────────────


def validate(fsm: FSM) -> list[str]:
    """Return a list of problems. Empty = OK to write."""
    problems: list[str] = []

    passed, errors = lint_layer2(fsm)
    if not passed:
        problems.extend(f"lint_layer2: {e}" for e in errors)

    state_ids = {s.id for s in fsm.layer1.states}
    for t in fsm.layer1.transitions:
        if t.from_state not in state_ids:
            problems.append(f"transition from_state not in states: {t.from_state}")
        if t.to_state not in state_ids:
            problems.append(f"transition to_state not in states: {t.to_state}")

    ids = [s.id for s in fsm.layer1.states]
    if len(ids) != len(set(ids)):
        problems.append(f"duplicate state ids: {ids}")

    names = [s.name for s in fsm.layer1.strategies]
    if len(names) != len(set(names)):
        problems.append(f"duplicate strategy names: {names}")

    # Round-trip: to_json → from_json must reproduce the same structure
    round_tripped = FSM.from_json(fsm.to_json())
    if round_tripped.to_json() != fsm.to_json():
        problems.append("FSM does not round-trip through to_json/from_json")

    return problems


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    had_failure = False

    for app, patch_fn in PATCHES.items():
        src = FSM_DIR / f"{app}.json"
        backup = BACKUP_DIR / f"{app}.json"

        if not backup.exists():
            shutil.copy2(src, backup)
            print(f"[{app}] backup -> {backup}")
        else:
            print(f"[{app}] backup exists, leaving untouched: {backup}")

        fsm = FSM.from_json(json.loads(src.read_text()))
        before = (
            len(fsm.layer1.states),
            len(fsm.layer1.transitions),
            len(fsm.layer1.strategies),
        )
        fsm = patch_fn(fsm)
        after = (
            len(fsm.layer1.states),
            len(fsm.layer1.transitions),
            len(fsm.layer1.strategies),
        )

        problems = validate(fsm)
        if problems:
            print(f"[{app}] VALIDATION FAILED — not writing:")
            for p in problems:
                print(f"   {p}")
            had_failure = True
            continue

        src.write_text(
            json.dumps(fsm.to_json(), indent=2, ensure_ascii=False) + "\n"
        )
        dS, dT, dSt = (after[0] - before[0], after[1] - before[1], after[2] - before[2])
        sign = lambda n: f"+{n}" if n >= 0 else str(n)
        print(
            f"[{app}] OK  states:{before[0]}->{after[0]} ({sign(dS)})  "
            f"transitions:{before[1]}->{after[1]} ({sign(dT)})  "
            f"strategies:{before[2]}->{after[2]} ({sign(dSt)})"
        )

    return 1 if had_failure else 0


if __name__ == "__main__":
    sys.exit(main())

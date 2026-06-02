"""App and task taxonomy for EvoFSM-RL.

Two layers exposed:
- App  → Play Store category   (12 categories used across our 25 active apps).
- Task → single-label task_type (collapsed from AndroidWorld's multi-label `tags`
  via a fixed priority rule; Plus-repo tasks not in `task_metadata.json` fall
  back to "generic" pending annotation).

Sources of truth:
- `EvoFSM-RL/configs/splits.yaml` — per-app category assignment.
- `android_world_plus/android_world/task_metadata.json` — per-task tags.

Citations: Play Store category scheme follows Android Control (Li et al. 2024).
See `EvoFSM-RL/docs/taxonomy.md`.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import yaml

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # EvoFSM-RL/
_SPLITS_YAML = _PROJECT_ROOT / "configs" / "splits.yaml"
_TASK_METADATA_JSON = (
    _PROJECT_ROOT.parent  # android_world_plus/
    / "android_world_plus"
    / "android_world"
    / "task_metadata.json"
)

# ──────────────────────────────────────────────────────────────────────
# Task-type priority rule
#
# AndroidWorld's `tags` field is multi-label across 15 tags + `requires_setup`.
# We collapse to one label per task by taking the highest-priority tag present.
# Priority order (first match wins):
# ──────────────────────────────────────────────────────────────────────
_TASK_TYPE_PRIORITY = (
    "information_retrieval",
    "search",
    "data_entry",
    "data_edit",
    "transcription",
    "screen_reading",
    "math_counting",
    "verification",
    "memorization",
    "complex_ui_understanding",
    "multi_app",
    "game_playing",
    "repetition",
    "parameterized",
    "requires_setup",
)
_GENERIC = "generic"


# ──────────────────────────────────────────────────────────────────────
# Plus-repo task annotations
#
# The 78 tasks added by android_world_plus (BMOCA + AndroidLab) are not in
# `task_metadata.json`. We hand-annotate them here following the same
# task_type vocabulary, derived from each task's goal_template (see
# `task_evals/single/{bluecoins,calculator,maps_me,pimusic,snapseed,wikipedia}.py`).
#
# Heuristics applied:
#   - Aggregate / cross-row queries  → information_retrieval
#   - Single-item lookup or "find X" → search
#   - Add new entity                 → data_entry
#   - Modify existing entity / setting / sort / playback control → data_edit
#   - Calculator math operations     → math_counting
#   - Pure app-open / tab-switch     → generic
# ──────────────────────────────────────────────────────────────────────
_PLUS_TASK_TYPES: dict[str, str] = {
    # ── bluecoins (15) ────────────────────────────────────────────
    "BluecoinsQuerySpendingOnDate": "information_retrieval",
    "BluecoinsQuerySpendingCategory": "information_retrieval",
    "BluecoinsQueryTotalSpendingOnDate": "information_retrieval",
    "BluecoinsQueryTransactionCount": "information_retrieval",
    "BluecoinsQueryCategorySpending": "information_retrieval",
    "BluecoinsAddExpense": "data_entry",
    "BluecoinsAddIncomeWithLabel": "data_entry",
    "BluecoinsAddExpenseOnDate": "data_entry",
    "BluecoinsAddIncomeOnDateWithNote": "data_entry",
    "BluecoinsAddExpenseOnDateWithLabel": "data_entry",
    "BluecoinsEditExpenseAmount": "data_edit",
    "BluecoinsEditIncomeDateAndAmount": "data_edit",
    "BluecoinsEditTransactionType": "data_edit",
    "BluecoinsEditTransactionTypeAmountNote": "data_edit",
    "BluecoinsEditExpenseDateAmountNote": "data_edit",
    # ── calculator (19) ────────────────────────────────────────────
    "CalculatorOpen": "generic",
    "CalculatorInput1": "math_counting",
    "CalculatorInput1Plus1": "math_counting",
    "CalculatorInput3Times5": "math_counting",
    "CalculatorInput2Plus24Div3": "math_counting",
    "CalculatorInput17Times23": "math_counting",
    "CalculatorInputCos60": "math_counting",
    "CalculatorInputCos180": "math_counting",
    "CalculatorInputFactorial6": "math_counting",
    "CalculatorInputSqrt25": "math_counting",
    "CalculatorInputLn1234": "math_counting",
    "CalculatorInput5Choose2": "math_counting",
    "CalculatorInput10Choose2": "math_counting",
    "CalculatorInputPercent50Of28": "math_counting",
    "CalculatorGeometricMean": "math_counting",
    "CalculatorHarmonicMean": "math_counting",
    "CalculatorConvert45DegreesToRadians": "math_counting",
    "CalculatorSumFirst5Fibonacci": "math_counting",
    "CalculatorSumFirst5Primes": "math_counting",
    # ── maps_me (15) ───────────────────────────────────────────────
    "MapsMeCheckWalkingDistanceTime": "information_retrieval",
    "MapsMeCheckDrivingDistanceTime": "information_retrieval",
    "MapsMeCheckRidingTime": "information_retrieval",
    "MapsMeCheckPublicTransportRoute": "information_retrieval",
    "MapsMeCompareRidingVsPublicTransport": "information_retrieval",
    "MapsMeCheckNearestPlace": "information_retrieval",
    "MapsMeCheckNearestPlaceWalkTime": "information_retrieval",
    "MapsMeCheckNearestHotel": "information_retrieval",
    "MapsMeCheckNearestPlaceDriveTime": "information_retrieval",
    "MapsMeAddWorkPlace": "data_entry",
    "MapsMeNavigateToLocation": "search",
    "MapsMeNavigateToStanford": "search",
    "MapsMeNavigateToUniversitySouth": "search",
    "MapsMeNavigateToOpenAI": "search",
    "MapsMeNavigateToBerkeley": "search",
    # ── pi_music (12) ──────────────────────────────────────────────
    "PiMusicQueryTotalSongs": "information_retrieval",
    "PiMusicQueryArtistSongCount": "information_retrieval",
    "PiMusicQuerySongAlbum": "information_retrieval",
    "PiMusicQueryLongestSongDuration": "information_retrieval",
    "PiMusicQuerySortedSongsByTitle": "information_retrieval",
    "PiMusicQueryArtistTotalDuration": "information_retrieval",
    "PiMusicPlayFromPlaylist": "search",
    "PiMusicPlaySongByTitleArtist": "search",
    "PiMusicCreatePlaylist": "data_entry",
    "PiMusicPauseAndSeek": "data_edit",
    "PiMusicSortByDurationAscending": "data_edit",
    "PiMusicSortByDurationDescending": "data_edit",
    # ── snapseed (11) ──────────────────────────────────────────────
    "SnapseedTask1": "generic",          # Open the Snapseed app
    "SnapseedTask2": "data_entry",       # Open an image
    "SnapseedTask3": "data_edit",        # Apply noir Pop filter
    "SnapseedTask4": "data_edit",        # Apply portrait filter
    "SnapseedTask5": "data_entry",       # Open image + go to tools tab
    "SnapseedTask6": "data_edit",        # Set dark theme
    "SnapseedTask7": "data_edit",        # Set format quality
    "SnapseedTask8": "data_edit",        # Set image sizing
    "SnapseedTask9": "data_edit",        # Apply filter after dark theme
    "SnapseedTask10": "data_edit",       # Apply filter after format
    "SnapseedTask11": "data_edit",       # Apply filter after sizing
    # ── wikipedia (6) ──────────────────────────────────────────────
    "WikipediaOpen": "generic",
    "WikipediaGoToSearchTab": "generic",
    "WikipediaGoToSavedTab": "generic",
    "WikipediaIncreaseTextSize180": "data_edit",
    "WikipediaDecreaseTextSize50": "data_edit",
    "WikipediaDisablePreviewAndFeed": "data_edit",
}


# ──────────────────────────────────────────────────────────────────────
# Lazy-loaded singletons
# ──────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def _load_splits() -> dict:
    with _SPLITS_YAML.open() as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=1)
def _app_to_category() -> dict[str, str]:
    """app_label → Play Store category."""
    splits = _load_splits()
    out: dict[str, str] = {}
    for pool_name in ("source_pool", "tier_B_held_out", "tier_C_held_out"):
        for app, info in splits[pool_name].items():
            out[app] = info["category"]
    return out


@functools.lru_cache(maxsize=1)
def _task_to_tags() -> dict[str, list[str]]:
    """task_name → list of tags from task_metadata.json (vanilla AndroidWorld only)."""
    with _TASK_METADATA_JSON.open() as f:
        meta = json.load(f)
    return {row["task_name"]: row.get("tags", []) for row in meta}


# Public API ──────────────────────────────────────────────────────────
APP_TO_PLAY_CATEGORY = _app_to_category  # call to materialize


def play_category_of(app: str) -> str:
    """Return the Play Store category of an app label.

    Args:
        app: Snake-case app key as used in `splits.yaml`
            (e.g. "simple_calendar_pro", "bluecoins", "system_settings").

    Raises:
        KeyError: if `app` is not in any of source / Tier-B / Tier-C pools.
    """
    mapping = _app_to_category()
    if app not in mapping:
        raise KeyError(
            f"Unknown app {app!r}. Known apps: {sorted(mapping)}"
        )
    return mapping[app]


def task_type_of(task_name: str) -> str:
    """Return a single task_type label for a task template.

    Resolution:
        1. Look up the task in `task_metadata.json` (vanilla AndroidWorld). If
           present, scan its tags in `_TASK_TYPE_PRIORITY` order and return the
           first match.
        2. If absent from metadata, fall back to the hand-annotated
           `_PLUS_TASK_TYPES` table (Plus-repo: bluecoins / calculator /
           maps_me / pi_music / snapseed / wikipedia).
        3. Otherwise return "generic".

    Args:
        task_name: PascalCase template class name (e.g. "MarkorCreateNote").
    """
    tags = _task_to_tags().get(task_name)
    if tags is not None:
        tag_set = set(tags)
        for priority_tag in _TASK_TYPE_PRIORITY:
            if priority_tag in tag_set:
                return priority_tag
        return _GENERIC
    return _PLUS_TASK_TYPES.get(task_name, _GENERIC)


def known_apps() -> list[str]:
    """All 25 active primary-task apps (sorted)."""
    return sorted(_app_to_category())


def known_play_categories() -> list[str]:
    """All Play Store categories used across the 25 active apps (sorted, unique)."""
    return sorted(set(_app_to_category().values()))


__all__ = [
    "play_category_of",
    "task_type_of",
    "known_apps",
    "known_play_categories",
    "APP_TO_PLAY_CATEGORY",
]

"""Accessibility-tree helpers for EvoFSM-RL agent — Story 1.5 (Qwen3-VL-M3A).

Builds two artifacts from an AndroidWorld ``State.ui_elements`` list:

  1. A verbose JSON-per-element text description (matches M3A's
     ``_generate_ui_elements_description_list`` byte-for-byte).
  2. A set-of-marks annotated screenshot — the same image with a green
     bounding box + integer index drawn on every validated element via
     M3A's own ``add_ui_element_mark`` (which honors
     ``physical_frame_boundary`` / ``orientation`` for landscape apps).

**Indexing convention (CRITICAL — matches M3A):**
  The integer index for each UI element is its position in the FULL
  ``state.ui_elements`` list (i.e. ``enumerate(state.ui_elements)``).
  Validation filters only decide whether an element gets a SoM mark
  drawn AND whether it gets a description line — but the index never
  shifts. So index 7 always refers to the 8th raw element, regardless
  of how many earlier elements failed validation. This matters because
  ``env.execute_action`` resolves ``action.index`` against the same
  raw list.

We DO NOT re-implement the drawing or validation logic — both are
imported directly from ``android_world.agents.m3a_utils`` so any
upstream fix lands here automatically.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from android_world.agents import m3a_utils
from android_world.env import representation_utils


# ── Re-exports from m3a_utils ───────────────────────────────────────────
# Importing under our names keeps callers dependency-clean (they import
# from evofsm_rl.agent.a11y, not from the AndroidWorld submodule).

validate_ui_element = m3a_utils.validate_ui_element
add_ui_element_mark = m3a_utils.add_ui_element_mark
add_screenshot_label = m3a_utils.add_screenshot_label


# ── Text rendering — verbose JSON-per-element (matches M3A) ─────────────


def _generate_ui_element_description(
    ui_element: representation_utils.UIElement, index: int
) -> str:
    """Render one element as a JSON-shaped one-liner.

    Verbatim port of m3a._generate_ui_element_description so the prompt
    text matches M3A's downstream eval inputs exactly.
    """
    element_description = f'UI element {index}: {{"index": {index}, '
    if ui_element.text:
        element_description += f'"text": "{ui_element.text}", '
    if ui_element.content_description:
        element_description += (
            f'"content_description": "{ui_element.content_description}", '
        )
    if ui_element.hint_text:
        element_description += f'"hint_text": "{ui_element.hint_text}", '
    if ui_element.tooltip:
        element_description += f'"tooltip": "{ui_element.tooltip}", '
    element_description += (
        f'"is_clickable": {"True" if ui_element.is_clickable else "False"}, '
    )
    element_description += (
        '"is_long_clickable":'
        f' {"True" if ui_element.is_long_clickable else "False"}, '
    )
    element_description += (
        f'"is_editable": {"True" if ui_element.is_editable else "False"}, '
    )
    if ui_element.is_scrollable:
        element_description += '"is_scrollable": True, '
    if ui_element.is_focusable:
        element_description += '"is_focusable": True, '
    element_description += (
        f'"is_selected": {"True" if ui_element.is_selected else "False"}, '
    )
    element_description += (
        f'"is_checked": {"True" if ui_element.is_checked else "False"}, '
    )
    return element_description[:-2] + '}'


def generate_ui_elements_description_list(
    ui_elements: list[representation_utils.UIElement],
    logical_screen_size: tuple[int, int],
) -> str:
    """Return the multi-line UI element list shown to the model.

    Same indexing rule as M3A: ``enumerate(ui_elements)`` decides the
    index, ``validate_ui_element`` decides whether the element gets a
    line at all. So the printed indices may be sparse (e.g. 0, 2, 3, 7…)
    when some intervening elements failed validation, and that sparsity
    is intentional — it lines up with the SoM marks drawn by
    ``draw_set_of_marks``.
    """
    lines = []
    for index, ui_element in enumerate(ui_elements):
        if validate_ui_element(ui_element, logical_screen_size):
            lines.append(_generate_ui_element_description(ui_element, index))
    return '\n'.join(lines)


# ── Set-of-marks drawing ────────────────────────────────────────────────


def draw_set_of_marks(
    screenshot: np.ndarray,
    ui_elements: list[representation_utils.UIElement],
    logical_screen_size: tuple[int, int],
    physical_frame_boundary: tuple[int, int, int, int],
    orientation: int,
) -> np.ndarray:
    """Return a copy of ``screenshot`` with M3A-style boxes/indices drawn.

    Each validated element is annotated by ``m3a_utils.add_ui_element_mark``,
    using its position in the ORIGINAL ``ui_elements`` list as the index.
    Indices line up with both the prompt's UI element text list and with
    ``action.index`` references.

    ``screenshot`` is mutated in-place by ``add_ui_element_mark``, so we
    copy first to keep the raw frame intact.
    """
    annotated = screenshot.copy()
    for index, ui_element in enumerate(ui_elements):
        if validate_ui_element(ui_element, logical_screen_size):
            add_ui_element_mark(
                annotated,
                ui_element,
                index,
                logical_screen_size,
                physical_frame_boundary,
                orientation,
            )
    return annotated


# ── Index → logical pixel center (kept for action.resolve_index_to_xy) ──


def element_center_xy(
    ui_element: representation_utils.UIElement,
    logical_screen_size: tuple[int, int],
) -> tuple[int, int] | None:
    """Return the (x, y) logical-pixel center of ``ui_element``'s bbox.

    Returns None if the element has no bbox. Values are clamped to
    ``logical_screen_size`` so an off-by-a-pixel bbox can't push the
    eventual tap off-screen. Used by ``action.resolve_index_to_xy``
    for tools that need explicit pixel coordinates (kept for
    backward-compat with logging / FSM serialization).
    """
    bbox = ui_element.bbox_pixels
    if bbox is None:
        return None
    cx = int((bbox.x_min + bbox.x_max) / 2)
    cy = int((bbox.y_min + bbox.y_max) / 2)
    lw, lh = logical_screen_size
    if lw:
        cx = max(0, min(cx, lw - 1))
    if lh:
        cy = max(0, min(cy, lh - 1))
    return cx, cy


__all__ = [
    "add_screenshot_label",
    "add_ui_element_mark",
    "draw_set_of_marks",
    "element_center_xy",
    "generate_ui_elements_description_list",
    "validate_ui_element",
]

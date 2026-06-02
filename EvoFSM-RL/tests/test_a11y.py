"""Tests for evofsm_rl.agent.a11y — Story 1.5 (M3A-clone API).

Covers:
    * element filtering (invisible / out-of-screen / zero-area bbox)
    * UI elements text list (M3A verbose JSON-per-element format,
      indices reflect FULL-list positions including skipped invalids)
    * SoM drawing (shape preserved, annotated image differs from source)
    * center-xy helpers + clamping

Run:
    python -m pytest tests/test_a11y.py -v
or  python tests/test_a11y.py   (no pytest required)
"""

from __future__ import annotations

import numpy as np

from android_world.env import representation_utils

from evofsm_rl.agent import a11y


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


def _element(
    *,
    x_min: float = 0.0,
    y_min: float = 0.0,
    x_max: float = 100.0,
    y_max: float = 100.0,
    text: str | None = None,
    content_description: str | None = None,
    class_name: str | None = "android.widget.Button",
    is_visible: bool | None = True,
    is_clickable: bool | None = False,
    is_editable: bool | None = False,
    is_scrollable: bool | None = False,
    is_long_clickable: bool | None = False,
) -> representation_utils.UIElement:
    return representation_utils.UIElement(
        text=text,
        content_description=content_description,
        class_name=class_name,
        is_visible=is_visible,
        is_clickable=is_clickable,
        is_editable=is_editable,
        is_scrollable=is_scrollable,
        is_long_clickable=is_long_clickable,
        bbox_pixels=representation_utils.BoundingBox(
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
        ),
    )


# Pixel 6 portrait: physical and logical frame are identical.
_LOGICAL = (1080, 2400)
_PHYSICAL = (0, 0, 1080, 2400)
_ORIENTATION = 0


# ─────────────────────────────────────────────────────────────────────────
# validate_ui_element  (re-exported from m3a_utils)
# ─────────────────────────────────────────────────────────────────────────


def test_validate_accepts_visible_on_screen_bbox():
    el = _element(x_min=10, y_min=10, x_max=200, y_max=200)
    assert a11y.validate_ui_element(el, _LOGICAL)


def test_validate_rejects_invisible():
    el = _element(is_visible=False)
    assert not a11y.validate_ui_element(el, _LOGICAL)


def test_validate_rejects_zero_area_bbox():
    el = _element(x_min=50, y_min=50, x_max=50, y_max=100)
    assert not a11y.validate_ui_element(el, _LOGICAL)


def test_validate_rejects_off_screen_bbox():
    # x_min >= screen_width
    el = _element(x_min=1200, y_min=0, x_max=1300, y_max=100)
    assert not a11y.validate_ui_element(el, _LOGICAL)
    # x_max <= 0
    el = _element(x_min=-200, y_min=0, x_max=-100, y_max=100)
    assert not a11y.validate_ui_element(el, _LOGICAL)


def test_validate_accepts_missing_bbox():
    """No bbox is not a filter reason on its own — visibility still counts."""
    el = representation_utils.UIElement(is_visible=True)
    assert a11y.validate_ui_element(el, _LOGICAL)


# ─────────────────────────────────────────────────────────────────────────
# generate_ui_elements_description_list — verbose JSON-per-element format
# ─────────────────────────────────────────────────────────────────────────


def test_description_includes_index_text_and_clickable():
    elements = [
        _element(text="Save", is_clickable=True),
    ]
    out = a11y.generate_ui_elements_description_list(elements, _LOGICAL)
    assert 'UI element 0' in out
    assert '"index": 0' in out
    assert '"text": "Save"' in out
    assert '"is_clickable": True' in out


def test_description_skips_invalid_but_keeps_global_indices():
    """The CRITICAL M3A invariant: indices = enumerate(full_list) position.

    Invalid elements get NO line, but valid ones keep their original
    enumerate index — so the printed indices may be sparse (0, 2, 4…).
    """
    elements = [
        _element(text="ok0"),                            # valid → "UI element 0"
        _element(is_visible=False),                      # dropped, idx 1 absent
        _element(text="ok2", x_min=200, x_max=300),      # valid → "UI element 2"
        _element(x_min=-500, x_max=-400),                # dropped, idx 3 absent
        _element(text="ok4", x_min=400, x_max=500),      # valid → "UI element 4"
    ]
    out = a11y.generate_ui_elements_description_list(elements, _LOGICAL)
    assert 'UI element 0' in out
    assert 'UI element 1' not in out
    assert 'UI element 2' in out
    assert 'UI element 3' not in out
    assert 'UI element 4' in out
    lines = [ln for ln in out.split('\n') if ln.strip()]
    assert len(lines) == 3


def test_description_falls_back_to_content_description():
    elements = [_element(text=None, content_description="Back")]
    out = a11y.generate_ui_elements_description_list(elements, _LOGICAL)
    assert '"content_description": "Back"' in out


def test_description_empty_for_no_valid_elements():
    out = a11y.generate_ui_elements_description_list([], _LOGICAL)
    assert out == ''


def test_description_marks_editable_and_scrollable_when_set():
    elements = [_element(text="body", is_editable=True, is_scrollable=True)]
    out = a11y.generate_ui_elements_description_list(elements, _LOGICAL)
    assert '"is_editable": True' in out
    assert '"is_scrollable": True' in out


# ─────────────────────────────────────────────────────────────────────────
# draw_set_of_marks
# ─────────────────────────────────────────────────────────────────────────


def test_draw_preserves_shape_and_is_a_copy():
    screenshot = np.zeros((2400, 1080, 3), dtype=np.uint8)
    elements = [_element(text="ok", x_min=100, y_min=200, x_max=300, y_max=400)]
    annotated = a11y.draw_set_of_marks(
        screenshot, elements, _LOGICAL, _PHYSICAL, _ORIENTATION,
    )
    assert annotated.shape == screenshot.shape
    assert annotated.dtype == screenshot.dtype
    # Green rectangle + label backdrop → at least some non-zero pixels.
    assert (annotated != 0).any()
    # Input must not be mutated.
    assert (screenshot == 0).all()


def test_draw_handles_no_elements():
    screenshot = np.full((120, 100, 3), 42, dtype=np.uint8)
    annotated = a11y.draw_set_of_marks(
        screenshot, [], (100, 120), (0, 0, 100, 120), _ORIENTATION,
    )
    # With no elements, no drawing; should still round-trip intact.
    assert (annotated == screenshot).all()
    assert annotated is not screenshot  # still a copy


def test_draw_skips_invalid_elements():
    """Invalid elements should not produce any pixel changes."""
    screenshot = np.zeros((2400, 1080, 3), dtype=np.uint8)
    elements = [_element(is_visible=False, x_min=10, y_min=10, x_max=200, y_max=200)]
    annotated = a11y.draw_set_of_marks(
        screenshot, elements, _LOGICAL, _PHYSICAL, _ORIENTATION,
    )
    assert (annotated == screenshot).all()


# ─────────────────────────────────────────────────────────────────────────
# element_center_xy
# ─────────────────────────────────────────────────────────────────────────


def test_center_xy_basic():
    el = _element(x_min=100, y_min=200, x_max=300, y_max=400)
    assert a11y.element_center_xy(el, _LOGICAL) == (200, 300)


def test_center_xy_clamps_to_screen():
    el = _element(x_min=1050, y_min=2380, x_max=1200, y_max=2500)
    cx, cy = a11y.element_center_xy(el, _LOGICAL)
    assert 0 <= cx < 1080
    assert 0 <= cy < 2400


def test_center_xy_returns_none_when_no_bbox():
    el = representation_utils.UIElement(is_visible=True)
    assert a11y.element_center_xy(el, _LOGICAL) is None


# ─────────────────────────────────────────────────────────────────────────
# add_screenshot_label  (re-export sanity)
# ─────────────────────────────────────────────────────────────────────────


def test_screenshot_label_writes_text_region():
    screenshot = np.zeros((400, 400, 3), dtype=np.uint8)
    a11y.add_screenshot_label(screenshot, "before")
    # The label is drawn in the bottom-right corner; that region should
    # contain non-zero pixels (white backdrop + black text).
    bottom_right = screenshot[370:, 250:, :]
    assert (bottom_right != 0).any()


# ─────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import traceback

    ns = dict(globals())
    tests = [(name, fn) for name, fn in ns.items()
             if name.startswith("test_") and callable(fn)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  ✔ {name}")
        except Exception:
            failed += 1
            print(f"  ✘ {name}")
            traceback.print_exc()
    print(f"\n{passed}/{passed + failed} tests passed")
    raise SystemExit(0 if failed == 0 else 1)

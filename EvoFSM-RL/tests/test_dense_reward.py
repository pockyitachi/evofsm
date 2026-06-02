"""Unit tests for the opt-in dense-reward (partial credit) path.

Design rule (CLAUDE.md 2026-05-20): ``is_successful`` stays binary
{0.0, 1.0}; ``get_dense_reward`` is parallel and only consumed when the
caller opts in via ``--use-dense-reward``. These tests check the
counting helpers + the harness flag's plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass

from android_world.task_evals.common_validators import sqlite_validators as sv
from android_world.task_evals.task_eval import TaskEval


@dataclass
class _Row:
    name: str
    amount: float
    category: str
    note: str


# ── count_added_reference_rows ───────────────────────────────────────
def test_count_added_one_of_one():
    before = [_Row("a", 1, "food", "x")]
    after = [_Row("a", 1, "food", "x"), _Row("b", 2, "transport", "y")]
    ref = [_Row("b", 2, "transport", "y")]
    assert sv.count_added_reference_rows(
        before, after, ref,
        compare_fields=["name", "amount", "category", "note"],
        free_form_fields=["name", "note"],
    ) == 1


def test_count_added_one_of_two_partial():
    before = [_Row("a", 1, "food", "x")]
    after = [_Row("a", 1, "food", "x"), _Row("b", 2, "transport", "y")]
    ref = [_Row("b", 2, "transport", "y"), _Row("c", 3, "food", "z")]
    assert sv.count_added_reference_rows(
        before, after, ref,
        compare_fields=["name", "amount", "category", "note"],
        free_form_fields=["name", "note"],
    ) == 1


def test_count_added_zero_of_two_none_added():
    before = [_Row("a", 1, "food", "x")]
    after = [_Row("a", 1, "food", "x")]
    ref = [_Row("b", 2, "transport", "y"), _Row("c", 3, "food", "z")]
    assert sv.count_added_reference_rows(
        before, after, ref,
        compare_fields=["name", "amount", "category", "note"],
        free_form_fields=["name", "note"],
    ) == 0


# ── count_deleted_reference_rows ─────────────────────────────────────
def test_count_deleted_full():
    before = [_Row("a", 1, "x", "q"), _Row("b", 2, "y", "q")]
    after = [_Row("a", 1, "x", "q")]
    to_delete = [_Row("b", 2, "y", "q")]
    assert sv.count_deleted_reference_rows(before, after, to_delete) == 1


def test_count_deleted_partial():
    before = [_Row("a", 1, "x", "q"), _Row("b", 2, "y", "q"),
              _Row("c", 3, "z", "q")]
    after = [_Row("a", 1, "x", "q"), _Row("b", 2, "y", "q")]
    to_delete = [_Row("b", 2, "y", "q"), _Row("c", 3, "z", "q")]
    assert sv.count_deleted_reference_rows(before, after, to_delete) == 1


# ── TaskEval base-class fallback ─────────────────────────────────────
def test_base_get_dense_reward_falls_back_to_is_successful():
    """If a task doesn't override get_dense_reward, it returns binary."""

    class _Task(TaskEval):
        complexity = 1
        template = ""
        app_names = ()
        schema = {}

        @classmethod
        def generate_random_params(cls):
            return {}

        def is_successful(self, env):
            return 1.0

    t = _Task({})
    t.initialized = True
    assert t.get_dense_reward(None) == 1.0


def test_base_get_dense_reward_zero():
    class _Task(TaskEval):
        complexity = 1
        template = ""
        app_names = ()
        schema = {}

        @classmethod
        def generate_random_params(cls):
            return {}

        def is_successful(self, env):
            return 0.0

    t = _Task({})
    t.initialized = True
    assert t.get_dense_reward(None) == 0.0

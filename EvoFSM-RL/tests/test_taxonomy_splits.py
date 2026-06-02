"""Sanity tests for taxonomy + splits modules.

Run: python -m pytest tests/   (from EvoFSM-RL/)
Or:  python tests/test_taxonomy_splits.py    (no pytest required)
"""

from __future__ import annotations

from evofsm_rl import splits, taxonomy


def test_total_app_count():
    assert len(taxonomy.known_apps()) == 25, "Expected 25 active primary-task apps"


def test_total_template_count():
    s = splits.split_summary()
    total = (
        s["source_templates"]
        + s["tier_B_T_adapt"]
        + s["tier_B_T_eval"]
        + s["tier_C_T_adapt"]
        + s["tier_C_T_eval"]
    )
    assert total == 192, f"Expected 192 app-attributable templates, got {total}"


def test_pool_partition_disjoint():
    src = set(splits.get_source_pool())
    b = set(splits.get_tier_B_apps())
    c = set(splits.get_tier_C_apps())
    assert src.isdisjoint(b), "source ∩ tier_B should be empty"
    assert src.isdisjoint(c), "source ∩ tier_C should be empty"
    assert b.isdisjoint(c), "tier_B ∩ tier_C should be empty"
    assert src | b | c == set(taxonomy.known_apps())


def test_T_adapt_T_eval_template_disjoint():
    """Within every held-out app, T_adapt and T_eval share no template."""
    for app, info in splits.all_held_out_apps().items():
        a, e = set(info.T_adapt), set(info.T_eval)
        assert a.isdisjoint(e), f"{app}: T_adapt ∩ T_eval = {a & e}"


def test_tier_B_categories_in_source():
    """Tier-B = held-out apps whose category IS in source pool."""
    src_cats = {info.category for info in splits.get_source_pool().values()}
    for app, info in splits.get_tier_B_apps().items():
        assert info.category in src_cats, (
            f"Tier-B app {app!r} has category {info.category!r} not present in source pool"
        )


def test_tier_C_categories_NOT_in_source():
    """Tier-C = held-out apps whose category is NOT in source pool."""
    src_cats = {info.category for info in splits.get_source_pool().values()}
    for app, info in splits.get_tier_C_apps().items():
        assert info.category not in src_cats, (
            f"Tier-C app {app!r} has category {info.category!r} that IS in source pool"
        )


def test_taxonomy_resolves_for_every_app():
    for app in taxonomy.known_apps():
        cat = taxonomy.play_category_of(app)
        assert isinstance(cat, str) and cat


def test_task_type_resolves_for_every_template():
    """Every registered template returns a non-empty task_type string."""
    for info in splits.get_source_pool().values():
        for t in info.templates:
            assert taxonomy.task_type_of(t)
    for info in splits.all_held_out_apps().values():
        for t in (*info.T_adapt, *info.T_eval):
            assert taxonomy.task_type_of(t)


def test_lookup_unknown_app_raises():
    try:
        taxonomy.play_category_of("nonexistent_app")
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError for unknown app")


def test_seed_counts_loaded():
    counts = splits.get_seed_counts()
    assert counts == {"K_source": 5, "K_adapt": 5, "K_eval": 3}


def test_seed_lists_disjoint_adapt_vs_eval():
    """T_adapt seeds and T_eval seeds must not overlap."""
    sl = splits.get_seed_lists()
    assert set(sl["T_adapt"]).isdisjoint(set(sl["T_eval"])), (
        "T_adapt and T_eval seed lists should be disjoint"
    )


def test_loader_is_deterministic():
    """Same call returns identical output across invocations (cache hit)."""
    a = splits.split_summary()
    b = splits.split_summary()
    assert a == b


# ────────────────────────────────────────────
# Standalone runner (no pytest needed)
# ────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import traceback

    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print()
    print(f"{len(tests) - failed} / {len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)

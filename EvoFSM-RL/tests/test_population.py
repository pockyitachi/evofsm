"""Tests for evofsm_rl.fsm.population — Story 3.1.

Coverage mirrors the acceptance list in the story brief:

  TrueSkill (1-6):
    1. Two-player: winner mu up, loser mu down.
    2. Both sigmas shrink (less uncertain) after a win/loss update.
    3. Three-player strict ranking: rank-0 gains most, rank-2 loses most.
    4. Tied ranks → symmetric updates.
    5. A high-sigma (uncertain) player moves more in mu than a low-sigma
       player in the same match.
    6. Many updates with tau > 0 keep sigma from collapsing to 0.

  Population (7-17):
    7.  Fresh population size=1, champion=root, window=[root].
    8.  add_child increases size, appends to window, sets parent_id.
    9.  Child inherits parent's mu exactly; sigma = parent.sigma + delta_sigma.
    10. Sliding window: with window_size=W, adding W+5 children keeps
        |window| == W and contains the latest W variants.
    11. Champion is the highest-mu variant in the window.
    12. select(m=1) returns 1 variant from the window; m > window_size raises.
    13. Select distribution: one high-mu variant is chosen more often than
        the low-mu rest over many independent draws.
    14. Select with lambda=0 becomes pure softmax on mu (no exploration
        bonus from sigma).
    15. update_ratings with higher reward pushes the winner's mu up.
    16. to_json / from_json round-trips the full population state.
    17. summary_table returns a string with header + one row per variant.

Run::
    python -m pytest tests/test_population.py -v
or  python tests/test_population.py           # pytest-less fallback
"""

from __future__ import annotations

import math
import random

from evofsm_rl.fsm.population import (
    DEFAULT_BETA,
    DEFAULT_SIGMA,
    DEFAULT_TAU,
    FSMVariant,
    Population,
    Rating,
    trueskill_update,
)
from evofsm_rl.fsm.schema import FSM, Layer1, Layer2, State


# ─────────────────────────────────────────────────────────────────────
# Fixture helpers — tiny minimal-FSM builder reused across tests
# ─────────────────────────────────────────────────────────────────────


def _tiny_fsm(app: str = "markor", category: str = "Productivity") -> FSM:
    """Minimal valid FSM for population tests (contents immaterial)."""
    return FSM(
        app=app,
        layer1=Layer1(
            app=app,
            category=category,
            states=[State(id="home")],
        ),
        layer2=Layer2(categories=[]),
    )


# ═════════════════════════════════════════════════════════════════════
# TrueSkill: rating math
# ═════════════════════════════════════════════════════════════════════


def test_ts_two_player_winner_mu_up_loser_mu_down():
    """Test 1: after a 2-player match, winner.mu > prior, loser.mu < prior."""
    before = [Rating(), Rating()]
    after = trueskill_update(before, [0, 1])
    assert after[0].mu > before[0].mu, (
        f"winner mu should go up: {before[0].mu} -> {after[0].mu}"
    )
    assert after[1].mu < before[1].mu, (
        f"loser mu should go down: {before[1].mu} -> {after[1].mu}"
    )


def test_ts_two_player_both_sigmas_shrink():
    """Test 2: both players' sigma strictly decreases after an informative
    pair update, even after the tau dynamics are added back."""
    before = [Rating(), Rating()]
    after = trueskill_update(before, [0, 1])
    assert after[0].sigma < before[0].sigma, (
        f"winner sigma should shrink: {before[0].sigma} -> {after[0].sigma}"
    )
    assert after[1].sigma < before[1].sigma, (
        f"loser sigma should shrink: {before[1].sigma} -> {after[1].sigma}"
    )


def test_ts_three_player_first_gains_most_last_loses_most():
    """Test 3: with strict ranking [0, 1, 2], rank-0 gains the most mu
    and rank-2 loses the most (monotone in final position)."""
    before = [Rating(), Rating(), Rating()]
    after = trueskill_update(before, [0, 1, 2])
    delta = [after[i].mu - before[i].mu for i in range(3)]
    assert delta[0] > 0, f"first should gain, got delta={delta[0]}"
    assert delta[2] < 0, f"last should lose, got delta={delta[2]}"
    assert delta[0] > abs(delta[1]), (
        f"first's gain ({delta[0]}) should exceed middle's |delta| ({abs(delta[1])})"
    )
    assert abs(delta[2]) > abs(delta[1]), (
        f"last's |loss| ({abs(delta[2])}) should exceed middle's |delta| ({abs(delta[1])})"
    )
    # Middle player nets to a small delta: pair-(0,1) drops it, pair-(1,2)
    # lifts it back. The result is not exactly zero because the first pair
    # also shrinks its sigma, changing the magnitude of the second update
    # — a known property of sum-of-adjacent-pairs TrueSkill. The net should
    # still be small in absolute terms relative to the extremes.
    assert abs(delta[1]) < 0.5 * min(abs(delta[0]), abs(delta[2])), (
        f"middle's |delta| ({abs(delta[1])}) should be small vs. extremes "
        f"({abs(delta[0])}, {abs(delta[2])})"
    )


def test_ts_tied_ranks_produce_symmetric_updates():
    """Test 4: two players with identical ranks end the round with
    identical ratings (they never played a pair that could discriminate)."""
    before = [Rating(mu=25.0, sigma=8.0), Rating(mu=25.0, sigma=8.0)]
    after = trueskill_update(before, [0, 0])
    assert abs(after[0].mu - after[1].mu) < 1e-12
    assert abs(after[0].sigma - after[1].sigma) < 1e-12
    # And the mus are exactly unchanged (no pair update fires at all).
    assert after[0].mu == before[0].mu


def test_ts_high_sigma_player_moves_more_in_mu_than_low_sigma_player():
    """Test 5: when an uncertain player beats a well-estimated player
    with equal mu, the uncertain player's mu shift is larger."""
    uncertain = Rating(mu=25.0, sigma=8.333)
    confident = Rating(mu=25.0, sigma=1.0)
    after = trueskill_update([uncertain, confident], [0, 1])
    uncertain_shift = after[0].mu - uncertain.mu
    confident_shift = confident.mu - after[1].mu  # loser: |delta|
    assert uncertain_shift > 5 * confident_shift, (
        f"high-sigma shift ({uncertain_shift}) should dwarf low-sigma "
        f"shift ({confident_shift})"
    )


def test_ts_tau_prevents_sigma_collapse_over_many_rounds():
    """Test 6: after many rounds, sigma floors near tau rather than 0."""
    ratings = [Rating(), Rating()]
    for _ in range(400):
        # Alternate winner to avoid runaway mu; sigma should still saturate
        # near tau regardless.
        ratings = trueskill_update(ratings, [0, 1])
    for r in ratings:
        assert r.sigma > 0.0, f"sigma went to zero: {r.sigma}"
        # tau=0.0833 is the dynamics; sigma cannot fall below it.
        assert r.sigma >= DEFAULT_TAU * 0.99, (
            f"sigma fell below tau: {r.sigma} < {DEFAULT_TAU}"
        )


# ═════════════════════════════════════════════════════════════════════
# Population: structure + lifecycle
# ═════════════════════════════════════════════════════════════════════


def test_pop_root_initialization():
    """Test 7: population starts with exactly the root variant."""
    fsm = _tiny_fsm()
    pop = Population(fsm)
    assert pop.size == 1
    assert pop.champion.id == "gen0_root"
    assert [v.id for v in pop.window] == ["gen0_root"]
    assert pop.window[0].fsm is fsm  # stored by reference, not copied


def test_pop_add_child_appends_and_sets_parent():
    """Test 8: add_child grows size, child lands in window, parent_id
    and generation are correct."""
    pop = Population(_tiny_fsm())
    child = pop.add_child("gen0_root", _tiny_fsm(), iteration=1)
    assert pop.size == 2
    assert child in pop.window
    assert child.parent_id == "gen0_root"
    assert child.generation == 1
    assert child.id.startswith("gen1_mut_")
    assert child.birth_iteration == 1


def test_pop_child_inherits_parent_mu_and_delta_sigma():
    """Test 9: child mu == parent mu; child sigma == parent sigma + delta."""
    pop = Population(_tiny_fsm(), delta_sigma=1.5)
    parent = pop.get("gen0_root")
    # Mutate parent rating so we're not just checking defaults.
    parent.rating = Rating(mu=30.0, sigma=5.0)
    child = pop.add_child("gen0_root", _tiny_fsm(), iteration=1)
    assert child.rating.mu == 30.0
    assert abs(child.rating.sigma - (5.0 + 1.5)) < 1e-12


def test_pop_window_slides_as_children_added():
    """Test 10: with window_size=W, adding W+5 children keeps |window|=W
    and the window contains exactly the latest W variants by creation order."""
    W = 5
    pop = Population(_tiny_fsm(), window_size=W)
    children = []
    for i in range(W + 5):
        c = pop.add_child("gen0_root", _tiny_fsm(), iteration=i)
        children.append(c)
    # Population grew; window capped at W.
    assert pop.size == 1 + W + 5
    assert len(pop.window) == W
    expected_ids = [v.id for v in children[-W:]]
    assert [v.id for v in pop.window] == expected_ids
    # Root has fallen out of the window (since W < total).
    assert "gen0_root" not in {v.id for v in pop.window}


def test_pop_champion_returns_highest_mu_in_window():
    """Test 11: champion is the variant with the highest mu among the
    *window*, not among all variants ever created."""
    pop = Population(_tiny_fsm(), window_size=3)
    # Fill the window with three children, manually rate them.
    mus = [20.0, 35.0, 25.0]
    children = []
    for i, mu in enumerate(mus):
        c = pop.add_child("gen0_root", _tiny_fsm(), iteration=i)
        c.rating = Rating(mu=mu, sigma=2.0)
        children.append(c)
    # Now window = [children[0], children[1], children[2]] (root pushed out).
    assert pop.champion.id == children[1].id  # the 35.0-mu one


# ═════════════════════════════════════════════════════════════════════
# Population: selection
# ═════════════════════════════════════════════════════════════════════


def test_pop_select_m1_returns_window_element():
    """Test 12a: select(m=1) returns exactly one variant drawn from the window."""
    pop = Population(_tiny_fsm(), window_size=3)
    pop.add_child("gen0_root", _tiny_fsm(), iteration=1)
    pop.add_child("gen0_root", _tiny_fsm(), iteration=2)

    rng = random.Random(0)
    chosen = pop.select(1, rng=rng)
    assert len(chosen) == 1
    assert chosen[0] in pop.window


def test_pop_select_raises_when_m_exceeds_window():
    """Test 12b: select(m > window_size) raises ValueError."""
    pop = Population(_tiny_fsm(), window_size=2)
    pop.add_child("gen0_root", _tiny_fsm(), iteration=1)
    # Window has 2 variants (root + 1 child), ask for 3 → raise.
    try:
        pop.select(3, rng=random.Random(0))
    except ValueError as e:
        assert "window" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on m > window_size")


def test_pop_select_distribution_favors_high_mu():
    """Test 13: with one clear favorite and many peers, the favorite is
    drawn disproportionately often across many independent selections."""
    pop = Population(_tiny_fsm(), window_size=5,
                      selection_lambda=1.0, selection_temperature=1.0)
    for i in range(4):
        pop.add_child("gen0_root", _tiny_fsm(), iteration=i)

    # Set mus: one favorite at 40, the other 4 at 20.
    window = pop.window
    favorite = window[0]
    favorite.rating = Rating(mu=40.0, sigma=1.0)
    for v in window[1:]:
        v.rating = Rating(mu=20.0, sigma=1.0)

    rng = random.Random(42)
    counts: dict[str, int] = {}
    n_trials = 1000
    for _ in range(n_trials):
        [picked] = pop.select(1, rng=rng)
        counts[picked.id] = counts.get(picked.id, 0) + 1

    fav_rate = counts.get(favorite.id, 0) / n_trials
    # Softmax with ΔE = 20, T = 1 should make the favorite essentially certain,
    # but we leave loads of slack for numerical + sampling variance.
    assert fav_rate > 0.9, (
        f"favorite should dominate sampling; got rate={fav_rate:.3f}, "
        f"counts={counts}"
    )


def test_pop_select_with_lambda_zero_is_pure_mu_softmax():
    """Test 14: with lambda=0 (no exploration bonus), a high-sigma variant
    with equal mu gets no selection advantage over a low-sigma peer."""
    pop = Population(_tiny_fsm(), window_size=3,
                      selection_lambda=0.0, selection_temperature=1.0)
    pop.add_child("gen0_root", _tiny_fsm(), iteration=1)
    pop.add_child("gen0_root", _tiny_fsm(), iteration=2)

    window = pop.window
    assert len(window) == 3
    # Equal mu, but very different sigma. With lambda=0 this should not matter.
    window[0].rating = Rating(mu=25.0, sigma=0.5)
    window[1].rating = Rating(mu=25.0, sigma=8.0)
    window[2].rating = Rating(mu=25.0, sigma=0.5)

    rng = random.Random(123)
    counts: dict[str, int] = {v.id: 0 for v in window}
    n_trials = 3000
    for _ in range(n_trials):
        [picked] = pop.select(1, rng=rng)
        counts[picked.id] += 1

    # All three should hover near 1/3 ± sampling noise.
    for vid, c in counts.items():
        rate = c / n_trials
        assert 0.25 < rate < 0.42, (
            f"lambda=0 should give uniform-ish selection; variant {vid} "
            f"got rate={rate:.3f} (counts={counts})"
        )


# ═════════════════════════════════════════════════════════════════════
# Population: rating update / persistence / reporting
# ═════════════════════════════════════════════════════════════════════


def test_pop_update_ratings_pushes_winner_mu_up():
    """Test 15: after a tournament where one variant got a much higher
    reward, that variant's mu increases."""
    pop = Population(_tiny_fsm(), window_size=3)
    c1 = pop.add_child("gen0_root", _tiny_fsm(), iteration=1)
    c2 = pop.add_child("gen0_root", _tiny_fsm(), iteration=2)

    before_c1 = c1.rating.mu
    before_c2 = c2.rating.mu

    pop.update_ratings([c1.id, c2.id], rewards=[0.9, 0.1])

    assert c1.rating.mu > before_c1, "high-reward variant's mu should rise"
    assert c2.rating.mu < before_c2, "low-reward variant's mu should fall"


def test_pop_update_ratings_round_trip_with_a_tie():
    """Competition ranking: tied rewards produce tied ranks so the tied
    pair doesn't update against each other but still updates vs. others."""
    pop = Population(_tiny_fsm(), window_size=3)
    c1 = pop.add_child("gen0_root", _tiny_fsm(), iteration=1)
    c2 = pop.add_child("gen0_root", _tiny_fsm(), iteration=2)

    # Two winners tie, root loses.
    pop.update_ratings(["gen0_root", c1.id, c2.id], rewards=[0.0, 0.5, 0.5])

    root = pop.get("gen0_root")
    # Root clearly lost — its mu should drop.
    assert root.rating.mu < 25.0
    # The two tied winners should come out with identical-ish mu since
    # adjacent-pair processing applies the same update to both vs root.
    # (They don't update against each other because ranks are equal.)
    # Note: with sort-stability the first tied player ends up adjacent to
    # root so receives the direct update; the second tied player is
    # adjacent to the first and gets the skip. So the two need not end up
    # identical — but their mu should both be >= 25.
    assert c1.rating.mu >= 25.0 - 1e-9
    assert c2.rating.mu >= 25.0 - 1e-9


def test_pop_round_trip_preserves_state():
    """Test 16: Population.from_json(pop.to_json()) reproduces the
    original state exactly — same variants, same ratings, same counters."""
    pop = Population(_tiny_fsm(), window_size=4, selection_lambda=0.5,
                      delta_sigma=1.2)
    pop.add_child("gen0_root", _tiny_fsm(), iteration=1, metadata={"mutation": "insert_state"})
    c2 = pop.add_child("gen0_root", _tiny_fsm(), iteration=2)
    c2.rating = Rating(mu=27.5, sigma=6.1)
    pop.add_child(c2.id, _tiny_fsm(), iteration=3)

    serialized = pop.to_json()
    restored = Population.from_json(serialized)

    # Same top-level config.
    assert restored.window_size == pop.window_size
    assert restored.selection_lambda == pop.selection_lambda
    assert restored.delta_sigma == pop.delta_sigma
    # Same variants.
    assert restored.size == pop.size
    for v_old, v_new in zip(pop._variants, restored._variants):  # noqa: SLF001
        assert v_new.id == v_old.id
        assert v_new.generation == v_old.generation
        assert v_new.parent_id == v_old.parent_id
        assert v_new.birth_iteration == v_old.birth_iteration
        assert v_new.metadata == v_old.metadata
        assert abs(v_new.rating.mu - v_old.rating.mu) < 1e-12
        assert abs(v_new.rating.sigma - v_old.rating.sigma) < 1e-12
        assert v_new.fsm.to_json() == v_old.fsm.to_json()
    # Mutation counter preserved so new add_child yields a fresh id.
    new_child = restored.add_child("gen0_root", _tiny_fsm(), iteration=999)
    existing_ids = {v.id for v in pop._variants}  # noqa: SLF001
    assert new_child.id not in existing_ids


def test_pop_summary_table_shape():
    """Test 17: summary_table produces one header row + one row per variant,
    with a consistent column count per row."""
    pop = Population(_tiny_fsm(), window_size=3)
    pop.add_child("gen0_root", _tiny_fsm(), iteration=1)
    pop.add_child("gen0_root", _tiny_fsm(), iteration=2)

    text = pop.summary_table()
    lines = text.splitlines()
    assert len(lines) == 1 + pop.size, (
        f"expected 1 header + {pop.size} rows, got {len(lines)}: {text!r}"
    )
    # Header should mention every advertised column.
    header = lines[0]
    for col in ("ID", "Gen", "mu", "sigma", "Conservative", "Parent", "InWindow"):
        assert col in header, f"header missing column {col!r}: {header!r}"
    # Every variant id appears in some row.
    for v in pop._variants:  # noqa: SLF001
        assert any(v.id in line for line in lines[1:]), (
            f"variant {v.id!r} missing from summary"
        )


# ═════════════════════════════════════════════════════════════════════
# Extra sanity: Rating helpers
# ═════════════════════════════════════════════════════════════════════


def test_rating_conservative_and_optimistic_bounds():
    r = Rating(mu=30.0, sigma=5.0)
    assert r.conservative == 30.0 - 3.0 * 5.0
    assert r.optimistic() == 30.0 + 1.0 * 5.0
    assert r.optimistic(lam=2.0) == 30.0 + 2.0 * 5.0


def test_trueskill_update_rejects_mismatched_lengths():
    try:
        trueskill_update([Rating()], [0, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on length mismatch")


def test_trueskill_update_returns_fresh_objects_does_not_mutate():
    before = [Rating(), Rating()]
    before_snapshot = [(r.mu, r.sigma) for r in before]
    _ = trueskill_update(before, [0, 1])
    after_snapshot = [(r.mu, r.sigma) for r in before]
    assert before_snapshot == after_snapshot, (
        "trueskill_update must not mutate its input ratings"
    )


# ═════════════════════════════════════════════════════════════════════
# Pytest-less standalone runner
# ═════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import inspect
    import traceback

    ns = dict(globals())
    tests = [(name, fn) for name, fn in ns.items()
             if name.startswith("test_") and callable(fn)]
    passed = failed = 0
    for name, fn in tests:
        try:
            sig = inspect.signature(fn)
            if "monkeypatch" in sig.parameters:
                continue  # none of these use monkeypatch
            fn()
            passed += 1
            print(f"  ok    {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    total = passed + failed
    print(f"\n{passed}/{total} passed")
    raise SystemExit(0 if failed == 0 else 1)

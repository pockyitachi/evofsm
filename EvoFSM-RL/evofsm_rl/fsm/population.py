"""FSM population + TrueSkill rating — Story 3.1 (Epic 3 / B3).

This module contains three things, in ascending order of specificity:

  1. :class:`Rating` — a lightweight TrueSkill-style Bayesian rating
     (mu, sigma) with conservative / optimistic projections.
  2. :func:`trueskill_update` — the sum-of-adjacent-pairs TrueSkill
     update from Herbrich 2007 (Thurstone-Mosteller approximation),
     implemented from first principles in ~80 lines. We do *not* take
     a dependency on the ``trueskill`` PyPI package; the free-for-all
     / tournament variant we need is small enough to own directly.
  3. :class:`Population` — a growing-tree FSM population with
     sliding-window selection, optimistic (UCB-style) sampling for
     exploration, and full-state JSON (de)serialization for
     checkpointing.

Downstream consumers (the evolution loop in Story 3.5, the RL trainer
in later Epic 4 stories) only need to know :class:`Population`'s public
surface. The rating math is exposed for unit testing and for the rare
caller that wants to run ad-hoc tournaments on non-FSM objects.

Algorithmic references
----------------------
- Hyperparameter defaults (mu_0=25, sigma_0=mu/3, beta=mu/6,
  tau=sigma_0/100, lambda=1.0, temperature=1.0, window=15, delta_sigma=1.5)
  come from ``plan/algorithm_design.md`` §1.2 and §9.
- Selection formula (softmax over mu + lambda*sigma) is §3.2.
- Growing-tree / sliding-window rationale is §6.3.

Deviations from the story spec (documented)
-------------------------------------------
- ``Rating.optimistic`` is a regular method, not a ``@property``, since
  Python properties cannot accept arguments. ``Rating.conservative``
  remains a property. Call sites that need the default lambda write
  ``rating.optimistic()``; the :class:`Population` selection path uses
  ``mu + self.selection_lambda * sigma`` directly and never touches the
  method.
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any

from evofsm_rl.fsm.schema import FSM


# ─────────────────────────────────────────────────────────────────────
# Hyperparameter defaults (match plan/algorithm_design.md §9)
# ─────────────────────────────────────────────────────────────────────

DEFAULT_MU = 25.0
DEFAULT_SIGMA = 25.0 / 3.0           # 8.333
DEFAULT_BETA = DEFAULT_MU / 6.0      # 4.167 — performance noise
DEFAULT_TAU = DEFAULT_SIGMA / 100.0  # 0.0833 — dynamics / skill drift

DEFAULT_WINDOW_SIZE = 15
DEFAULT_SELECTION_LAMBDA = 1.0
DEFAULT_SELECTION_TEMPERATURE = 1.0
DEFAULT_DELTA_SIGMA = 1.5


# ─────────────────────────────────────────────────────────────────────
# Part 1: Rating
# ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Rating:
    """TrueSkill-style Bayesian skill rating for one FSM variant."""

    mu: float = DEFAULT_MU
    sigma: float = DEFAULT_SIGMA

    @property
    def conservative(self) -> float:
        """Pessimistic lower bound on skill: ``mu - 3*sigma``.

        Useful for promoting a variant to frozen-baseline status only
        when we're confident it's genuinely good (the lower bound is
        still high). Not used in selection — selection uses the
        optimistic bound instead, to encourage exploration.
        """
        return self.mu - 3.0 * self.sigma

    def optimistic(self, lam: float = DEFAULT_SELECTION_LAMBDA) -> float:
        """UCB-style upper bound: ``mu + lam * sigma``.

        Used in :meth:`Population.select` to blend exploitation
        (prefer high mu) with exploration (prefer high sigma — we
        don't know yet if it's good).
        """
        return self.mu + lam * self.sigma

    def to_json(self) -> dict[str, float]:
        return {"mu": self.mu, "sigma": self.sigma}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Rating":
        return cls(mu=float(data["mu"]), sigma=float(data["sigma"]))


# ─────────────────────────────────────────────────────────────────────
# Part 2: TrueSkill update (pure math)
# ─────────────────────────────────────────────────────────────────────


# math.erf-based Gaussian helpers. Using the standard library here is
# deliberate: scipy is a heavyweight dep and we only need pdf/cdf of the
# standard normal, which is a dozen lines of math.

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)


def _std_pdf(x: float) -> float:
    """PDF of the standard normal at ``x``."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _std_cdf(x: float) -> float:
    """CDF of the standard normal at ``x``, via ``math.erf``."""
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def _v_win(t: float) -> float:
    """TrueSkill 'v' for a win: truncated-normal mean at threshold ``-t``.

    ``v = pdf(t) / cdf(t)`` (letting cdf(t) floor at a tiny positive
    value to avoid division by zero on very lopsided upsets).
    """
    cdf = _std_cdf(t)
    if cdf < 1e-300:
        cdf = 1e-300
    return _std_pdf(t) / cdf


def _w_win(t: float, v: float) -> float:
    """TrueSkill 'w' for a win: ``v * (v + t)``, clamped to ``[0, 1]``.

    The [0, 1] clamp tracks the closed-form guarantee that w is a valid
    variance-reduction fraction; numerical jitter on extreme t values
    can push it slightly out of bounds in practice.
    """
    w = v * (v + t)
    if w < 0.0:
        return 0.0
    if w > 1.0:
        return 1.0
    return w


def trueskill_update(
    ratings: list[Rating],
    ranks: list[int],
    *,
    beta: float = DEFAULT_BETA,
    tau: float = DEFAULT_TAU,
) -> list[Rating]:
    """Update TrueSkill ratings after one tournament round.

    Uses the sum-of-adjacent-pairs Thurstone-Mosteller approximation:
    sort participants by rank (0 = best), and for every adjacent pair
    with strictly distinct ranks apply the standard winner/loser update.
    Tied adjacent pairs are skipped — their only change for this round
    is the dynamics (``+tau^2``) term applied uniformly to every
    participant at the end.

    Args:
        ratings: Pre-round ratings for the participants.
        ranks: Finishing position, 0 = best. Ties are allowed (equal
            rank values). Must have the same length as ``ratings``.
        beta: Performance noise scale. Standard default
            ``mu_0 / 6 ≈ 4.167``.
        tau: Skill-drift standard deviation added each round to prevent
            sigma from collapsing to zero over time. Standard default
            ``sigma_0 / 100 ≈ 0.0833``.

    Returns:
        A fresh ``list[Rating]`` aligned with the input order. Inputs
        are not mutated.

    Raises:
        ValueError if the two arguments disagree on length.
    """
    if len(ratings) != len(ranks):
        raise ValueError(
            f"trueskill_update: ratings and ranks must be equal length; "
            f"got {len(ratings)} vs {len(ranks)}"
        )
    n = len(ratings)
    if n == 0:
        return []

    # Work on mutable copies of mu and sigma^2 indexed by the input order.
    mu = [r.mu for r in ratings]
    sig2 = [r.sigma * r.sigma for r in ratings]

    # Sort input indices by rank. Stable sort keeps input order among
    # equal-rank participants — harmless since tied pairs are skipped.
    order = sorted(range(n), key=lambda i: ranks[i])

    # Walk adjacent pairs in rank order; apply winner/loser update on
    # each strictly-ordered pair.
    for k in range(n - 1):
        i = order[k]       # higher-ranked (smaller rank value) = winner
        j = order[k + 1]   # lower-ranked = loser
        if ranks[i] == ranks[j]:
            continue  # tied: skip pair update (dynamics still apply below)

        c2 = 2.0 * beta * beta + sig2[i] + sig2[j]
        c = math.sqrt(c2)
        t = (mu[i] - mu[j]) / c
        v = _v_win(t)
        w = _w_win(t, v)

        # Winner (i)
        mu[i] += sig2[i] / c * v
        sig2[i] = max(0.0, sig2[i] * (1.0 - sig2[i] / c2 * w))

        # Loser (j): same magnitude of mu shift in the opposite direction
        mu[j] -= sig2[j] / c * v
        sig2[j] = max(0.0, sig2[j] * (1.0 - sig2[j] / c2 * w))

    # Dynamics: inflate every sigma by tau so uncertainty cannot drain
    # to zero even after many rounds.
    tau2 = tau * tau
    return [
        Rating(mu=mu[i], sigma=math.sqrt(sig2[i] + tau2))
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────
# Part 3: FSMVariant + Population
# ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class FSMVariant:
    """One FSM variant in the population."""

    id: str
    fsm: FSM
    rating: Rating
    parent_id: str | None
    generation: int
    birth_iteration: int
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class Population:
    """Growing-tree FSM population with sliding-window selection.

    Lifecycle
    ---------
    - Construct with a single root ``FSM``. The root becomes variant
      ``"gen0_root"`` with the default rating (mu=25, sigma=8.333).
    - Grow via :meth:`add_child`. Each child inherits its parent's mu
      and gets ``sigma = parent.sigma + delta_sigma`` — we just created
      it, so we're less sure about its skill than the parent.
    - Evaluate via the caller's own evolution loop; feed rewards back
      through :meth:`update_ratings`.
    - Sample for new rounds via :meth:`select`, which draws from the
      sliding window only (the latest ``window_size`` variants, in
      creation order).

    Old variants never leave the population (they're kept for
    provenance / the full tree), but only the window is eligible for
    selection.
    """

    def __init__(
        self,
        root_fsm: FSM,
        *,
        window_size: int = DEFAULT_WINDOW_SIZE,
        selection_lambda: float = DEFAULT_SELECTION_LAMBDA,
        selection_temperature: float = DEFAULT_SELECTION_TEMPERATURE,
        delta_sigma: float = DEFAULT_DELTA_SIGMA,
        trueskill_beta: float = DEFAULT_BETA,
        trueskill_tau: float = DEFAULT_TAU,
    ):
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if selection_temperature <= 0:
            raise ValueError(
                f"selection_temperature must be positive, got {selection_temperature}"
            )
        if delta_sigma < 0:
            raise ValueError(f"delta_sigma must be >= 0, got {delta_sigma}")

        self.window_size = window_size
        self.selection_lambda = selection_lambda
        self.selection_temperature = selection_temperature
        self.delta_sigma = delta_sigma
        self.trueskill_beta = trueskill_beta
        self.trueskill_tau = trueskill_tau

        self._variants: list[FSMVariant] = []
        self._variants_by_id: dict[str, FSMVariant] = {}
        self._mut_counter: int = 0

        # Seed the root.
        root = FSMVariant(
            id="gen0_root",
            fsm=root_fsm,
            rating=Rating(),
            parent_id=None,
            generation=0,
            birth_iteration=0,
            metadata={"role": "root"},
        )
        self._variants.append(root)
        self._variants_by_id[root.id] = root

    # ── Read-only views ────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Total variants ever created (root + every add_child), including
        variants that have since fallen outside the sliding window."""
        return len(self._variants)

    @property
    def window(self) -> list[FSMVariant]:
        """The last ``window_size`` variants in creation order — the
        subset eligible for :meth:`select`. Always non-empty as long as
        the population contains at least the root."""
        if self.window_size >= len(self._variants):
            return list(self._variants)
        return list(self._variants[-self.window_size:])

    @property
    def champion(self) -> FSMVariant:
        """Variant in the current window with the highest mu.

        Ties broken by later creation (a more recent variant with the
        same mu is preferred, matching the "fresher is better" bias).
        """
        win = self.window
        if not win:
            raise RuntimeError("Population is empty — no champion available")
        best = win[0]
        for v in win[1:]:
            if v.rating.mu >= best.rating.mu:
                best = v
        return best

    def get(self, variant_id: str) -> FSMVariant:
        """Look up a variant by id. Raises KeyError if unknown."""
        return self._variants_by_id[variant_id]

    # ── Mutation / selection ──────────────────────────────────────

    def add_child(
        self,
        parent_id: str,
        child_fsm: FSM,
        iteration: int,
        metadata: dict[str, Any] | None = None,
    ) -> FSMVariant:
        """Add a mutated child and return it.

        The child's initial rating is ``mu = parent.mu``,
        ``sigma = parent.sigma + delta_sigma`` — same expected skill,
        more uncertainty. Its id is ``f"gen{generation}_mut_{counter}"``
        where ``counter`` is a monotonic mutation index across the
        whole population and ``generation = parent.generation + 1``.
        """
        if parent_id not in self._variants_by_id:
            raise KeyError(f"parent variant {parent_id!r} not in population")
        parent = self._variants_by_id[parent_id]

        self._mut_counter += 1
        child_generation = parent.generation + 1
        child_id = f"gen{child_generation}_mut_{self._mut_counter}"

        child_rating = Rating(
            mu=parent.rating.mu,
            sigma=parent.rating.sigma + self.delta_sigma,
        )
        child = FSMVariant(
            id=child_id,
            fsm=child_fsm,
            rating=child_rating,
            parent_id=parent_id,
            generation=child_generation,
            birth_iteration=iteration,
            metadata=dict(metadata or {}),
        )
        self._variants.append(child)
        self._variants_by_id[child_id] = child
        return child

    def select(
        self,
        m: int,
        rng: random.Random | None = None,
    ) -> list[FSMVariant]:
        """Sample ``m`` variants from the window (without replacement).

        Probability of selecting variant ``i`` on a given draw is
        proportional to ``exp((mu_i + lambda * sigma_i) / T)`` where
        ``lambda = selection_lambda`` and ``T = selection_temperature``.
        Once a variant is drawn it's removed from the pool for
        subsequent draws.

        Args:
            m: Number of variants to return.
            rng: An injected :class:`random.Random` for reproducibility.
                If ``None``, constructs a fresh unseeded RNG.

        Raises:
            ValueError on ``m < 0`` or ``m > len(window)``.
        """
        if m < 0:
            raise ValueError(f"m must be >= 0, got {m}")
        if m == 0:
            return []

        rng = rng or random.Random()
        win = self.window
        if m > len(win):
            raise ValueError(
                f"Cannot select m={m}; window has only {len(win)} variants"
            )

        # Weighted sampling without replacement using the
        # "exponential rank" (Gumbel-softmax) trick would be fastest
        # but the window is tiny (≤ ~15), so a straightforward
        # normalize-and-draw loop is clearer.
        scores = [
            v.rating.mu + self.selection_lambda * v.rating.sigma
            for v in win
        ]
        # Subtract max before exp for numerical stability.
        max_score = max(scores)
        weights = [
            math.exp((s - max_score) / self.selection_temperature)
            for s in scores
        ]

        remaining = list(zip(win, weights))
        chosen: list[FSMVariant] = []
        for _ in range(m):
            total = sum(w for _, w in remaining)
            draw = rng.random() * total
            cum = 0.0
            for idx, (variant, w) in enumerate(remaining):
                cum += w
                if cum >= draw:
                    chosen.append(variant)
                    remaining.pop(idx)
                    break
            else:
                # Floating-point slop — take the last element.
                chosen.append(remaining.pop()[0])
        return chosen

    def update_ratings(
        self,
        variant_ids: list[str],
        rewards: list[float],
    ) -> None:
        """Apply a TrueSkill update to the named variants in place.

        Rewards are converted to ranks (highest reward = rank 0) using
        competition ranking — tied rewards produce tied ranks, and the
        next distinct reward skips to the post-tie position (standard
        "1-2-2-4" ranking, 0-indexed).

        Args:
            variant_ids: IDs of the variants that just played a round.
                Must all exist in the population; they need not be in
                the window (historical variants can participate).
            rewards: Parallel list of scalar rewards.
        """
        if len(variant_ids) != len(rewards):
            raise ValueError(
                f"variant_ids and rewards must be equal length; "
                f"got {len(variant_ids)} vs {len(rewards)}"
            )
        if not variant_ids:
            return

        ranks = _rewards_to_ranks(rewards)
        variants = [self._variants_by_id[vid] for vid in variant_ids]
        new_ratings = trueskill_update(
            [v.rating for v in variants],
            ranks,
            beta=self.trueskill_beta,
            tau=self.trueskill_tau,
        )
        for v, r in zip(variants, new_ratings):
            v.rating = r

    # ── Serialization ─────────────────────────────────────────────

    def to_json(self) -> dict[str, Any]:
        """Serialize the full population state to a JSON-ready dict.

        Captures the construction config, every variant (full FSM
        included), and the mutation counter so IDs remain unique after
        a round trip.
        """
        return {
            "config": {
                "window_size": self.window_size,
                "selection_lambda": self.selection_lambda,
                "selection_temperature": self.selection_temperature,
                "delta_sigma": self.delta_sigma,
                "trueskill_beta": self.trueskill_beta,
                "trueskill_tau": self.trueskill_tau,
            },
            "variants": [
                {
                    "id": v.id,
                    "fsm": v.fsm.to_json(),
                    "rating": v.rating.to_json(),
                    "parent_id": v.parent_id,
                    "generation": v.generation,
                    "birth_iteration": v.birth_iteration,
                    "metadata": dict(v.metadata),
                }
                for v in self._variants
            ],
            "mut_counter": self._mut_counter,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Population":
        """Restore a population from :meth:`to_json`'s output.

        Requires at least one variant (the root) in ``data["variants"]``.
        The first variant is treated as the root for the purpose of the
        constructor; subsequent variants are re-appended with their
        ratings and metadata preserved verbatim (no re-running of
        add_child's rating-inheritance logic).
        """
        if not data.get("variants"):
            raise ValueError("from_json: data must contain at least one variant")

        cfg = data.get("config", {})
        first = data["variants"][0]
        root_fsm = FSM.from_json(first["fsm"])
        pop = cls(
            root_fsm=root_fsm,
            window_size=cfg.get("window_size", DEFAULT_WINDOW_SIZE),
            selection_lambda=cfg.get("selection_lambda", DEFAULT_SELECTION_LAMBDA),
            selection_temperature=cfg.get(
                "selection_temperature", DEFAULT_SELECTION_TEMPERATURE,
            ),
            delta_sigma=cfg.get("delta_sigma", DEFAULT_DELTA_SIGMA),
            trueskill_beta=cfg.get("trueskill_beta", DEFAULT_BETA),
            trueskill_tau=cfg.get("trueskill_tau", DEFAULT_TAU),
        )
        # Overwrite the auto-seeded root with the persisted one (may
        # carry mutated metadata / ratings / id).
        root = FSMVariant(
            id=first["id"],
            fsm=root_fsm,
            rating=Rating.from_json(first["rating"]),
            parent_id=first.get("parent_id"),
            generation=int(first.get("generation", 0)),
            birth_iteration=int(first.get("birth_iteration", 0)),
            metadata=dict(first.get("metadata", {})),
        )
        pop._variants = [root]
        pop._variants_by_id = {root.id: root}

        for raw in data["variants"][1:]:
            variant = FSMVariant(
                id=raw["id"],
                fsm=FSM.from_json(raw["fsm"]),
                rating=Rating.from_json(raw["rating"]),
                parent_id=raw.get("parent_id"),
                generation=int(raw.get("generation", 0)),
                birth_iteration=int(raw.get("birth_iteration", 0)),
                metadata=dict(raw.get("metadata", {})),
            )
            pop._variants.append(variant)
            pop._variants_by_id[variant.id] = variant

        pop._mut_counter = int(data.get("mut_counter", 0))
        return pop

    # ── Reporting ─────────────────────────────────────────────────

    def summary_table(self) -> str:
        """Render the current state of the population as a text table.

        Columns: ID | Gen | mu | sigma | Conservative | Parent | InWindow.
        Rows are every variant in the population (not just the window),
        in creation order, with an ``InWindow`` flag on the last
        ``window_size`` rows.
        """
        window_ids = {v.id for v in self.window}

        headers = ("ID", "Gen", "mu", "sigma", "Conservative", "Parent", "InWindow")
        rows: list[tuple[str, ...]] = [headers]
        for v in self._variants:
            rows.append((
                v.id,
                str(v.generation),
                f"{v.rating.mu:.2f}",
                f"{v.rating.sigma:.2f}",
                f"{v.rating.conservative:.2f}",
                v.parent_id or "-",
                "yes" if v.id in window_ids else "no",
            ))

        col_widths = [max(len(r[c]) for r in rows) for c in range(len(headers))]
        out_lines: list[str] = []
        for r in rows:
            out_lines.append("  ".join(
                cell.ljust(col_widths[c]) for c, cell in enumerate(r)
            ))
        return "\n".join(out_lines)


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────


def _rewards_to_ranks(rewards: list[float]) -> list[int]:
    """Convert rewards to competition ranks (0 = highest reward).

    Tied rewards get the same rank; the next distinct reward skips to
    the post-tie position so ranks correspond to "how many players
    strictly above me". Example::

        rewards = [5, 10, 5, 3]  →  ranks = [1, 0, 1, 3]
    """
    n = len(rewards)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: -rewards[i])
    ranks = [0] * n
    ranks[order[0]] = 0
    for k in range(1, n):
        idx = order[k]
        prev_idx = order[k - 1]
        if rewards[idx] == rewards[prev_idx]:
            ranks[idx] = ranks[prev_idx]
        else:
            ranks[idx] = k
    return ranks


__all__ = [
    "DEFAULT_MU",
    "DEFAULT_SIGMA",
    "DEFAULT_BETA",
    "DEFAULT_TAU",
    "DEFAULT_WINDOW_SIZE",
    "DEFAULT_SELECTION_LAMBDA",
    "DEFAULT_SELECTION_TEMPERATURE",
    "DEFAULT_DELTA_SIGMA",
    "FSMVariant",
    "Population",
    "Rating",
    "trueskill_update",
]

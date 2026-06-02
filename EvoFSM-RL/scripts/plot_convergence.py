#!/usr/bin/env python3
"""Convergence-plot utility for B3 / B4 evolution runs.

Reads the JSONL logs produced by ``evofsm_rl.fsm.evolution.run_evolution``
and renders a multi-panel figure summarising:

  Panel 1: per-iteration success (``best_reward``) with a sliding average.
  Panel 2 (B4 only): GRPO loss over time, one point per weight update.
  Panel 3 / final panel: Champion TrueSkill ``mu`` across iterations.

Usage as library (called at end of ``run_b3_evolution.py`` /
``run_b4_evolution.py``)::

    from scripts.plot_convergence import plot_convergence
    plot_convergence(output_dir, app_name, n_iterations, has_grpo=False)

Usage as CLI (retroactive generation over existing traces)::

    python EvoFSM-RL/scripts/plot_convergence.py \\
        --log-dir EvoFSM-RL/traces/b3_evolution/pro_expense \\
        --app pro_expense --n-iterations 20

``matplotlib`` uses the ``Agg`` backend so no display is required. If
``matplotlib`` isn't installed the function logs a warning and returns
``None`` instead of crashing — the evolution runs themselves never
depend on the plot.

Input files expected under ``log_dir``:
  * ``iterations.jsonl`` — required. One line per iteration. Fields used:
    ``iteration``, ``best_reward``, ``champion_mu``. Missing
    ``champion_mu`` (present only from Story 4.4 onward) falls back to
    the default rating mean 25.0, which renders as a flat line —
    harmless for plot legibility.
  * ``grpo_metrics.jsonl`` — optional. Only read when
    ``has_grpo=True``. One line per GRPO update with ``iteration`` and
    ``loss`` fields. If absent, the loss panel shows a placeholder.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


logger = logging.getLogger("plot_convergence")


def _sliding_avg(values, window: int = 5):
    """Return running mean over ``window`` with ``valid`` mode.

    Shorter-than-window inputs return the original values so callers
    still have something to plot on very small runs.
    """
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_convergence(
    log_dir: Path | str,
    app_name: str,
    n_iterations: int,
    *,
    has_grpo: bool = False,
    output_filename: str = "convergence.png",
) -> Path | None:
    """Generate a convergence plot for one evolution run.

    Args:
        log_dir: Directory holding ``iterations.jsonl`` (and, for B4,
            ``grpo_metrics.jsonl``). Typically the evolution output dir.
        app_name: App label for the figure title.
        n_iterations: Total configured iterations. Used only in the
            suptitle; the plot itself reflects whatever rows are in
            ``iterations.jsonl``, which may be fewer on a partial run.
        has_grpo: ``True`` for B4 runs → include a dedicated GRPO-loss
            panel. ``False`` for B3 → 2-panel figure (SR + champion mu).
        output_filename: Name of the PNG written under ``log_dir``.

    Returns:
        Absolute path to the saved PNG, or ``None`` if matplotlib is
        missing or ``iterations.jsonl`` is absent.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning(
            "matplotlib not installed — skipping convergence plot. "
            "Install with: pip install matplotlib",
        )
        return None

    log_dir = Path(log_dir)
    iter_path = log_dir / "iterations.jsonl"
    if not iter_path.exists():
        logger.warning(
            "No iterations.jsonl at %s — skipping convergence plot.",
            iter_path,
        )
        return None

    iterations: list[int] = []
    successes: list[float] = []
    champion_mus: list[float] = []
    with iter_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            iterations.append(int(row["iteration"]))
            successes.append(float(row.get("best_reward", 0.0)))
            # champion_mu is populated from Story 4.4 onward. Older
            # runs fall back to the default TrueSkill prior (25.0).
            mu = row.get("champion_mu")
            if mu is None:
                mu = row.get("best_mu", 25.0)
            champion_mus.append(float(mu))

    if not iterations:
        logger.warning("iterations.jsonl at %s is empty.", iter_path)
        return None

    iterations_arr = np.asarray(iterations)
    successes_arr = np.asarray(successes)
    mus_arr = np.asarray(champion_mus)

    n_panels = 3 if has_grpo else 2
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(10, 3.5 * n_panels), sharex=False,
    )
    if n_panels == 1:
        axes = [axes]

    # ── Panel 1: T_adapt Success Rate ────────────────────────────
    ax = axes[0]
    ax.scatter(
        iterations_arr, successes_arr,
        alpha=0.35, s=18, color="steelblue", label="Per-iteration",
    )
    window = 5
    sr_smooth = _sliding_avg(successes_arr, window=window)
    if len(sr_smooth) < len(successes_arr):
        x_smooth = iterations_arr[window - 1 : window - 1 + len(sr_smooth)]
        ax.plot(
            x_smooth, sr_smooth,
            color="darkblue", linewidth=2, label=f"Sliding avg (w={window})",
        )
    ax.set_ylabel("best_reward (per iter)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax.set_title(f"{app_name} — T_adapt Success Rate")
    ax.grid(True, alpha=0.3)

    # ── Panel 2 (B4 only): GRPO Loss ─────────────────────────────
    if has_grpo:
        ax = axes[1]
        grpo_path = log_dir / "grpo_metrics.jsonl"
        if grpo_path.exists():
            grpo_iters: list[int] = []
            grpo_losses: list[float] = []
            with grpo_path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    grpo_iters.append(int(entry["iteration"]))
                    grpo_losses.append(float(entry["loss"]))
            if grpo_iters:
                ax.plot(
                    grpo_iters, grpo_losses,
                    marker="o", markersize=5, color="crimson", linewidth=1.5,
                )
                ax.set_ylabel("GRPO loss")
                ax.set_title("GRPO Weight-Update Loss")
                ax.grid(True, alpha=0.3)
            else:
                ax.text(
                    0.5, 0.5, "No GRPO updates recorded",
                    ha="center", va="center", transform=ax.transAxes,
                )
                ax.set_axis_off()
        else:
            ax.text(
                0.5, 0.5, "grpo_metrics.jsonl not found",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_axis_off()

    # ── Final panel: Champion TrueSkill mu ───────────────────────
    ax = axes[-1]
    ax.plot(
        iterations_arr, mus_arr,
        color="forestgreen", linewidth=2, marker=".", markersize=4,
    )
    ax.set_ylabel("Champion mu")
    ax.set_xlabel("Iteration")
    ax.set_title("Population Champion TrueSkill Rating")
    ax.grid(True, alpha=0.3)

    tag = "B4" if has_grpo else "B3"
    fig.suptitle(
        f"{app_name} - {tag} Evolution Convergence "
        f"({n_iterations} iterations planned, {len(iterations)} logged)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    out_path = log_dir / output_filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Convergence plot saved to %s", out_path)
    return out_path


def _cli() -> int:
    p = argparse.ArgumentParser(
        description="Generate a convergence plot for one evolution run.",
    )
    p.add_argument("--log-dir", type=Path, required=True,
                   help="Directory holding iterations.jsonl (and optionally "
                        "grpo_metrics.jsonl).")
    p.add_argument("--app", type=str, required=True,
                   help="App label for the figure title.")
    p.add_argument("--n-iterations", type=int, default=20,
                   help="Configured iteration count (for the suptitle).")
    p.add_argument("--has-grpo", action="store_true",
                   help="Include a GRPO-loss panel (B4 runs).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    out = plot_convergence(
        log_dir=args.log_dir,
        app_name=args.app,
        n_iterations=args.n_iterations,
        has_grpo=args.has_grpo,
    )
    return 0 if out is not None else 1


if __name__ == "__main__":
    sys.exit(_cli())

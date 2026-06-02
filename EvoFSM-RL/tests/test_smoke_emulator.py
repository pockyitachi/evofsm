"""Smoke test: boot AVD → connect → initialize tasks → check harness wiring.

This test verifies Story 1.1 (ADR-001): the Android emulator is reachable,
AndroidWorld's env_launcher can connect, and tasks can be initialized and
evaluated without an agent.

Prerequisites:
    1. AVD created: ./scripts/bootstrap_avd.sh
    2. Emulator running: emulator -avd Pixel_6_API_33 -no-snapshot-load -gpu auto
    3. Python env active with android_world importable.

Run:
    cd EvoFSM-RL/
    PYTHONPATH=../android_world_plus:. python tests/test_smoke_emulator.py

    Or with pytest:
    PYTHONPATH=../android_world_plus:. python -m pytest tests/test_smoke_emulator.py -v

What it does NOT test:
    - Agent quality / success rate (no LLM involved).
    - Model loading (Story 1.2).
    - Snapshot round-trip (Story 1.3).

Expected: all tasks return success=0.0 (no agent acts), with no exceptions.
If a task raises, it's captured in TemplateResult.error and reported as FAIL.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time

from evofsm_rl.env import harness

# ── Configuration ────────────────────────────────────────────────────
DEFAULT_N_TEMPLATES = 10
DEFAULT_SEED = 30

logger = logging.getLogger(__name__)


def run_smoke_test(
    n_templates: int = DEFAULT_N_TEMPLATES,
    seed: int = DEFAULT_SEED,
    console_port: int = harness.DEFAULT_CONSOLE_PORT,
    grpc_port: int = harness.DEFAULT_GRPC_PORT,
    templates: list[str] | None = None,
) -> tuple[int, int]:
    """Run smoke test: connect to emulator, init N tasks, check results.

    Args:
        n_templates: How many templates to test.
        seed: Random seed for both template selection and task params.
        console_port: Emulator console port.
        grpc_port: Emulator gRPC port.
        templates: Explicit list of templates to test. If None, randomly
            samples from LIGHTWEIGHT_TEMPLATES.

    Returns:
        (n_passed, n_total) — a template "passes" if it initializes and
        returns a TemplateResult with error=None.
    """
    # Select templates from actual registry
    if templates is None:
        all_templates = harness.list_all_templates()
        rng = random.Random(seed)
        n = min(n_templates, len(all_templates))
        templates = rng.sample(all_templates, n)

    print(f"\n{'='*60}")
    print(f"EvoFSM-RL Smoke Test — Story 1.1")
    print(f"  Templates: {len(templates)}")
    print(f"  Seed:      {seed}")
    print(f"  Ports:     console={console_port}, grpc={grpc_port}")
    print(f"{'='*60}\n")

    # Connect to emulator
    print("Connecting to emulator...")
    t0 = time.monotonic()
    env = harness.connect(
        console_port=console_port,
        grpc_port=grpc_port,
    )
    connect_time = time.monotonic() - t0
    print(f"  Connected in {connect_time:.1f}s\n")

    # Run each template (no agent — just init + check + teardown)
    passed = 0
    failed = 0
    results: list[harness.TemplateResult] = []

    for i, template_name in enumerate(templates, 1):
        print(f"[{i}/{len(templates)}] {template_name} ... ", end="", flush=True)
        result = harness.run_template(
            template_name=template_name,
            seed=seed,
            env=env,
            agent=None,  # no agent — smoke test only
        )
        results.append(result)

        if result.error is None:
            passed += 1
            # success=0.0 is expected (no agent acted)
            print(f"OK  (success={result.success:.1f}, {result.wall_seconds:.1f}s)")
        else:
            failed += 1
            print(f"FAIL  ({result.error})")

    # Close environment
    print("\nClosing emulator connection...")
    env.close()

    # Summary
    total = passed + failed
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed > 0:
        print("\nFailed templates:")
        for r in results:
            if r.error is not None:
                print(f"  {r.task_name}: {r.error}")
    print(f"{'='*60}\n")

    return passed, total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EvoFSM-RL Story 1.1 smoke test — verify emulator harness."
    )
    parser.add_argument(
        "-n",
        "--n-templates",
        type=int,
        default=DEFAULT_N_TEMPLATES,
        help=f"Number of templates to test (default: {DEFAULT_N_TEMPLATES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--console-port",
        type=int,
        default=harness.DEFAULT_CONSOLE_PORT,
        help=f"Emulator console port (default: {harness.DEFAULT_CONSOLE_PORT})",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=harness.DEFAULT_GRPC_PORT,
        help=f"Emulator gRPC port (default: {harness.DEFAULT_GRPC_PORT})",
    )
    parser.add_argument(
        "--templates",
        nargs="+",
        default=None,
        help="Explicit list of template names to test",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    passed, total = run_smoke_test(
        n_templates=args.n_templates,
        seed=args.seed,
        console_port=args.console_port,
        grpc_port=args.grpc_port,
        templates=args.templates,
    )

    sys.exit(0 if passed == total else 1)


# ── pytest entry point ───────────────────────────────────────────────
def test_smoke_emulator():
    """pytest-compatible wrapper.

    Runs 5 templates (fewer than CLI default) to keep CI fast.
    Skips if emulator is not reachable.
    """
    try:
        passed, total = run_smoke_test(n_templates=5)
    except Exception as exc:
        import pytest
        pytest.skip(f"Emulator not reachable: {exc}")
        return
    assert passed == total, f"Smoke test: {passed}/{total} passed"


if __name__ == "__main__":
    main()

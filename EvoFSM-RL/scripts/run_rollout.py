#!/usr/bin/env python3
"""Run a single AndroidWorld task with the Qwen3-VL baseline agent.

This is Story 1.4.3's executable proof: load the pinned Qwen3-VL-8B policy,
connect to a running emulator, pick one task template, run the agent, and
print the rule-based ``is_successful`` score.

Story 1.4.4 is "make this work end-to-end on one task"; Story 1.4.5 loops
this over ~10 tasks to produce the baseline number comparable to
AndroidLab 2025's ~28%.

Usage:
    # Emulator already booted (Pixel 6 API 33, console 5554, grpc 8554)
    cd /path/to/android_world_plus
    source .venv/bin/activate
    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      python EvoFSM-RL/scripts/run_rollout.py \\
      --template CalculatorInput1Plus1 --seed 30

    # Change step budget multiplier (default 10 × task.complexity)
    ... --max-steps-multiplier 5

    # Force device (default auto-detect)
    ... --device mps

Exit codes:
    0 — task ran to completion (success OR failure is both a PASS here;
        we only exit non-zero on hard infrastructure errors)
    1 — model / env / infra failure before we got a score
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("run_rollout")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )


def _write_trace(trace_path: Path, step_data_list: list[dict[str, Any]]) -> None:
    """Persist a JSON trace of the episode for offline inspection.

    Strips numpy arrays (screenshots) since they bloat the file and aren't
    meaningful in JSON form — write separately as PNGs if visual review is
    needed in a later story.
    """
    import numpy as np

    def _scrub(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return f"<ndarray shape={x.shape} dtype={x.dtype}>"
        if dataclasses.is_dataclass(x):
            return dataclasses.asdict(x)
        if hasattr(x, "as_dict"):
            # JSONAction has as_dict(skip_none=True)
            try:
                return x.as_dict(skip_none=True)
            except TypeError:
                pass
        if isinstance(x, dict):
            return {k: _scrub(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_scrub(v) for v in x]
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        return repr(x)

    scrubbed = [_scrub(sd) for sd in step_data_list]
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w") as fh:
        json.dump(scrubbed, fh, indent=2)
    logger.info("wrote trace to %s", trace_path)


def _flatten_step_data(result_step_data: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Un-transpose the runner's dol-of-list step_data back to a list-of-dicts."""
    if not result_step_data:
        return []
    keys = list(result_step_data.keys())
    n = len(result_step_data[keys[0]])
    return [{k: result_step_data[k][i] for k in keys} for i in range(n)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--template", required=True,
        help="Task template name (e.g. 'CalculatorInput1Plus1').",
    )
    p.add_argument("--seed", type=int, default=30,
                   help="Random seed for task param generation.")
    p.add_argument("--device", choices=("mps", "cuda", "cpu"), default=None,
                   help="Force torch device (default: auto-detect).")
    p.add_argument(
        "--max-steps-multiplier", type=float, default=10.0,
        help="Episode step budget = int(multiplier * task.complexity). "
             "Default 10 matches AW framework convention.",
    )
    p.add_argument(
        "--console-port", type=int, default=5554,
        help="emulator -avd Pixel_6_API_33 console port (default 5554)",
    )
    p.add_argument(
        "--grpc-port", type=int, default=8554,
        help="android_env gRPC port (default 8554)",
    )
    p.add_argument(
        "--adb-path", type=str, default=None,
        help="Path to adb binary (default: $ANDROID_HOME/platform-tools/adb, "
             "falling back to ~/Library/Android/sdk/platform-tools/adb on Mac).",
    )
    p.add_argument(
        "--trace-dir", type=Path, default=Path("traces"),
        help="Where to write per-step JSON trace. "
             "Created if missing. Relative paths resolve against CWD.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    _setup_logging(args.verbose)

    # Late imports so --help doesn't pay the torch/transformers cost.
    from evofsm_rl.agent.rollout import GenerationConfig, Qwen3VLAgent
    from evofsm_rl.env import harness
    from evofsm_rl.model import load_base_model, load_model_config, resolve_device

    # 1. Load model.
    resolved_device = args.device or resolve_device()
    logger.info("Loading Qwen3-VL-8B on device='%s'...", resolved_device)
    t0 = time.monotonic()
    model, processor = load_base_model(device=resolved_device)
    logger.info("Model loaded in %.1fs.", time.monotonic() - t0)

    # Pick up generation config from yaml (greedy baseline by default).
    cfg = load_model_config(device=resolved_device)
    gen_cfg = GenerationConfig.from_yaml(cfg.raw.get("generation", {}))
    logger.info("Generation config: %s", gen_cfg)

    # 2. Connect to emulator.
    logger.info("Connecting to emulator...")
    connect_kwargs: dict[str, Any] = {
        "console_port": args.console_port,
        "grpc_port": args.grpc_port,
    }
    if args.adb_path:
        connect_kwargs["adb_path"] = args.adb_path
    env = harness.connect(**connect_kwargs)

    # 3. Build agent.
    agent = Qwen3VLAgent(
        model=model,
        processor=processor,
        env=env,
        device=resolved_device,
        generation_config=gen_cfg,
    )

    # 4. Run template via harness (handles init/success-check/tear_down).
    try:
        logger.info("Running template %s (seed=%d)...", args.template, args.seed)
        result = harness.run_template(
            template_name=args.template,
            seed=args.seed,
            env=env,
            agent=agent,
            max_steps_multiplier=args.max_steps_multiplier,
        )
    finally:
        try:
            env.close()
        except Exception:
            logger.warning("env.close() failed", exc_info=True)

    # 5. Persist trace (only on successful run — error path has no data).
    trace_path = args.trace_dir / f"{args.template}_seed{args.seed}.json"
    # harness stores the per-step dicts inside episode_runner.EpisodeResult,
    # but run_template doesn't expose them. Emit a summary file here; the
    # full per-step trace is captured via Qwen3VLAgent's history state,
    # which we can re-serialize from the agent directly.
    summary = {
        "task_name": result.task_name,
        "seed": result.seed,
        "success": result.success,
        "n_steps": result.n_steps,
        "wall_seconds": result.wall_seconds,
        "error": result.error,
        "device": resolved_device,
        "model_name": cfg.name,
        "model_revision": cfg.revision,
        "agent_history": agent.history,  # last-episode history (M3A-style summaries)
    }
    _write_trace(trace_path, [summary])

    # 6. Report.
    print("=" * 64)
    print(f"Task:           {result.task_name}")
    print(f"Seed:           {result.seed}")
    print(f"Success:        {result.success:.2f}")
    print(f"Steps taken:    {result.n_steps}")
    print(f"Wall time:      {result.wall_seconds:.1f}s")
    if result.error:
        print(f"ERROR:          {result.error}")
    print(f"Trace:          {trace_path}")
    print("=" * 64)

    # Hard infra failures exit non-zero; "success=0.0 but ran clean" is
    # still exit 0 because the agent DID produce a judgeable outcome.
    return 1 if result.error else 0


if __name__ == "__main__":
    sys.exit(main())

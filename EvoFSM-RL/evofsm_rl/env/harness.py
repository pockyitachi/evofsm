"""Thin wrapper over AndroidWorld's env_launcher + task runner.

Provides a single entry point for EvoFSM-RL to:
  1. Connect to a running Android emulator.
  2. Initialize a task template.
  3. (Optionally) run an agent episode.
  4. Check success and tear down.

All gRPC / ADB wiring is delegated to `android_world.env.env_launcher`.
See ADR-001 for emulator path decisions.

Usage (smoke test — no agent, just verify harness):
    env_handle = connect()
    result = run_template("CalculatorInput1Plus1", seed=30, env=env_handle)
    print(result)  # TemplateResult(task_name=..., success=0.0, ...)
    env_handle.close()
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from typing import TYPE_CHECKING, Any

from android_world import registry as aw_registry
from android_world import suite_utils
from android_world.env import env_launcher

if TYPE_CHECKING:
    from android_world.env import interface
    from android_world.task_evals import task_eval

logger = logging.getLogger(__name__)

# ── Defaults (per ADR-001) ───────────────────────────────────────────
DEFAULT_CONSOLE_PORT = 5554
DEFAULT_GRPC_PORT = 8554


def _default_adb_path() -> str:
    """Resolve a sensible adb default: honor $ANDROID_HOME if set, else Mac layout."""
    sdk = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
    if sdk:
        return os.path.join(sdk, "platform-tools", "adb")
    return "~/Library/Android/sdk/platform-tools/adb"


DEFAULT_ADB_PATH = _default_adb_path()


# ── Result dataclass ─────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class TemplateResult:
    """Outcome of running (or just initializing) a single task template."""

    task_name: str
    seed: int
    success: float  # 0.0 or 1.0 (or partial for composite tasks)
    n_steps: int  # 0 if no agent was run
    wall_seconds: float
    error: str | None = None  # non-None if an exception occurred


# ── Environment handle ───────────────────────────────────────────────
def connect(
    *,
    console_port: int = DEFAULT_CONSOLE_PORT,
    grpc_port: int = DEFAULT_GRPC_PORT,
    adb_path: str = DEFAULT_ADB_PATH,
    emulator_setup: bool = False,
) -> interface.AsyncEnv:
    """Connect to a running emulator and return an AsyncEnv handle.

    The emulator must already be booted (e.g. via
    ``emulator -avd Pixel_6_API_33 -no-snapshot-load -gpu auto``).

    Args:
        console_port: ADB console port (default 5554).
        grpc_port: gRPC port for android_env (default 8554).
        adb_path: Path to ``adb`` binary.
        emulator_setup: If True, run first-time app installation.
            Only needed once per fresh AVD image.

    Returns:
        An ``AsyncEnv`` handle. Caller is responsible for calling
        ``.close()`` when done.
    """
    logger.info(
        "Connecting to emulator (console=%d, grpc=%d, adb=%s)",
        console_port,
        grpc_port,
        adb_path,
    )
    env = env_launcher.load_and_setup_env(
        console_port=console_port,
        grpc_port=grpc_port,
        adb_path=adb_path,
        emulator_setup=emulator_setup,
    )
    logger.info("Connected to emulator successfully.")
    return env


# ── Task registry helper ─────────────────────────────────────────────
def _get_task_class(template_name: str) -> type[task_eval.TaskEval]:
    """Look up a task class by its PascalCase template name.

    Searches the ``android_world`` family (which includes both vanilla
    AndroidWorld and the Plus-repo tasks).
    """
    task_registry = aw_registry.TaskRegistry()
    all_tasks = task_registry.get_registry(family="android_world")
    if template_name not in all_tasks:
        raise KeyError(
            f"Unknown template {template_name!r}. "
            f"Available: {sorted(all_tasks)[:10]}... ({len(all_tasks)} total)"
        )
    return all_tasks[template_name]


def list_all_templates() -> list[str]:
    """Return sorted list of all registered template names."""
    task_registry = aw_registry.TaskRegistry()
    return sorted(task_registry.get_registry(family="android_world"))


# ── Core runner ──────────────────────────────────────────────────────
def run_template(
    template_name: str,
    seed: int,
    env: interface.AsyncEnv,
    *,
    agent: Any | None = None,
    max_steps_multiplier: float = 10.0,
    use_dense_reward: bool = False,
) -> TemplateResult:
    """Initialize a task, optionally run an agent, check success, tear down.

    Args:
        template_name: PascalCase class name (e.g. "CalculatorInput1Plus1").
        seed: Random seed for ``generate_random_params``.
        env: A connected ``AsyncEnv`` (from :func:`connect`).
        agent: If provided, must expose ``step(goal)`` per AndroidWorld's
            ``EnvironmentInteractingAgent`` protocol. If None, the task is
            initialized and immediately checked (useful for smoke tests).
        max_steps_multiplier: Step budget = int(multiplier * task.complexity).
        use_dense_reward: If True, call ``task.get_dense_reward(env)`` for the
            success field instead of ``task.is_successful(env)``. Default
            False (binary {0,1}). Opt-in for RL training only — T_eval must
            stay binary. See CLAUDE.md "Dense reward design rule".

    Returns:
        A :class:`TemplateResult` with success score, step count, and timing.
    """
    task_cls = _get_task_class(template_name)

    # Create task instance with seeded params
    suite = suite_utils.create_suite(
        {template_name: task_cls},
        n_task_combinations=1,
        seed=seed,
    )
    # suite is {template_name: [task_instance, ...]}
    task_instances = suite[template_name]
    task = task_instances[0]

    n_steps = 0
    t0 = time.monotonic()

    try:
        # Initialize task (sets up app state on device)
        logger.info("Initializing task %s (seed=%d)", template_name, seed)
        task.initialize_task(env)

        if agent is not None:
            # Run agent episode
            from android_world import episode_runner

            max_n_steps = int(max_steps_multiplier * task.complexity)
            episode_result = episode_runner.run_episode(
                goal=task.goal,
                agent=agent,
                max_n_steps=max_n_steps,
                start_on_home_screen=task.start_on_home_screen,
            )
            # Count steps from episode data
            if episode_result.step_data:
                # step_data is a dict of lists; length of any list = n_steps
                first_key = next(iter(episode_result.step_data))
                n_steps = len(episode_result.step_data[first_key])

        # Check success — binary by default; dense (partial credit) opt-in.
        if use_dense_reward:
            success = task.get_dense_reward(env)
        else:
            success = task.is_successful(env)
        logger.info(
            "Task %s: success=%.2f, steps=%d", template_name, success, n_steps
        )

        return TemplateResult(
            task_name=template_name,
            seed=seed,
            success=float(success),
            n_steps=n_steps,
            wall_seconds=time.monotonic() - t0,
        )

    except Exception as exc:
        logger.exception("Task %s failed with exception", template_name)
        return TemplateResult(
            task_name=template_name,
            seed=seed,
            success=0.0,
            n_steps=n_steps,
            wall_seconds=time.monotonic() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )

    finally:
        # Always tear down to leave device in a clean state
        try:
            task.tear_down(env)
        except Exception:
            logger.warning("tear_down failed for %s", template_name, exc_info=True)

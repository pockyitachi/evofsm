#!/usr/bin/env python3
"""Pretrain the PPO value head on existing trajectory data.

The value head is a ``Linear(hidden_size → 1)`` projection of the
frozen base model's last-token hidden state. We pretrain it via MSE
regression on trajectory-level returns (binary 0/1 = task success)
**before** kicking off the full PPO training loop. This gives PPO a
non-trivial advantage signal from the first iteration instead of
zero-baseline noise.

Algorithm (per step in each loaded trajectory):
  1. Load before-step screenshot + UI element text + history summaries
     from ``episode.jsonl`` + per-step PNGs (Story 2.0 schema).
  2. Build the M3A action-selection prompt (same as the agent's
     rollout-time prompt).
  3. Forward the base model (LoRA NOT attached) with
     ``output_hidden_states=True``.
  4. Predict V(s) = head(hidden[:, -1, :]).
  5. MSE loss against the trajectory's terminal reward (binary
     ``meta["success"]``). Adam step on the value head only.

We do NOT need a real emulator or GPU-side LoRA — only the base model
and the value head. Runs cleanly on a single GPU. ~1-2 hours on the
source_pool 480-episode collection (5000+ steps) per epoch.

Example::

    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      python EvoFSM-RL/scripts/run_ppo_value_pretrain.py \\
        --data-dirs EvoFSM-RL/traces/source_pool_trajectories \\
        --epochs 3 --batch-size 8 --lr 1e-3 \\
        --output EvoFSM-RL/artifacts/value_head_v01.pt \\
        --device cuda

For testing / dry-run::

    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      python EvoFSM-RL/scripts/run_ppo_value_pretrain.py \\
        --data-dirs EvoFSM-RL/traces/source_pool_trajectories \\
        --epochs 1 --max-traj 5 \\
        --output /tmp/vh_smoke.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("run_ppo_value_pretrain")


# ─────────────────────────────────────────────────────────────────────
# Trajectory loading
# ─────────────────────────────────────────────────────────────────────


def _discover_episodes(data_dirs: list[Path]) -> list[Path]:
    """Walk each data dir and yield every episode directory found.

    An episode dir is anything containing both ``meta.json`` and
    ``episode.jsonl``. Walks recursively because B4 sweep traces live
    at e.g. ``traces/b4_k4_unified/<app>/episodes/<template>_seed<N>/``.
    """
    eps: list[Path] = []
    for root in data_dirs:
        root = Path(root)
        if not root.exists():
            logger.warning("Data dir does not exist: %s", root)
            continue
        for meta_path in root.rglob("meta.json"):
            ep_dir = meta_path.parent
            if (ep_dir / "episode.jsonl").exists():
                eps.append(ep_dir)
    return sorted(eps)


def _load_episode(ep_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return ``(meta, steps)`` for one episode."""
    meta = json.loads((ep_dir / "meta.json").read_text())
    steps: list[dict[str, Any]] = []
    with (ep_dir / "episode.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            steps.append(json.loads(line))
    return meta, steps


# ─────────────────────────────────────────────────────────────────────
# Value-head training
# ─────────────────────────────────────────────────────────────────────


def _build_step_prompt(
    *,
    goal: str,
    history_summaries: list[str],
    before_ui_elements_text: str,
):
    """Compose the M3A action-selection prompt text for one step.

    Mirrors ``Qwen3VLAgent.step()``'s prompt construction (without L_C
    injection and without per-task guidelines — we deliberately use
    the byte-identical B1 prompt shape during value-head pretraining,
    because the head should learn a value of state under the *prompt
    distribution it will see at PPO time*).
    """
    from evofsm_rl.agent.prompts import build_action_prompt

    history_lines = [
        f"Step {i + 1}- {summary}"
        for i, summary in enumerate(history_summaries)
    ]
    return build_action_prompt(
        goal=goal,
        history=history_lines,
        ui_elements=before_ui_elements_text or "Not available",
        additional_guidelines=None,
        l_c_prompt_text=None,
    )


def _forward_value_head(
    model: Any,
    value_head: Any,
    processor: Any,
    prompt_text: str,
    raw_screenshot: Any,
    som_screenshot: Any,
    device: str,
):
    """Run the base model + value head on one step's (text, images).

    Returns the scalar V(s) tensor with gradient flowing through the
    head only.
    """
    import torch
    from evofsm_rl.agent.prompts import build_action_messages

    action_messages = build_action_messages(
        prompt_text, raw_screenshot, som_screenshot,
    )
    text = processor.apply_chat_template(
        action_messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[raw_screenshot, som_screenshot],
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Last-layer hidden state; detach so only the head receives grad.
    hidden = outputs.hidden_states[-1].detach()
    return value_head(hidden)  # shape (1,)


def _draw_som_screenshot(raw_pil, ui_elements, logical_size):
    """Cheap helper: render SoM marks on a copy of the raw screenshot.

    Mirrors what the agent does at rollout time. UI elements come from
    the agent's per-step pickle if available; otherwise we fall back to
    the raw screenshot unmarked (still feeds the value head — slightly
    different distribution from training-time but acceptable for
    pretraining).
    """
    # The episode.jsonl does not carry the raw UI element objects (only
    # the text rendering). Rather than try to reconstruct them, we
    # pass the raw screenshot for both image slots. The value head
    # learns from the (text + image) joint distribution; the second
    # SoM image differs from training time but is structurally the
    # same number of tokens — the head's V(s) is still calibrated on
    # what matters (the goal/history/UI-text + screenshot).
    return raw_pil


def train_value_head(
    *,
    data_dirs: list[Path],
    output_path: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    max_traj: int | None,
    seed: int = 42,
    log_every: int = 20,
) -> dict[str, Any]:
    """End-to-end pretraining loop. Returns final metrics dict."""
    import torch
    from PIL import Image

    from evofsm_rl.model import load_base_model, resolve_device
    from evofsm_rl.rl_ppo.value_head import (
        attach_value_head, save_value_head,
    )

    resolved_device = device or resolve_device()
    logger.info("Loading base Qwen3-VL on device=%s", resolved_device)
    t0 = time.monotonic()
    model, processor = load_base_model(device=resolved_device)
    logger.info("Base model loaded in %.1fs", time.monotonic() - t0)
    model.eval()

    value_head = attach_value_head(model, resolved_device)
    optimizer = torch.optim.AdamW(value_head.parameters(), lr=lr)

    episodes = _discover_episodes(data_dirs)
    if not episodes:
        raise RuntimeError(
            f"No episodes found under {data_dirs}. Check the paths."
        )
    rng = random.Random(seed)
    rng.shuffle(episodes)
    if max_traj is not None:
        episodes = episodes[: int(max_traj)]
    logger.info("Loaded %d episodes for value-head pretraining", len(episodes))

    # Pre-load (meta, steps) for each episode (lightweight — just JSON).
    loaded: list[tuple[Path, dict[str, Any], list[dict[str, Any]]]] = []
    total_steps = 0
    for ep_dir in episodes:
        try:
            meta, steps = _load_episode(ep_dir)
        except Exception as e:
            logger.warning("Failed to load %s: %s", ep_dir, e)
            continue
        if not steps:
            continue
        loaded.append((ep_dir, meta, steps))
        total_steps += len(steps)
    logger.info(
        "Total (episode, step) pairs to process: %d across %d episodes",
        total_steps, len(loaded),
    )

    metrics: dict[str, Any] = {
        "total_episodes": len(loaded),
        "total_steps": total_steps,
        "epoch_losses": [],
    }

    for epoch in range(1, int(epochs) + 1):
        logger.info("=== Epoch %d/%d ===", epoch, epochs)
        ep_indices = list(range(len(loaded)))
        rng.shuffle(ep_indices)

        running_loss = 0.0
        running_n = 0
        batch_loss_t = None
        batch_count_in_step = 0

        t_epoch = time.monotonic()
        for ep_i in ep_indices:
            ep_dir, meta, steps = loaded[ep_i]
            target_value = float(meta.get("success", 0.0))
            # Binary {0, 1} target — even partial success 0.5 is allowed
            # (the agent records it but at MC return interpretation it
            # is still a scalar regression target).
            target_t = torch.tensor(
                [target_value], device=resolved_device, dtype=torch.float32,
            )

            goal = steps[0].get("goal", "") if steps else ""
            history_summaries: list[str] = []

            for step in steps:
                # Build the prompt as the agent would have seen it.
                prompt_text = _build_step_prompt(
                    goal=goal,
                    history_summaries=history_summaries,
                    before_ui_elements_text=step.get(
                        "before_ui_elements_text", "",
                    ),
                )
                # Load the before-screenshot PNG.
                png_rel = step.get("before_screenshot_path")
                if not png_rel:
                    # Some older traces omit this; skip silently.
                    continue
                png_path = ep_dir / png_rel
                if not png_path.exists():
                    continue
                try:
                    raw_pil = Image.open(png_path).convert("RGB")
                except Exception as e:
                    logger.warning("Failed to open %s: %s", png_path, e)
                    continue
                som_pil = _draw_som_screenshot(raw_pil, None, None)

                try:
                    v_pred = _forward_value_head(
                        model, value_head, processor,
                        prompt_text, raw_pil, som_pil,
                        device=resolved_device,
                    )
                except Exception:
                    logger.exception(
                        "Value-head forward failed at %s step %d",
                        ep_dir.name, step.get("step", -1),
                    )
                    continue

                # MSE loss step-by-step. Targets are constant within a
                # trajectory (= terminal success), which means the head
                # learns a state-conditional Monte-Carlo return.
                step_loss = torch.nn.functional.mse_loss(v_pred, target_t)

                if batch_loss_t is None:
                    batch_loss_t = step_loss
                else:
                    batch_loss_t = batch_loss_t + step_loss
                batch_count_in_step += 1

                running_loss += float(step_loss.detach().item())
                running_n += 1

                if batch_count_in_step >= batch_size:
                    avg_batch_loss = batch_loss_t / batch_count_in_step
                    optimizer.zero_grad()
                    avg_batch_loss.backward()
                    optimizer.step()
                    batch_loss_t = None
                    batch_count_in_step = 0
                    if running_n % log_every == 0 or running_n == 1:
                        logger.info(
                            "epoch=%d step=%d running_mean_loss=%.4f "
                            "last_target=%.2f last_pred=%.3f",
                            epoch, running_n,
                            running_loss / max(1, running_n),
                            target_value, float(v_pred.detach().item()),
                        )

                # Update history with the actual summary the agent wrote.
                summary = step.get("summary") or ""
                if summary:
                    history_summaries.append(summary)

                if torch.cuda.is_available() and resolved_device.startswith("cuda"):
                    torch.cuda.empty_cache()

        # Flush remaining partial batch at end of epoch.
        if batch_loss_t is not None and batch_count_in_step > 0:
            avg_batch_loss = batch_loss_t / batch_count_in_step
            optimizer.zero_grad()
            avg_batch_loss.backward()
            optimizer.step()

        epoch_loss = running_loss / max(1, running_n)
        wall = time.monotonic() - t_epoch
        logger.info(
            "Epoch %d done: mean_loss=%.4f (n=%d) wall=%.1fs",
            epoch, epoch_loss, running_n, wall,
        )
        metrics["epoch_losses"].append({
            "epoch": epoch,
            "mean_loss": epoch_loss,
            "n_steps": running_n,
            "wall_seconds": wall,
        })

        # Save after each epoch so partial runs are recoverable.
        save_value_head(value_head, output_path)
        logger.info("Saved value head -> %s", output_path)

    return metrics


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description="Pretrain the PPO linear value head via MSE on "
                    "trajectory-level returns (binary 0/1 = success).",
    )
    p.add_argument(
        "--data-dirs", nargs="+", required=True, type=Path,
        help="One or more directories containing episode/meta.json data. "
             "Walks recursively for any meta.json+episode.jsonl pairs. "
             "Typical: traces/source_pool_trajectories and the per-app "
             "traces/b4_k4_unified/<app>/episodes/.",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Output path for the value-head state_dict. E.g. "
             "EvoFSM-RL/artifacts/value_head_v01.pt",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Number of per-step losses to accumulate before "
                        "each optimizer step.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", choices=("mps", "cuda", "cpu"), default=None)
    p.add_argument(
        "--max-traj", type=int, default=None,
        help="If set, sample at most N trajectories (testing).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for trajectory shuffle order.",
    )
    p.add_argument(
        "--log-every", type=int, default=20,
        help="Log running mean loss every N steps.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    metrics = train_value_head(
        data_dirs=args.data_dirs,
        output_path=args.output,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        max_traj=args.max_traj,
        seed=args.seed,
        log_every=args.log_every,
    )

    print()
    print("=" * 64)
    print("Value-head pretraining complete.")
    print(f"  output:           {args.output}")
    print(f"  episodes used:    {metrics['total_episodes']}")
    print(f"  total steps:      {metrics['total_steps']}")
    for entry in metrics["epoch_losses"]:
        print(
            f"  epoch {entry['epoch']:2d}: mean_loss={entry['mean_loss']:.4f} "
            f"n={entry['n_steps']} ({entry['wall_seconds']:.1f}s)"
        )
    print("=" * 64)

    # Persist a sidecar JSON with the pretraining metrics so we can
    # inspect convergence offline without re-running.
    sidecar = args.output.with_suffix(args.output.suffix + ".meta.json")
    sidecar.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics sidecar -> {sidecar}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

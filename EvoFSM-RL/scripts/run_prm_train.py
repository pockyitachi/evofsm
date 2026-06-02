#!/usr/bin/env python3
"""Train a Process Reward Model (PRM) on Sonnet step-level labels.

The PRM is a ``Linear(hidden_size → 1)`` head on the frozen base model.
Inputs mirror what Sonnet saw at labeling time:
  - GOAL (string)
  - BEFORE screenshot + UI elements text
  - ACTION JSON + agent reasoning
  - AFTER screenshot + UI elements text
  - Agent summary

Training target: the Sonnet step score (float ∈ {0, 0.25, 0.5, 0.75, 1.0}).
Loss: MSE on PRM(input) vs Sonnet score.

At PPO time, the PRM is queried after every step in a rollout to produce
a dense per-step reward signal r_t = PRM(before, action, after, ...) ∈ [0, 1].

Usage::

    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      python EvoFSM-RL/scripts/run_prm_train.py \\
        --labels EvoFSM-RL/data/step_labels/source_pool_sonnet.jsonl \\
                 EvoFSM-RL/data/step_labels/b4_k4_unified_sonnet.jsonl \\
        --epochs 3 \\
        --batch-size 4 \\
        --lr 1e-3 \\
        --output EvoFSM-RL/artifacts/prm_v01.pt
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

logger = logging.getLogger("run_prm_train")


# ── Prompt construction (matches Sonnet's labeling prompt structure) ─
PROMPT_TEMPLATE = """Goal: {goal}

BEFORE action:
[image 1 above]
UI elements: {before_ui}

Action taken: {action}
Reasoning: {action_reason}

AFTER action:
[image 2 above]
UI elements: {after_ui}
Reflection: {summary}

Evaluate this step's progress toward the goal."""


def _build_prm_messages(prompt_text: str, before_pil, after_pil) -> list[dict]:
    """Same format as evofsm_rl.agent.prompts.build_action_messages but with
    [before, after] images. Sonnet saw the images in this order at label time."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": before_pil},
                {"type": "image", "image": after_pil},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


# ── Label loading ────────────────────────────────────────────────────
def _load_labels(label_paths: list[Path]) -> list[dict]:
    """Load all valid labels from JSONL files, filtering out errors + ambiguous."""
    examples = []
    for p in label_paths:
        with p.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("judge_error") or rec.get("judge_parse_error"):
                    continue
                score = rec.get("judge_score")
                if score == "ambiguous" or score is None:
                    continue
                # Verify required fields + paths exist
                if not all(rec.get(k) is not None for k in (
                    "before_screenshot_path", "after_screenshot_path",
                    "goal", "action", "action_reason",
                    "before_ui_elements_truncated", "after_ui_elements_truncated",
                    "agent_summary",
                )):
                    continue
                if not Path(rec["before_screenshot_path"]).exists():
                    continue
                if not Path(rec["after_screenshot_path"]).exists():
                    continue
                examples.append({
                    "before_screenshot_path": rec["before_screenshot_path"],
                    "after_screenshot_path": rec["after_screenshot_path"],
                    "before_ui": rec["before_ui_elements_truncated"][:2000],
                    "after_ui": rec["after_ui_elements_truncated"][:2000],
                    "goal": rec["goal"],
                    "action": rec["action"],
                    "action_reason": rec["action_reason"],
                    "agent_summary": rec["agent_summary"],
                    "score": float(score),
                    "trajectory_id": rec.get("trajectory_id", "?"),
                    "step_idx": rec.get("step_idx", -1),
                    "app": rec.get("app", "?"),
                })
    return examples


# ── Forward pass through frozen base + PRM head ──────────────────────
def _prm_forward(model, processor, prm_head, ex: dict, device: str):
    """Build prompt, forward through frozen base, return PRM scalar."""
    import torch
    from PIL import Image

    before_pil = Image.open(ex["before_screenshot_path"]).convert("RGB")
    after_pil = Image.open(ex["after_screenshot_path"]).convert("RGB")
    prompt_text = PROMPT_TEMPLATE.format(
        goal=ex["goal"],
        before_ui=ex["before_ui"],
        after_ui=ex["after_ui"],
        action=json.dumps(ex["action"]),
        action_reason=ex["action_reason"],
        summary=ex["agent_summary"],
    )
    messages = _build_prm_messages(prompt_text, before_pil, after_pil)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[before_pil, after_pil],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Detach so only the head receives grad
    hidden = outputs.hidden_states[-1].detach()
    score = prm_head(hidden)  # shape (1,)
    return score


# ── Main training loop ───────────────────────────────────────────────
def train_prm(
    examples: list[dict],
    base_model,
    processor,
    prm_head,
    optimizer,
    *,
    epochs: int,
    device: str,
    output_path: Path,
    checkpoint_every: int = 1000,
    log_every: int = 50,
) -> dict:
    import torch

    n_total = len(examples)
    logger.info("Training: %d examples × %d epochs = %d total steps",
                n_total, epochs, n_total * epochs)

    base_model.eval()  # frozen
    prm_head.train()

    step = 0
    running_loss = 0.0
    running_n = 0
    t_start = time.monotonic()
    epoch_summaries = []

    for epoch in range(1, epochs + 1):
        random.shuffle(examples)
        epoch_loss_sum = 0.0
        epoch_n = 0
        skipped_in_epoch = 0
        for ex in examples:
            try:
                pred = _prm_forward(base_model, processor, prm_head, ex, device)
            except Exception as e:
                logger.warning("Skip example %s step %d: %s",
                               ex.get("trajectory_id"), ex.get("step_idx"), e)
                skipped_in_epoch += 1
                continue

            target = torch.tensor([ex["score"]], dtype=pred.dtype, device=pred.device)
            loss = torch.nn.functional.mse_loss(pred, target)
            loss_val = float(loss.item())

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(prm_head.parameters(), max_norm=10.0)
            optimizer.step()

            step += 1
            running_loss += loss_val
            running_n += 1
            epoch_loss_sum += loss_val
            epoch_n += 1

            if step % log_every == 0:
                mean_running = running_loss / max(running_n, 1)
                elapsed = time.monotonic() - t_start
                logger.info(
                    "epoch=%d step=%d running_mean_loss=%.4f last_target=%.2f last_pred=%.3f grad=%.2f wall=%.1f min",
                    epoch, step, mean_running, ex["score"], float(pred.detach().item()),
                    float(grad_norm), elapsed / 60,
                )
                running_loss = 0.0
                running_n = 0

            if step % checkpoint_every == 0:
                ckpt_path = output_path.with_name(f"{output_path.stem}_step{step:06d}{output_path.suffix}")
                torch.save({"state_dict": prm_head.state_dict()}, ckpt_path)
                logger.info("Saved checkpoint: %s", ckpt_path)

        epoch_mean = epoch_loss_sum / max(epoch_n, 1)
        epoch_summaries.append({
            "epoch": epoch,
            "n": epoch_n,
            "skipped": skipped_in_epoch,
            "mean_loss": epoch_mean,
        })
        logger.info("=== Epoch %d done: n=%d skipped=%d mean_loss=%.4f ===",
                    epoch, epoch_n, skipped_in_epoch, epoch_mean)

    # Save final
    torch.save({"state_dict": prm_head.state_dict()}, output_path)
    logger.info("Saved final PRM head: %s", output_path)

    return {
        "n_examples_total": n_total,
        "epochs": epochs,
        "epoch_summaries": epoch_summaries,
        "total_steps": step,
        "wall_min": (time.monotonic() - t_start) / 60,
        "final_checkpoint": str(output_path),
    }


# ── CLI ──────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, nargs="+", required=True,
                   help="Path(s) to JSONL label files (e.g. source_pool_sonnet.jsonl).")
    p.add_argument("--output", type=Path, required=True,
                   help="Output .pt path for trained PRM head state_dict.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Forward batch size. Current impl is per-sample (batch=1) "
                        "because of variable image sizes — batching needs padding.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-examples", type=int, default=None,
                   help="If set, cap training set size (for smoke testing).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    random.seed(args.seed)

    import torch
    from evofsm_rl.model import load_base_model, resolve_device
    from evofsm_rl.rl_ppo.value_head import attach_value_head  # reused architecture

    resolved_device = args.device or resolve_device()
    logger.info("Loading base Qwen3-VL on device=%s", resolved_device)
    t0 = time.monotonic()
    model, processor = load_base_model(device=resolved_device)
    logger.info("Base model loaded in %.1fs", time.monotonic() - t0)

    # PRM head architecture identical to value head — just different training target
    prm_head = attach_value_head(model, resolved_device)
    logger.info("PRM head attached: %d trainable params",
                sum(p.numel() for p in prm_head.parameters()))

    optimizer = torch.optim.AdamW(
        prm_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    examples = _load_labels(args.labels)
    logger.info("Loaded %d valid labeled examples from %d file(s)",
                len(examples), len(args.labels))
    if args.max_examples is not None:
        random.shuffle(examples)
        examples = examples[: args.max_examples]
        logger.info("Capped to %d examples (--max-examples).", len(examples))

    # Save config for reproducibility
    run_log = {
        "args": vars(args),
        "n_examples": len(examples),
        "label_files": [str(p) for p in args.labels],
    }
    run_log_str = json.dumps(run_log, default=str, indent=2)
    with args.output.with_suffix(".run_log.json").open("w") as f:
        f.write(run_log_str)

    metrics = train_prm(
        examples=examples,
        base_model=model,
        processor=processor,
        prm_head=prm_head,
        optimizer=optimizer,
        epochs=args.epochs,
        device=resolved_device,
        output_path=args.output,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
    )

    # Save sidecar metrics
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    with meta_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics sidecar -> %s", meta_path)

    print()
    print("=" * 60)
    print("PRM training complete.")
    print(f"  output:           {args.output}")
    print(f"  examples used:    {metrics['n_examples_total']}")
    for s in metrics["epoch_summaries"]:
        print(f"  epoch {s['epoch']:2d}: mean_loss={s['mean_loss']:.4f} n={s['n']} skipped={s['skipped']}")
    print(f"  wall:             {metrics['wall_min']:.1f} min")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

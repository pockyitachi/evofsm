#!/usr/bin/env python3
"""Rejection Fine-Tuning (RFT) baseline for EvoFSM-RL.

Plain SFT a LoRA on Qwen3-VL-8B over **only the successful trajectories**
from the source pool. No reward shaping, no FSM, no L_C — just imitation
learning on the M3A action-selection turn.

Source data
-----------
``EvoFSM-RL/traces/source_pool_trajectories/`` — 480 per-episode dirs
written by ``scripts/baseline_10task.py``. Filter to
``meta.json["success"] == 1.0``; from each successful episode pull one
SFT example per ``episode.jsonl`` step where ``action_raw_response`` is
present and ``parse_error`` is None.

Per example
-----------
* Input  = M3A action-selection prompt
  (``build_action_prompt(goal, history_lines, before_ui_elements_text)``)
  packaged via ``build_action_messages(prompt, raw_pil, raw_pil)``.
  **IMPORTANT**: SoM-annotated screenshots were NOT saved by the
  trajectory collector (we only have ``step_N_before.png`` = raw).
  So we feed the raw screenshot in BOTH image slots. At eval, the agent
  passes ``[raw, som_annotated]``, so there is a slight distribution
  mismatch — we accept this as the cheapest viable RFT path.
* Target = ``action_raw_response`` (the literal text "Reason: …\\nAction: …").
* Loss = next-token cross-entropy over the target token positions only;
  prompt tokens are label-masked with -100.

Hyperparameters (calibrated to match prior LoRA runs)
-----------------------------------------------------
* LoRA rank=16, alpha=32, dropout=0.05, targets ``("q_proj", "v_proj")``
* AdamW lr=3e-4, weight_decay=0.01
* batch_size=1 (each example has 2 images + long text), epochs=2
* gradient checkpointing ON
* Save adapter every --checkpoint-every steps + at end

Usage::

    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      CUDA_VISIBLE_DEVICES=5 \\
      nohup python EvoFSM-RL/scripts/run_rft.py \\
        --source-pool EvoFSM-RL/traces/source_pool_trajectories \\
        --output-dir EvoFSM-RL/traces/rft_v01 \\
        --epochs 2 \\
        --checkpoint-every 500 \\
        > EvoFSM-RL/traces/rft_v01/train.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("run_rft")


# ─────────────────────────────────────────────────────────────────────
# Dataset assembly
# ─────────────────────────────────────────────────────────────────────


def _scan_successful_examples(
    source_pool: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Walk source_pool dirs and collect one example per successful step.

    Returns (examples, stats) where each example is::

        {
            "episode_dir": Path,
            "step_idx": int,
            "goal": str,
            "ui_elements_text": str,
            "history_lines": list[str],   # prior step summaries
            "target": str,                # action_raw_response
            "before_screenshot_path": Path,
        }

    A step is included iff all of:
      - meta.json["success"] == 1.0
      - episode.jsonl row has action_raw_response (not None, non-empty)
      - parse_error is None / falsy
      - before_screenshot_path file exists on disk
    """
    examples: list[dict[str, Any]] = []
    stats = {
        "n_episodes_total": 0,
        "n_episodes_success": 0,
        "n_steps_total": 0,
        "n_steps_kept": 0,
        "n_steps_skip_no_raw": 0,
        "n_steps_skip_parse_err": 0,
        "n_steps_skip_missing_img": 0,
    }

    for ep_dir in sorted(source_pool.iterdir()):
        if not ep_dir.is_dir():
            continue
        meta_p = ep_dir / "meta.json"
        ep_p = ep_dir / "episode.jsonl"
        if not (meta_p.exists() and ep_p.exists()):
            continue
        stats["n_episodes_total"] += 1
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            continue
        if float(meta.get("success", 0.0)) != 1.0:
            continue
        stats["n_episodes_success"] += 1

        # Re-build history from prior step summaries as the agent would
        # have rendered it at action-selection time.
        prior_summaries: list[str] = []
        with ep_p.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                stats["n_steps_total"] += 1
                row = json.loads(line)
                step_idx = int(row.get("step", 0))
                goal = row.get("goal") or ""
                ui_text = row.get("before_screenshot_path")  # placeholder; corrected below
                ui_elements_text = row.get("before_ui_elements_text") or ""
                target = row.get("action_raw_response")
                parse_err = row.get("parse_error")
                before_img_rel = row.get("before_screenshot_path")
                summary = row.get("summary") or ""

                history_for_this_step = list(prior_summaries)

                # Update history AFTER snapshotting (next step uses this one's summary)
                prior_summaries.append(
                    f"Step {step_idx}- {summary}" if summary else f"Step {step_idx}- (no summary)"
                )

                if not target or not isinstance(target, str):
                    stats["n_steps_skip_no_raw"] += 1
                    continue
                if parse_err:
                    stats["n_steps_skip_parse_err"] += 1
                    continue
                if not before_img_rel:
                    stats["n_steps_skip_missing_img"] += 1
                    continue
                img_path = ep_dir / before_img_rel
                if not img_path.exists():
                    stats["n_steps_skip_missing_img"] += 1
                    continue

                # Render history exactly as the agent did: lines like
                # "Step N- <summary>" (one per prior step). build_action_prompt
                # joins this list with newlines internally.
                history_lines = [
                    f"Step {i + 1}- {s}"
                    for i, s in enumerate(
                        [h.split("- ", 1)[1] if "- " in h else h
                         for h in history_for_this_step]
                    )
                ] if history_for_this_step else []

                examples.append({
                    "episode_dir": ep_dir,
                    "step_idx": step_idx,
                    "goal": goal,
                    "ui_elements_text": ui_elements_text,
                    "history_lines": history_lines,
                    "target": target,
                    "before_screenshot_path": img_path,
                })
                stats["n_steps_kept"] += 1

    return examples, stats


# ─────────────────────────────────────────────────────────────────────
# Single-step training fn
# ─────────────────────────────────────────────────────────────────────


def _build_labeled_inputs(
    example: dict[str, Any],
    processor: Any,
    device: str,
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    """Render one SFT example to (input_ids, labels, image kwargs).

    Returns a dict with everything needed for one ``model(**kwargs)`` call:
    ``input_ids``, ``attention_mask``, ``labels``, ``pixel_values``,
    ``image_grid_thw``, and any other field the processor emits.

    The target tokens (action_raw_response + EOS) are appended after the
    rendered prompt; prompt tokens have label = -100 so loss is computed
    only on target positions.
    """
    import torch
    from PIL import Image

    from evofsm_rl.agent.prompts import build_action_messages, build_action_prompt

    prompt_text = build_action_prompt(
        goal=example["goal"],
        history=example["history_lines"],
        ui_elements=example["ui_elements_text"],
        additional_guidelines=None,
        l_c_prompt_text=None,
    )

    raw_pil = Image.open(example["before_screenshot_path"]).convert("RGB")
    # Distribution-mismatch workaround: source_pool didn't save SoM
    # screenshots; feed raw in both slots. Documented in module docstring.
    messages = build_action_messages(prompt_text, raw_pil, raw_pil)

    # 1) Render prompt with add_generation_prompt=True — this is what the
    #    agent feeds to .generate() at inference time. The processor adds
    #    the trailing "<|im_start|>assistant\n" so the next token belongs
    #    to the model.
    prompt_str = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_inputs = processor(
        text=[prompt_str], images=[raw_pil, raw_pil],
        return_tensors="pt", padding=True,
    )
    prompt_ids = prompt_inputs["input_ids"]
    prompt_len = int(prompt_ids.shape[-1])

    # 2) Tokenize the target text WITHOUT special tokens (the chat-template
    #    already added the assistant role open token). Append EOS so the
    #    model learns to stop.
    tokenizer = getattr(processor, "tokenizer", processor)
    eos_id = tokenizer.eos_token_id
    target_text = example["target"]
    target_ids = tokenizer(
        target_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    if eos_id is not None:
        target_ids = torch.cat(
            [target_ids, torch.tensor([[eos_id]], dtype=target_ids.dtype)], dim=-1,
        )
    target_len = int(target_ids.shape[-1])

    # 3) Concatenate to a single sequence.
    full_ids = torch.cat([prompt_ids, target_ids], dim=-1)
    full_len = int(full_ids.shape[-1])

    # 4) Labels: -100 on prompt positions, target_ids on target positions.
    labels = full_ids.clone()
    labels[:, :prompt_len] = -100

    # 5) Attention mask: 1 everywhere (no padding in a batch of 1).
    attn_mask = torch.ones_like(full_ids)

    # 6) Pass through every other field the processor emitted (pixel_values,
    #    image_grid_thw, mm_token_type_ids if present, ...). Per-token fields
    #    of length prompt_len need to be extended by target_len with zeros
    #    (target tokens are text, no image content).
    out_kwargs: dict[str, Any] = {
        "input_ids": full_ids.to(device),
        "attention_mask": attn_mask.to(device),
        "labels": labels.to(device),
    }
    for key, val in prompt_inputs.items():
        if key in ("input_ids", "attention_mask"):
            continue
        if not isinstance(val, torch.Tensor):
            continue
        # Per-token fields whose last dim equals prompt_len: extend to full_len.
        if val.dim() >= 2 and val.shape[-1] == prompt_len:
            pad_shape = list(val.shape)
            pad_shape[-1] = full_len - prompt_len
            pad = torch.zeros(pad_shape, dtype=val.dtype)
            val = torch.cat([val, pad], dim=-1)
        out_kwargs[key] = val.to(device)

    return out_kwargs


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--source-pool", type=Path,
        default=Path("EvoFSM-RL/traces/source_pool_trajectories"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("EvoFSM-RL/traces/rft_v01"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Per-device batch size. Must be 1 (variable seq lens).")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", type=str, default="q_proj,v_proj")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on examples (for smoke-testing only).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    if args.batch_size != 1:
        logger.error("Only batch_size=1 supported (variable seq lens with images).")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "run_log.json"
    metrics_path = args.output_dir / "train_metrics.jsonl"
    ckpt_root = args.output_dir / "lora_checkpoints"
    ckpt_root.mkdir(exist_ok=True)

    logger.info("Scanning %s for successful trajectories ...", args.source_pool)
    examples, stats = _scan_successful_examples(args.source_pool)
    logger.info("Scan stats: %s", json.dumps(stats, indent=2))
    if not examples:
        logger.error("No SFT examples found — abort.")
        return 2

    if args.limit:
        examples = examples[: args.limit]
        logger.info("Capped to %d examples (--limit).", len(examples))

    rng = random.Random(args.seed)

    # Persist scan stats up front so a crashed run still has the dataset description.
    log_path.write_text(json.dumps(
        {
            "args": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()},
            "scan_stats": stats,
            "n_examples": len(examples),
            "start_time": time.time(),
        },
        indent=2,
    ))

    # ── Load model + attach LoRA ────────────────────────────────────
    from evofsm_rl.model import load_base_model, load_model_config, resolve_device
    from evofsm_rl.model.lora import (
        attach_lora,
        count_trainable_params,
        save_lora_checkpoint,
    )

    device = args.device or resolve_device()
    logger.info("Loading Qwen3-VL-8B on device=%s", device)
    t0 = time.monotonic()
    model, processor = load_base_model(device=device)
    logger.info("Model loaded in %.1fs", time.monotonic() - t0)

    target_modules = tuple(
        m.strip() for m in args.lora_target_modules.split(",") if m.strip()
    )
    model = attach_lora(
        model,
        rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    counts = count_trainable_params(model)
    logger.info(
        "LoRA attached: trainable=%d / total=%d (%.3f%%)",
        counts["trainable"], counts["total"], counts["percent"],
    )

    # Gradient checkpointing — same rationale as B4 sweep.
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        logger.info("Gradient checkpointing enabled (use_reentrant=False)")
    except Exception as e:
        logger.warning("Failed to enable gradient checkpointing: %s", e)

    model.train()

    # ── Optimizer ──────────────────────────────────────────────────
    import torch
    from torch.optim import AdamW

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
    )

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(
        tokenizer, "eos_token_id", 0,
    )

    # ── Training loop ──────────────────────────────────────────────
    metrics_fh = metrics_path.open("w")
    total_examples = len(examples)
    total_steps = total_examples * args.epochs
    step = 0
    n_skipped = 0
    losses_window: list[float] = []
    first_losses: list[float] = []
    last_losses: list[float] = []
    t_train_start = time.monotonic()

    logger.info(
        "Training: %d examples × %d epochs = %d optimizer steps; lr=%.2e",
        total_examples, args.epochs, total_steps, args.lr,
    )

    for epoch in range(args.epochs):
        rng.shuffle(examples)
        for ex_idx, example in enumerate(examples):
            step += 1
            try:
                fwd_kwargs = _build_labeled_inputs(
                    example, processor, device, pad_token_id=pad_id,
                )
            except Exception as e:
                logger.warning(
                    "Skipped example (build_labeled_inputs failed): %s "
                    "ep=%s step=%d err=%r",
                    example["episode_dir"].name, example["step_idx"], step, e,
                )
                n_skipped += 1
                continue

            try:
                outputs = model(**fwd_kwargs)
                loss = outputs.loss
                if loss is None or not torch.isfinite(loss):
                    logger.warning(
                        "Non-finite loss at step %d (ep=%s step=%d); skipping.",
                        step, example["episode_dir"].name, example["step_idx"],
                    )
                    n_skipped += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, args.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            except torch.cuda.OutOfMemoryError as e:
                logger.warning(
                    "OOM at step %d (full_len=%d); skipping. %s",
                    step, fwd_kwargs["input_ids"].shape[-1], e,
                )
                n_skipped += 1
                optimizer.zero_grad(set_to_none=True)
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
            except Exception:
                logger.exception(
                    "Training step %d crashed — skipping.", step,
                )
                n_skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            loss_val = float(loss.item())
            grad_val = float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)
            losses_window.append(loss_val)
            if len(first_losses) < 50:
                first_losses.append(loss_val)
            last_losses.append(loss_val)
            if len(last_losses) > 50:
                last_losses = last_losses[-50:]

            metrics_fh.write(json.dumps({
                "step": step, "epoch": epoch, "loss": loss_val,
                "grad_norm": grad_val,
                "seq_len": int(fwd_kwargs["input_ids"].shape[-1]),
            }) + "\n")
            metrics_fh.flush()

            if step % args.log_every == 0:
                avg = sum(losses_window) / len(losses_window)
                losses_window = []
                logger.info(
                    "[ep %d/%d step %d/%d] loss=%.4f grad=%.3f seq_len=%d wall=%.1fmin",
                    epoch + 1, args.epochs, step, total_steps,
                    avg, grad_val, fwd_kwargs["input_ids"].shape[-1],
                    (time.monotonic() - t_train_start) / 60,
                )

            if step % args.checkpoint_every == 0:
                ckpt_path = ckpt_root / f"step_{step:06d}"
                save_lora_checkpoint(model, ckpt_path)
                logger.info("Saved checkpoint: %s", ckpt_path)

            # Cheap memory hygiene every few hundred steps.
            if device == "cuda" and step % 100 == 0:
                torch.cuda.empty_cache()

    final_path = ckpt_root / "final"
    save_lora_checkpoint(model, final_path)
    logger.info("Saved final LoRA: %s", final_path)

    wall_train = (time.monotonic() - t_train_start) / 60
    first_avg = sum(first_losses) / len(first_losses) if first_losses else None
    last_avg = sum(last_losses) / len(last_losses) if last_losses else None

    summary = {
        "n_examples_total": total_examples,
        "epochs": args.epochs,
        "n_steps_attempted": step,
        "n_steps_skipped": n_skipped,
        "first_50_loss_avg": first_avg,
        "last_50_loss_avg": last_avg,
        "wall_train_min": round(wall_train, 1),
        "final_checkpoint": str(final_path),
    }
    summary_path = args.output_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Train summary:\n%s", json.dumps(summary, indent=2))

    metrics_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

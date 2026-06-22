#!/usr/bin/env python3
"""Step-level reward labeling via Claude Sonnet 4.5.

For each step in each successful/failed trajectory, send Claude:
  - The episode goal
  - Before & after screenshots
  - Before & after UI elements (truncated)
  - The action taken + agent's reasoning + agent's reflection

Claude returns a [0, 1] scalar progress score + brief reason.

Designed for resumable execution: output is appended to a JSONL file;
already-labeled (trajectory_id, step_idx) pairs are skipped.

Concurrency: a ThreadPoolExecutor sends N concurrent requests to Anthropic;
defaults to 10. Each request takes ~3s; 5012 steps × 3s / 10 ≈ 25 min.

Usage:

    python EvoFSM-RL/scripts/label_step_rewards.py \\
        --trajectories EvoFSM-RL/traces/source_pool_trajectories \\
        --output EvoFSM-RL/data/step_labels/source_pool_sonnet.jsonl \\
        --model claude-sonnet-4-5-20250929 \\
        --concurrency 10 \\
        --limit-steps 20   # for testing; omit for full sweep
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger("label_step_rewards")


# ── Prompt template ──────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are evaluating one step of a GUI agent's trajectory toward a goal.

GOAL: {goal}

BEFORE-ACTION STATE:
[image 1 above: before screenshot]
UI elements (truncated):
{before_ui}

ACTION TAKEN: {action}
AGENT REASONING: {action_reason}

AFTER-ACTION STATE:
[image 2 above: after screenshot]
UI elements (truncated):
{after_ui}
Agent's own reflection: {summary}

TASK: Did this step make progress toward the goal?

Score guide:
- 1.0  : Clear progress (action correctly advances toward goal)
- 0.75 : Likely progress (action plausibly contributes)
- 0.5  : Neutral (valid action but no clear advance)
- 0.25 : Likely regress (action seems off-track)
- 0.0  : Clear regress or error (wrong action, error, stuck-loop)
- ambiguous : Insufficient information to decide

Respond in EXACTLY this format (no extra text):
SCORE: <0.0|0.25|0.5|0.75|1.0|ambiguous>
REASON: <one sentence>
"""

VALID_SCORES = {"0.0", "0.25", "0.5", "0.75", "1.0", "ambiguous"}

# Truncate UI elements to this many chars per side
UI_TRUNC_CHARS = 2000


# ── Helpers ──────────────────────────────────────────────────────────
def _png_to_b64(path: Path) -> str:
    with path.open("rb") as f:
        return base64.standard_b64encode(f.read()).decode("ascii")


def _strip_action_prefix(summary: str) -> str:
    """The agent's summary always starts with `Action selected: {...}\n` which
    is just a copy of the action JSON. Drop it to save input tokens."""
    if summary.startswith("Action selected:"):
        # Drop until first newline (or two newlines for safety)
        idx = summary.find("\n")
        if idx != -1:
            return summary[idx + 1 :].lstrip()
    return summary


def _parse_response(text: str) -> tuple[str | float, str]:
    """Parse SCORE: <x>  REASON: <y> response. Returns (score, reason).

    score is a float in {0, 0.25, 0.5, 0.75, 1.0} OR the string "ambiguous".
    """
    score_m = re.search(r"SCORE:\s*(\S+)", text)
    reason_m = re.search(r"REASON:\s*(.+?)(?:\n|$)", text, re.S)
    if not score_m or not reason_m:
        raise ValueError(f"Could not parse SCORE / REASON in response: {text!r}")
    raw = score_m.group(1).strip().rstrip(".")
    if raw not in VALID_SCORES:
        # Some lenience: strip trailing punctuation, try once more
        cleaned = raw.rstrip(",.")
        if cleaned not in VALID_SCORES:
            raise ValueError(f"Invalid score {raw!r} (expected one of {VALID_SCORES})")
        raw = cleaned
    if raw == "ambiguous":
        score: str | float = "ambiguous"
    else:
        score = float(raw)
    return score, reason_m.group(1).strip()


# ── Anthropic call ───────────────────────────────────────────────────
def _call_claude(client, model: str, before_b64: str, after_b64: str, prompt: str, max_retries: int = 4) -> dict:
    """Send a single labeling request to Claude. Retry on rate limits."""
    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.monotonic()
            resp = client.messages.create(
                model=model,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/png", "data": before_b64,
                        }},
                        {"type": "image", "source": {
                            "type": "base64", "media_type": "image/png", "data": after_b64,
                        }},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            wall = time.monotonic() - t0
            return {
                "raw_text": resp.content[0].text,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "wall_s": wall,
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e).lower()
            # Retry on rate-limit / 5xx / transient
            if "rate" in msg or "overload" in msg or "503" in msg or "529" in msg:
                backoff = 2 ** attempt + (attempt * 0.5)
                logger.warning("Claude transient error (attempt %d/%d): %s — backing off %.1fs",
                               attempt + 1, max_retries, e, backoff)
                time.sleep(backoff)
                continue
            # Non-retriable
            raise
    raise RuntimeError(f"Exhausted retries; last error: {last_err}")


# ── Step iteration ───────────────────────────────────────────────────
def _iter_steps(trajectories_dir: Path) -> list[dict]:
    """Yield one dict per (trajectory, step). Each dict carries everything
    needed to build the prompt + the output record fields.

    Supports two on-disk layouts:
      * Flat: ``trajectories_dir/<ep_id>/{meta.json,episode.jsonl,*.png}``
        — used by ``source_pool_trajectories``.
      * Per-app nested: ``trajectories_dir/<app>/episodes/<ep_id>/...``
        — used by B4 sweep dirs like ``b4_k4_unified``.
    """
    # Collect candidate episode directories. Auto-detect layout by trying
    # flat first and falling back to nested.
    ep_dirs: list[Path] = []
    for child in sorted(trajectories_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "meta.json").exists() and (child / "episode.jsonl").exists():
            ep_dirs.append(child)
        elif (child / "episodes").is_dir():
            for ep in sorted((child / "episodes").iterdir()):
                if ep.is_dir() and (ep / "meta.json").exists() and (ep / "episode.jsonl").exists():
                    ep_dirs.append(ep)

    out = []
    for ep_dir in ep_dirs:
        meta_p = ep_dir / "meta.json"
        jsonl_p = ep_dir / "episode.jsonl"
        if not meta_p.exists() or not jsonl_p.exists():
            continue
        try:
            with meta_p.open() as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skip %s: meta unreadable: %s", ep_dir.name, e)
            continue
        with jsonl_p.open() as f:
            steps = [json.loads(line) for line in f if line.strip()]
        trajectory_success = meta.get("success")
        n_steps = meta.get("n_steps", len(steps))
        for step_idx, step in enumerate(steps):
            # require all fields present AND non-None
            if not all(step.get(k) is not None for k in (
                "goal", "action", "action_reason", "summary",
                "before_ui_elements_text", "after_ui_elements_text",
                "before_screenshot_path", "after_screenshot_path",
            )):
                continue
            # Resolve screenshot paths (they're stored as basenames)
            before_p = ep_dir / Path(step["before_screenshot_path"]).name
            after_p = ep_dir / Path(step["after_screenshot_path"]).name
            if not before_p.exists() or not after_p.exists():
                continue
            out.append({
                "trajectory_id": ep_dir.name,
                "step_idx": step_idx,
                "app": meta.get("app"),
                "template": meta.get("template"),
                "goal": step["goal"],
                "trajectory_success": trajectory_success,
                "trajectory_n_steps": n_steps,
                "action": step["action"],
                "action_reason": step["action_reason"],
                "agent_summary": _strip_action_prefix(step["summary"]),
                "before_ui_elements_truncated": step["before_ui_elements_text"][:UI_TRUNC_CHARS],
                "after_ui_elements_truncated": step["after_ui_elements_text"][:UI_TRUNC_CHARS],
                "before_screenshot_path": str(before_p),
                "after_screenshot_path": str(after_p),
            })
    return out


# ── Resume support ───────────────────────────────────────────────────
def _load_done(output_path: Path) -> set[tuple[str, int]]:
    if not output_path.exists():
        return set()
    done: set[tuple[str, int]] = set()
    with output_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["trajectory_id"], int(r["step_idx"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return done


# ── Worker ───────────────────────────────────────────────────────────
def _label_one(client, model: str, step_rec: dict, write_lock: Lock, fh) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        goal=step_rec["goal"],
        before_ui=step_rec["before_ui_elements_truncated"],
        after_ui=step_rec["after_ui_elements_truncated"],
        action=json.dumps(step_rec["action"]),
        action_reason=step_rec["action_reason"],
        summary=step_rec["agent_summary"],
    )
    before_b64 = _png_to_b64(Path(step_rec["before_screenshot_path"]))
    after_b64 = _png_to_b64(Path(step_rec["after_screenshot_path"]))
    try:
        api_resp = _call_claude(client, model, before_b64, after_b64, prompt)
    except Exception as e:  # noqa: BLE001
        # Record the failure but don't crash the worker
        out_rec = {
            **{k: step_rec[k] for k in (
                "trajectory_id", "step_idx", "app", "template", "goal",
                "trajectory_success", "trajectory_n_steps",
                "action", "action_reason", "agent_summary",
                "before_ui_elements_truncated", "after_ui_elements_truncated",
                "before_screenshot_path", "after_screenshot_path",
            )},
            "judge_model": model,
            "judge_error": f"{type(e).__name__}: {e}",
            "labeled_at": dt.datetime.now(dt.UTC).isoformat(),
        }
        with write_lock:
            fh.write(json.dumps(out_rec) + "\n")
            fh.flush()
        return out_rec
    try:
        score, reason = _parse_response(api_resp["raw_text"])
        parse_error = None
    except ValueError as e:
        score, reason = None, None
        parse_error = str(e)
    out_rec = {
        **{k: step_rec[k] for k in (
            "trajectory_id", "step_idx", "app", "template", "goal",
            "trajectory_success", "trajectory_n_steps",
            "action", "action_reason", "agent_summary",
            "before_ui_elements_truncated", "after_ui_elements_truncated",
            "before_screenshot_path", "after_screenshot_path",
        )},
        "judge_model": model,
        "judge_score": score,
        "judge_reason": reason,
        "judge_raw_text": api_resp["raw_text"],
        "judge_input_tokens": api_resp["input_tokens"],
        "judge_output_tokens": api_resp["output_tokens"],
        "judge_wall_s": api_resp["wall_s"],
        "judge_parse_error": parse_error,
        "labeled_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    with write_lock:
        fh.write(json.dumps(out_rec) + "\n")
        fh.flush()
    return out_rec


# ── Main ─────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories", type=Path, required=True,
                   help="Directory containing one episode subdir per trajectory.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output JSONL path (one labeled step per line).")
    p.add_argument("--model", default="claude-sonnet-4-5-20250929")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--limit-steps", type=int, default=None,
                   help="If set, only process this many new steps (for testing).")
    p.add_argument("--cost-cap-usd", type=float, default=200.0,
                   help="Abort if cumulative spend exceeds this.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import anthropic
    except ImportError:
        logger.error("anthropic SDK not installed: pip install anthropic")
        return 1
    client = anthropic.Anthropic()

    logger.info("Enumerating steps in %s ...", args.trajectories)
    all_steps = _iter_steps(args.trajectories)
    logger.info("Found %d total steps across %d episodes.",
                len(all_steps),
                len({s["trajectory_id"] for s in all_steps}))

    done = _load_done(args.output)
    logger.info("Already labeled: %d steps. Skipping those.", len(done))

    pending = [s for s in all_steps if (s["trajectory_id"], s["step_idx"]) not in done]
    if args.limit_steps is not None:
        pending = pending[: args.limit_steps]
    logger.info("Pending: %d steps (will label this run).", len(pending))

    if not pending:
        logger.info("Nothing to do.")
        return 0

    # Cost projection
    est_in_per = 4500  # rough avg input tokens (mostly images)
    est_out_per = 180  # SCORE+REASON
    sonnet_in = 3.0 / 1e6
    sonnet_out = 15.0 / 1e6
    est_cost = len(pending) * (est_in_per * sonnet_in + est_out_per * sonnet_out)
    logger.info("Projected cost (Sonnet 4.5): ~$%.2f for %d steps.", est_cost, len(pending))
    if est_cost > args.cost_cap_usd:
        logger.error("Projected cost $%.2f > cap $%.2f; aborting.", est_cost, args.cost_cap_usd)
        return 1

    # Open output file for appending
    fh = args.output.open("a")
    write_lock = Lock()

    t_start = time.monotonic()
    n_done = 0
    n_errors = 0
    n_parse_errors = 0
    tokens_in_total = 0
    tokens_out_total = 0

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(_label_one, client, args.model, s, write_lock, fh): s for s in pending}
            for fut in as_completed(futures):
                rec = fut.result()
                n_done += 1
                if rec.get("judge_error"):
                    n_errors += 1
                if rec.get("judge_parse_error"):
                    n_parse_errors += 1
                tokens_in_total += rec.get("judge_input_tokens") or 0
                tokens_out_total += rec.get("judge_output_tokens") or 0
                if n_done % max(1, len(pending) // 50) == 0 or n_done <= 10:
                    elapsed = time.monotonic() - t_start
                    rate = n_done / max(elapsed, 1)
                    eta_s = (len(pending) - n_done) / max(rate, 0.01)
                    spent = tokens_in_total * sonnet_in + tokens_out_total * sonnet_out
                    logger.info(
                        "Progress: %d/%d (%.1f%%) | %.1f/s | api_err=%d parse_err=%d | "
                        "tok in=%d out=%d | spent=$%.2f | ETA=%.0fs",
                        n_done, len(pending), 100 * n_done / len(pending),
                        rate, n_errors, n_parse_errors,
                        tokens_in_total, tokens_out_total, spent, eta_s,
                    )
    finally:
        fh.close()

    elapsed = time.monotonic() - t_start
    spent = tokens_in_total * sonnet_in + tokens_out_total * sonnet_out
    logger.info("=" * 60)
    logger.info("DONE. %d steps in %.1fs. Errors: api=%d parse=%d",
                n_done, elapsed, n_errors, n_parse_errors)
    logger.info("Tokens: in=%d out=%d → $%.2f", tokens_in_total, tokens_out_total, spent)
    logger.info("Output: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

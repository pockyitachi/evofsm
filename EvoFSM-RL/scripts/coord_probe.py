#!/usr/bin/env python3
"""Coordinate calibration probe — Story 1.4.3b.

**What this proves**

The agent prompt tells Qwen3-VL that the phone is 1080×2400 and instructs
it to emit x,y in absolute original-resolution pixels. But the processor
``smart_resize``s the image down to ~800k pixels before the vision tower
sees it. This script verifies the model really does output original-
resolution coords (not resized-canvas coords).

**How it works**

For each clickable, text-bearing UI element in the current emulator
screen, we:

    1. Ask the model to click "the element whose text is X"
    2. Get its (x,y) via the normal rollout generation path
    3. Compare to the element's ground-truth bbox center from AW's
       accessibility tree

If coordinates are systematically offset by ~40% (the smart_resize
factor), the coord convention is wrong.  If errors are random <150px,
the convention is right and that's just VLM grounding noise.

**Usage**

    # Emulator booted, on any screen with clickable elements:
    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      python EvoFSM-RL/scripts/coord_probe.py --n-probes 5

    # Run on the home screen specifically (good for app-icon targets)
    PYTHONPATH=android_world_plus:EvoFSM-RL \\
      python EvoFSM-RL/scripts/coord_probe.py --go-home --n-probes 5
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from typing import Any

logger = logging.getLogger("coord_probe")


def _bbox_center_px(el: Any, screen_wh: tuple[int, int]) -> tuple[int, int] | None:
    """Return (cx, cy) in absolute pixels for a UIElement, or None if unusable.

    AW exposes two possible coordinate sources on a UIElement:
      * ``bbox_pixels`` — already in physical pixels (what we want)
      * ``bbox``        — normalized or logical; requires multiplying by screen
    Preference order: bbox_pixels > bbox (logical, multiplied).
    """
    screen_w, screen_h = screen_wh
    bbp = getattr(el, "bbox_pixels", None)
    if bbp is not None:
        # BoundingBox.{x_min, y_min, x_max, y_max} in pixels
        try:
            return (
                int((bbp.x_min + bbp.x_max) / 2),
                int((bbp.y_min + bbp.y_max) / 2),
            )
        except AttributeError:
            pass
    bb = getattr(el, "bbox", None)
    if bb is not None:
        try:
            cx = int((bb.x_min + bb.x_max) / 2 * screen_w)
            cy = int((bb.y_min + bb.y_max) / 2 * screen_h)
            return cx, cy
        except AttributeError:
            pass
    return None


def _pick_probe_targets(ui_elements: list[Any], screen_wh: tuple[int, int],
                        n: int) -> list[tuple[str, int, int]]:
    """Pick ``n`` UI elements usable as probe targets.

    Must: be clickable + have short text + have a valid bbox.
    Returns list of (text, truth_cx, truth_cy).
    """
    targets: list[tuple[str, int, int]] = []
    seen_texts: set[str] = set()
    for el in ui_elements:
        is_clickable = bool(getattr(el, "is_clickable", False))
        text = (getattr(el, "text", None) or getattr(el, "content_description", None))
        if not is_clickable or not text:
            continue
        text = text.strip()
        if not text or len(text) > 40 or text in seen_texts:
            continue
        center = _bbox_center_px(el, screen_wh)
        if center is None:
            continue
        targets.append((text, *center))
        seen_texts.add(text)
        if len(targets) >= n:
            break
    return targets


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-probes", type=int, default=5,
                   help="Number of UI elements to probe (default 5).")
    p.add_argument("--go-home", action="store_true",
                   help="Reset to home screen before probing.")
    p.add_argument("--device", choices=("mps", "cuda", "cpu"), default=None)
    p.add_argument("--console-port", type=int, default=5554)
    p.add_argument("--grpc-port", type=int, default=8554)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # Late imports (same rationale as run_rollout.py).
    from PIL import Image

    from evofsm_rl.agent.action import parse_action
    from evofsm_rl.agent.prompts import build_messages
    from evofsm_rl.agent.rollout import GenerationConfig, generate_action_text
    from evofsm_rl.env import harness
    from evofsm_rl.model import load_base_model, load_model_config, resolve_device

    # 1. Model.
    resolved_device = args.device or resolve_device()
    print(f"Loading Qwen3-VL-8B on '{resolved_device}'...")
    model, processor = load_base_model(device=resolved_device)
    cfg = load_model_config(device=resolved_device)
    gen_cfg = GenerationConfig.from_yaml(cfg.raw.get("generation", {}))

    # 2. Emulator.
    print("Connecting to emulator...")
    env = harness.connect(
        console_port=args.console_port, grpc_port=args.grpc_port,
    )

    try:
        if args.go_home:
            env.reset(go_home=True)
            time.sleep(1.5)

        state = env.get_state(wait_to_stabilize=True)
        try:
            screen_w, screen_h = env.logical_screen_size
        except Exception:
            screen_h, screen_w = state.pixels.shape[:2]
        print(f"Screen: {screen_w}x{screen_h}, UI elements: {len(state.ui_elements)}")

        # 3. Pick targets.
        targets = _pick_probe_targets(
            state.ui_elements, (screen_w, screen_h), n=args.n_probes
        )
        if not targets:
            print("ERROR: no clickable text-bearing UI elements on this screen. "
                  "Try --go-home, or navigate somewhere with obvious buttons.")
            return 1

        print(f"\nProbing {len(targets)} targets:\n")

        # 4. Loop: each target gets one probe.
        results: list[dict[str, Any]] = []
        screenshot = Image.fromarray(state.pixels)
        for text, truth_x, truth_y in targets:
            goal = f"Click the UI element whose text is exactly {text!r}."
            messages = build_messages(
                task_goal=goal,
                screenshot=screenshot,
                history=[],
                screen_width=screen_w,
                screen_height=screen_h,
            )
            raw, dt, ntok = generate_action_text(
                model, processor, messages,
                screenshot=screenshot,
                device=resolved_device,
                generation_config=gen_cfg,
            )
            result = parse_action(raw, screen_width=screen_w, screen_height=screen_h)

            rec: dict[str, Any] = {
                "target_text": text,
                "truth_xy": (truth_x, truth_y),
                "model_raw": raw,
                "parsed_ok": result.ok,
                "parse_error": result.error,
                "gen_seconds": round(dt, 2),
                "input_tokens": ntok,
            }
            if result.ok and result.action.action_type == "click":
                pred_x, pred_y = result.action.x, result.action.y
                dx = pred_x - truth_x
                dy = pred_y - truth_y
                dist = math.hypot(dx, dy)
                rec["pred_xy"] = (pred_x, pred_y)
                rec["err_dx"] = dx
                rec["err_dy"] = dy
                rec["err_px"] = round(dist, 1)
            else:
                rec["pred_xy"] = None
                rec["err_px"] = None
            results.append(rec)

            mark = "✔" if rec["err_px"] is not None and rec["err_px"] < 150 else "?"
            print(f"  {mark} {text!r:<40s} "
                  f"truth=({truth_x},{truth_y}) "
                  f"pred={rec['pred_xy']} "
                  f"err={rec['err_px']}px "
                  f"[{dt:.1f}s]")
            if not result.ok:
                print(f"      parse_error: {result.error}")

        # 5. Summary.
        errors = [r["err_px"] for r in results if r["err_px"] is not None]
        if errors:
            import statistics
            print("\n" + "=" * 64)
            print(f"Probes: {len(errors)}/{len(results)} parsed as click + had bbox truth")
            print(f"Median error:    {statistics.median(errors):6.1f} px")
            print(f"Mean   error:    {statistics.mean(errors):6.1f} px")
            print(f"Max    error:    {max(errors):6.1f} px")
            # A systematic 40% scaling bug would look like median > 0.3 * max(screen_w, screen_h).
            expected_ceiling = 0.2 * max(screen_w, screen_h)
            verdict = (
                "✅ coord convention looks correct (median < 20% of screen)"
                if statistics.median(errors) < expected_ceiling
                else "⚠  large systematic offset — check coord convention!"
            )
            print(f"\n{verdict}")
            print("=" * 64)
        else:
            print("\nNo probes produced a parseable click — re-check prompt + model.")
            return 1

        return 0

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

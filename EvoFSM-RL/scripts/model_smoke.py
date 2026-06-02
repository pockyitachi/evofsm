#!/usr/bin/env python3
"""Story 1.2 smoke test — Qwen3-VL-8B-Instruct forward-pass wiring.

What this proves (per ticket acceptance criteria):
  1. `evofsm_rl.model.load_base_model()` loads the pinned checkpoint.
  2. Revision + fingerprint gates work (or write the lock file on first run).
  3. The model accepts a real AndroidWorld screenshot and emits logits.
  4. Prints the first-token log-prob so we can eyeball that the model is
     producing sensible predictions and not garbage.

What this DOES NOT prove:
  - Anything about task success, agent quality, or downstream loops.
  - Long-form generation (Story 1.4 covers that).

Usage:
    # Most common — Mac dev loop with emulator already booted
    cd /path/to/android_world_plus
    source .venv/bin/activate
    PYTHONPATH=android_world_plus:EvoFSM-RL python EvoFSM-RL/scripts/model_smoke.py

    # Skip emulator, use a blank synthetic image (useful in CI or offline)
    PYTHONPATH=android_world_plus:EvoFSM-RL python EvoFSM-RL/scripts/model_smoke.py --synthetic

    # Force device (default: auto-detect)
    PYTHONPATH=android_world_plus:EvoFSM-RL python EvoFSM-RL/scripts/model_smoke.py --device mps

Exit codes:
    0 — forward pass ok, fingerprint written/matched, memory reported
    1 — any failure (load / forward / fingerprint)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)

PROMPT = "Describe what you see on this Android screen in one sentence."

# ─────────────────────────────────────────────────────────────────────────
# Image source — prefer real emulator screenshot, fall back to synthetic
# ─────────────────────────────────────────────────────────────────────────


def grab_emulator_screenshot():
    """Connect to emulator via EvoFSM-RL harness and return (H, W, 3) uint8 RGB.

    Fails cleanly if emulator isn't up — caller can switch to synthetic.
    """
    from evofsm_rl.env import harness

    print("  Connecting to emulator (console=5554, grpc=8554)...")
    env = harness.connect()
    try:
        state = env.get_state(wait_to_stabilize=False)
        pixels = state.pixels  # numpy (H, W, 3) uint8
        print(f"  Got screenshot: {pixels.shape} dtype={pixels.dtype}")
        return pixels
    finally:
        env.close()


def synthetic_screenshot():
    """Return a 1080×600×3 synthetic test image (gradient + text-ish block).

    Used when emulator isn't available. The gradient gives the vision tower
    something non-trivial to tokenize; result is not meaningful, just non-zero.
    """
    import numpy as np

    h, w = 1080, 600
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Horizontal gradient in R, vertical in G, constant B = 128
    img[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    img[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    img[..., 2] = 128
    # Fake "UI bar" band at top
    img[:80, :, :] = 240
    print(f"  Synthetic screenshot: shape={img.shape} dtype={img.dtype}")
    return img


# ─────────────────────────────────────────────────────────────────────────
# Memory reporting
# ─────────────────────────────────────────────────────────────────────────
#
# On macOS + MPS, the bulk of the model lives in Metal-allocated pages. These
# DO show up against the process as unified memory, but `psutil.rss` reports
# only CPU-side resident pages — GPU driver allocations are tracked separately.
# safetensors also mmaps weights, so a lot of host pages aren't resident.
#
# To honestly answer "does this fit in 18 GB with the emulator?" we report
# both:
#   (a) host_rss   — CPU-side resident (ps / psutil) : MB-scale, housekeeping
#   (b) mps_alloc  — torch.mps.driver_allocated_memory(): the real weight bill
# and a `total` that approximates what Activity Monitor shows under "Memory".


def host_rss_gb() -> float:
    """CPU-side resident-set size in GB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    except ImportError:
        try:
            import resource
            # Linux gives KB, macOS gives bytes.
            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return maxrss / (1024**3) if maxrss > 1e9 else maxrss / (1024**2)
        except Exception:
            return float("nan")


def mps_alloc_gb() -> float:
    """Metal driver allocation in GB (0.0 when MPS isn't in use)."""
    try:
        import torch
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            # driver_allocated_memory reflects what the Metal driver has claimed,
            # which is the memory pressure we actually care about.
            return torch.mps.driver_allocated_memory() / (1024**3)
    except Exception:
        pass
    return 0.0


def cuda_alloc_gb() -> float:
    """CUDA reserved memory in GB (0.0 on non-CUDA hosts)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / (1024**3)
    except Exception:
        pass
    return 0.0


def mem_report() -> tuple[float, float, float]:
    """Return (host_rss_gb, accel_alloc_gb, total_gb) — total is our budget check."""
    host = host_rss_gb()
    accel = max(mps_alloc_gb(), cuda_alloc_gb())
    return host, accel, host + accel


def fmt_mem(label: str) -> str:
    h, a, t = mem_report()
    return f"{label:<38s} host={h:5.2f} GB  accel={a:6.2f} GB  total={t:6.2f} GB"


# ─────────────────────────────────────────────────────────────────────────
# Main smoke flow
# ─────────────────────────────────────────────────────────────────────────


def run_smoke(device: str | None, use_synthetic: bool) -> int:
    from PIL import Image

    print("=" * 64)
    print("EvoFSM-RL model smoke — Story 1.2")
    print("=" * 64)

    t0 = time.monotonic()
    print("\n[1/4] " + fmt_mem("Memory before any import:"))

    # ── Load model via the public loader ───────────────────────────
    from evofsm_rl.model import load_base_model, resolve_device

    resolved_device = device or resolve_device()
    print(f"\n[2/4] Loading Qwen3-VL-8B on device='{resolved_device}'...")
    t_load = time.monotonic()
    try:
        model, processor = load_base_model(device=resolved_device)
    except Exception as e:
        print(f"  ❌ load_base_model failed: {type(e).__name__}: {e}")
        return 1
    print(f"  Loaded in {time.monotonic() - t_load:.1f}s")
    print("  " + fmt_mem("Memory after load:"))

    # ── Get an image ────────────────────────────────────────────────
    print("\n[3/4] Acquiring test image...")
    if use_synthetic:
        pixels = synthetic_screenshot()
    else:
        try:
            pixels = grab_emulator_screenshot()
        except Exception as e:
            print(f"  ⚠ emulator unreachable ({e}); falling back to synthetic.")
            pixels = synthetic_screenshot()
    image = Image.fromarray(pixels)

    # ── Build inputs and run forward ────────────────────────────────
    print("\n[4/4] Running forward pass...")
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        print(f"  ❌ processor.apply_chat_template failed: {type(e).__name__}: {e}")
        return 1

    try:
        inputs = processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        )
    except Exception as e:
        print(f"  ❌ processor(...) failed: {type(e).__name__}: {e}")
        return 1

    # Move tensors to device (skip non-tensor entries)
    inputs = {
        k: (v.to(resolved_device) if hasattr(v, "to") else v)
        for k, v in inputs.items()
    }
    print(f"  Input tokens: {inputs['input_ids'].shape[-1]}")

    t_fwd = time.monotonic()
    model.eval()
    with torch.no_grad():
        out = model(**inputs)
    print(f"  Forward in {time.monotonic() - t_fwd:.2f}s")

    logits = out.logits  # (batch, seq, vocab)
    last = logits[0, -1, :].float()  # last-position logits for next-token pred
    log_probs = torch.log_softmax(last, dim=-1)
    top_k = torch.topk(log_probs, k=5)
    tokenizer = getattr(processor, "tokenizer", processor)
    top_tokens = [tokenizer.decode([i.item()]) for i in top_k.indices]

    print("\n  Top-5 next tokens (by log-prob):")
    for tok, lp in zip(top_tokens, top_k.values.tolist()):
        print(f"    {tok!r:20s}  log_prob={lp:+.3f}")

    # ── Summary ─────────────────────────────────────────────────────
    host, accel, total = mem_report()
    print("\n" + "=" * 76)
    print(f"✅ Smoke PASS — {time.monotonic() - t0:.1f}s total")
    print(f"   Device:                              {resolved_device}")
    print(f"   Host RSS (CPU side):                 {host:6.2f} GB")
    print(f"   Accelerator allocation (MPS/CUDA):   {accel:6.2f} GB")
    print(f"   Total memory footprint:              {total:6.2f} GB")
    print(f"   First-token argmax log-prob:         {top_k.values[0].item():+.3f}")
    # Budget check against yaml hint (18 GB on Mac, 70 GB on A100)
    try:
        from evofsm_rl.model import load_model_config
        cfg = load_model_config(device=resolved_device)
        if cfg.max_memory_gb_hint is not None:
            status = "✅ under budget" if total <= cfg.max_memory_gb_hint else "⚠ OVER BUDGET"
            print(f"   Budget (max_memory_gb_hint):         {cfg.max_memory_gb_hint:6.2f} GB  {status}")
    except Exception:
        pass
    print("=" * 76)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--device",
        choices=("mps", "cuda", "cpu"),
        default=None,
        help="Force a device (default: auto-detect via resolve_device())",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Skip emulator and use a synthetic gradient image.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    sys.exit(run_smoke(device=args.device, use_synthetic=args.synthetic))


if __name__ == "__main__":
    main()

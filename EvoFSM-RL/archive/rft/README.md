# Archived route: RFT (abandoned)

This directory archives an **abandoned** non-main-line exploration: rejection /
reward fine-tuning (RFT). The main line is GRPO-based test-time adaptation.

## Contents
- `scripts/` — `run_rft.py`, `run_rft_teval.py` (entry points only; RFT reused
  main-line modules and had no separate package).
- `traces_backup.zip` — original `rft_v01` + `rft_v01_teval` trajectories
  (~1 GB raw, 900 MB zipped; **gitignored**). The original `traces/rft_v01*`
  directories were deleted after this zip was verified.

## Status
Restorable by moving the scripts back. No main-line code depended on these.

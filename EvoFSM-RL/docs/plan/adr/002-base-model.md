# ADR-002 — Base model: Qwen3-VL-8B

**Status:** Accepted (2026-04-15)
**Context:** Epic 1 kickoff. The policy π that executes actions (and later gets LoRA-adapted at test time) needs a concrete base VLM. Two realistic candidates: **Qwen3-VL-8B** (dense 8B) and **Qwen3-VL-30B-A3B** (30B-param MoE, ~3B active per token).

---

## Decision

Use **Qwen3-VL-8B-Instruct** as the base policy for both pretraining (Phase 1) and test-time LoRA adaptation (Phase 3), on both dev host and training cloud.

## Hardware constraints driving this

- **Dev host:** M5 Pro Mac, **24 GB unified memory**. Must run inference (not training) during local dev, plus hold the Android emulator (~4 GB) + IDE + agent harness overhead.
- **Training host:** rented **single A100 80 GB** (user decision 2026-04-15 — not Tinker API, not multi-GPU). This is the only hardware that does any backprop.

### What fits

| Model | Inference (8-bit) | LoRA training (bf16 + grad ckpt) |
|---|---|---|
| Qwen3-VL-8B | ~9 GB → fits Mac 24 GB with emulator ✅ | ~28 GB on A100 80 GB ✅ (comfortable, room for batch) |
| Qwen3-VL-30B-A3B | ~18 GB (weights must be fully resident despite MoE active-3B) → tight on Mac, fails with emulator loaded ❌ | ~55 GB on A100 80 GB ⚠️ (fits but no headroom; batch size 1) |

The MoE "3B active" doesn't help memory — all 30B params must be loaded. It only helps throughput.

## Why not 30B-A3B

1. **Doesn't fit dev loop.** Can't iterate locally → forced to rent A100 even for prompt debugging. Same problem ADR-001 rejects.
2. **Single-A100 training is tight.** 55 GB resident leaves no room for LoRA optimizer states at reasonable batch. Would need to move to 2×A100 or H100, doubling rental cost.
3. **Ablation story muddier.** If 30B beats baselines, reviewers will ask "is it EvoFSM-RL or just a bigger VLM?" An 8B base lets us claim the gain is from the method, not scale. We can later add a 30B appendix run if budget allows.

## Why not 7B / Llama-VL / InternVL

- Qwen3-VL is the current SOTA open-weights VLM on Android/GUI benchmarks (per AndroidLab 2025 and related leaderboards as of our knowledge cutoff). Going smaller (7B) to an older family would underperform on the vanilla leaderboard tasks before our method even kicks in — making it hard to show TTA gains above a noisy floor.
- InternVL / Llama-3.2-Vision are respectable alternatives but lack strong evidence of GUI-grounding at 8B. Defer as "future work / reviewer-requested ablation."

## Consequences

**Positive**
- Inference runs locally → fast iteration on prompts, trajectory templates, FSM edits.
- Single-A100 training → predictable rental cost (~$1.50–2.50/hr × expected ~40 hr for Phase 1 pretraining = ~$60–100 per full run).
- Base model is a recognized strong baseline; any improvement from EvoFSM-RL has clean attribution.

**Negative / risk**
- 8B may plateau below SOTA before our method is applied → smaller absolute numbers than a 30B headline. Paper should report **relative gain over base**, not just raw success rate, to stay defensible.
- Need to pin a specific checkpoint hash for reproducibility (HF revision pin in `configs/model.yaml` — Story 1.3).

## Follow-up actions

- [ ] **Story 1.3** — add `configs/model.yaml` pinning `Qwen/Qwen3-VL-8B-Instruct@<revision>`, loader wrapper with 8-bit quantization on Mac / bf16 on A100.
- [ ] **Story 1.4** — baseline eval: run vanilla Qwen3-VL-8B with AndroidWorld's default M3A agent on the 192-task inventory, K=1. This is the "no-method" floor for all later comparisons.
- [ ] **Backlog / camera-ready** — optional 30B-A3B ablation on A100×2 if reviewers push on scale.

---

*Related: ADR-001 (emulator path), ADR-003 (snapshot schema).*

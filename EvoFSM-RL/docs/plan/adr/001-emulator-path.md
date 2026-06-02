# ADR-001 — Emulator path: native Mac AVD (arm64)

**Status:** Accepted (2026-04-15)
**Context:** Epic 1 kickoff. Need a reproducible Android emulator to run AndroidWorld + Plus tasks against our agent during both dev iteration (on J's laptop) and paper-time eval runs (on rented cloud GPUs).

---

## Decision

Use the **Android Studio `emulator` binary with a dedicated AVD** (Android Virtual Device) running natively on the host. No Docker wrapper.

- **Dev host:** Apple Silicon M5 Pro, 24 GB RAM → arm64 AVD image.
- **Cloud host:** Linux A100 box (rented for training + large-scale eval) → x86_64 AVD image, same API level, same snapshot contract.
- **AVD spec:** `Pixel_6_API_33` (Android 13 / API 33), Google APIs image (not Google Play), default RAM 4 GB, heap 512 MB, cold-boot snapshot disabled (we use AndroidWorld's own reset-via-adb path).
- **Harness:** `android_world.env.env_launcher.load_and_setup_env()` — same function vanilla AndroidWorld uses. We call it directly; no re-implementation.

## Why not Docker

Evaluated `budtmo/docker-android` and `google/android-emulator-container-scripts`. Rejected:

1. **Nested virtualization** — Mac doesn't expose KVM/HAXM inside Docker Desktop. AVD inside Linux container on Mac = software rendering = ~10× slower per step. Kills dev-loop velocity.
2. **arm64 vs x86_64 image mismatch** — Docker images ship x86_64 by default; running x86_64 AVD via qemu-user on M5 Pro = very slow; running arm64 AVD needs a custom container we'd have to maintain.
3. **Reproducibility benefit is marginal** — AndroidWorld tasks are already deterministic via `task_random_seed=30` + adb-level state reset. The emulator itself doesn't need containerizing.

## Why not cloud-only (skip native dev AVD)

Tried conceptually: SSH to A100 box, run x86_64 AVD there, display-forward to Mac. Rejected because:
- Iteration latency on debugging prompts / observations becomes painful (every change = round-trip to cloud).
- A100 rental $ burns for every minute we're just writing prompts — wasteful vs a free local AVD.

## Consequences

**Positive**
- Dev loop runs entirely offline on M5 Pro, $0/hr.
- Identical AndroidWorld harness code on both hosts — only difference is `sdkmanager "system-images;android-33;google_apis;arm64-v8a"` vs `...;x86_64`.
- Snapshots (see ADR-003) are host-agnostic because they serialize FSM + LoRA weights, not emulator state.

**Negative / risk**
- Two AVD images to maintain (arm64 + x86_64). Mitigation: both pin `android-33 google_apis`, differ only in arch; bootstrap script parameterized by arch.
- Any architecture-dependent flakiness in tasks becomes an integration bug to find. Mitigation: Story 1.2 includes a smoke test that runs ≥10 random templates on both arches and asserts identical `TaskEval.is_successful()` outputs.

## Follow-up actions

- [ ] **Story 1.2** — write `scripts/bootstrap_avd.sh {arm64|x86_64}` that installs the right system image and creates the AVD.
- [ ] **Story 1.3** — include a CI job that boots the arm64 AVD and runs the smoke test on every PR touching the agent code.

---

*Related: ADR-002 (base model), ADR-003 (snapshot schema).*

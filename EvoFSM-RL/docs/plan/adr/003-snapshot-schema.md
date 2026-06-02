# ADR-003 — Snapshot schema for frozen-checkpoint eval

**Status:** Accepted (2026-04-15)
**Context:** The core experimental claim of EvoFSM-RL is: *"after adapting on `T_adapt` of a target app, the agent generalizes to unseen templates `T_eval` of that same app."* This requires a **frozen checkpoint** of everything that changed during TTA, captured at the end of the `T_adapt` loop, then replayed read-only on `T_eval`. This ADR pins the snapshot format.

---

## Decision

Each TTA run produces one directory per `(target_app, seed)` pair:

```
runs/{run_id}/snapshots/{target_app}_seed{seed}/
├── snapshot.json            ← metadata + pointers
├── fsm_L1.json              ← serialized app-specific FSM (full graph)
├── L_C_target_row.json      ← the target-category row of L_C (the only L_C row mutated)
└── lora_delta/              ← PEFT adapter weights (safetensors + adapter_config.json)
    ├── adapter_model.safetensors
    └── adapter_config.json
```

### `snapshot.json` schema

```json
{
  "snapshot_id": "evofsm-rl_2026-04-22_simple_calendar_pro_seed30",
  "schema_version": "1.0",
  "created_utc": "2026-04-22T14:32:11Z",

  "target_app": "simple_calendar_pro",
  "play_category": "productivity",
  "tier": "tier_B",
  "base_seed": 30,

  "base_model": {
    "name": "Qwen/Qwen3-VL-8B-Instruct",
    "revision": "abc123def456"
  },

  "adapt_protocol": {
    "T_adapt_templates": ["SimpleCalendarAddOneEvent", "..."],
    "T_adapt_seeds": [30, 31, 32, 33, 34],
    "n_adapt_trajectories": 35,
    "n_adapt_steps": 40,
    "evolve_rl_config_hash": "sha256:..."
  },

  "artifacts": {
    "fsm_L1": "fsm_L1.json",
    "L_C_target_row": "L_C_target_row.json",
    "lora_delta": "lora_delta/"
  },

  "adapt_metrics": {
    "T_adapt_success_rate_pre":  0.14,
    "T_adapt_success_rate_post": 0.63,
    "fsm_nodes_added":   7,
    "fsm_edges_added":  12,
    "L_C_row_edits":     3,
    "lora_params_trained": 12582912
  },

  "eval_protocol": {
    "T_eval_templates": ["SimpleCalendarDeleteEvents", "..."],
    "T_eval_seeds": [40, 41, 42],
    "frozen": true
  }
}
```

### Invariants (checked by `evofsm_rl.snapshots.validate()`)

1. `schema_version` must match the currently-supported version range.
2. `base_model.name` + `revision` must be resolvable and hash-match the local weights before loading.
3. `adapt_protocol.T_adapt_templates` ∩ `eval_protocol.T_eval_templates` = ∅ (enforced against `splits.yaml`).
4. `play_category` must equal `taxonomy.play_category_of(target_app)`.
5. All three artifact paths must exist and be non-empty.
6. `evolve_rl_config_hash` must match the recorded config blob — guards against config drift between adapt and eval.
7. `frozen=true` — eval pipeline refuses to load a snapshot with `frozen=false`.

## Why a single JSON + sidecar files (vs monolithic pickle / vs W&B only)

- **JSON is greppable, diffable, human-inspectable.** Critical for debugging "why did two snapshots of the same app differ?"
- **Sidecar artifacts keep large binaries out of JSON** — LoRA weights as safetensors (partially readable, mmap-friendly, no pickle-arbitrary-code risk).
- **W&B / MLflow are logging layers, not formats.** We log _into_ W&B, but the artifact of record is the on-disk directory. Portable across environments (including reviewers rerunning us without a W&B account).

## Why not per-task snapshots

Considered snapshotting after each `T_adapt` task. Rejected:
- Blows up storage (hundreds of LoRA deltas per app). 
- No experimental claim hinges on the per-task curve — only the post-`T_adapt` endpoint matters for the T_eval comparison.
- We _do_ log per-task metrics in `adapt_metrics_trace.jsonl` for plotting, but that's observation, not artifact.

## Consequences

**Positive**
- Eval pipeline is dead simple: `load_snapshot(path) → run T_eval → report`. No state to accidentally mutate.
- Reviewers / other groups can re-run our T_eval from a shared snapshot tarball without the adapt pipeline.
- Ablations (e.g. "FSM only, no LoRA" / "LoRA only, no FSM") are straightforward — just zero out one artifact before loading.

**Negative / risk**
- Schema will grow. Mitigation: `schema_version` + an explicit migration function in `evofsm_rl.snapshots.migrate()` for v1→v2.
- LoRA + FSM together may still leave hidden mutable state in the base model if we're not careful (e.g. KV-cache, RNG). Mitigation: eval harness re-seeds every episode with the seed from `T_eval_seeds` and constructs a fresh model instance per snapshot load (Story 1.5 will enforce this).

## Follow-up actions

- [ ] **Story 1.5** — implement `evofsm_rl/snapshots.py` with `save()`, `load()`, `validate()`, `migrate()`.
- [ ] **Story 1.5** — unit tests: round-trip save/load, invariant violations, cross-tier rejection.
- [ ] **Story 2.x** — the adapt loop ends with a `save()` call; the eval loop starts with a `load()` + `validate()` call. No other code path writes snapshots.

---

*Related: ADR-001 (emulator path), ADR-002 (base model).*

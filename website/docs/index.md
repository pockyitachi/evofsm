# EvoFSM

**Test-time adaptation for mobile GUI agents on _unseen_ apps**

Deployed to a new app, the agent adapts on a small budget by **jointly** evolving
a two-layer FSM prior (symbolic) and fine-tuning a LoRA policy (sub-symbolic),
reusing a per-category abstract-action library `L_C` learned during pretraining.

## The problem

Mobile GUI agents are reliable on apps they were trained on but brittle on apps
they have never seen. Today's 60–80% benchmark numbers are measured where train
and test share the same task templates — they say little about the
deployment-relevant question:

> When a user installs a banking app, a niche notes tool, or a ride-hailing
> service the agent has never encountered, can it become reliable from a handful
> of adaptation examples?

EvoFSM targets exactly this unseen-app, small-budget regime. See [Method](method.md).

## Two studies

EvoFSM is evaluated at two levels of generalization:

| | [Within-benchmark](within-benchmark.md) | [Cross-benchmark](cross-benchmark.md) |
|---|---|---|
| **Setting** | held-out apps of one benchmark | a separately-authored benchmark |
| **Train → test** | AndroidWorld+ | AndroidWorld+ → MobileWorld |
| **Headline** | B1 38.6 → B2 (+9.3) → B3 (+3.7) → B4 **52.9** | symbolic TTA ≈ static prior on two models; joint B4 (π^pre + lessons-only) under evaluation |

## Where to go

- **[Method](method.md)** — the two-layer FSM and the joint adaptation loop
- **[Within-benchmark study](within-benchmark.md)** — AndroidWorld+ results
- **[Cross-benchmark study](cross-benchmark.md)** — MobileWorld results
- **[Dataset & splits](dataset.md)** — how the data is partitioned at each level
- **[Reproduce](reproduce.md)** — environment, emulator, and run commands

Code: [github.com/pockyitachi/evofsm](https://github.com/pockyitachi/evofsm)

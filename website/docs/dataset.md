# Dataset & splits

EvoFSM is evaluated at two levels of generalization, each with its own dataset
and partitioning protocol. This page is the self-contained reference for both;
the machine-readable sources of truth are `EvoFSM-RL/configs/splits.yaml`
(within-benchmark) and `EvoFSM-MW/configs/mobileworld_splits.yaml`
(cross-benchmark).

!!! note "Two transfer axes, one shared idea"
    Both studies test **category-level transfer** — whether the per-category
    abstract-action library `L_C`, learned during pretraining, helps an app of
    the *same Google Play category*. Tiers are defined purely by Play-Store
    category, never by app identity. "Seen / novel app" claims are never made.

---

## Within-benchmark (AndroidWorld+)

**AndroidWorld+** is a 25-app, 12-Play-category benchmark built on Google's
AndroidWorld (19 active apps) plus 6 apps imported from BMOCA (Calculator,
Snapseed, Wikipedia) and AndroidLab (Bluecoins, Maps.me, Pi Music). Of the ~194
templates in the metadata, **192 are app-attributable** (the other two are
generic/composite and excluded from the per-app split). Categories are taken
directly from each app's Play listing, an externally validated taxonomy that
also defines the equivalence class for category-generic `L_C`.

The split decomposes into **three independent, disjoint levels** — the pool an
app belongs to, the category-transfer tier it anchors, and the template-disjoint
adapt/eval partition within each app.

### Level 1 — Pool (train vs. target apps)

Standard app-level train/test separation. The **source pool** is used for
Phase-1 pretraining (per-app static FSM extraction and per-category `L_C`
synthesis); target apps are never touched during pretraining and only appear at
test-time adaptation.

| Pool | Role | Apps | Templates |
|---|---|---:|---:|
| Source pool | Phase-1 pretraining (FSM + `L_C`) | 12 | 96 |
| Tier-B (near) | Target — category seen in source | 6 | 50 |
| Tier-C (far) | Target — category novel | 7 | 46 |

The 12 source-pool apps span 6 Play categories (Productivity, Tools, Finance,
Music & Audio, Photography, Communication). These are exactly the categories for
which a matching `L_C` exists at test-time hand-off.

### Level 2 — Category tier (near vs. far transfer)

Target apps are split by whether their Play category is represented in the
source pool. Measuring near- and far-transfer separately is more informative
than one conflated number, and mirrors Android Control's `unseen-app` (≈ Tier-B)
vs. `unseen-category` (≈ Tier-C) distinction.

| Tier | Transfer | Categories | Apps | `T_adapt` | `T_eval` |
|---|---|---|---:|---:|---:|
| **Tier-B** | Near — category seen, matching `L_C` exists | Productivity, Tools, Finance, Music & Audio, Photography, Communication | 6 | 32 | 18 |
| **Tier-C** | Far — category novel, no matching `L_C` | Maps & Navigation, Food & Drink, Health & Fitness, Books & Reference, Video, Art & Design | 7 | 29 | 17 |

!!! info "Tier-C is a null control"
    Tier-C apps have no source-pool category, so `resolve_l_c_for_app()` returns
    `None` and they fall back to the B1 (no-injection) path. Tier-C therefore
    acts as a null control for the `L_C` injection mechanism.

    `simple_draw_pro` (Art & Design) has a single template, assigned entirely to
    `T_adapt`; its `T_eval` is empty and it is excluded from the Tier-C aggregate
    `T_eval` score.

### Level 3 — Template split (adapt vs. eval, within each app)

Within every target app, templates are partitioned into two **template-disjoint**
sets — not merely seed-disjoint, the task-template classes themselves differ.
This is what makes `T_eval` a measurement of adaptation generality rather than
parameter robustness on memorized templates, following the support/query
protocol of meta-learning (MAML) and the train-level/test-level protocol of
generalization-focused RL benchmarks (Procgen).

| Set | Used for | Split | Seeds (K) |
|---|---|---|---|
| `T_adapt` | Online adaptation (B3 FSM evolution, B4 joint TTA) | 60% (deterministic alphabetical) | 5 — `[30, 31, 32, 33, 34]` |
| `T_eval` | Single frozen evaluation pass after adaptation; the headline number | 40% | 3 — `[40, 41, 42]` |

`T_eval` seeds are disjoint from `T_adapt` seeds, so even a template that shared
parameter values across the two sets would realize a different initial state.
The partition rule degrades gracefully for small apps: `N = 2` splits 1/1, and
`N = 1` puts the single template in `T_adapt` with an empty `T_eval`.

---

## Cross-benchmark (MobileWorld)

The cross-benchmark study trains on **AndroidWorld+ (193 tasks, all source)** and
test-time-adapts on **MobileWorld GUI-only (161 tasks)**, a separately authored,
pure-vision benchmark. MobileWorld is eval-only: its 161 tasks are hard-coded
single instances with **no seed (K = 1 inherent)** and no built-in adapt split,
so adaptation tasks are carved out *within* MobileWorld via a task-disjoint
partition.

### Tiers (category-level)

Tiers are assigned by the Play categories of the apps a task touches, relative to
the 12 categories seen during AndroidWorld+ training. The only **novel**
categories are **Social** (Mastodon, Mattermost) and **Shopping** (Taodian); all
other MobileWorld apps fall in seen categories.

| Tier | Definition | Tasks | `L_C` injection |
|---|---|---:|---|
| **Tier-B** | Category-seen — every app's category was seen in training | 91 | Ready-made category `L_C` |
| **Tier-C** | Category-novel — all apps in novel categories (Social, Shopping) | 43 | Bootstrap `L_C` (from target-app trajectories) |
| **Tier-A** | Mixed — task touches both seen and novel categories | 27 | Seen ready-made + novel bootstrap |
| **Total** | | **161** | |

Each tier is composed of single-app and cross-app cases:

| Tier | Composition |
|---|---|
| Tier-B (91) | 36 single-app (all seen) + 55 cross-app (all seen) |
| Tier-C (43) | 40 single-app (all novel) + 3 cross-app (all novel) |
| Tier-A (27) | cross-app tasks mixing one novel app with 1–3 seen apps |

!!! warning "Shared package names ≠ unseen-app generalization"
    Six system apps (Chrome, Contacts, Settings, Clock, Camera, Files) share the
    same package name across AndroidWorld+ and MobileWorld. The project does **not**
    special-case them — tiers are category-only. Because the method's contribution
    is reported as a *difference* (with `L_C` − without `L_C`, policy fixed), any
    same-app advantage cancels and does not bias the `L_C` claim. Absolute Tier-B
    accuracy should **not** be read as generalization to unseen apps.

### Adapt / eval split (task-disjoint)

The split is deterministic (sorted by class name, no RNG) with
`adapt_fraction = 0.40`. Tier-A is assigned **entirely to eval** — it is the
headline "compose seen + novel knowledge" set and has no single-app tasks to
partition cleanly — which pulls the overall adapt fraction down to ~32%.
Multi-app tasks may enter `T_adapt`: under symbolic per-app evolution, one
multi-app adapt task simultaneously evolves the FSM/`L_C` of every app it touches,
so apps with no single-app tasks still gain adapt coverage, and task-level
disjointness prevents leakage.

| Tier | Tasks | `T_adapt` | `T_eval` |
|---|---:|---:|---:|
| **Tier-B** (near) | 91 | 33 | 58 |
| **Tier-C** (far) | 43 | 18 | 25 |
| **Tier-A** (mixed) | 27 | 0 | 27 |
| **Total** | **161** | **51** | **110** |

Because there are no seeds, each of the 51 adapt tasks is reused as a fitness
signal across the full B4 evolution loop (~20 iterations × K rollouts), so the
40% task budget amounts to hundreds of rollout experiences driving the symbolic
evolution. The disjoint 110-task `T_eval` (including multi-app combinations)
verifies genuine generalization rather than memorization.

!!! info "L_C source mapping"
    Injection resolves **per app**, with no tier branching: each app a task
    touches gets its category's ready-made `L_C` if the category is seen, or a
    bootstrapped `L_C` otherwise, then entries are de-duplicated by category.
    Seen MobileWorld apps map to AndroidWorld+ source apps of the same category
    (e.g. Mail/Messages/Contacts/Chrome → Communication; Files/Settings/Clock →
    Tools; Calendar → Productivity; Maps → Maps & Navigation; Camera/Gallery →
    Photography; Docreader → Books & Reference). Mastodon, Mattermost, and
    Taodian have no source and use bootstrap `L_C`.

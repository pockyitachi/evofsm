"""Split loader for EvoFSM-RL.

Reads `EvoFSM-RL/configs/splits.yaml` (the K=1 baseline split: which apps go to
the source pool / Tier-B / Tier-C, and which templates go to T_adapt / T_eval
inside each held-out app) and exposes typed accessors over it.

Optionally reads `EvoFSM-RL/configs/multi_seed_config.yaml` for K_source /
K_adapt / K_eval seed counts.

Both YAMLs are the source of truth; this module is a thin loader.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import yaml

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # EvoFSM-RL/
_SPLITS_YAML = _PROJECT_ROOT / "configs" / "splits.yaml"
_MULTI_SEED_YAML = _PROJECT_ROOT / "configs" / "multi_seed_config.yaml"


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HeldOutAppSplit:
    app: str
    category: str
    T_adapt: tuple[str, ...]
    T_eval: tuple[str, ...]
    note: str | None = None

    @property
    def n_templates(self) -> int:
        return len(self.T_adapt) + len(self.T_eval)


@dataclass(frozen=True)
class SourceAppSplit:
    app: str
    category: str
    templates: tuple[str, ...]

    @property
    def n_templates(self) -> int:
        return len(self.templates)


# ──────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def _load_splits() -> dict:
    with _SPLITS_YAML.open() as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=1)
def _load_multi_seed() -> dict:
    with _MULTI_SEED_YAML.open() as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────
# Public API — apps
# ──────────────────────────────────────────────────────────────────────
def get_source_pool() -> dict[str, SourceAppSplit]:
    """Source-pool apps used for Phase-1 multi-source pretraining."""
    raw = _load_splits()["source_pool"]
    return {
        app: SourceAppSplit(
            app=app,
            category=info["category"],
            templates=tuple(info["templates"]),
        )
        for app, info in raw.items()
    }


def get_tier_B_apps() -> dict[str, HeldOutAppSplit]:
    """Tier-B (near-transfer) held-out apps: category is in source pool."""
    return _held_out("tier_B_held_out")


def get_tier_C_apps() -> dict[str, HeldOutAppSplit]:
    """Tier-C (far-transfer) held-out apps: category absent from source pool."""
    return _held_out("tier_C_held_out")


def _held_out(key: str) -> dict[str, HeldOutAppSplit]:
    raw = _load_splits()[key]
    return {
        app: HeldOutAppSplit(
            app=app,
            category=info["category"],
            T_adapt=tuple(info["T_adapt"]),
            T_eval=tuple(info["T_eval"]),
            note=info.get("note"),
        )
        for app, info in raw.items()
    }


def all_held_out_apps() -> dict[str, HeldOutAppSplit]:
    """Tier-B ∪ Tier-C, single dict."""
    return {**get_tier_B_apps(), **get_tier_C_apps()}


# ──────────────────────────────────────────────────────────────────────
# Public API — per-app template lookups (for held-out apps)
# ──────────────────────────────────────────────────────────────────────
def get_T_adapt(app: str) -> tuple[str, ...]:
    """Return the T_adapt template list for a held-out app."""
    return _lookup_held_out(app).T_adapt


def get_T_eval(app: str) -> tuple[str, ...]:
    """Return the T_eval template list for a held-out app."""
    return _lookup_held_out(app).T_eval


def _lookup_held_out(app: str) -> HeldOutAppSplit:
    held = all_held_out_apps()
    if app not in held:
        raise KeyError(
            f"App {app!r} is not in any held-out pool. "
            f"Tier-B: {sorted(get_tier_B_apps())}; "
            f"Tier-C: {sorted(get_tier_C_apps())}"
        )
    return held[app]


def tier_of(app: str) -> str:
    """Return 'source', 'tier_B', or 'tier_C' for any registered app."""
    if app in get_source_pool():
        return "source"
    if app in get_tier_B_apps():
        return "tier_B"
    if app in get_tier_C_apps():
        return "tier_C"
    raise KeyError(f"App {app!r} not found in any pool.")


# ──────────────────────────────────────────────────────────────────────
# Public API — multi-seed overlay (K_source / K_adapt / K_eval)
# ──────────────────────────────────────────────────────────────────────
def get_seed_counts() -> dict[str, int]:
    """Return {K_source, K_adapt, K_eval} from multi_seed_config.yaml."""
    cfg = _load_multi_seed()
    return {
        "K_source": cfg["phase1_pretraining"]["K_source"],
        "K_adapt": cfg["phase3_TTA"]["K_adapt"],
        "K_eval": cfg["phase3_TTA"]["K_eval"],
    }


def get_seed_lists() -> dict[str, list[int]]:
    """Return the actual seed integer lists for {source, T_adapt, T_eval}."""
    cfg = _load_multi_seed()
    return {
        "source": cfg["phase1_pretraining"]["task_random_seeds"],
        "T_adapt": cfg["phase3_TTA"]["task_random_seeds"]["T_adapt"],
        "T_eval": cfg["phase3_TTA"]["task_random_seeds"]["T_eval"],
    }


# ──────────────────────────────────────────────────────────────────────
# Convenience summary
# ──────────────────────────────────────────────────────────────────────
def split_summary() -> dict:
    """Aggregate counts — useful for sanity-check prints."""
    src = get_source_pool()
    b = get_tier_B_apps()
    c = get_tier_C_apps()
    return {
        "source_apps": len(src),
        "source_templates": sum(s.n_templates for s in src.values()),
        "tier_B_apps": len(b),
        "tier_B_T_adapt": sum(len(s.T_adapt) for s in b.values()),
        "tier_B_T_eval": sum(len(s.T_eval) for s in b.values()),
        "tier_C_apps": len(c),
        "tier_C_T_adapt": sum(len(s.T_adapt) for s in c.values()),
        "tier_C_T_eval": sum(len(s.T_eval) for s in c.values()),
    }


__all__ = [
    "HeldOutAppSplit",
    "SourceAppSplit",
    "get_source_pool",
    "get_tier_B_apps",
    "get_tier_C_apps",
    "all_held_out_apps",
    "get_T_adapt",
    "get_T_eval",
    "tier_of",
    "get_seed_counts",
    "get_seed_lists",
    "split_summary",
]

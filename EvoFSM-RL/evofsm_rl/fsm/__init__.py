"""FSM package — two-layer schema for the EvoFSM-RL agent.

Story 2.2 onwards. The two-layer split is the keystone that lets us
transfer learned knowledge across apps:

  - LAYER 1 (APP_SPECIFIC): non-transferable. App name, concrete state
    visual cues, resource ids, concrete transitions. Discarded on
    hand-off to a new app.
  - LAYER 2 (GENERIC): transferable. Category-indexed abstract
    workflows, failure modes, verification checklists. App-agnostic.
    This is what gets carried across apps via L_C.

See `plan/algorithm_design.md` §3.7 for the canonical schema spec.
"""

from evofsm_rl.fsm.aggregator import aggregate_L_C, category_to_slug, load_L_C
from evofsm_rl.fsm.builder import build_fsm, compress_trajectories
from evofsm_rl.fsm.injection import resolve_l_c_for_app
from evofsm_rl.fsm.linter import lint_L_C, lint_layer2
from evofsm_rl.fsm.schema import (
    SCHEMA_VERSION,
    AbstractCategory,
    FSM,
    Layer1,
    Layer2,
    SchemaVersionError,
    State,
    Strategy,
    Transition,
)

__all__ = [
    "SCHEMA_VERSION",
    "AbstractCategory",
    "FSM",
    "Layer1",
    "Layer2",
    "SchemaVersionError",
    "State",
    "Strategy",
    "Transition",
    "aggregate_L_C",
    "build_fsm",
    "category_to_slug",
    "compress_trajectories",
    "lint_L_C",
    "lint_layer2",
    "load_L_C",
    "resolve_l_c_for_app",
]

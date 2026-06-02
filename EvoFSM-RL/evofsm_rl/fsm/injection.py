"""App-name → L_C prompt-text resolver — Story B2.

When the B2 baseline driver iterates over the T_eval template list of a
held-out app, it needs to decide two things per app:

  1. Does this app's Play Store category appear in the source pool?
     (Tier-B ⇒ yes; Tier-C ⇒ no.)
  2. If yes, where is the corresponding ``artifacts/L_C/{slug}.json``?

Both questions are answered by :func:`resolve_l_c_for_app`, which
centralizes the lookup so the runner and the tests agree on the logic.
Source-pool apps are also accepted (useful for validation / debugging /
future self-transfer experiments); held-out apps whose category has no
L_C file (Tier-C) return ``None``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from evofsm_rl.fsm.aggregator import category_to_slug, load_L_C


def _find_category(splits_data: dict[str, Any], app_name: str) -> str | None:
    """Scan source_pool / tier_B_held_out / tier_C_held_out for ``app_name``.

    Returns the Play Store category string or ``None`` if the app is not
    registered in any of the three pools.
    """
    for pool_key in ("source_pool", "tier_B_held_out", "tier_C_held_out"):
        pool = splits_data.get(pool_key) or {}
        if app_name in pool:
            return pool[app_name].get("category")
    return None


def resolve_l_c_for_app(
    app_name: str,
    splits_yaml_path: str | Path,
    l_c_dir: str | Path,
) -> str | None:
    """Return the L_C prompt-text for one app, or ``None`` if unavailable.

    Args:
        app_name: canonical snake_case app key (must match the keys in
            ``configs/splits.yaml``).
        splits_yaml_path: path to ``configs/splits.yaml``.
        l_c_dir: directory holding ``{slug}.json`` files written by
            ``scripts/build_L_C.py``.

    Returns:
        - Non-empty prompt text (ready to splice into the agent's action
          prompt via ``agent.set_l_c_prompt_text``) when the app's
          category has a corresponding L_C file.
        - ``None`` when either (a) the app is unknown to splits.yaml,
          or (b) the app's category has no L_C file on disk (Tier-C
          fallthrough — B2 degrades to B1 for these).

    The returned string includes the ``L_C CATEGORY: <name>`` tag so
    the agent knows which category the transferred knowledge belongs to.
    """
    splits_path = Path(splits_yaml_path)
    with splits_path.open() as fh:
        splits_data = yaml.safe_load(fh)

    category = _find_category(splits_data, app_name)
    if category is None:
        return None

    slug = category_to_slug(category)
    lc_path = Path(l_c_dir) / f"{slug}.json"
    if not lc_path.exists():
        # Tier-C categories (no source-pool coverage) won't have a file
        # in artifacts/L_C/. That's the designed B2 degradation path.
        return None

    _, layer2 = load_L_C(lc_path)
    return layer2.to_prompt_text(category=category)


__all__ = ["resolve_l_c_for_app"]

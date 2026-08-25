"""Remote-aligned, spot-only frozen state recipes.

This package is a vendored copy of the state-definition portion of the
20260813 remote freeze-alignment package.  It is intentionally kept separate
from the candidate logic so that the three-state generation path can point to
one auditable state source without importing the remote package's result
files, notebooks, or external comparison baseline.
"""

from .spot_eight_state import (
    EIGHT_STATE_COLUMNS,
    FULL_END,
    FULL_START,
    ROLLING_MIN_PERIODS,
    ROLLING_WINDOW,
    assign_eight_base_state,
    build_economic_features_eight,
    compute_eight_states,
    load_frozen_recipes,
)

__all__ = [
    "EIGHT_STATE_COLUMNS",
    "FULL_START",
    "FULL_END",
    "ROLLING_MIN_PERIODS",
    "ROLLING_WINDOW",
    "assign_eight_base_state",
    "build_economic_features_eight",
    "compute_eight_states",
    "load_frozen_recipes",
]

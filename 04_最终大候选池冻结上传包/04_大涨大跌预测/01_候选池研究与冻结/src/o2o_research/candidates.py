from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .specs import VersionSpec


COVERAGE_GRID = (0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.105, 0.115, 0.130, 0.150)
COMPONENT_POWERS = (0.75, 1.0, 1.25, 1.5)
AGREEMENT_WEIGHTS = (0.0, 0.15, 0.30, 0.45)


def _weight_profiles() -> tuple[tuple[float, float, float, float], ...]:
    rows: list[tuple[float, float, float, float]] = [(0.25, 0.25, 0.25, 0.25)]
    for dominant in range(4):
        row = [0.15] * 4
        row[dominant] = 0.55
        rows.append(tuple(row))
    for a, b in combinations(range(4), 2):
        row = [0.15] * 4
        row[a] = row[b] = 0.35
        rows.append(tuple(row))
    base = (0.40, 0.30, 0.20, 0.10)
    for shift in range(4):
        rows.append(tuple(base[(i - shift) % 4] for i in range(4)))
    rows.append(tuple(reversed(base)))
    if len(rows) != 16 or len(set(rows)) != 16:
        raise AssertionError("weight registry must contain sixteen unique economic profiles")
    return tuple(rows)


WEIGHT_PROFILES = _weight_profiles()


@dataclass(frozen=True)
class BaseCandidate:
    base_candidate_id: str
    weight_profile: int
    w1: float
    w2: float
    w3: float
    w4: float
    component_power: float
    agreement_weight: float

    def to_dict(self) -> dict:
        return asdict(self)


def base_candidates() -> list[BaseCandidate]:
    rows: list[BaseCandidate] = []
    for profile_index, weights in enumerate(WEIGHT_PROFILES, start=1):
        for power in COMPONENT_POWERS:
            for agreement_weight in AGREEMENT_WEIGHTS:
                rows.append(BaseCandidate(
                    base_candidate_id=f"base_{len(rows) + 1:03d}",
                    weight_profile=profile_index,
                    w1=weights[0], w2=weights[1], w3=weights[2], w4=weights[3],
                    component_power=power,
                    agreement_weight=agreement_weight,
                ))
    if len(rows) != 256:
        raise AssertionError("each version must define 256 base parameter combinations")
    return rows


BASE_CANDIDATES = base_candidates()


def raw_candidate_count() -> int:
    return len(BASE_CANDIDATES) * len(COVERAGE_GRID)


def _resolve_component(frame: pd.DataFrame, token: str) -> np.ndarray:
    if ":" in token:
        operation, column = token.split(":", 1)
    else:
        operation, column = "plain", token
    if column not in frame:
        raise KeyError(f"missing candidate component: {column}")
    values = frame[column].astype(float).clip(0, 1).fillna(0.5).to_numpy()
    if operation == "plain":
        return values
    if operation == "inv":
        return 1 - values
    if operation == "center":
        return np.abs(values - 0.5) * 2
    if operation == "low":
        return np.maximum(0, 0.5 - values) * 2
    if operation == "high":
        return np.maximum(0, values - 0.5) * 2
    raise KeyError(f"unknown component operation: {operation}")


def component_matrix(frame: pd.DataFrame, spec: VersionSpec, side: str) -> np.ndarray:
    tokens = spec.down_components if side == "down" else spec.up_components
    matrix = np.column_stack([_resolve_component(frame, token) for token in tokens])
    if matrix.shape != (len(frame), 4):
        raise AssertionError(matrix.shape)
    return np.clip(matrix, 0, 1)


def _aggregate(components: np.ndarray, weights: np.ndarray, aggregator: str, cut: float) -> np.ndarray:
    # components: n x 4; weights: 4 x m; return n x m.
    weighted_mean = components @ weights
    if aggregator == "weighted_mean":
        return weighted_mean
    if aggregator == "geometric":
        return np.exp(np.log(np.clip(components, 1e-8, 1)) @ weights)
    if aggregator == "minimum":
        return 0.35 * weighted_mean + 0.65 * components.min(axis=1, keepdims=True)
    if aggregator == "robust_median":
        return 0.35 * weighted_mean + 0.65 * np.median(components, axis=1, keepdims=True)
    if aggregator == "hurdle":
        direction = components[:, :2] @ weights[:2] / np.maximum(weights[:2].sum(axis=0), 1e-12)
        magnitude = components[:, 2:] @ weights[2:] / np.maximum(weights[2:].sum(axis=0), 1e-12)
        return np.sqrt(np.clip(direction * magnitude, 0, 1))
    if aggregator == "vote":
        vote = (components >= cut).mean(axis=1, keepdims=True)
        return 0.35 * weighted_mean + 0.65 * vote
    if aggregator == "margin":
        disagreement = components.std(axis=1, keepdims=True)
        return weighted_mean * np.clip(1 - 0.75 * disagreement, 0, 1)
    if aggregator == "state_gate":
        direction = components[:, :2] @ weights[:2] / np.maximum(weights[:2].sum(axis=0), 1e-12)
        state = components[:, 2:] @ weights[2:] / np.maximum(weights[2:].sum(axis=0), 1e-12)
        return direction * (0.35 + 0.65 * state)
    if aggregator == "distance":
        distance = np.sqrt(((1 - components) ** 2) @ weights)
        return np.clip(1 - distance, 0, 1)
    if aggregator == "abstain":
        conflict = components.max(axis=1, keepdims=True) - components.min(axis=1, keepdims=True)
        return weighted_mean * np.clip(1 - 0.85 * conflict, 0, 1)
    if aggregator == "hysteresis":
        # Applied below because it needs the time axis.
        return weighted_mean
    raise KeyError(f"unknown aggregator: {aggregator}")


def score_matrix(frame: pd.DataFrame, spec: VersionSpec, side: str) -> tuple[np.ndarray, pd.DataFrame]:
    raw_components = component_matrix(frame, spec, side)
    scores: list[np.ndarray] = []
    metadata: list[dict] = []
    for power in COMPONENT_POWERS:
        powered = np.power(np.clip(raw_components, 0, 1), power)
        candidates_for_power = [row for row in BASE_CANDIDATES if row.component_power == power]
        weights = np.array([[row.w1, row.w2, row.w3, row.w4] for row in candidates_for_power], dtype=float).T
        base = _aggregate(powered, weights, spec.aggregator, spec.component_cut)
        if spec.aggregator == "hysteresis":
            base = pd.DataFrame(base).rolling(3, min_periods=1).apply(
                lambda x: float(np.dot(x, np.array((0.2, 0.3, 0.5))[-len(x):]) / np.array((0.2, 0.3, 0.5))[-len(x):].sum()),
                raw=True,
            ).to_numpy()
        agreement = (raw_components >= spec.component_cut).mean(axis=1, keepdims=True)
        for j, row in enumerate(candidates_for_power):
            score = (1 - row.agreement_weight) * base[:, j] + row.agreement_weight * agreement[:, 0]
            scores.append(np.clip(score, 0, 1))
            metadata.append({**row.to_dict(), "aggregator": spec.aggregator, "component_cut": spec.component_cut})
    matrix = np.column_stack(scores)
    meta = pd.DataFrame(metadata)
    # Looping by power changes order, so restore the canonical base IDs.
    order = np.argsort(meta.base_candidate_id.to_numpy())
    matrix = matrix[:, order]
    meta = meta.iloc[order].reset_index(drop=True)
    if matrix.shape != (len(frame), 256) or meta.base_candidate_id.nunique() != 256:
        raise AssertionError(f"invalid score matrix: {matrix.shape}")
    return matrix, meta


def candidate_parameter_table(spec: VersionSpec, side: str) -> pd.DataFrame:
    rows = []
    for base in BASE_CANDIDATES:
        for coverage in COVERAGE_GRID:
            rows.append({
                "candidate_id": f"{base.base_candidate_id}_cov_{coverage:.3f}",
                "base_candidate_id": base.base_candidate_id,
                "coverage_config": coverage,
                "side": side,
                "core_logic_name": spec.core_logic_name,
                "aggregator": spec.aggregator,
                "component_cut": spec.component_cut,
                "w1": base.w1, "w2": base.w2, "w3": base.w3, "w4": base.w4,
                "component_power": base.component_power,
                "agreement_weight": base.agreement_weight,
                "down_components": "|".join(spec.down_components),
                "up_components": "|".join(spec.up_components),
            })
    frame = pd.DataFrame(rows)
    if len(frame) != 3072 or frame.candidate_id.nunique() != 3072:
        raise AssertionError("each side/version must have 3,072 explicit candidates")
    return frame

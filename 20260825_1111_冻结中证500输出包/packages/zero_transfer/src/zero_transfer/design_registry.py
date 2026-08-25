from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np

from .logic_registry import LOGICS, LOGIC_BY_VERSION


# These values are preregistered before any version Test result is opened.  Every
# version owns its own candidate pool and freezes its own Top1; there is no
# cross-version winner table used to decide Test entry.
DEFAULT_WINDOWS = (3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 90, 120)
HURST_WINDOWS = (8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 90, 105, 120, 150, 180)
DTW_WINDOWS = (4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 90, 120, 180)

MIN_ZERO_AGES = (1, 2, 3, 5)
ENTRY_QUANTILES = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
CONFIRMATION_DAYS = (1, 2, 3)


@dataclass(frozen=True)
class HoldingPackage:
    package_id: str
    min_hold_days: int
    max_hold_days: int
    release_quantile_gap: float
    cooldown_days: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


HOLDING_PACKAGES = (
    HoldingPackage("H01", 1, 3, 0.05, 1),
    HoldingPackage("H02", 2, 5, 0.10, 2),
    HoldingPackage("H03", 3, 10, 0.15, 3),
    HoldingPackage("H04", 5, 20, 0.20, 5),
)

RAW_CANDIDATES_PER_SIDE_VERSION = (
    16
    * len(MIN_ZERO_AGES)
    * len(ENTRY_QUANTILES)
    * len(CONFIRMATION_DAYS)
    * len(HOLDING_PACKAGES)
)


JOINT_SCORE_WEIGHTS = {
    "dev_mean_directional_o2o_h1": 0.15,
    "val_mean_directional_o2o_h1": 0.18,
    "pooled_mean_directional_o2o_h1": 0.10,
    "worst_phase_mean_directional_o2o_h1": 0.07,
    "pooled_h1_hit_rate": 0.08,
    "worst_phase_rank_ic": 0.06,
    "pooled_mean_directional_o2o_h3": 0.03,
    "h1_h3_path_consistency": 0.03,
    "pooled_target_transition_rate": 0.15,
    "pooled_transition_purity": 0.10,
    "one_minus_rapid_restore_rate": 0.05,
}

MINIMUM_COMPUTABILITY = {
    "development_selected_days": 20,
    "validation_selected_days": 10,
}

# Local research badge and local freeze preference. The company package deliberately
# removes these stability gates and uses only MINIMUM_COMPUTABILITY.
LOCAL_FORMAL_PASS_GATES = {
    "dev_mean_directional_o2o_h1_strictly_positive": True,
    "val_mean_directional_o2o_h1_strictly_positive": True,
    "dev_improvement_vs_all_eligible_zero_strictly_positive": True,
    "val_improvement_vs_all_eligible_zero_strictly_positive": True,
    "pooled_h1_hit_rate_min": 0.52,
    "dev_rank_ic_min": 0.0,
    "val_rank_ic_min": 0.0,
    "pooled_mean_directional_o2o_h3_strictly_positive": True,
    "h1_h3_path_consistency_min": 0.50,
    # These are deliberately modest because 0->+/-1 transitions are rare.
    # They prevent a high O2O result from being accepted when the selected
    # days contain mostly neutral continuation or the opposite transition.
    "pooled_target_transition_rate_min": 0.02,
    "pooled_transition_purity_min": 0.50,
    "pooled_target_transition_lift_min": 0.0,
    "rapid_restore_rate_max": 0.50,
    "pooled_eligible_coverage_min": 0.02,
    "pooled_eligible_coverage_max": 0.35,
}

BOOTSTRAP_PLAN = {
    "method": "moving_block_bootstrap_on_full_trading_day_panel",
    "block_length_trading_days": 20,
    "replications": 2000,
    "seed": 15452026,
    "confidence_level": 0.95,
    "scope": "frozen_top20_and_top1_diagnostics_only_not_selection",
}

MONOTONICITY_PLAN = {
    "groups": 5,
    "cutpoints_fit_on": "development_eligible_zero_scores_only",
    "validation_and_test_use_frozen_cutpoints": True,
    "tie_policy": "merge_adjacent_equal_cutpoints_and_report_actual_group_count",
}


def score_variant_configs(version: str) -> tuple[dict[str, object], ...]:
    """Return the exact 16 score variants preregistered for a version."""

    method = LOGIC_BY_VERSION[version].method_key
    reserve_axes: dict[str, tuple[tuple[str, tuple[object, ...]], tuple[str, tuple[object, ...]]]] = {
        "kernel_mmd": (("window", (10, 20, 40, 60)), ("bandwidth_multiplier", (0.5, 1.0, 2.0, 4.0))),
        "energy_distance": (("window", (10, 20, 40, 60)), ("distance_exponent", (0.5, 1.0, 1.5, 1.9))),
        "sliced_wasserstein": (("window", (10, 20, 40, 60)), ("projection_count", (4, 8, 16, 32))),
        "rulsif": (("window", (10, 20, 40, 60)), ("relative_alpha", (0.1, 0.3, 0.5, 0.7))),
        "graph_scan": (("window", (10, 20, 40, 60)), ("knn_k", (1, 3, 5, 8))),
        "permutation_entropy": (("embedding_m", (3, 4, 5, 6)), ("delay", (1, 2, 3, 5))),
        "rqa": (("embedding_m", (2, 3, 4, 5)), ("delay", (1, 2, 3, 5))),
        "visibility_graph": (("window", (20, 40, 60, 120)), ("statistic", ("end_degree", "degree_asymmetry", "kl_irreversibility", "slope_degree"))),
        "matrix_profile": (("subsequence_length", (5, 10, 20, 40)), ("neighbor_count", (3, 5, 10, 20))),
        "sax": (("window", (10, 20, 40, 60)), ("alphabet_size", (3, 4, 5, 6))),
        "transfer_entropy": (("driver", ("volume", "range", "gap", "intraday")), ("driver_lag", (1, 2, 3, 5))),
        "directional_change": (("volatility_window", (10, 20, 40, 60)), ("event_threshold_sigma", (0.5, 0.75, 1.0, 1.5))),
        "hawkes": (("event_threshold_sigma", (0.0, 0.5, 1.0, 1.5)), ("half_life", (2, 5, 10, 20))),
        "l1_trend_filter": (("window", (20, 40, 60, 120)), ("lambda_scale", (0.5, 1.0, 2.0, 4.0))),
        "ssa": (("window", (20, 40, 60, 120)), ("rank", (1, 2, 3, 4))),
        "wavelet": (("wavelet", ("haar", "db2", "db4", "sym4")), ("level", (1, 2, 3, 4))),
        "dmd": (("delay_depth", (2, 3, 5, 10)), ("rank", (1, 2, 3, 4))),
        "path_signature": (("path_length", (5, 10, 20, 40)), ("signature_order", (1, 2, 3, 4))),
        "persistent_homology": (("embedding_m", (2, 3, 4, 5)), ("delay", (1, 2, 3, 5))),
        "edm_simplex": (("embedding_e", (2, 3, 4, 5)), ("neighbor_multiplier", (1, 2, 3, 4))),
        "hsmm": (("state_count", (2, 3, 4, 5)), ("max_duration", (10, 20, 40, 80))),
        "fine_gray": (("horizon", (1, 2, 3, 5)), ("l2", (0.1, 1.0, 10.0, 100.0))),
        "aalen": (("horizon", (1, 2, 3, 5)), ("ridge", (0.1, 1.0, 10.0, 100.0))),
        "gam": (("basis_dimension", (3, 4, 5, 6)), ("smoothing_penalty", (0.1, 1.0, 10.0, 100.0))),
        "quantile_regression": (("tau", (0.05, 0.10, 0.20, 0.30)), ("l2", (0.1, 1.0, 10.0, 100.0))),
        "expectile": (("tau", (0.05, 0.10, 0.20, 0.30)), ("ridge", (0.1, 1.0, 10.0, 100.0))),
        "elastic_net": (("l1_ratio", (0.1, 0.3, 0.6, 0.9)), ("alpha", (0.001, 0.01, 0.1, 1.0))),
        "gaussian_process": (("kernel", ("rbf", "matern15", "matern25", "rq")), ("noise", (1e-4, 1e-3, 1e-2, 1e-1))),
        "svr": (("C", (0.1, 1.0, 10.0, 100.0)), ("gamma_multiplier", (0.25, 0.5, 1.0, 2.0))),
        "random_forest": (("max_depth", (2, 3, 4, 6)), ("min_samples_leaf", (10, 20, 40, 80))),
        "rda": (("covariance_shrinkage", (0.0, 0.33, 0.67, 1.0)), ("class_pooling", (0.0, 0.33, 0.67, 1.0))),
        "setar": (("ar_lag", (1, 2, 3, 5)), ("threshold_quantile", (0.25, 0.40, 0.60, 0.75))),
        "star": (("ar_lag", (1, 2, 3, 5)), ("smoothness", (1.0, 2.0, 5.0, 10.0))),
        "qar": (("ar_lag", (1, 2, 3, 5)), ("tau", (0.05, 0.10, 0.20, 0.30))),
        "gas": (("distribution", ("gaussian", "t5", "t10", "asymmetric_laplace")), ("persistence", (0.80, 0.90, 0.95, 0.98))),
        "split_conformal": (("ridge", (0.1, 1.0, 10.0, 100.0)), ("miscoverage", (0.05, 0.10, 0.15, 0.20))),
        "conformal_martingale": (("calibration_window", (20, 40, 60, 120)), ("betting_epsilon", (0.5, 0.7, 0.9, "mixture"))),
        "sparse_varx": (("lag_order", (1, 2, 3, 5)), ("structured_penalty", (0.001, 0.01, 0.1, 1.0))),
        "echo_state": (("reservoir_size", (20, 50, 100, 200)), ("spectral_radius", (0.20, 0.50, 0.80, 0.95))),
        "projection_pursuit": (("ridge_function_count", (1, 2, 3, 4)), ("smoother_span", (0.10, 0.20, 0.30, 0.40))),
        # V91--V120 paper-derived reserve axes.  Exactly two four-level axes
        # give each new logic its own 16-score sub-grid before entry/holding
        # parameters are expanded.
        "garch": (("omega_scale", (0.02, 0.05, 0.10, 0.20)), ("persistence", (0.80, 0.90, 0.95, 0.98))),
        "egarch": (("leverage", (0.05, 0.15, 0.30, 0.50)), ("persistence", (0.80, 0.90, 0.95, 0.98))),
        "gjr_garch": (("threshold_gamma", (0.05, 0.15, 0.30, 0.50)), ("persistence", (0.80, 0.90, 0.95, 0.98))),
        "aparch": (("power", (0.50, 0.75, 1.00, 1.50)), ("asymmetry", (0.00, 0.20, 0.40, 0.60))),
        "figarch": (("fractional_d", (0.10, 0.30, 0.50, 0.70)), ("memory_length", (20, 40, 80, 120))),
        "har_rv": (("weekly_weight", (0.25, 0.50, 0.75, 1.00)), ("monthly_weight", (0.25, 0.50, 0.75, 1.00))),
        "caviar": (("tau", (0.05, 0.10, 0.20, 0.30)), ("persistence", (0.80, 0.90, 0.95, 0.98))),
        "novas": (("volatility_window", (5, 10, 20, 40)), ("shock_clip", (1.5, 2.0, 3.0, 4.0))),
        "evt_pot": (("threshold_quantile", (0.80, 0.90, 0.95, 0.98)), ("tail_window", (40, 80, 120, 240))),
        "hill_tail": (("tail_fraction", (0.05, 0.10, 0.20, 0.30)), ("window", (40, 80, 120, 240))),
        "smap": (("embedding", (2, 3, 4, 5)), ("theta", (0.0, 1.0, 2.0, 4.0))),
        "analog_knn": (("embedding", (2, 3, 4, 5)), ("neighbor_count", (3, 5, 10, 20))),
        "shapelet": (("shape_length", (4, 6, 8, 12)), ("shape_count", (2, 4, 8, 12))),
        "sample_entropy": (("embedding_m", (2, 3, 4, 5)), ("tolerance_scale", (0.10, 0.20, 0.30, 0.50))),
        "multiscale_entropy": (("embedding_m", (2, 3, 4, 5)), ("max_scale", (2, 3, 4, 5))),
        "lz_complexity": (("alphabet_size", (2, 3, 4, 5)), ("window", (20, 40, 80, 120))),
        "higuchi_fd": (("k_max", (4, 6, 8, 12)), ("window", (20, 40, 80, 120))),
        "hilbert_phase": (("window", (20, 40, 80, 120)), ("amplitude_weight", (0.25, 0.50, 0.75, 1.00))),
        "emd_residual": (("sift_passes", (1, 2, 3, 4)), ("window", (20, 40, 80, 120))),
        "vmd": (("mode_count", (2, 3, 4, 5)), ("bandwidth", (0.25, 0.50, 1.00, 2.00))),
        "rocket": (("kernel_count", (32, 64, 128, 256)), ("max_kernel_length", (5, 9, 15, 21))),
        "elm": (("hidden_units", (16, 32, 64, 128)), ("activation", ("tanh", "relu", "sin", "sigmoid"))),
        "rff_ridge": (("feature_count", (32, 64, 128, 256)), ("bandwidth", (0.25, 0.50, 1.00, 2.00))),
        "pls": (("components", (1, 2, 3, 4)), ("scale_mode", ("standard", "robust", "none", "unit"))),
        "huber_regression": (("epsilon", (1.10, 1.35, 1.75, 2.50)), ("alpha", (0.001, 0.01, 0.10, 1.00))),
        "ransac_trend": (("residual_threshold", (0.5, 1.0, 1.5, 2.0)), ("min_samples_fraction", (0.50, 0.60, 0.75, 0.90))),
        "logistic_hazard": (("C", (0.01, 0.10, 1.00, 10.00)), ("class_weight", ("balanced", "none", "up_weighted", "down_weighted"))),
        "qda": (("reg_param", (0.00, 0.10, 0.30, 0.60)), ("class_prior", ("empirical", "uniform", "up_prior", "down_prior"))),
        "naive_bayes": (("var_smoothing", (1e-11, 1e-9, 1e-7, 1e-5)), ("prior_mode", ("empirical", "uniform", "up_prior", "down_prior"))),
        "extra_trees": (("max_depth", (2, 3, 4, 6)), ("min_samples_leaf", (10, 20, 40, 80))),
    }
    if method in reserve_axes:
        (left_name, left_values), (right_name, right_values) = reserve_axes[method]
        rows = [
            {left_name: left, right_name: right, "reserve_variant": number}
            for number, (left, right) in enumerate(product(left_values, right_values))
        ]
        if len(rows) != 16:
            raise AssertionError(f"{version} reserve axes must form exactly 16 variants")
        return tuple(rows)
    if method == "zero_hazard":
        rows = [
            {
                "momentum_window": int(momentum_window),
                "age_bin_width": int(age_bin_width),
                "beta_smoothing": float(beta_smoothing),
            }
            for momentum_window, age_bin_width, beta_smoothing in product(
                (3, 5, 10, 20), (2, 5), (1.0, 5.0)
            )
        ]
    elif method == "gradient_boosting":
        rows = [
            {
                "max_leaf_nodes": int(leaves),
                "max_depth": int(depth),
                "learning_rate": float(rate),
                "l2_regularization": float(l2),
                "max_iter": 100,
                "min_samples_leaf": 20,
                "random_state": 20260813,
            }
            for leaves, depth, rate, l2 in product(
                (5, 9), (2, 3), (0.03, 0.07), (1.0, 10.0)
            )
        ]
    else:
        if method == "hurst_rs":
            windows = HURST_WINDOWS
        elif method == "dtw_template":
            windows = DTW_WINDOWS
        else:
            windows = DEFAULT_WINDOWS
        rows = []
        kalman_q = np.logspace(-5.0, -1.0, 16)
        for number, window in enumerate(windows):
            row: dict[str, object] = {"window": int(window)}
            if method == "ema_crossover":
                row.update(fast_span=max(2, window // 3), slow_span=window)
            elif method == "macd":
                row.update(
                    fast_span=max(2, window // 3),
                    slow_span=max(max(2, window // 3) + 1, window),
                    signal_span=max(2, int(np.sqrt(window))),
                )
            elif method == "cusum":
                row["allowance"] = 0.10 + 0.04 * (number % 4)
            elif method == "page_hinkley":
                row.update(
                    delta=0.02 * (1 + number % 4),
                    decay=0.95 + 0.01 * (number % 4),
                )
            elif method in {"kalman_slope", "kalman_innovation"}:
                row["process_to_observation_variance_ratio"] = float(kalman_q[number])
            elif method == "bocpd_proxy":
                row["half_window"] = max(2, window // 2)
            elif method == "pelt_proxy":
                row["half_window"] = max(2, window // 2)
            elif method == "variance_ratio":
                row["aggregation_lag"] = 2 + number % 4
            elif method == "autocorrelation":
                row["lag"] = 1 + number % 4
            elif method == "volatility_regime":
                row["short_window"] = max(2, window // 4)
            elif method in {"turnover_surprise", "volume_breakout", "drawdown_recovery"}:
                row["direction_or_recovery_lag"] = max(1, window // 4)
            elif method == "kernel_path":
                row["bandwidth_divisor"] = 3 + number % 4
            elif method == "hamilton_filter":
                row.update(
                    persistence=0.90 + 0.006 * number,
                    state_mean_scale=0.20 + 0.05 * (number % 4),
                )
            rows.append(row)

    if len(rows) != 16:
        raise AssertionError(f"{version} must have exactly 16 score variants, got {len(rows)}")
    if len({repr(sorted(row.items())) for row in rows}) != 16:
        raise AssertionError(f"{version} contains duplicate score-variant parameter rows")
    return tuple(rows)


SCORE_VARIANTS_BY_VERSION = {
    spec.version: score_variant_configs(spec.version) for spec in LOGICS
}


def candidate_grid_summary() -> dict[str, object]:
    return {
        "score_variants": 16,
        "minimum_zero_ages": list(MIN_ZERO_AGES),
        "entry_quantiles": list(ENTRY_QUANTILES),
        "confirmation_days": list(CONFIRMATION_DAYS),
        "holding_packages": [row.to_dict() for row in HOLDING_PACKAGES],
        "raw_candidates_per_side_version": RAW_CANDIDATES_PER_SIDE_VERSION,
        "raw_candidates_all_120_versions_two_sides": RAW_CANDIDATES_PER_SIDE_VERSION * 120 * 2,
        "raw_candidates_v91_v120_two_sides": RAW_CANDIDATES_PER_SIDE_VERSION * 30 * 2,
    }


if len(SCORE_VARIANTS_BY_VERSION) != 120:
    raise AssertionError("The design registry must cover exactly V01-V120")
if RAW_CANDIDATES_PER_SIDE_VERSION != 6144:
    raise AssertionError("The preregistered raw pool must contain 6,144 candidates")
if not np.isclose(sum(JOINT_SCORE_WEIGHTS.values()), 1.0):
    raise AssertionError("Joint score weights must sum to one")

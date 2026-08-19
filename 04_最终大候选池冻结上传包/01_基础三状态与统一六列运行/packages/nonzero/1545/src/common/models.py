from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .indicators import rolling_slope, rolling_zscore, true_range
from .registry import BY_ID, SCORE_VARIANTS
from .scores import exit_sign, side_state


SEED = 1545
DEV_END = pd.Timestamp("2022-12-31")


def build_model_feature_groups(panel: pd.DataFrame, side: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    sign = exit_sign(side)
    threshold = 0.42 if side == "minus" else 0.58
    close = panel["close"].astype(float)
    returns = close.pct_change(fill_method=None)
    span = panel["high"].sub(panel["low"]).replace(0.0, np.nan)
    prior_close = close.shift(1)
    true_range_value = true_range(panel)

    output = pd.DataFrame(index=panel.index)
    output["state_age"] = panel["state_age"].astype(float)
    output["log_state_age"] = np.log1p(output["state_age"])
    output["exit_rule_margin"] = sign * panel["rule_axis"].sub(threshold)
    output["exit_rule_continuous_margin"] = sign * panel["rule_axis_continuous"].sub(threshold)
    output["exit_rule_band_margin"] = sign * panel["rule_axis_band"].sub(threshold)
    output["exit_fast_slow_spread"] = sign * panel["fast_engine"].sub(panel["slow_engine"])
    allowed_state_names = (
        "position", "reversal", "tail", "trend", "path", "intraday", "heat", "volume_price"
    )
    for name in allowed_state_names:
        output[f"level_{name}"] = panel[f"cv_{name}"].astype(float)

    for lag in (1, 2, 3, 5):
        output[f"axis_change_{lag}"] = sign * panel["rule_axis"].diff(lag).div(lag)
        output[f"fast_change_{lag}"] = sign * panel["fast_engine"].diff(lag).div(lag)
        output[f"slow_change_{lag}"] = sign * panel["slow_engine"].diff(lag).div(lag)

    for lag in (1, 2, 3, 5, 10, 20, 40):
        output[f"exit_return_{lag}"] = sign * close.pct_change(lag, fill_method=None)
    output["exit_intraday_body"] = sign * panel["close"].sub(panel["open"]).div(span)
    close_location = panel["close"].sub(panel["low"]).div(span)
    output["exit_close_location"] = close_location if side == "minus" else 1.0 - close_location
    output["exit_gap"] = sign * panel["open"].div(prior_close).sub(1)
    output["range_fraction"] = span.div(prior_close)
    output["atr_14_fraction"] = true_range_value.rolling(14, min_periods=7).mean().div(close)
    output["lower_wick_fraction"] = (
        pd.concat([panel["open"], panel["close"]], axis=1).min(axis=1).sub(panel["low"]).div(span)
    )
    output["upper_wick_fraction"] = (
        panel["high"].sub(pd.concat([panel["open"], panel["close"]], axis=1).max(axis=1)).div(span)
    )
    for window in (10, 20, 40):
        output[f"exit_price_slope_{window}"] = sign * rolling_slope(np.log(close), window)
        low = close.rolling(window, min_periods=window).min()
        high = close.rolling(window, min_periods=window).max()
        location = close.sub(low).div(high.sub(low).replace(0.0, np.nan))
        output[f"exit_channel_location_{window}"] = location if side == "minus" else 1.0 - location

    log_volume = np.log1p(panel["volume"])
    log_amount = np.log1p(panel["total_turnover"])
    for lag in (1, 5, 20):
        output[f"volume_change_{lag}"] = log_volume.diff(lag).div(lag)
        output[f"amount_change_{lag}"] = log_amount.diff(lag).div(lag)
    output["volume_z_20"] = rolling_zscore(log_volume, 20)
    output["volume_z_60"] = rolling_zscore(log_volume, 60)
    output["amount_z_20"] = rolling_zscore(log_amount, 20)
    output["signed_volume_10"] = (
        np.sign(sign * returns).fillna(0).mul(panel["volume"]).rolling(10, min_periods=10).sum()
        .div(panel["volume"].rolling(10, min_periods=10).sum().replace(0.0, np.nan))
    )

    output["age_x_margin"] = output["log_state_age"] * output["exit_rule_margin"]
    output["axis_x_return"] = output["exit_rule_margin"] * output["exit_return_3"]
    output["price_x_volume"] = output["exit_return_5"] * output["volume_z_20"]
    output["fast_x_path"] = output["exit_fast_slow_spread"] * output["exit_return_3"]

    rule_level = [
        "state_age", "log_state_age", "exit_rule_margin", "exit_rule_continuous_margin",
        "exit_rule_band_margin", "exit_fast_slow_spread",
        *[f"level_{name}" for name in allowed_state_names],
    ]
    rule_change = [f"{prefix}_{lag}" for prefix in ("axis_change", "fast_change", "slow_change") for lag in (1,2,3,5)]
    short_path = [f"exit_return_{lag}" for lag in (1,2,3,5)] + ["exit_intraday_body", "exit_close_location"]
    long_path = [f"exit_return_{lag}" for lag in (10,20,40)] + [
        f"exit_price_slope_{window}" for window in (10,20,40)
    ] + [f"exit_channel_location_{window}" for window in (10,20,40)]
    range_gap = ["exit_gap", "range_fraction", "atr_14_fraction", "lower_wick_fraction", "upper_wick_fraction"]
    spot_volume = [
        *[f"volume_change_{lag}" for lag in (1,5,20)],
        *[f"amount_change_{lag}" for lag in (1,5,20)],
        "volume_z_20", "volume_z_60", "amount_z_20", "signed_volume_10",
    ]
    compact = [
        "log_state_age", "exit_rule_margin", "exit_fast_slow_spread", "level_reversal",
        "level_tail", "level_path", "level_heat", "axis_change_1", "axis_change_3",
        "exit_return_1", "exit_return_3", "exit_return_10", "exit_intraday_body",
        "exit_close_location", "atr_14_fraction", "exit_gap", "volume_z_20", "signed_volume_10",
    ]
    interactions = ["age_x_margin", "axis_x_return", "price_x_volume", "fast_x_path"]
    groups = {
        "rule_level": rule_level,
        "rule_level_change": list(dict.fromkeys(rule_level + rule_change)),
        "short_path": short_path,
        "long_path_range": list(dict.fromkeys(long_path + range_gap)),
        "rule_short_range": list(dict.fromkeys(rule_level + rule_change + short_path + range_gap)),
        "rule_spot_volume": list(dict.fromkeys(rule_level + rule_change + spot_volume)),
        "compact_all": compact,
        "all_plus_interactions": list(dict.fromkeys(compact + rule_change + long_path + range_gap + spot_volume + interactions)),
    }
    return output.replace([np.inf, -np.inf], np.nan), groups


def _variant_configs(version_id: str) -> list[dict[str, Any]]:
    feature_sets = [
        "rule_level", "rule_level_change", "short_path", "long_path_range",
        "rule_short_range", "rule_spot_volume", "compact_all", "all_plus_interactions",
    ]
    if version_id == "V41":
        params = [(1.10,1e-5),(1.20,1e-4),(1.35,1e-4),(1.50,1e-3),(1.75,1e-3),(2.0,1e-2),(2.5,1e-2),(3.0,1e-1)]
        return [{"feature_set": f, "epsilon": e, "alpha": a} for f, (e, a) in zip(feature_sets, params)]
    if version_id in {"V42", "V43"}:
        values = (.01,.03,.10,.30,1.0,3.0,10.0,30.0)
        return [{"feature_set": f, "C": c} for f, c in zip(feature_sets, values)]
    if version_id == "V44":
        params = ((.03,.10),(.10,.20),(.30,.35),(1.0,.50),(1.0,.65),(3.0,.80),(10.0,.90),(30.0,.95))
        return [{"feature_set": f, "C": c, "l1_ratio": ratio} for f, (c, ratio) in zip(feature_sets, params)]
    if version_id == "V45":
        params = ((3,2,.1),(4,2,.3),(5,2,1.0),(6,2,3.0),(3,3,.1),(4,3,.3),(5,3,1.0),(6,3,3.0))
        return [{"feature_set": f, "n_knots": knots, "degree": degree, "C": c} for f, (knots, degree, c) in zip(feature_sets, params)]
    if version_id == "V46":
        values = (.005,.02,.05,.20,.50,2.0,8.0,20.0)
        return [{"feature_set": f, "C": c, "class_weight": "balanced" if number >= 4 else None} for number, (f, c) in enumerate(zip(feature_sets, values))]
    if version_id == "V47":
        params = ((1,8),(2,8),(2,12),(3,8),(3,12),(4,12),(4,20),(5,20))
        return [{"feature_set": f, "max_depth": depth, "min_samples_leaf": leaf} for f, (depth, leaf) in zip(feature_sets, params)]
    if version_id in {"V48", "V49"}:
        params = ((2,4,.5),(3,4,.7),(3,8,.5),(4,8,.7),(5,8,1.0),(6,12,.7),(8,12,.8),(None,12,1.0))
        return [{"feature_set": f, "max_depth": depth, "min_samples_leaf": leaf, "max_features": fraction, "n_estimators": 300} for f, (depth, leaf, fraction) in zip(feature_sets, params)]
    if version_id == "V50":
        params = (
            (.03,150,7,1.0,15),(.05,120,7,2.0,12),(.05,160,15,2.0,12),(.08,120,15,3.0,10),
            (.10,100,15,5.0,10),(.05,200,31,5.0,12),(.08,160,31,8.0,15),(.10,150,31,10.0,20),
        )
        return [
            {"feature_set": f, "learning_rate": lr, "max_iter": iterations, "max_leaf_nodes": leaves, "l2_regularization": l2, "min_samples_leaf": min_leaf}
            for f, (lr, iterations, leaves, l2, min_leaf) in zip(feature_sets, params)
        ]
    raise ValueError(version_id)


def _estimator(version_id: str, config: dict[str, Any]) -> tuple[Any, str]:
    impute_scale = [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if version_id == "V41":
        estimator = Pipeline(impute_scale + [("model", HuberRegressor(epsilon=config["epsilon"], alpha=config["alpha"], max_iter=1000))])
        return estimator, "regression"
    if version_id in {"V42", "V43", "V44"}:
        penalty = {"V42": "l2", "V43": "l1", "V44": "elasticnet"}[version_id]
        solver = "lbfgs" if version_id == "V42" else ("liblinear" if version_id == "V43" else "saga")
        kwargs: dict[str, Any] = {"C": config["C"], "penalty": penalty, "solver": solver, "max_iter": 5000, "random_state": SEED}
        if version_id == "V44":
            kwargs["l1_ratio"] = config["l1_ratio"]
        return Pipeline(impute_scale + [("model", LogisticRegression(**kwargs))]), "classification"
    if version_id == "V45":
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(n_knots=config["n_knots"], degree=config["degree"], include_bias=False)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=config["C"], penalty="l2", solver="lbfgs", max_iter=5000, random_state=SEED)),
        ])
        return estimator, "classification"
    if version_id == "V46":
        estimator = Pipeline(impute_scale + [("model", LinearSVC(C=config["C"], class_weight=config["class_weight"], random_state=SEED, dual="auto", max_iter=10000))])
        return estimator, "classification"
    if version_id == "V47":
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(max_depth=config["max_depth"], min_samples_leaf=config["min_samples_leaf"], class_weight="balanced", random_state=SEED)),
        ])
        return estimator, "classification"
    if version_id in {"V48", "V49"}:
        cls = RandomForestClassifier if version_id == "V48" else ExtraTreesClassifier
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", cls(
                n_estimators=config["n_estimators"], max_depth=config["max_depth"],
                min_samples_leaf=config["min_samples_leaf"], max_features=config["max_features"],
                class_weight="balanced_subsample" if version_id == "V48" else "balanced",
                random_state=SEED, n_jobs=1,
            )),
        ])
        return estimator, "classification"
    if version_id == "V50":
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=config["learning_rate"], max_iter=config["max_iter"],
                max_leaf_nodes=config["max_leaf_nodes"], l2_regularization=config["l2_regularization"],
                min_samples_leaf=config["min_samples_leaf"], random_state=SEED,
            )),
        ])
        return estimator, "classification"
    raise ValueError(version_id)


def _predict_score(estimator: Any, kind: str, features: pd.DataFrame) -> np.ndarray:
    if kind == "regression":
        return np.asarray(estimator.predict(features), dtype=float)
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)
        classes = list(estimator.named_steps["model"].classes_)
        return np.asarray(probabilities[:, classes.index(1)], dtype=float)
    return np.asarray(estimator.decision_function(features), dtype=float)


def _valid_training_target(target: pd.Series, kind: str) -> bool:
    finite = target.dropna()
    if len(finite) < 20:
        return False
    return kind == "regression" or finite.nunique() >= 2


def build_model_score_variants(panel: pd.DataFrame, version_id: str, side: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    spec = BY_ID[version_id]
    if spec.group != "model":
        raise ValueError(f"{version_id} is not a model version")
    feature_frame, feature_groups = build_model_feature_groups(panel, side)
    state = side_state(side)
    improvement = exit_sign(side) * panel["o2o_h1"]
    train_mask = (
        panel["base_state"].eq(state)
        & panel.index.to_series().le(DEV_END)
        & panel["exit_h1_date"].le(DEV_END)
        & improvement.notna()
    )
    configs = _variant_configs(version_id)
    scores: list[pd.Series] = []
    metadata: list[dict[str, Any]] = []

    for number, config in enumerate(configs, start=1):
        columns = feature_groups[config["feature_set"]]
        x = feature_frame[columns]
        positions = np.flatnonzero(train_mask.to_numpy())
        y_cont = improvement.iloc[positions].astype(float)
        estimator, kind = _estimator(version_id, config)
        if kind == "regression":
            lower, upper = y_cont.quantile([0.02, 0.98]).tolist()
            y = y_cont.clip(lower=lower, upper=upper)
        else:
            y = y_cont.gt(0).astype(int)
        output = pd.Series(np.nan, index=panel.index, dtype=float)
        failures: list[str] = []

        if _valid_training_target(y, kind):
            split_count = min(4, max(2, len(positions) // 25))
            splitter = TimeSeriesSplit(n_splits=split_count, gap=2)
            for fold, (train_local, test_local) in enumerate(splitter.split(positions), start=1):
                fold_positions = positions[train_local]
                test_positions = positions[test_local]
                fold_y = y.iloc[train_local]
                if not _valid_training_target(fold_y, kind):
                    failures.append(f"fold_{fold}_insufficient_target")
                    continue
                fold_estimator, _ = _estimator(version_id, config)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fold_estimator.fit(x.iloc[fold_positions], fold_y)
                    output.iloc[test_positions] = _predict_score(fold_estimator, kind, x.iloc[test_positions])
                except Exception as error:  # retained in metadata; version still produces a computability audit
                    failures.append(f"fold_{fold}_{type(error).__name__}:{error}")

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    estimator.fit(x.iloc[positions], y)
                future_positions = np.flatnonzero(
                    (panel["base_state"].eq(state) & panel.index.to_series().gt(DEV_END)).to_numpy()
                )
                if len(future_positions):
                    output.iloc[future_positions] = _predict_score(estimator, kind, x.iloc[future_positions])
            except Exception as error:
                failures.append(f"full_fit_{type(error).__name__}:{error}")
        else:
            failures.append("insufficient_development_target")

        scores.append(output)
        metadata.append({
            "score_variant": f"score_{number:02d}",
            "method": spec.core_logic_name,
            "parameters": config,
            "feature_columns": columns,
            "feature_count": len(columns),
            "development_fit_rows": int(len(positions)),
            "development_positive_rate": float(y_cont.gt(0).mean()) if len(y_cont) else np.nan,
            "development_oof_score_rows": int(output.loc[:DEV_END].notna().sum()),
            "fit_uses_validation": False,
            "fit_uses_test": False,
            "purge_trading_rows": 2,
            "failures": failures,
        })

    frame = pd.concat(scores, axis=1)
    frame.columns = [f"score_{number:02d}" for number in range(1, SCORE_VARIANTS + 1)]
    return frame.replace([np.inf, -np.inf], np.nan), metadata


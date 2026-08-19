from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VersionSpec:
    version: int
    core_logic_name: str
    title_cn: str
    group: str
    literature: str

    @property
    def version_id(self) -> str:
        return f"V{self.version:02d}"

    def as_dict(self) -> dict:
        return asdict(self) | {"version_id": self.version_id}


_ROWS = [
    (1, "parametric_state_age_hazard", "参数化状态年龄风险率", "duration", "Prentice–Gloeckler (1978)"),
    (2, "grouped_discrete_time_hazard", "分组离散时间风险率", "duration", "Prentice–Gloeckler (1978)"),
    (3, "age_rule_interaction_hazard", "年龄与规则证据交互风险率", "duration", "Prentice–Gloeckler (1978)"),
    (4, "causal_exit_survival_residual", "因果退出生存残差", "duration", "grouped survival adaptation"),
    (5, "online_exit_intensity", "在线退出事件强度", "duration", "causal counting-process adaptation"),
    (6, "rule_evidence_first_difference", "规则证据一阶变化", "rule_evidence", "frozen-rule derivative"),
    (7, "rule_evidence_second_difference", "规则证据二阶变化", "rule_evidence", "frozen-rule derivative"),
    (8, "rule_evidence_curvature", "规则证据多尺度曲率", "rule_evidence", "local polynomial derivative"),
    (9, "slow_fast_engine_cross", "快慢引擎交叉", "rule_evidence", "dual-engine crossover"),
    (10, "continuation_margin_erosion", "延续阈值边际侵蚀", "rule_evidence", "frozen-rule distance"),
    (11, "one_sided_cusum", "单侧 CUSUM", "changepoint", "Page (1954)"),
    (12, "two_sided_cusum_asymmetry", "双侧 CUSUM 不对称", "changepoint", "Page (1954)"),
    (13, "page_hinkley", "Page–Hinkley 变化检测", "changepoint", "Page-style sequential monitoring"),
    (14, "ewma_standardized_innovation", "EWMA 标准化创新", "changepoint", "sequential EWMA monitoring"),
    (15, "bayesian_online_changepoint", "贝叶斯在线变点", "changepoint", "Adams–MacKay (2007)"),
    (16, "penalized_last_break", "惩罚末端断点", "changepoint", "Killick–Fearnhead–Eckley (2012)"),
    (17, "binary_segmentation_slope_break", "二分斜率断裂", "changepoint", "causal binary segmentation"),
    (18, "theil_sen_slope_break", "Theil–Sen 稳健斜率断裂", "changepoint", "robust slope comparison"),
    (19, "chow_structural_break", "Chow 型结构断裂", "changepoint", "rolling structural-break test"),
    (20, "kalman_level_innovation", "Kalman 局部水平创新", "state_space", "linear Gaussian state space"),
    (21, "kalman_local_trend_reversal", "Kalman 局部趋势反转", "state_space", "local-linear-trend state space"),
    (22, "rolling_extreme_repair", "滚动极值修复", "spot_path", "causal rolling path"),
    (23, "drawdown_recovery_ratio", "回撤修复比例", "spot_path", "drawdown path decomposition"),
    (24, "failed_breakout", "突破失败", "spot_path", "prior-range breakout failure"),
    (25, "gap_body_reversal", "缺口实体反转", "spot_path", "daily gap/body decomposition"),
    (26, "wick_rejection", "影线拒绝", "spot_path", "OHLC candlestick geometry"),
    (27, "close_location_repair", "收盘位置修复", "spot_path", "close-location value"),
    (28, "volume_price_divergence", "量价背离", "spot_volume", "price-volume divergence"),
    (29, "obv_slope_reversal", "OBV 斜率反转", "spot_volume", "on-balance volume"),
    (30, "signed_volume_imbalance", "有向成交量失衡", "spot_volume", "signed-volume participation"),
    (31, "rsi_failure_swing", "RSI 失败摆动", "spot_path", "Wilder-style RSI"),
    (32, "stochastic_reversal", "随机指标反转", "spot_path", "stochastic oscillator"),
    (33, "macd_histogram_rollover", "MACD 柱体滚落", "spot_path", "EMA convergence/divergence"),
    (34, "bollinger_reentry", "布林带再进入", "spot_path", "rolling location-scale channel"),
    (35, "atr_normalized_reversal", "ATR 标准化反转", "spot_path", "true-range normalization"),
    (36, "keltner_channel_failure", "Keltner 通道失败", "spot_path", "EMA/ATR channel"),
    (37, "efficiency_ratio_collapse", "效率比崩解", "spot_path", "Kaufman efficiency ratio"),
    (38, "path_curvature", "价格路径曲率", "spot_path", "rolling quadratic path"),
    (39, "return_sign_entropy", "收益符号熵", "spot_path", "binary Shannon entropy"),
    (40, "directional_run_exhaustion", "同向序列耗竭", "spot_path", "run-length exhaustion"),
    (41, "huber_robust_regression", "Huber 稳健回归", "model", "Huber (1964)"),
    (42, "ridge_logistic", "L2 Logistic", "model", "regularized GLM"),
    (43, "lasso_logistic", "L1 Logistic", "model", "Tibshirani (1996)"),
    (44, "elastic_net_logistic", "Elastic Net Logistic", "model", "Zou–Hastie (2005)"),
    (45, "spline_gam_logistic", "样条加性 Logistic", "model", "Hastie–Tibshirani (1986)"),
    (46, "linear_svm", "线性支持向量机", "model", "linear large-margin classification"),
    (47, "decision_tree", "浅层决策树", "model", "CART class"),
    (48, "random_forest", "随机森林", "model", "Breiman (2001)"),
    (49, "extra_trees", "极端随机树", "model", "randomized tree ensemble"),
    (50, "hist_gradient_boosting", "直方图梯度提升", "model", "Friedman (2001)"),
]

VERSIONS = tuple(VersionSpec(*row) for row in _ROWS)
BY_ID = {item.version_id: item for item in VERSIONS}

SCORE_QUANTILES = (0.50, 0.56, 0.62, 0.68, 0.74, 0.80, 0.85, 0.90, 0.94)
MIN_STATE_AGES = (1, 3, 5, 8)
CONFIRM_DAYS = (1, 2, 3)
COOLDOWN_DAYS = (0, 3, 7)
SCORE_VARIANTS = 8
RAW_CANDIDATES_PER_SIDE = (
    SCORE_VARIANTS
    * len(SCORE_QUANTILES)
    * len(MIN_STATE_AGES)
    * len(CONFIRM_DAYS)
    * len(COOLDOWN_DAYS)
)

assert len(VERSIONS) == 50
assert RAW_CANDIDATES_PER_SIDE == 2592


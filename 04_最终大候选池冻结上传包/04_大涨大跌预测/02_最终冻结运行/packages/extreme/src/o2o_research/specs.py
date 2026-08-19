from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VersionSpec:
    version: str
    core_logic_name: str
    title_zh: str
    hypothesis: str
    down_components: tuple[str, str, str, str]
    up_components: tuple[str, str, str, str]
    aggregator: str = "weighted_mean"
    component_cut: float = 0.65
    literature_keys: tuple[str, ...] = ()
    uses_1545: bool = False
    uses_nine_state: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# Component syntax: a plain name uses its causal [0, 1] rolling rank;
# inv:name reverses it; center:name measures distance from the neutral rank;
# low:name/high:name measure one-sided distance from the neutral rank.
_SPECS = [
    ("V01", "o2o_multihorizon_consensus", "历史 O2O 多尺度共识", "已完成的开盘到开盘收益在短中窗同向聚集时，下一腿尾部风险可能更高。", ("inv:oo_ret_1_rank252", "inv:oo_ret_5_rank252", "inv:oo_ret_20_rank252", "oo_vol_20_rank252"), ("oo_ret_1_rank252", "oo_ret_5_rank252", "oo_ret_20_rank252", "oo_vol_20_rank252"), "weighted_mean", ("path_vol",)),
    ("V02", "o2o_sign_persistence", "历史 O2O 符号持续", "开盘收益正负占比的持续偏斜可构成独立于收益均值的方向证据。", ("oo_down_share_20_rank252", "inv:oo_up_share_20_rank252", "inv:oo_ret_1_rank252", "risk_expansion_rank"), ("oo_up_share_20_rank252", "inv:oo_down_share_20_rank252", "oo_ret_1_rank252", "risk_expansion_rank"), "weighted_mean", ("direction",)),
    ("V03", "gap_pressure", "跳空压力", "当日跳空、日收益、收盘位置与振幅共同刻画开盘定价压力是否延续。", ("inv:gap_rank252", "inv:ret_1_rank252", "inv:close_location_rank252", "true_range_pct_rank252"), ("gap_rank252", "ret_1_rank252", "close_location_rank252", "true_range_pct_rank252"), "weighted_mean", ("technical",)),
    ("V04", "gap_intraday_divergence", "跳空与日内背离", "跳空方向与日内修复方向发生背离时，隔日开盘段可能出现非对称延续或回补。", ("gap_intraday_divergence_rank", "inv:gap_rank252", "inv:intraday_ret_rank252", "true_range_pct_rank252"), ("gap_intraday_divergence_rank", "gap_rank252", "intraday_ret_rank252", "true_range_pct_rank252"), "geometric", ("technical",)),
    ("V05", "candle_rejection", "K线拒绝形态", "上影/下影、收盘位置与真实振幅可量化盘中拒绝及承接。", ("upper_shadow_share_rank252", "inv:close_location_rank252", "true_range_pct_rank252", "inv:lower_shadow_share_rank252"), ("lower_shadow_share_rank252", "close_location_rank252", "true_range_pct_rank252", "inv:upper_shadow_share_rank252"), "geometric", ("technical", "range_vol")),
    ("V06", "close_location_pressure", "收盘位置压力", "收盘位于日内区间边缘并伴随方向收益与成交确认时，信息可能延续到下一开盘段。", ("inv:close_location_rank252", "inv:ret_1_rank252", "amount_ratio_20_rank252", "true_range_pct_rank252"), ("close_location_rank252", "ret_1_rank252", "amount_ratio_20_rank252", "true_range_pct_rank252"), "minimum", ("technical",)),
    ("V07", "body_range_impulse", "实体与振幅冲击", "大实体、宽振幅和方向一致收盘构成可验证的当日冲击强度。", ("inv:body_pct_rank252", "abs_body_pct_rank252", "true_range_pct_rank252", "inv:close_location_rank252"), ("body_pct_rank252", "abs_body_pct_rank252", "true_range_pct_rank252", "close_location_rank252"), "hurdle", ("range_vol",)),
    ("V08", "shadow_imbalance", "上下影线失衡", "影线失衡在不依赖主观图形命名的情况下表达盘中反转压力。", ("upper_shadow_share_rank252", "inv:lower_shadow_share_rank252", "inv:close_location_rank252", "abs_body_pct_rank252"), ("lower_shadow_share_rank252", "inv:upper_shadow_share_rank252", "close_location_rank252", "abs_body_pct_rank252"), "margin", ("technical",)),
    ("V09", "parkinson_range_volatility", "Parkinson 区间波动", "高低价区间波动与方向压力结合，可比单纯收盘收益更充分利用日频现货信息。", ("parkinson_vol_10_rank252", "parkinson_vol_20_rank252", "inv:ret_1_rank252", "inv:close_location_rank252"), ("parkinson_vol_10_rank252", "parkinson_vol_20_rank252", "ret_1_rank252", "close_location_rank252"), "hurdle", ("parkinson", "range_vol")),
    ("V10", "ohlc_range_volatility_ensemble", "OHLC 区间波动组合", "多种 OHLC 波动估计的共同高位代表区间风险，而方向由收盘和开盘路径确认。", ("gk_vol_20_rank252", "rs_vol_20_rank252", "yz_vol_20_rank252", "inv:ret_1_rank252"), ("gk_vol_20_rank252", "rs_vol_20_rank252", "yz_vol_20_rank252", "ret_1_rank252"), "robust_median", ("range_vol", "yang_zhang")),
    ("V11", "volatility_term_structure", "短长波动期限差", "短窗波动相对长窗突然抬升时，尾部概率可能改变。", ("vol_term_spread_rank252", "vol_5_rank252", "inv:ret_5_rank252", "risk_expansion_rank"), ("vol_term_spread_rank252", "vol_5_rank252", "ret_5_rank252", "risk_expansion_rank"), "hurdle", ("vol_news",)),
    ("V12", "volatility_compression_release", "波动压缩释放", "低长窗波动之后出现真实振幅与短窗波动扩张，方向由当日收益确认。", ("compression_release_rank", "true_range_pct_rank252", "vol_5_rank252", "inv:ret_1_rank252"), ("compression_release_rank", "true_range_pct_rank252", "vol_5_rank252", "ret_1_rank252"), "geometric", ("path_vol",)),
    ("V13", "volatility_clustering", "平方收益波动聚集", "递减核平方收益活动量刻画波动聚集，并用方向收益区分两侧。", ("pdv_activity_fast_rank252", "pdv_activity_slow_rank252", "inv:ret_1_rank252", "inv:ret_5_rank252"), ("pdv_activity_fast_rank252", "pdv_activity_slow_rank252", "ret_1_rank252", "ret_5_rank252"), "weighted_mean", ("path_vol", "vol_news")),
    ("V14", "leverage_path_kernel", "杠杆效应路径核", "递减核历史负收益与活动量共同表达下跌后波动抬升；上涨侧使用对称正向冲击核作对照。", ("pdv_negative_shock_fast_rank252", "pdv_negative_shock_slow_rank252", "pdv_activity_fast_rank252", "pdv_activity_slow_rank252"), ("pdv_positive_shock_fast_rank252", "pdv_positive_shock_slow_rank252", "pdv_activity_fast_rank252", "pdv_activity_slow_rank252"), "weighted_mean", ("path_vol", "vol_news")),
    ("V15", "path_dependent_trend_activity", "路径趋势与活动双核", "短长记忆趋势核和活动核的方向一致性用于预测单侧尾部。", ("inv:pdv_trend_fast_rank252", "inv:pdv_trend_slow_rank252", "pdv_activity_fast_rank252", "pdv_activity_slow_rank252"), ("pdv_trend_fast_rank252", "pdv_trend_slow_rank252", "pdv_activity_fast_rank252", "pdv_activity_slow_rank252"), "geometric", ("path_vol",)),
    ("V16", "semivariance_asymmetry", "上下半方差非对称", "下行与上行半方差的相对扩张可能改变下一腿尾部分布。", ("downside_vol_20_rank252", "inv:upside_vol_20_rank252", "inv:ret_5_rank252", "risk_expansion_rank"), ("upside_vol_20_rank252", "inv:downside_vol_20_rank252", "ret_5_rank252", "risk_expansion_rank"), "margin", ("vol_news",)),
    ("V17", "tail_cluster", "尾部收益簇", "损失/收益簇与符号占比共同反映尾部冲击是否聚集。", ("loss_cluster_5_rank252", "negative_share_20_rank252", "inv:ret_5_rank252", "risk_expansion_rank"), ("gain_cluster_5_rank252", "inv:negative_share_20_rank252", "ret_5_rank252", "risk_expansion_rank"), "weighted_mean", ("extreme_value",)),
    ("V18", "sign_run_persistence", "涨跌连续段", "连续涨跌日数、符号占比与区间扩张共同表达短期状态持续。", ("down_run_rank252", "negative_share_10_rank252", "inv:ret_3_rank252", "true_range_pct_rank252"), ("up_run_rank252", "inv:negative_share_10_rank252", "ret_3_rank252", "true_range_pct_rank252"), "vote", ("direction",)),
    ("V19", "jump_intensity", "跳变强度", "近期大幅日收益的方向、占比与当前振幅共同刻画跳变延续风险。", ("down_jump_share_20_rank252", "jump_intensity_20_rank252", "inv:ret_1_rank252", "vol_5_rank252"), ("up_jump_share_20_rank252", "jump_intensity_20_rank252", "ret_1_rank252", "vol_5_rank252"), "hurdle", ("extreme_value",)),
    ("V20", "range_breakout", "区间突破", "相对前期高低点的突破距离、趋势效率与成交额确认决定方向。", ("support_break_rank252", "inv:trend_efficiency_20_rank252", "amount_ratio_20_rank252", "true_range_pct_rank252"), ("resistance_break_rank252", "trend_efficiency_20_rank252", "amount_ratio_20_rank252", "true_range_pct_rank252"), "geometric", ("technical", "momentum")),
    ("V21", "amihud_illiquidity", "Amihud 非流动性冲击", "绝对收益相对成交额的价格冲击与方向收益结合，测试流动性冲击是否预示尾部。", ("amihud_rank252", "down_liquidity_impact_rank252", "inv:ret_1_rank252", "inv:volume_ratio_20_rank252"), ("amihud_rank252", "up_liquidity_impact_rank252", "ret_1_rank252", "volume_ratio_20_rank252"), "hurdle", ("amihud",)),
    ("V22", "volume_price_confirmation", "量价确认", "收益方向得到成交量、成交额与收盘位置共同确认时，延续概率可能提高。", ("inv:ret_5_rank252", "volume_ratio_20_rank252", "amount_ratio_20_rank252", "inv:close_location_rank252"), ("ret_5_rank252", "volume_ratio_20_rank252", "amount_ratio_20_rank252", "close_location_rank252"), "minimum", ("momentum",)),
    ("V23", "amount_surprise", "成交额异常", "成交额相对自身历史的异常变化与方向冲击共同表达信息到达。", ("amount_z_20_rank252", "amount_ratio_5_rank252", "inv:ret_1_rank252", "true_range_pct_rank252"), ("amount_z_20_rank252", "amount_ratio_5_rank252", "ret_1_rank252", "true_range_pct_rank252"), "hurdle", ("amihud",)),
    ("V24", "liquidity_adjusted_return", "流动性调整收益", "方向收益乘以成交冲击后再与成交活跃度确认，区分大成交趋势与脆弱冲击。", ("down_liquidity_impact_rank252", "amihud_rank252", "inv:amount_ratio_20_rank252", "risk_expansion_rank"), ("up_liquidity_impact_rank252", "inv:amihud_rank252", "amount_ratio_20_rank252", "risk_expansion_rank"), "geometric", ("amihud",)),
    ("V25", "drawdown_depth", "回撤深度", "距滚动高点的深度、区间位置和短期方向共同区分跌势与修复。", ("inv:drawdown_60_rank252", "inv:range_position_60_rank252", "inv:ret_5_rank252", "vol_20_rank252"), ("drawdown_60_rank252", "range_position_60_rank252", "ret_5_rank252", "vol_20_rank252"), "weighted_mean", ("mean_reversion",)),
    ("V26", "drawdown_velocity", "回撤速度", "回撤在短窗内加深或收敛的速度比静态回撤水平提供不同路径信息。", ("inv:drawdown_change_5_rank252", "inv:drawdown_60_rank252", "inv:ret_5_rank252", "true_range_pct_rank252"), ("drawdown_change_5_rank252", "drawdown_60_rank252", "ret_5_rank252", "true_range_pct_rank252"), "margin", ("path_vol",)),
    ("V27", "trend_efficiency", "趋势效率", "净位移占累计路径长度的比例与方向共同识别顺畅趋势。", ("trend_efficiency_20_rank252", "inv:ret_20_rank252", "inv:close_location_rank252", "inv:vol_20_rank252"), ("trend_efficiency_20_rank252", "ret_20_rank252", "close_location_rank252", "inv:vol_20_rank252"), "geometric", ("momentum",)),
    ("V28", "momentum_continuation", "多尺度动量延续", "短中期收益、趋势效率与收盘位置的共同方向用于检验延续。", ("inv:ret_5_rank252", "inv:ret_20_rank252", "trend_efficiency_20_rank252", "inv:close_location_rank252"), ("ret_5_rank252", "ret_20_rank252", "trend_efficiency_20_rank252", "close_location_rank252"), "minimum", ("momentum",)),
    ("V29", "momentum_curvature", "动量曲率", "短期方向相对中期趋势发生加速或减速时，下一腿尾部概率可能变化。", ("inv:momentum_curvature_rank252", "inv:ret_5_rank252", "ret_20_rank252", "upper_shadow_share_rank252"), ("momentum_curvature_rank252", "ret_5_rank252", "inv:ret_20_rank252", "lower_shadow_share_rank252"), "weighted_mean", ("momentum",)),
    ("V30", "mean_reversion_overshoot", "均值回归超调", "单日极端、低趋势效率和边缘收盘共同识别可能过度反应。", ("high:ret_1_rank252", "inv:trend_efficiency_20_rank252", "close_location_rank252", "drawdown_60_rank252"), ("low:ret_1_rank252", "inv:trend_efficiency_20_rank252", "inv:close_location_rank252", "inv:drawdown_60_rank252"), "geometric", ("mean_reversion",)),
    ("V31", "rebound_confirmation", "回撤修复确认", "深回撤后的短期收益、下影与收盘修复共同表达反弹；上涨后的转弱作镜像。", ("drawdown_60_rank252", "inv:ret_1_rank252", "upper_shadow_share_rank252", "inv:close_location_rank252"), ("inv:drawdown_60_rank252", "ret_1_rank252", "lower_shadow_share_rank252", "close_location_rank252"), "margin", ("mean_reversion",)),
    ("V32", "support_resistance_position", "支撑阻力位置", "价格相对滚动区间的位置、突破距离和振幅共同刻画支撑/阻力附近的非线性。", ("inv:range_position_20_rank252", "support_break_rank252", "inv:close_location_rank252", "true_range_pct_rank252"), ("range_position_20_rank252", "resistance_break_rank252", "close_location_rank252", "true_range_pct_rank252"), "geometric", ("technical",)),
    ("V33", "cross_horizon_majority_vote", "跨窗口多数表决", "多个方向窗口同时落入尾部时才提高预警，降低单一窗口噪声。", ("inv:ret_1_rank252", "inv:ret_5_rank252", "inv:ret_20_rank252", "inv:close_location_rank252"), ("ret_1_rank252", "ret_5_rank252", "ret_20_rank252", "close_location_rank252"), "vote", ("direction",)),
    ("V34", "rank_intersection", "分位秩交集", "用最弱证据约束多条件共识，避免某一强分量补偿其他反向分量。", ("inv:ret_1_rank252", "inv:ret_5_rank252", "inv:close_location_rank252", "risk_expansion_rank"), ("ret_1_rank252", "ret_5_rank252", "close_location_rank252", "risk_expansion_rank"), "minimum", ("quantile",)),
    ("V35", "robust_median_components", "稳健分量中位数", "多条现货方向证据取稳健中位，降低异常单分量的影响。", ("inv:ret_1_rank252", "inv:ret_5_rank252", "inv:close_location_rank252", "upper_shadow_share_rank252"), ("ret_1_rank252", "ret_5_rank252", "close_location_rank252", "lower_shadow_share_rank252"), "robust_median", ("robust",)),
    ("V36", "magnitude_direction_hurdle", "幅度-方向两阶段门槛", "先由区间/波动确认大幅事件，再由收益与收盘位置确认方向。", ("inv:ret_1_rank252", "inv:close_location_rank252", "true_range_pct_rank252", "risk_expansion_rank"), ("ret_1_rank252", "close_location_rank252", "true_range_pct_rank252", "risk_expansion_rank"), "hurdle", ("rare_event", "range_vol")),
    ("V37", "volatility_state_gate", "波动状态门控", "同一方向信号在高低波动状态下的尾部含义不同，使用因果波动秩做软门控。", ("inv:ret_5_rank252", "inv:close_location_rank252", "vol_regime_causal", "risk_expansion_rank"), ("ret_5_rank252", "close_location_rank252", "vol_regime_causal", "risk_expansion_rank"), "state_gate", ("local_state", "direction")),
    ("V38", "trend_state_gate", "趋势状态门控", "方向冲击是否与中期趋势状态一致，决定延续与反转的相对权重。", ("inv:ret_1_rank252", "inv:close_location_rank252", "inv:trend_regime_causal", "trend_efficiency_20_rank252"), ("ret_1_rank252", "close_location_rank252", "trend_regime_causal", "trend_efficiency_20_rank252"), "state_gate", ("local_state", "momentum")),
    ("V39", "drawdown_state_gate", "回撤状态门控", "相同短期收益在深回撤和高位环境中的含义不同，用因果回撤秩软门控。", ("inv:ret_1_rank252", "inv:close_location_rank252", "inv:drawdown_regime_causal", "risk_expansion_rank"), ("ret_1_rank252", "close_location_rank252", "drawdown_regime_causal", "risk_expansion_rank"), "state_gate", ("local_state", "mean_reversion")),
    ("V40", "causal_state_transition", "因果状态转移", "波动与趋势秩的短期变化量表达状态切换速度，方向由当前收益确认。", ("state_transition_rank", "inv:ret_1_rank252", "risk_expansion_rank", "inv:close_location_rank252"), ("state_transition_rank", "ret_1_rank252", "risk_expansion_rank", "close_location_rank252"), "hurdle", ("local_state",)),
    ("V41", "local_state_similarity", "当前状态局部相似度", "当前多维状态距方向尾部原型的距离形成无标签、跨时期可定义的局部相似分数。", ("inv:ret_5_rank252", "vol_20_rank252", "inv:close_location_rank252", "down_liquidity_impact_rank252"), ("ret_5_rank252", "vol_20_rank252", "close_location_rank252", "up_liquidity_impact_rank252"), "distance", ("local_state",)),
    ("V42", "time_decay_path_features", "时间衰减路径", "近期样本获得更高权重的指数核趋势与活动量，测试短记忆和长记忆组合。", ("inv:ewm_ret_hl3_rank252", "inv:ewm_ret_hl10_rank252", "ewm_absret_hl3_rank252", "ewm_absret_hl20_rank252"), ("ewm_ret_hl3_rank252", "ewm_ret_hl10_rank252", "ewm_absret_hl3_rank252", "ewm_absret_hl20_rank252"), "weighted_mean", ("local_state", "path_vol")),
    ("V43", "multi_scale_kernel_path", "多尺度核路径", "短、中、长半衰期的方向核与活动核共同捕捉不同记忆尺度。", ("inv:ewm_ret_hl3_rank252", "inv:ewm_ret_hl20_rank252", "ewm_sqret_hl5_rank252", "ewm_sqret_hl60_rank252"), ("ewm_ret_hl3_rank252", "ewm_ret_hl20_rank252", "ewm_sqret_hl5_rank252", "ewm_sqret_hl60_rank252"), "geometric", ("path_vol",)),
    ("V44", "directional_corner_distance", "方向角点距离", "把方向、波动、收盘和流动性视为四维状态，使用到理想尾部角点的加权距离。", ("inv:ret_5_rank252", "vol_5_rank252", "inv:close_location_rank252", "down_liquidity_impact_rank252"), ("ret_5_rank252", "vol_5_rank252", "close_location_rank252", "up_liquidity_impact_rank252"), "distance", ("local_state", "robust")),
    ("V45", "event_concurrence", "事件并发计数", "只有多个独立现货事件同时越过因果分位阈值才形成高分。", ("inv:ret_1_rank252", "inv:ret_5_rank252", "true_range_pct_rank252", "down_liquidity_impact_rank252"), ("ret_1_rank252", "ret_5_rank252", "true_range_pct_rank252", "up_liquidity_impact_rank252"), "vote", ("rare_event",)),
    ("V46", "conflict_aware_abstention", "冲突感知弃权", "方向分量分歧越大越降低置信度，用概率弃权思想减少反向极端。", ("inv:ret_1_rank252", "inv:ret_5_rank252", "inv:close_location_rank252", "upper_shadow_share_rank252"), ("ret_1_rank252", "ret_5_rank252", "close_location_rank252", "lower_shadow_share_rank252"), "abstain", ("conformal", "direction")),
    ("V47", "hysteresis_persistence", "迟滞式持续确认", "当前信号与过去两日同向分数共同确认，减少一天噪声造成的切换。", ("inv:ret_1_rank252", "inv:ret_3_rank252", "down_run_rank252", "inv:close_location_rank252"), ("ret_1_rank252", "ret_3_rank252", "up_run_rank252", "close_location_rank252"), "hysteresis", ("local_state", "direction")),
    ("V48", "ensemble_disagreement_margin", "多通道分歧边际", "动量、K线、波动和流动性通道的共同边际高时才预警。", ("inv:ret_5_rank252", "upper_shadow_share_rank252", "vol_5_rank252", "down_liquidity_impact_rank252"), ("ret_5_rank252", "lower_shadow_share_rank252", "vol_5_rank252", "up_liquidity_impact_rank252"), "margin", ("robust", "conformal")),
    ("V49", "quantile_band_distance", "因果分位带距离", "方向特征偏离历史中性分位带的距离与风险扩张共同构成尾部分数。", ("low:ret_1_rank252", "low:ret_5_rank252", "low:close_location_rank252", "risk_expansion_rank"), ("high:ret_1_rank252", "high:ret_5_rank252", "high:close_location_rank252", "risk_expansion_rank"), "weighted_mean", ("quantile",)),
    ("V50", "tail_uncertainty_gate", "尾部不确定性门控", "方向共识必须同时获得区间波动、路径脆弱和流动性冲击确认，作为尾部不确定性过滤。", ("inv:ret_5_rank252", "tail_uncertainty_rank", "inv:close_location_rank252", "down_liquidity_impact_rank252"), ("ret_5_rank252", "tail_uncertainty_rank", "close_location_rank252", "up_liquidity_impact_rank252"), "hurdle", ("extreme_value", "conformal")),
]


VERSION_SPECS: dict[str, VersionSpec] = {}
for row in _SPECS:
    version, name, title, hypothesis, down, up, aggregator, refs = row
    VERSION_SPECS[version] = VersionSpec(
        version=version,
        core_logic_name=name,
        title_zh=title,
        hypothesis=hypothesis,
        down_components=down,
        up_components=up,
        aggregator=aggregator,
        literature_keys=refs,
    )


def get_version_spec(version: str) -> VersionSpec:
    normalized = version.upper()
    if normalized not in VERSION_SPECS:
        raise KeyError(f"unknown version: {version}")
    return VERSION_SPECS[normalized]


if list(VERSION_SPECS) != [f"V{i:02d}" for i in range(1, 51)]:
    raise AssertionError("version registry must contain contiguous V01-V50")

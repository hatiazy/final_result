from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LogicSpec:
    version: str
    core_logic_name: str
    method_key: str
    logic_family: str
    description: str
    source_title: str
    source_url: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


PAGE = ("Page (1954), Continuous Inspection Schemes", "https://doi.org/10.1093/biomet/41.1-2.100")
KALMAN = ("Kalman (1960), A New Approach to Linear Filtering and Prediction Problems", "https://doi.org/10.1115/1.3662552")
BOCPD = ("Adams & MacKay (2007), Bayesian Online Changepoint Detection", "https://arxiv.org/abs/0710.3742")
PELT = ("Killick, Fearnhead & Eckley (2012), Optimal Detection of Changepoints", "https://arxiv.org/abs/1101.1438")
HAMILTON = ("Hamilton (1989), A New Approach to Nonstationary Time Series", "https://doi.org/10.2307/1912559")
RABINER = ("Rabiner (1989), A Tutorial on Hidden Markov Models", "https://doi.org/10.1109/5.18626")
TSMOM = ("Moskowitz, Ooi & Pedersen (2012), Time Series Momentum", "https://doi.org/10.1016/j.jfineco.2011.11.003")
BROCK = ("Brock, Lakonishok & LeBaron (1992), Simple Technical Trading Rules", "https://doi.org/10.1111/j.1540-6261.1992.tb04681.x")
LO_PATH = ("Lo, Mamaysky & Wang (2000), Foundations of Technical Analysis", "https://www.nber.org/papers/w7613")
LO_VR = ("Lo & MacKinlay (1988), Stock Market Prices Do Not Follow Random Walks", "https://www.nber.org/papers/w2168")
PARKINSON = ("Parkinson (1980), Extreme Value Method for Estimating Variance", "https://doi.org/10.1086/296071")
COX = ("Cox (1972), Regression Models and Life-Tables", "https://doi.org/10.1111/j.2517-6161.1972.tb00899.x")
FRIEDMAN = ("Friedman (2001), Greedy Function Approximation", "https://doi.org/10.1214/aos/1013203451")

# The second block is deliberately kept in the executable registry now that the
# user has asked for V51–V90 to be run as independent candidate versions.  Each
# row still names one mathematical core; the implementation adaptations live in
# reserve_features.py and never mix methods within a version.
MMD = ("Gretton et al. (2012), A Kernel Two-Sample Test", "https://jmlr.org/papers/v13/gretton12a.html")
ENERGY = ("Matteson & James (2014), A Nonparametric Approach to Detecting Changes in a Multivariate Time Series", "https://doi.org/10.1080/01621459.2013.849605")
SW = ("Peyre & Cuturi (2019), Computational Optimal Transport", "https://arxiv.org/abs/1803.00567")
RULSIF = ("Liu et al. (2013), Direct Density-Ratio Estimation for Change-Point Detection", "https://doi.org/10.1016/j.neunet.2013.01.012")
GRAPH_SCAN = ("Chen & Zhang (2015), Graph-Based Change-Point Detection", "https://doi.org/10.1214/14-AOS1269")
PERM_ENTROPY = ("Bandt & Pompe (2002), Permutation Entropy", "https://doi.org/10.1103/PhysRevLett.88.174102")
RQA = ("Marwan et al. (2007), Recurrence Plots for the Analysis of Complex Systems", "https://doi.org/10.1016/j.physrep.2006.11.001")
VISIBILITY = ("Lacasa et al. (2009), From Time Series to Complex Networks", "https://doi.org/10.1073/pnas.0709247105")
MATRIX_PROFILE = ("Yeh et al. (2016), Matrix Profile I", "https://doi.org/10.1109/ICDM.2016.0179")
SAX = ("Lin et al. (2007), Experiencing SAX", "https://doi.org/10.1007/s10618-007-0064-z")
TRANSFER_ENTROPY = ("Schreiber (2000), Measuring Information Transfer", "https://doi.org/10.1103/PhysRevLett.85.461")
DIRECTIONAL_CHANGE = ("Glattfelder et al. (2010), Patterns in the Foreign Exchange Market", "https://doi.org/10.2139/ssrn.1973471")
HAWKES = ("Hawkes (1971), Spectra of Some Self-Exciting and Mutually Exciting Point Processes", "https://doi.org/10.1093/biomet/58.1.83")
L1_TREND = ("Kim et al. (2009), l1 Trend Filtering", "https://doi.org/10.1137/070690274")
SSA = ("Golyandina et al. (2001), Analysis of Time Series Structure", "https://doi.org/10.1007/978-3-642-34913-3")
WAVELET = ("Mallat (1989), A Theory for Multiresolution Signal Decomposition", "https://doi.org/10.1109/34.192463")
DMD = ("Schmid (2010), Dynamic Mode Decomposition of Numerical and Experimental Data", "https://doi.org/10.1017/S0022112010001217")
SIGNATURE = ("Lyons et al. (2016), A Machine Learning Approach to Signature Methods", "https://arxiv.org/abs/1603.03788")
TDA = ("Gidea & Katz (2018), Topological Data Analysis of Financial Time Series", "https://doi.org/10.1016/j.physa.2017.09.028")
EDM = ("Sugihara & May (1990), Nonlinear Forecasting as a Way of Distinguishing Chaos from Measurement Error", "https://doi.org/10.1038/344734a0")
HSMM = ("Yu (2010), Hidden Semi-Markov Models", "https://doi.org/10.1016/j.artint.2009.11.011")
FINE_GRAY = ("Fine & Gray (1999), A Proportional Hazards Model for the Subdistribution of a Competing Risk", "https://doi.org/10.1080/01621459.1999.10474144")
AALEN = ("Aalen (1989), A Linear Regression Model for the Analysis of Life Times", "https://doi.org/10.1002/sim.4780080803")
GAM = ("Hastie & Tibshirani (1990), Generalized Additive Models", "https://doi.org/10.1214/ss/1177013604")
QUANTILE = ("Koenker & Bassett (1978), Regression Quantiles", "https://doi.org/10.2307/1913643")
EXPECTILE = ("Newey & Powell (1987), Asymmetric Least Squares Estimation", "https://doi.org/10.2307/1911031")
ELASTIC_NET = ("Zou & Hastie (2005), Regularization and Variable Selection via the Elastic Net", "https://doi.org/10.1111/j.1467-9868.2005.00503.x")
GP = ("Rasmussen & Williams (2006), Gaussian Processes for Machine Learning", "https://direct.mit.edu/books/oa-monograph/2320/Gaussian-Processes-for-Machine-Learning")
SVR = ("Drucker et al. (1997), Support Vector Regression Machines", "https://doi.org/10.1007/BF00994018")
RANDOM_FOREST = ("Breiman (2001), Random Forests", "https://doi.org/10.1023/A:1010933404324")
RDA = ("Friedman (1989), Regularized Discriminant Analysis", "https://doi.org/10.1080/01621459.1989.10478752")
SETAR = ("Tong (1980), Threshold Autoregression", "https://doi.org/10.1111/j.2517-6161.1980.tb01126.x")
STAR = ("van Dijk et al. (2002), Smooth Transition Autoregressive Models", "https://doi.org/10.1080/01621459.1994.10476462")
QAR = ("Koenker & Xiao (2006), Quantile Autoregression", "https://doi.org/10.1198/016214506000000672")
GAS = ("Creal et al. (2013), Generalized Autoregressive Score Models", "https://doi.org/10.1002/jae.1279")
CONFORMAL = ("Shafer & Vovk (2008), A Tutorial on Conformal Prediction", "https://jmlr.csail.mit.edu/beta/papers/v9/shafer08a.html")
CONFORMAL_MART = ("Vovk et al. (2022), Testing Exchangeability Online", "https://proceedings.mlr.press/v152/vovk21b.html")
VARX = ("Basu et al. (2019), Sparse Group Lasso for VARX", "https://arxiv.org/abs/1508.07497")
ESN = ("Lukosevicius (2012), A Practical Guide to Applying Echo State Networks", "https://publica.fraunhofer.de/entities/publication/7d4a7eec-a22c-4df0-903d-93f9cd5aca02")
PPR = ("Friedman & Stuetzle (1981), Projection Pursuit Regression", "https://doi.org/10.1080/01621459.1981.10477729")

# V91--V120: a deliberately separate reserve block.  These methods were
# selected because their core transformation is not a rename of V01--V90:
# conditional-volatility recursion, tail laws, local nonlinear maps,
# complexity/phase geometry, random convolutional features, and new small-
# sample estimators/classifiers are all represented explicitly.
GARCH = ("Bollerslev (1986), Generalized Autoregressive Conditional Heteroskedasticity", "https://doi.org/10.2307/1913109")
EGARCH = ("Nelson (1991), Conditional Heteroskedasticity in Asset Returns", "https://doi.org/10.2307/2938260")
GJR = ("Glosten, Jagannathan & Runkle (1993), On the Relation between the Expected Value and the Volatility of the Nominal Excess Return", "https://doi.org/10.2307/2328882")
APARCH = ("Ding, Granger & Engle (1993), A Long Memory Property of Stock Market Returns", "https://doi.org/10.1016/0304-4076(93)90006-D")
FIGARCH = ("Baillie, Bollerslev & Mikkelsen (1996), Fractionally Integrated Generalized Autoregressive Conditional Heteroskedasticity", "https://doi.org/10.2307/2171802")
HAR_RV = ("Corsi (2009), A Simple Approximate Long-Memory Model of Realized Volatility", "https://doi.org/10.1016/j.jempfin.2008.12.002")
CAVIAR = ("Engle & Manganelli (2004), CAViaR: Conditional Autoregressive Value at Risk by Regression Quantiles", "https://doi.org/10.1198/073500104000000370")
NOVAS = ("Politis (2007), Model-Free Prediction of Time Series Based on NoVaS", "https://doi.org/10.1111/j.1467-9892.2007.00566.x")
EVT = ("Pickands (1975), Statistical Inference Using Extreme Order Statistics", "https://doi.org/10.1214/aos/1176343003")
HILL = ("Hill (1975), A Simple General Approach to Inference About the Tail of a Distribution", "https://doi.org/10.1214/aos/1176343247")
SMAP = ("Sugihara (1994), Nonlinear Forecasting for the Classification of Natural Time Series", "https://doi.org/10.1016/0167-2789(94)90205-4")
ANALOG = ("Sugihara & May (1990), Nonlinear Forecasting as a Way of Distinguishing Chaos from Measurement Error", "https://doi.org/10.1038/344734a0")
SHAPELET = ("Ye & Keogh (2011), Time-Series Shapelets: A New Primitive for Data Mining", "https://www.jmlr.org/papers/v12/ye11a.html")
SAMPLE_ENTROPY = ("Richman & Moorman (2000), Physiological Time-Series Analysis Using Approximate Entropy and Sample Entropy", "https://doi.org/10.1152/ajpheart.2000.278.6.H2039")
MULTISCALE_ENTROPY = ("Costa, Goldberger & Peng (2005), Multiscale Entropy Analysis of Complex Physiologic Time Series", "https://doi.org/10.1103/PhysRevLett.89.068102")
LZ = ("Lempel & Ziv (1976), On the Complexity of Finite Sequences", "https://doi.org/10.1109/TIT.1976.1055501")
HIGUCHI = ("Higuchi (1988), Approach to an Irregular Time Series on the Basis of the Fractal Theory", "https://doi.org/10.1016/0022-460X(88)90081-4")
HILBERT = ("Huang et al. (1998), The Empirical Mode Decomposition and the Hilbert Spectrum", "https://doi.org/10.1098/rspa.1998.0193")
EMD = ("Huang et al. (1999), A New View of Nonlinear Water Waves", "https://doi.org/10.1016/S0165-0114(99)00015-8")
VMD = ("Dragomiretskiy & Zosso (2014), Variational Mode Decomposition", "https://doi.org/10.1109/TSP.2013.2288675")
ROCKET = ("Dempster, Petitjean & Webb (2020), ROCKET: Exceptionally Fast and Accurate Time Series Classification Using Random Convolutional Kernels", "https://arxiv.org/abs/1910.13051")
ELM = ("Huang, Zhu & Siew (2006), Extreme Learning Machine: Theory and Applications", "https://doi.org/10.1162/neco.2006.18.6.132")
RFF = ("Rahimi & Recht (2007), Random Features for Large-Scale Kernel Machines", "https://papers.neurips.cc/paper/3182-random-features-for-large-scale-kernel-machines")
PLS = ("Wold, Sjöström & Eriksson (2001), PLS-Regression: A Basic Tool of Chemometrics", "https://doi.org/10.1002/14356007.a20_245")
HUBER = ("Huber (1964), Robust Estimation of a Location Parameter", "https://doi.org/10.1214/aoms/1177703732")
RANSAC = ("Fischler & Bolles (1981), Random Sample Consensus", "https://doi.org/10.1016/0004-3702(81)90081-7")
LOGISTIC = ("Cox (1958), The Regression Analysis of Binary Sequences", "https://doi.org/10.2307/2983890")
QDA = ("Friedman (1989), Regularized Discriminant Analysis", "https://doi.org/10.1080/01621459.1989.10478752")
NAIVE_BAYES = ("John & Langley (1995), Estimating Continuous Distributions in Bayesian Classifiers", "https://www.cs.cmu.edu/~knn/naive-bayes.html")
EXTRA_TREES = ("Geurts, Ernst & Wehenkel (2006), Extremely Randomized Trees", "https://doi.org/10.1007/s10994-006-6226-1")


_ROWS = [
    ("V01", "收盘时间序列动量", "close_momentum", "现货价格路径", "不同回看窗的收盘对数收益按已实现波动缩放。", *TSMOM),
    ("V02", "开盘时间序列动量", "open_momentum", "现货价格路径", "只用历史开盘路径构造多尺度动量，直接贴近 O2O 执行锚。", *TSMOM),
    ("V03", "日内实体持续性", "intraday_persistence", "现货日内路径", "聚合收盘相对开盘的日内实体方向与稳定性。", *LO_PATH),
    ("V04", "隔夜跳空持续性", "gap_persistence", "现货隔夜路径", "聚合当日开盘相对前收的可见跳空方向。", *LO_PATH),
    ("V05", "收盘区间位置持续性", "close_location", "现货日内路径", "用收盘在当日高低区间的位置刻画承接方向。", *LO_PATH),
    ("V06", "简单均线距离", "sma_distance", "移动平均规则", "收盘相对不同回看均线的距离按波动缩放。", *BROCK),
    ("V07", "指数均线交叉", "ema_crossover", "移动平均规则", "快慢指数均线差异形成因果趋势滤波。", *BROCK),
    ("V08", "交易区间突破", "donchian_breakout", "区间突破规则", "收盘相对历史高低通道的位置与突破幅度。", *BROCK),
    ("V09", "布林标准化偏离", "bollinger_z", "标准化价格路径", "价格相对滚动均值的标准差距离。", *BROCK),
    ("V10", "滚动线性趋势斜率", "ols_slope", "局部趋势模型", "对数价格的后向滚动 OLS 斜率按残差波动缩放。", *LO_PATH),
    ("V11", "多滞后稳健中位趋势", "robust_median_trend", "稳健价格路径", "多个子滞后年化斜率取中位数，降低单日异常影响。", *LO_PATH),
    ("V12", "价格路径加速度", "momentum_acceleration", "二阶价格路径", "短窗与长窗单位时间动量之差刻画方向加速。", *LO_PATH),
    ("V13", "Wilder相对强弱", "rsi", "涨跌幅平衡", "上涨与下跌绝对变动的滚动平衡映射到有符号强度。", *BROCK),
    ("V14", "随机振荡区间位置", "stochastic", "区间位置", "收盘位于多日最高最低区间的相对位置。", *BROCK),
    ("V15", "MACD差分趋势", "macd", "指数滤波趋势", "快慢 EMA 差及其信号线差刻画趋势形成。", *BROCK),
    ("V16", "双侧CUSUM均值变点", "cusum", "在线变点", "对标准化现货收益运行双侧 CUSUM，比较正负累积证据。", *PAGE),
    ("V17", "Page-Hinkley方向变点", "page_hinkley", "在线变点", "用递归均值偏离和回撤重置识别方向均值突变。", *PAGE),
    ("V18", "EWMA控制图方向", "ewma_control", "在线控制图", "EWMA 收益均值相对 EWMA 波动的标准化偏移。", *PAGE),
    ("V19", "Kalman局部水平斜率", "kalman_slope", "状态空间", "单变量局部水平 Kalman 过滤后的因果水平变化。", *KALMAN),
    ("V20", "Kalman创新方向", "kalman_innovation", "状态空间", "观测相对一步预测的标准化创新。", *KALMAN),
    ("V21", "贝叶斯在线变点代理", "bocpd_proxy", "在线变点", "相邻窗口均值差的贝叶斯证据代理乘以变动方向。", *BOCPD),
    ("V22", "PELT局部成本下降代理", "pelt_proxy", "离线成本变点", "当前切分相对不切分的高斯 SSE 成本改善及均值方向。", *PELT),
    ("V23", "Lo-MacKinlay方差比方向", "variance_ratio", "随机游走偏离", "滚动方差比与最近价格方向结合，区分延续和均值回归。", *LO_VR),
    ("V24", "收益自相关延续", "autocorrelation", "序列依赖", "滚动滞后自相关乘以最近收益，形成下一步延续方向。", *LO_VR),
    ("V25", "收益符号游程", "sign_runs", "符号序列", "涨跌符号不平衡和游程持续性共同刻画方向。", *LO_VR),
    ("V26", "重标极差持久性", "hurst_rs", "长记忆路径", "滚动 R/S 持久性偏离 0.5 后与价格方向结合。", *LO_VR),
    ("V27", "波动状态下的方向效率", "volatility_regime", "波动调整路径", "方向动量按短长波动状态及路径效率缩放。", *TSMOM),
    ("V28", "Parkinson区间波动调整", "parkinson", "高低价波动", "用高低极差波动对方向动量做尺度标准化。", *PARKINSON),
    ("V29", "Garman-Klass波动调整", "garman_klass", "OHLC波动", "以开高低收联合波动估计缩放方向动量。", *PARKINSON),
    ("V30", "Rogers-Satchell路径波动调整", "rogers_satchell", "OHLC路径波动", "用允许漂移的 OHLC 路径波动估计缩放方向。", *PARKINSON),
    ("V31", "量价相关确认", "volume_price_corr", "量价确认", "收益与成交量变化的滚动相关性调制价格方向。", *LO_PATH),
    ("V32", "OBV累积量斜率", "obv_slope", "量价累积", "按涨跌符号累积成交量并估计标准化斜率。", *LO_PATH),
    ("V33", "资金流量平衡", "money_flow", "典型价量平衡", "典型价格上涨和下跌日的成交金额流量比。", *LO_PATH),
    ("V34", "Chaikin累积分布", "chaikin_ad", "区间量价累积", "收盘区间位置乘成交量的累积路径斜率。", *LO_PATH),
    ("V35", "滚动成交量加权典型价偏离", "vwap_proxy", "成交量加权路径", "收盘相对日频典型价成交量加权均值的偏离。", *LO_PATH),
    ("V36", "成交参与度惊奇方向", "turnover_surprise", "成交参与路径", "成交额的因果标准化惊奇乘以当日与短窗方向。", *LO_PATH),
    ("V37", "有符号成交量不平衡", "signed_volume", "成交量方向", "按收益符号加权成交量相对总成交量的滚动比率。", *LO_PATH),
    ("V38", "成交量趋势突破确认", "volume_breakout", "成交量确认", "成交量相对历史分布的突破强度与价格方向共振。", *LO_PATH),
    ("V39", "收益符号熵方向", "sign_entropy", "路径信息熵", "方向不平衡乘以二元符号熵的可预测性补量。", *LO_PATH),
    ("V40", "价格路径效率比", "path_efficiency", "路径效率", "净位移相对逐日总位移的有符号效率。", *LO_PATH),
    ("V41", "回撤修复斜率", "drawdown_recovery", "回撤路径", "滚动峰值下的回撤水平及近期修复速度。", *LO_PATH),
    ("V42", "距滚动峰值标准化距离", "peak_distance", "位置路径", "收盘距历史峰值的对数距离按波动缩放。", *LO_PATH),
    ("V43", "距滚动谷值标准化距离", "trough_distance", "位置路径", "收盘距历史谷值的对数距离按波动缩放。", *LO_PATH),
    ("V44", "Ulcer回撤风险调整方向", "ulcer_direction", "回撤风险", "方向动量按滚动回撤平方均值缩放。", *LO_PATH),
    ("V45", "有利不利日内偏移平衡", "excursion_balance", "日内路径", "开盘到高低的有利/不利偏移在窗口内的方向平衡。", *LO_PATH),
    ("V46", "核平滑技术路径斜率", "kernel_path", "非参数路径", "对后向价格路径做核平滑后读取端点斜率。", *LO_PATH),
    ("V47", "DTW方向模板距离", "dtw_template", "模板路径", "尾部路径到单调上行/下行模板的动态时间规整距离差。", *LO_PATH),
    ("V48", "零段持续时间竞争风险率", "zero_hazard", "持续时间风险", "按零状态年龄与现货动量分层估计两侧离开零段的离散风险率。", *COX),
    ("V49", "Hamilton因果状态过滤", "hamilton_filter", "状态切换", "两状态高斯 Markov 过滤的正负均值状态后验差。", *HAMILTON),
    ("V50", "单一梯度提升树模型类", "gradient_boosting", "树模型", "仅用现货因果特征的浅层梯度提升回归；两侧独立拟合 O2O 方向化目标。", *FRIEDMAN),

    ("V51", "Kernel MMD分布变点", "kernel_mmd", "分布变点", "以RKHS双样本距离检测相邻现货特征分布变化，并用收益见证方向定向。", *MMD),
    ("V52", "Energy多元分布变点", "energy_distance", "分布变点", "以能量距离的U统计量检测相邻现货特征分布变化。", *ENERGY),
    ("V53", "Sliced-Wasserstein运输方向", "sliced_wasserstein", "分布几何", "以固定投影的一维分位数运输量表示相邻现货分布移动。", *SW),
    ("V54", "RuLSIF相对密度比", "rulsif", "分布变点", "直接估计近期相对历史现货特征的密度比。", *RULSIF),
    ("V55", "Graph Scan图边变点", "graph_scan", "图变点", "用相邻窗口样本相似图的跨窗口边比例检测分布变化。", *GRAPH_SCAN),
    ("V56", "Bandt-Pompe排列熵", "permutation_entropy", "序数动力学", "以多点序数排列而非二元符号统计路径可预测性与方向。", *PERM_ENTROPY),
    ("V57", "Recurrence Quantification", "rqa", "复现动力学", "用延迟嵌入复现点阵的确定性与层流结构构造方向分。", *RQA),
    ("V58", "Visibility Graph不可逆性", "visibility_graph", "时序图", "把单边价格路径转为可见图并用端点度与时间不可逆性定向。", *VISIBILITY),
    ("V59", "Matrix Profile历史模式", "matrix_profile", "历史模式", "在历史真实子序列中查找近邻模式及其已知后续。", *MATRIX_PROFILE),
    ("V60", "SAX符号词转移", "sax", "符号词", "以PAA压缩和离散词表示完整路径，再以历史词条件方向收缩。", *SAX),
    ("V61", "Transfer Entropy信息流", "transfer_entropy", "信息流", "以条件转移概率衡量量价通道对收益符号的有向信息增量。", *TRANSFER_ENTROPY),
    ("V62", "Directional Change事件时间", "directional_change", "事件时间", "以价格越过因果波动阈值的事件而非固定交易日窗表示方向。", *DIRECTIONAL_CHANGE),
    ("V63", "双变量Hawkes事件激发", "hawkes", "点过程", "以涨跌事件的自激发与互激发条件强度表示转移方向。", *HAWKES),
    ("V64", "L1 Trend Filtering", "l1_trend_filter", "稀疏折点", "以二阶差分L1稀疏惩罚获取分段线性现货趋势。", *L1_TREND),
    ("V65", "SSA低秩Hankel", "ssa", "低秩谱", "以历史Hankel矩阵低秩重建和递推预测路径。", *SSA),
    ("V66", "单边多分辨率小波", "wavelet", "时频分析", "以单边小波的低频方向和高频能量分解现货路径。", *WAVELET),
    ("V67", "DMD/Koopman局部算子", "dmd", "动力算子", "以延迟快照估计局部线性演化算子并预测两步路径。", *DMD),
    ("V68", "Path Signature", "path_signature", "路径签名", "以多通道路径迭代积分表示交互顺序和面积。", *SIGNATURE),
    ("V69", "Persistent Homology", "persistent_homology", "拓扑数据分析", "以延迟点云连通与环的持久性几何构造方向分。", *TDA),
    ("V70", "Empirical Dynamic Modeling", "edm_simplex", "非线性状态空间", "以延迟重构状态空间的simplex近邻预测局部轨道。", *EDM),
    ("V71", "Hidden Semi-Markov显式持续时间", "hsmm", "持续时间状态", "在隐状态转移中显式建模驻留时长并做因果前向过滤。", *HSMM),
    ("V72", "Fine-Gray竞争风险", "fine_gray", "竞争风险", "对离开零段的两侧累积发生率显式保留反侧竞争事件。", *FINE_GRAY),
    ("V73", "Aalen加性风险率", "aalen", "加性风险", "以加性计数过程估计协变量对离开零段强度的时变贡献。", *AALEN),
    ("V74", "GAM可加平滑", "gam", "可加模型", "以惩罚一维平滑函数的可加结构学习现货非线性。", *GAM),
    ("V75", "Regression Quantiles", "quantile_regression", "尾部回归", "以条件分位数而非均值预测本侧O2O尾部。", *QUANTILE),
    ("V76", "Expectile非对称平方损失", "expectile", "尾部回归", "以非对称平方损失学习对幅度敏感的条件尾部位置。", *EXPECTILE),
    ("V77", "Elastic Net稀疏线性", "elastic_net", "稀疏监督", "以全局线性、L1稀疏和L2收缩检验小样本可加信号。", *ELASTIC_NET),
    ("V78", "Gaussian Process", "gaussian_process", "概率监督", "以核函数上的函数分布同时输出预测均值与不确定性。", *GP),
    ("V79", "Support Vector Regression", "svr", "核监督", "以最大间隔和支持向量的核回归拟合方向化O2O。", *SVR),
    ("V80", "Random Forest", "random_forest", "袋装树模型", "以bootstrap和特征随机化的并行树集成建模现货特征。", *RANDOM_FOREST),
    ("V81", "Regularized Discriminant Analysis", "rda", "收缩判别", "以类条件协方差收缩的判别后验预测方向正确概率。", *RDA),
    ("V82", "SETAR阈值自回归", "setar", "体制自回归", "在阈值两侧拟合两套不同的开盘收益自回归动力。", *SETAR),
    ("V83", "STAR平滑转换自回归", "star", "体制自回归", "以logistic权重平滑连接两套自回归动力。", *STAR),
    ("V84", "Quantile Autoregression", "qar", "动态分位数", "让条件分位数直接随历史开盘收益滞后动态变化。", *QAR),
    ("V85", "Generalized Autoregressive Score", "gas", "score驱动状态", "以观测密度的scaled score递归更新时变位置与尺度。", *GAS),
    ("V86", "Split Conformal Prediction", "split_conformal", "共形不确定性", "以时间拆分校准残差构造本侧预测集并允许弃权。", *CONFORMAL),
    ("V87", "Conformal Test Martingale", "conformal_martingale", "在线变分布", "以共形p值的赌注鞅积累可交换性破坏方向证据。", *CONFORMAL_MART),
    ("V88", "Structured Sparse VARX", "sparse_varx", "多变量动态", "以结构稀疏VAR联合建模多通道现货滞后和两步预测。", *VARX),
    ("V89", "Echo State Network", "echo_state", "储备池动态", "以固定随机递归池压缩历史动态，仅拟合小型ridge读出。", *ESN),
    ("V90", "Projection Pursuit Regression", "projection_pursuit", "投影回归", "以少数线性投影上的一维平滑函数和表示非线性。", *PPR),

    ("V91", "GARCH条件波动递归", "garch", "条件波动", "以平方收益的条件方差递归和标准化方向冲击构造下一步支持度。", *GARCH),
    ("V92", "EGARCH杠杆波动", "egarch", "非对称条件波动", "在对数方差递归中分离冲击绝对值与符号，显式保留杠杆效应。", *EGARCH),
    ("V93", "GJR非对称GARCH", "gjr_garch", "非对称条件波动", "对负向冲击增加独立的门槛项，测试坏消息是否改变零段离开方向。", *GJR),
    ("V94", "APARCH幂次波动", "aparch", "幂次条件波动", "递归估计可变幂次的绝对冲击与非对称项，而不是仅用平方波动。", *APARCH),
    ("V95", "FIGARCH分数记忆", "figarch", "长记忆波动", "用分数差分权重保留长期波动记忆，再以当前冲击定向。", *FIGARCH),
    ("V96", "HAR多尺度实现波动", "har_rv", "多尺度波动", "以日、周、月实现波动的异质自回归结构提取波动状态与方向确认。", *HAR_RV),
    ("V97", "CAViaR条件分位数", "caviar", "条件尾部分位数", "递归更新本侧条件分位数，使用冲击和分位数滞后而非均值回归。", *CAVIAR),
    ("V98", "NoVaS波动归一化", "novas", "波动归一化", "用历史波动归一化收益并读取异常归一化冲击，避免直接假设收益分布。", *NOVAS),
    ("V99", "POT广义Pareto尾部", "evt_pot", "极值理论", "在因果阈值以上拟合广义Pareto超额，分别读取上尾/下尾条件风险。", *EVT),
    ("V100", "Hill尾指数不对称", "hill_tail", "极值尾指数", "滚动Hill估计比较正负尾厚度，形成极端风险方向分。", *HILL),

    ("V101", "S-Map局部非线性", "smap", "局部非线性状态空间", "在延迟嵌入上按距离自适应加权线性映射，允许动力随状态连续变化。", *SMAP),
    ("V102", "因果类比近邻", "analog_knn", "历史类比预测", "只在历史窗口寻找标准化轨迹类比，并用类比后的实际方向收缩。", *ANALOG),
    ("V103", "Shapelet形状片段", "shapelet", "局部形状", "从开发历史方向片段中提取可解释的局部形状距离与后续方向。", *SHAPELET),
    ("V104", "Sample Entropy样本熵", "sample_entropy", "复杂度熵", "以有限样本匹配概率估计序列复杂度，并与当前方向变化分离。", *SAMPLE_ENTROPY),
    ("V105", "Multiscale Entropy多尺度熵", "multiscale_entropy", "多尺度复杂度", "对多个粗粒化尺度计算样本熵的一致性，而非单尺度排列统计。", *MULTISCALE_ENTROPY),
    ("V106", "Lempel-Ziv复杂度", "lz_complexity", "符号复杂度", "以增量符号序列的新短语增长率衡量路径可压缩性和方向状态。", *LZ),
    ("V107", "Higuchi分形维数", "higuchi_fd", "分形几何", "用多步长折线长度估计局部分形维度，再由维度变化定向。", *HIGUCHI),
    ("V108", "Hilbert瞬时相位", "hilbert_phase", "解析信号", "用单边解析信号分离瞬时相位、振幅和相位速度，构造相位领先。", *HILBERT),
    ("V109", "EMD经验模态残差", "emd_residual", "自适应模态分解", "用因果sifting代理分解高频IMF和残差趋势，避免固定基函数。", *EMD),
    ("V110", "VMD变分模态", "vmd", "变分模态分解", "用固定频率中心的变分滤波代理提取低频模态相位与能量。", *VMD),

    ("V111", "ROCKET随机卷积核", "rocket", "随机卷积特征", "用多尺度随机卷积核的PPV和最大响应作为时间序列形状特征。", *ROCKET),
    ("V112", "Extreme Learning Machine", "elm", "随机特征监督", "固定随机隐层、只拟合线性读出，降低小样本非线性拟合自由度。", *ELM),
    ("V113", "Random Fourier Features", "rff_ridge", "核近似监督", "用固定随机傅里叶特征近似RBF核，再以ridge输出方向。", *RFF),
    ("V114", "Partial Least Squares", "pls", "潜变量回归", "在高相关现货特征中以监督潜变量逐步压缩并回归O2O方向。", *PLS),
    ("V115", "Huber稳健回归", "huber_regression", "稳健监督", "对异常O2O标签采用Huber折损，测试小样本下稳健线性关系。", *HUBER),
    ("V116", "RANSAC稳健趋势", "ransac_trend", "抗异常监督", "以随机一致性集拟合局部线性方向，显式隔离少量异常观测。", *RANSAC),
    ("V117", "Logistic离开风险", "logistic_hazard", "二元风险模型", "直接估计本侧离开零段的条件概率，而不是回归收益幅度。", *LOGISTIC),
    ("V118", "Quadratic Discriminant Analysis", "qda", "二次判别", "允许两侧协方差不同的二次判别边界，并以收缩避免小样本奇异。", *QDA),
    ("V119", "Gaussian Naive Bayes", "naive_bayes", "条件独立判别", "以每个现货特征的条件密度乘积形成两侧后验证据。", *NAIVE_BAYES),
    ("V120", "Extremely Randomized Trees", "extra_trees", "随机化树模型", "以完全随机化分裂和多树集成区别于bootstrap随机森林。", *EXTRA_TREES),
]


LOGICS: tuple[LogicSpec, ...] = tuple(LogicSpec(*row) for row in _ROWS)
LOGIC_BY_VERSION = {spec.version: spec for spec in LOGICS}

if len(LOGICS) != 120 or len(LOGIC_BY_VERSION) != 120:
    raise AssertionError("The executable registry must contain exactly V01-V120")

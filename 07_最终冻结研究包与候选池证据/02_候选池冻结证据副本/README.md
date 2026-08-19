# 候选池冻结证据副本说明

本目录中的 JSON/源码是从现有研究包复制的证据副本，便于在一个位置检查“候选池规模 → 最终冻结候选 → 冻结后运行”的链条。

## 非零退出

- `1545_rq_run_company_sides.json`：非零两侧候选池的核心记录。包括每个入选版本的原始候选数、两侧合计、去重和排名、最终 V55/V80 冻结记录，以及 Development/Validation/Test 分段指标。
- `1545_rq_reproduction_summary.json`：非零候选池与冻结结果的汇总复现记录。
- `remote_nonzero_metadata.json`：冻结运行时元数据，明确运行时只保留 V55/V80 各一个候选，`candidate_grid_rebuilt=false`。
- `nonzero_pool_runner.py`、`nonzero_candidates.py`、`nonzero_pool_registry.py`：非零研究阶段候选池的版本、参数网格和筛选/排序实现副本。

## 大涨/大跌

- `paper_engine.py`：`paper8` 的四个 8 级基础轴和 4,096 个 base 组合来源。
- `extreme_engine.py`：12 个 coverage 网格和候选参数表构造逻辑。
- `company_pool_runner.py`：研究阶段候选池选择逻辑，以及冻结后只计算一个 `base_index` 的运行逻辑。
- `极端候选元数据_paper_metadata.json`：V156/V189 的研究版本登记和候选网格说明。
- `FROZEN_FINAL_ONLY_V156_down.json`、`FROZEN_FINAL_ONLY_V189_up.json`：两个方向的最终冻结参数。
- `remote_down_extreme_summary.json`、`remote_up_extreme_summary.json`：冻结运行结果和 Test 只观察标记。

## 重要区分

远端冻结运行记录中的 `raw_candidate_count=1` 是正确的：它表示远端不再重建研究候选池，只计算一个已冻结候选。研究阶段的大候选池规模应看源码/研究记录，而不能拿运行时的 1 去替代。

V156 的元数据 count 字段未同步填值，见上级 `研究流程确认.md` 中的审计说明。这里保留了原文件，不篡改历史证据，只在登记表中写明由代码网格核验得到的 49,152。

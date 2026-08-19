# 大涨大跌候选池研究与冻结材料

这里保留大涨/大跌研究阶段的完整现货候选引擎、候选网格定义和冻结产物：

- src/o2o_research/paper_engine.py：paper8 基础组合和候选评分逻辑；
- src/o2o_research/extreme_engine.py：coverage 网格和极端事件候选逻辑；
- src/company_pool_runner.py：最终 V156/V189 的冻结后公司端复现入口；
- runtime_outputs/：最终冻结 JSON 和冻结前后登记材料。

研究阶段每侧候选网格为 4,096 × 12 = 49,152。当前日常运行包在冻结后只计算 V156 或 V189 一个候选，位于同级的 ../02_最终冻结运行/，不会把研究阶段候选池误当成日常重选流程。

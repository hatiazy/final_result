# 长0反转：现货八状态零段转移双侧最终大候选池

这是基于 V01-V120 corrected re-audit plus V121-V150 experimental quality audit; only V38/down and V57/up passed strict three-phase promotion 独立运行及外层入池后的公司端重选包。下、上两侧是两个独立候选池，可以来自不同版本和不同核心逻辑；公司端不携带本机 Top1、阈值或 Test 指标，在自己的 Development+Validation 中分别重选一个 Top1，冻结后才展示 Test。

## 版本侧别大池

- `0→-1` 下侧来源版本：V38
- `0→1` 上侧来源版本：V57
- 下侧完整候选网格：6,144
- 上侧完整候选网格：6,144
- 合计完整候选：12,288

每个入池版本/侧别的 6,144 条候选由公司端重建。`src/pool_registry.py` 只保存版本/侧别名单和规则编号，不保存本机候选结果、阈值或 Test 指标。

## 唯一运行时数据输入

运行时默认直接使用包内 `src/runtime_paths.py` 写好的参考远端现货路径模式，并把结果写入当前上传包文件夹。若挂载位置不同，可用以下环境变量覆盖；它们不再是必填项：

```text
COMPANY_SPOT_PATH       公司原始日频现货 OHLCV/amount 文件（Parquet、CSV 或 TSV）
COMPANY_OUTPUT_DIR      可选的公司端输出目录覆盖；默认就是当前上传包文件夹
```

公司端将 `COMPANY_SPOT_PATH` 设置为指定的 `CSI500_SPOT_md_eod_raw*最终版.parquet` 单一日频现货文件。文件字段为：

```text
index_code、trade_dt、crncy_code、preclose、open、high、low、close、change、pctchange、volume、amount、data_source、month
```

包内实际只使用 `trade_dt`、`preclose`、OHLCV 和 `amount`；其余列仅作现货文件元数据并被忽略。

包内 `src/zero_transfer/remote_frozen_state/spot_eight_state_config.json` 保存远端冻结的八个现货状态配方，`src/zero_transfer/spot_eight_state.py` 按同一配方从这一个现货文件重算八状态，再通过 `compute_eight_states → build_economic_features_eight → assign_eight_base_state` 生成远端一致的 `state∈{-1,0,1}` 和状态年龄。运行时不读取任何预计算 state、九状态、外部三状态基准、期货/OI/基差或其他旁路文件。

`src/spot_panel.py` 只读取 `COMPANY_SPOT_PATH` 这一个文件，并拒绝未知字段及期货/OI/基差/期限结构/流动性字段。formation/effective/exit 按该文件的实际交易日行对齐；O2O_H1 是唯一主选择口径，C2C_H1 只作观察。

以上“唯一输入”适用于 01、02、03 和 04 的正式运行链路。05 默认使用 `runtime_paths.py` 中写好的参考远端三状态基准路径，也可用 `REMOTE_THREE_STATE_PATH` 覆盖；它是单独的事后对比诊断，不会参与冻结，也不会改变上传包的现货-only 生产依赖。

## 选择与Test隔离

两侧分别在自己的完整候选池内使用公司 Development+Validation 独立排名；Test 从冻结后才计算，`test_used_for_selection=false`。每个侧别对完整 6,144 条网格逐条计算事件、状态和实际持仓段质量，并使用固定 Dev/Val 质量门冻结 Top1：Dev/Val 每段至少 20 个信号日，方向 H1/H3、H1 命中率、转移 lift/纯度、状态日/段胜率均达标，0 段平均长度控制在 3–25 天且单日 0 段占比不超过 15%，实际持仓段胜率至少 55%、平均长度 2–12 天且单日持仓不超过 10%。通过者按最差阶段质量维度排序。Test 只在冻结后读取一次，不参与候选参数或公司 Top1 选择；具体门槛和候选数量记录在 `company_freeze.json`。两侧不互相排名、不共享阈值、不把 Test 回写选择。

## 运行结果与上传边界

两个侧别 Notebook 固定侧别、无手动开关。运行时会在 Notebook 输出区打印阶段 0–5 的进度（候选数、核心逻辑分数进度、Dev/Val 扫描进度、冻结点、Test 完成），便于远端观察长任务是否仍在运行。运行结束后，冻结前代码格先展示 Dev/Val 候选概览与 Top20；最后一个代码格先展示全体候选池最终 Top1 的冻结参数和 `full`、`development`、`validation`、`test` 指标，再展示上传包内每个核心逻辑只用 Dev/Val 选出的局部 Top1 及冻结后 Test 诊断。局部 Top1 只作诊断，不改变主排序和状态质量护栏的选择，适合直接截图回传。第三个事件式导出 Notebook 会重新完成两侧冻结，并展示最终五列信号。

运行结果写入 `COMPANY_OUTPUT_DIR`，不属于上传包：`company_pre_freeze.json`（Test 锁定时的候选摘要）、`company_top20.csv`、`company_freeze.json`（全体候选池最终冻结参数与完整指标，含质量门记录）、`company_logic_top1.csv/json`（每个逻辑的 Dev/Val 局部 Top1 与冻结后 Test 诊断）、`company_logic_signal_overlap.csv/json`（冻结后实际入场日路径的跨逻辑重合诊断，不参与选择）。Notebook 最后一格还输出完整的全周期/分周期表、最近 30 个信号及 O2O H1/H3、分周期准确率与收益分位数，并画出 CSI500 收盘曲线：绿色点代表冻结 entry 后 H1 方向正确，红色点代表错误，点旁标注方向 O2O H1(bp)。两个侧别都运行后还会生成 `company_state_adjustment_diagnostics.json`：正式三状态只把下侧/上侧冻结 `entry_signal` 的当天改为 -1/+1，后续 0 天保持 0；同时单独审计实际冻结 `holding_signal` 连续段的长度、一天占比、胜率和收益分布。`persistent_relabel`、冲突延续与 1 日/2 日桥接均为非生产对照，不能进入上传信号。三个状态的逐日和完整段收益分布仍会完整输出；`quality_assessment` 只作冻结后审查，完全不参与筛选。包内有五个 Notebook、必要 `src`、README 和 requirements；没有数据、Top1、阈值、模型、结果、缓存、截图、凭据或绝对路径。

新增 `03_事件式三状态信号导出.ipynb`：它从同一个现货输入重新独立冻结下、上两侧，并输出五列事件式信号 `date / three_state / minus_entry_signal / plus_entry_signal / final_three_state`。其中 `final_three_state` 只在当天 entry signal 命中基础 0 时改为 -1/1，不会把后续整个 0 段持续重标；同日双侧冲突保留为 0。完整文件为 `company_event_three_state_signal.csv`，审计信息为 `company_event_three_state_signal.json`。

日期边界：形成日 `t` 的状态放在下一实际执行日 `t+1`。如果最新形成日是周五 `2026-08-14` 且输入中还没有下一行现货，五列导出的最后一行会暂时显示为 `2026-08-17`；等 08-17 现货进入数据后重新运行，会按实际交易行对齐。该待执行日没有未来 O2O 收益，因此 04 号审计不会把它纳入收益统计。

新增 `04_事件式段与收益审计.ipynb`：读取上述五列 CSV 和同一份现货文件，展示状态天数、分周期占比、连续段长度/碎片化、信号日 O2O_H1、逐日收益分布、持仓段复合收益分布，并绘制带状态颜色和信号标记的指数曲线。新增 `05_比较远端三状态.ipynb`：只在显式设置 `REMOTE_THREE_STATE_PATH` 时读取远端三状态基准，按实际执行日输出共同日期、列联表、逐状态天数、分周期一致率和全部差异日；缺少基准时会显示 `BASELINE_NOT_CONFIGURED`，不会误报为一致。04、05 都是冻结后诊断，不参与筛选和冻结。

## 推荐运行顺序

在远端将包放到任意目录即可。默认路径已写入 Notebook；如果需要覆盖挂载位置，可以设置：

```text
COMPANY_SPOT_PATH=<单一原始现货 Parquet>
COMPANY_OUTPUT_DIR=<可选的可写结果目录>
```

先运行 `01_零段向下_0到-1.ipynb` 和 `02_零段向上_0到1.ipynb` 可分别截图两侧完整候选扫描、冻结 Top1、全周期/分周期指标和冻结后 Test；运行 `03_事件式三状态信号导出.ipynb` 会重新独立冻结两侧并生成五列 CSV；随后运行 `04_事件式段与收益审计.ipynb` 查看段长度、碎片化、逐日/段收益和价格曲线。05 默认直接检查参考远端三状态路径；如果挂载不同，再设置 `REMOTE_THREE_STATE_PATH=<远端三状态 CSV>` 后运行。03 会重新扫描两侧候选，这是为了保证五列导出不依赖前面 Notebook 的缓存结果。

## 研究与运行边界

本版按最新运行口径只保留现货运行时输入。研究期和公司端使用同一份远端冻结对齐状态代码与 JSON 配方；唯一外部输入是原始现货数据。远端包中的 Notebook、截图和外部基准比较仅作参考/诊断，不进入上传包运行链路。

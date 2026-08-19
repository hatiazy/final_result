# 05_上传包_日期解析修正版：非零退出与大涨大跌远端验证

这个目录和 `03_汇报版本` 完全独立。当前汇报入口是 `03_汇报版本/20260818_1024_05汇报_基础三状态非零退出大涨大跌版`；本目录是把同一批冻结逻辑带到远端、只用远端现货重算，并把远端输出与本地数值结果逐日比较的工程包。本目录不包含 HTML，也不依赖汇报版本运行。

## 主入口

打开并运行：

```text
00_主入口_远端验证.ipynb
```

如果三路引擎已经运行完、只是想根据已有结果刷新最后一张简表，打开并运行：

```text
01_只更新最终执行日简表.ipynb
```

这个轻量 Notebook 只读取 `runtime_outputs/remote_nonzero_five_columns.csv`、`remote_down_extreme_daily.csv`、`remote_up_extreme_daily.csv` 和现货交易日历，不会重新跑候选池或信号引擎。它会覆盖更新 `runtime_outputs/最终执行日简表.csv`，并写入 `runtime_outputs/最终执行日简表_更新记录.json`。

如果主入口已经运行完、但只能通过截图反馈大涨/大跌的差异，打开并运行：

```text
02_截图诊断_大涨大跌一致性.ipynb
```

这个 Notebook 只读取 `runtime_outputs/大涨大跌_down_逐日对比.csv`、`大涨大跌_up_逐日对比.csv` 和 `最终一致性结论.json`，不会重跑任何引擎、不会重建候选，也不会修改结果。它会输出总体状态、按字段拆分的不一致行数，以及少量差异示例。只需把 Notebook 输出截图发回即可；优先截图“总体状态”和两张“字段级不一致统计”表。

如果需要把 `score` 的微小浮点差异按六位小数重新判断，打开并运行：

```text
03_六位小数一致性诊断.ipynb
```

这个 Notebook 同样只读取已有对比结果。最终以预测信号和收益为主，逐项输出预测标签、实际极端标签、命中、方向、O2O 和方向化 O2O 的不一致数；原始 `score` 和六位小数差异只作为辅助诊断。截图时优先看“信号和收益口径结论”表：预测、实际、命中、方向、O2O 和方向化 O2O 的不一致数应全部为 0，`信号和收益口径一致` 应为 `True`。

Notebook 会自动寻找此前包中约定的两个远端路径：

```text
现货：/home/hzy/cta/IC数据更新*最终固化版/现货最终版/CSI500_SPOT_md_eod_raw*最终版.parquet
三状态审计基准：/home/hzy/cta/三状态冻结/IC_1545_three_state_and_downside_warning.csv
```

远端只需要提供现货文件就能运行两个信号引擎；三状态文件只在引擎输出后读取，用来做逐日审计，不参与模型评分、候选选择或阈值拟合。该三状态审计文件本身就是执行日口径：文件中的 `date` 是实际执行日，模型在前一形成日收盘后计算，下一实际交易日开盘执行。Notebook 不要求你修改源码或填写路径。若在本地运行且默认远端路径不存在，Notebook 会自动使用 `data/local_smoke/` 中的本地自检现货，并把结果与 `expected/local_smoke/` 比较。

本包当前是“冻结参数复现包”，不是重新选参包。主入口只计算 V55、V80、V156、V189 四个最终登记参数；不会重建 V04/V49/V50/V57/V76/V70，也不会重建 V158/V168/V211，更不会根据远端 Test 重新选择版本。远端现货更新后，输出变化只来自新增现货和冻结公式本身。

当前固定参数为：

| 信号 | 冻结版本 | 冻结登记参数 |
|---|---|---|
| -1→0 | V55 | `score_02_q68_a1_c2_cd0`，阈值 `0.815842643776842` |
| +1→0 | V80 | `score_02_q90_a5_c1_cd0`，阈值 `0.4920211306764882` |
| 大跌 | V156 | `base_0621_cov_0.075`，阈值 `0.6832148298881413` |
| 大涨 | V189 | `base_1839_cov_0.055`，阈值 `0.8135114753699175` |

## 包内结构

- `src/remote_validation.py`：总控程序；解析路径、运行三个隔离子进程、读取本地参考结果、输出最终结论。
- `src/update_execution_summary.py`：只更新执行日简表；读取已有三路结果和现货日历，不重新运行信号引擎。
- `src/run_nonzero_remote.py`：只计算 V55/V80 冻结参数，输出五列 `date / three_state / minus_exit_signal / plus_exit_signal / final_three_state`。
- `src/run_extreme_side.py`：只计算 V156 大跌或 V189 大涨冻结参数，输出逐日得分、预测、实际阈值事件和冻结信息。
- `packages/nonzero/1545/src/`：从前一上传包中复制的非零退出冻结源码。
- `packages/extreme/src/`：从最终纯现货二分类上传包中复制的 V156/V189 现货冻结源码。
- `expected/report_freeze/`：与本次报告窗口对应的本地结果，默认远端运行会与它比较。
- `expected/local_smoke/`：随包提供的本地自检现货对应结果，保证上传前可以先完成一次闭环。
- `runtime_outputs/`：运行日志、远端结果、逐日对比表和机器可读结论。
- `runtime_outputs/最终执行日简表.csv`：对比完成后生成的简洁执行表，列为 `实际执行日 / 三状态 / +1反转 / -1反转 / 大涨 / 大跌`，各信号列使用 0/1。

## 成功条件

最后查看：

```text
runtime_outputs/最终一致性结论.json
```

只有下面三类比较都无日期或数值差异，`success` 才会为 `true`：

1. 远端重算的非零五列与本地非零五列参考结果逐日一致；
2. 远端重算的基础三状态与远端三状态审计文件逐日一致；
3. V156 大跌和 V189 大涨的逐日得分、预测、实际极端标签、O2O 和方向化 O2O 与本地参考结果逐日一致。

日期范围不一致也会判为失败，并在对比文件中区分 `GENERATED_ONLY`、`LOCAL_ONLY` 和 `MISMATCH`，不会只看共同日期而把截断输入误报成一致。浮点结果比较使用绝对误差 `1e-12`，用于消除 CSV 序列化的最后几位噪声；状态、信号、日期和分类字段仍按逐项值比较。

## 直接运行

在包根目录执行：

```bash
python src/remote_validation.py
```

如果远端需要明确指定现货文件，可以只设置环境变量，不改 Notebook 或源码：

```bash
export COMPANY_SPOT_PATH='/home/hzy/cta/IC数据更新*最终固化版/现货最终版/CSI500_SPOT_md_eod_raw*最终版.parquet'
export REMOTE_THREE_STATE_PATH='/home/hzy/cta/三状态冻结/IC_1545_three_state_and_downside_warning.csv'
python src/remote_validation.py
```

`COMPANY_SPOT_PATH` 只影响现货输入；`REMOTE_THREE_STATE_PATH` 只影响审计基准。运行时会生成：

如果三路详细结果已经存在，只想重新生成执行日简表，不需要重跑引擎：

```bash
python src/update_execution_summary.py
```

该命令也支持 `--spot` 指定现货文件、`--output` 指定 `runtime_outputs` 目录；不指定时沿用主入口的环境变量和默认路径解析规则。

- `remote_nonzero_five_columns.csv` 与 `非零五列逐日对比.csv`；
- `三状态逐日对比.csv`；
- `remote_down_extreme_daily.csv`、`remote_up_extreme_daily.csv` 及两个逐日对比文件；
- `最终执行日简表.csv`：把非零退出和两侧大涨大跌预测对齐到实际执行日；大涨大跌预测从形成日映射到下一实际现货交易日；
- `remote_nonzero_metadata.json`、两侧 `remote_*_extreme_summary.json`；
- `run_nonzero.log`、`run_extreme_down.log`、`run_extreme_up.log`；
- `最终一致性结论.json`。

## 本地自检解释

本地自检现货是已审计的纯现货 Parquet，最新现货交易日到 2026-08-17。按执行日口径，非零五列已经生成到 2026-08-18；其中 2026-08-18 是最新执行日状态行，但因为本地现货尚未有 2026-08-18 开盘价，所以该行没有完整 H1 O2O。最新完整的非零 H1 O2O 可计算到 2026-08-14（2026-08-14 开盘到 2026-08-17 开盘）；大涨大跌逐日评价需要两天未来开盘，本次完整逐日结果到 2026-08-13，对应的未来开盘可对齐到 2026-08-17。旧的 2026-08-07 本地 fixture 保留在包内作为历史文件，不作为当前自检默认输入。它们和报告主材料中的本地冻结结果分别保存在 `expected/local_smoke/` 与 `expected/report_freeze/`，不会被远端引擎读取，只在最后的比较步骤中读取。

运行前提是环境已经安装 `requirements.txt` 中的 pandas、numpy、scipy、scikit-learn、pyarrow 和 Jupyter 依赖。三个引擎分别在子进程中运行，避免两个冻结包中同名模块互相污染。

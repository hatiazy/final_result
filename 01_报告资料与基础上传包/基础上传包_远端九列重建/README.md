# 远端九列重建与逐日一致性审计上传包

这个包用于在远端重新运行两个最终冻结包，然后合并成与本地相同口径的 9 列结果，并逐日核对基础三状态。远端运行链路只需要两个外部数据输入：

1. 原始 CSI500 现货 Parquet/CSV/TSV；
2. 远端三状态基准文件。

两个默认路径来自此前最终上传包：

```text
COMPANY_SPOT_PATH=/home/hzy/cta/IC数据更新*最终固化版/现货最终版/CSI500_SPOT_md_eod_raw*最终版.parquet
REMOTE_THREE_STATE_PATH=/home/hzy/cta/三状态冻结/IC_1545_three_state_and_downside_warning.csv
```

如果远端挂载路径不同，只需设置这两个环境变量。现货文件至少要有此前包使用的 `trade_dt/open/high/low/close/volume/amount`，最好同时有 `preclose` 和 `index_code`；三状态文件至少要有 `date` 与 `three_state`。审计程序也支持常见别名，例如 `effective_date`、`state`、`base_state`。

## 运行

解压后进入本目录，先设置路径：

```bash
export COMPANY_SPOT_PATH='/home/hzy/cta/IC数据更新*最终固化版/现货最终版/CSI500_SPOT_md_eod_raw*最终版.parquet'
export REMOTE_THREE_STATE_PATH='/home/hzy/cta/三状态冻结/IC_1545_three_state_and_downside_warning.csv'
```

这两行就是此前最终上传包中的默认路径，可以直接复制执行；如果远端实际挂载位置不同，再改成实际路径即可。

然后打开并运行主 Notebook：

```text
01_远端重建九列并逐日审计.ipynb
```

也可以直接运行脚本：

```bash
python remote_rebuild.py
```

如果需要把结果写到其他目录，可以再设置：

```bash
export REMOTE_OUTPUT_DIR='/home/hzy/cta/远端九列审计结果'
```

两个冻结引擎会在隔离的子进程中运行，避免两个包中同名顶层模块相互污染。运行时间取决于远端候选池计算速度；运行时会实时打印 `1545` 和 `长0` 的进度，并保留 `run_1545.log`、`run_long0.log`。

## 输出与成功条件

默认写入 `远端输出/`：

- `remote_1545_five_columns.csv`：远端重建的 `date / three_state / minus_exit_signal / plus_exit_signal / final_three_state`；
- `remote_long0_five_columns.csv`：远端重建的 `date / three_state / minus_entry_signal / plus_entry_signal / final_three_state`；
- `remote_combined_nine_columns.csv`：最终 9 列：
  `date, three_state, minus_exit_signal, plus_exit_signal, reversal_final_three_state, minus_entry_signal, plus_entry_signal, transfer_final_three_state, combined_final_three_state`；
- `三状态逐日对比.csv`：1545 重建三状态、长0重建三状态、远端三状态基准的逐日并排比较；
- `本地1545五列逐日对比.csv`：远端 1545 五列与本地 1545 五列审计附件的逐列比较；
- `本地长0五列逐日对比.csv`：远端长0五列与本地长0五列审计附件的逐列比较；
- `本地九列逐日对比.csv`：远端 9 列与包内本地参考 9 列的逐日比较；
- `run_1545.log`、`run_long0.log`：两个冻结引擎的完整运行日志。

主 Notebook 的最后几格会直接在输出栏展示机器可读结论和逐日差异。重点看：

```text
all_three_state_checks_exact == True
local_five_audits['1545']['no_date_or_value_difference'] == True
local_five_audits['long0']['no_date_or_value_difference'] == True
local_nine_audit['no_date_or_value_difference'] == True
success == True
```

其中第一项要求：共同日期完整、1545 重建三状态与远端基准逐日一致、长0重建三状态与远端基准逐日一致、两个引擎彼此一致。第二项要求远端九列与包内本地参考九列在共同日期上的列值逐项一致。若日期范围不同，Notebook 会同时展示 `remote_only_rows`、`generated_only_rows`，不会把不完整的比较误报成一致。

## 九列口径

四个信号都是 event-only：

- `minus_exit_signal=1` 且基础为 `-1`：当天 `reversal_final_three_state=0`；
- `plus_exit_signal=1` 且基础为 `+1`：当天 `reversal_final_three_state=0`；
- `minus_entry_signal=1` 且基础为 `0`：当天 `transfer_final_three_state=-1`；
- `plus_entry_signal=1` 且基础为 `0`：当天 `transfer_final_three_state=+1`；
- 同日两个入口信号冲突时，最终保持 `0`；
- 后续日期不延续信号，重新读取基础三状态。

包内 `reference/本地1545五列参考结果.csv`、`reference/本地长0五列参考结果.csv` 和 `reference/本地九列参考结果.csv` 是当前本地 2091 行结果的审计附件，用于防止远端运行后只看行数、不看逐列值。两个冻结引擎运行期间不会读取或展示这些附件；只有两个远端五列结果都生成后，Notebook 的最后审计部分才会读取并逐列比较。它们不是远端运行的外部输入，也不会参与远端冻结选择。

## 日期注意事项

此前包的 `date` 是形成日收盘后对应的下一实际现货交易日执行日期；如果最新形成日后尚无下一行现货，最终待执行日期可能暂时显示为下一个工作日。该尾行没有完整未来行情，不应把它当作已经实现的收益，但仍应纳入九列和三状态逐日一致性核对。

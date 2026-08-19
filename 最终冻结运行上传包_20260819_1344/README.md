# 最终冻结运行上传包（20260819_1344）

这是日常运行的唯一冻结入口，结构和 `06_最终上传包` 一致，但只保留冻结参数直接运行所需的内容。包内没有候选池扫描包、汇报材料或重复套娃目录。

部署根目录必须是绝对路径：

```text
/home/hzy/cta/最终冻结运行上传包_20260819_1344
```

## 包内内容

- `01_只用远端现货生成六列.ipynb`：只用远端现货和包内冻结参数生成最新六列表；
- `02_六列表详细一致性检查.ipynb`：把生成结果与 `expected/local_freeze/最终执行日简表_本地冻结参考.csv` 逐日逐列核对；
- `src/`：运行入口、六列拼接和 02 检查脚本；
- `packages/nonzero/1545/`：三状态、V55、V80 冻结运行代码；
- `packages/extreme/`：V156、V189 冻结运行代码；
- `expected/local_freeze/`：02 使用的本地冻结参考；
- `requirements.txt`：运行依赖。

固定使用的版本是 V55、V80、V156、V189，不重新遍历候选池、不重新选择版本、不用本地结果参与生成。

输入历史按冻结血缘从 `2007-01-15` 起保留。米筐接口若返回更早的 2005–2006 行，运行代码会明确丢弃这些前置行并在审计元数据中记录；这是为了保持因果滚动特征与冻结参考一致，不是训练集起点。Development 仍从 `2018-01-01` 开始，Validation 为 2023–2024，Test 为 2025 至最新。

## 运行顺序

### 1. 生成六列表

打开 `01_只用远端现货生成六列.ipynb` 并运行。Notebook 通过绝对路径读取：

```text
/home/hzy/cta/IC数据更新*最终固化版/现货最终版/CSI500_SPOT_md_eod_raw*最终版.parquet
```

也可以设置绝对路径环境变量 `COMPANY_SPOT_PATH`，或在命令行显式传入绝对路径：

```bash
python /home/hzy/cta/最终冻结运行上传包_20260819_1344/src/generate_compact_output.py \
  --spot /home/hzy/cta/IC数据更新_最终固化版/现货最终版/CSI500_SPOT_md_eod_raw_最终版.parquet
```

输出目录为：

```text
/home/hzy/cta/最终冻结运行上传包_20260819_1344/runtime_outputs/
```

最终文件是：

```text
/home/hzy/cta/最终冻结运行上传包_20260819_1344/runtime_outputs/最终执行日简表.csv
```

最终 CSV 固定只有六列：

```text
实际执行日、三状态、+1反转、-1反转、大涨、大跌
```

同时会在 `/home/hzy/cta/最终冻结运行上传包_20260819_1344/runtime_outputs/_engine_outputs/` 保存运行所需的内部中间结果。其中，大涨/大跌的 `remote_up_extreme_predictions.csv` 和 `remote_down_extreme_predictions.csv` 是运行预测输入，包含最新形成日；`*_daily.csv` 只用于有未来 O2O 标签的历史评价。最终六列表只读取 `*_predictions.csv`，不会因等待评价标签而漏掉最新信号。最终 CSV 不包含成交记录和评价字段。

### 2. 运行 02 一致性检查

确认 01 已生成：

```text
/home/hzy/cta/最终冻结运行上传包_20260819_1344/runtime_outputs/最终执行日简表.csv
```

再打开 `02_六列表详细一致性检查.ipynb` 并运行。它读取包内绝对路径：

```text
/home/hzy/cta/最终冻结运行上传包_20260819_1344/expected/local_freeze/最终执行日简表_本地冻结参考.csv
```

并生成：

```text
/home/hzy/cta/最终冻结运行上传包_20260819_1344/runtime_outputs/六列表逐日对比.csv
/home/hzy/cta/最终冻结运行上传包_20260819_1344/runtime_outputs/六列表一致性结论.json
/home/hzy/cta/最终冻结运行上传包_20260819_1344/runtime_outputs/六列表不一致示例.csv
```

重点检查 `六列表一致性结论.json` 中的 `success`、日期范围、`remote_only_rows`、`local_only_rows`，以及五个信号字段的不一致行数。

## 日期和执行口径

所有信号严格按以下口径：

```text
t 日收盘形成信号 → t+1 实际交易日上午开盘执行
```

因此，最终六列表中的 `实际执行日` 不是形成信号的收盘日，而是下一实际交易日。比如现货最新一行是 18 日，下一实际交易日是 19 日，则 19 日这一行的数值全部由 18 日收盘及之前的数据计算得到，19 日只表示执行日，不是填入的默认数值。也就是说，执行日这一行可以在当天开盘直接使用。

非零信号导出层和大涨/大跌拼接层都带有严格的向后日期保护：执行日必须大于形成日。大涨/大跌引擎内部 `date` 是形成日，最终拼接时先按远端现货交易日历映射到下一实际交易日；如果最新形成日尚未有下一行现货数据，则按下一个工作日生成该次执行日，使最新预测仍然进入六列表。

## 绝对路径约束

包内 Notebook 默认使用 `FINAL_UPLOAD_PACKAGE_ROOT=/home/hzy/cta/最终冻结运行上传包_20260819_1344`。若覆盖该变量，必须仍然是绝对路径；`COMPANY_SPOT_PATH`、`1545_SPOT_PATH`、命令行 `--spot` 和 `--output` 也都拒绝相对路径。代码文件自身通过 `__file__` 解析包根目录，运行时得到绝对路径。

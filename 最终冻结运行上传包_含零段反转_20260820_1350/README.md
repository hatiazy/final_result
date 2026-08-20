# 最终冻结运行上传包（含零段反转）20260820_1350

这是一个独立的冻结运行包。在 20260819_1441 六列运行包的基础上新增两列：`0转-1`、`0转+1`。生产直算只读取一份远端原始现货文件；02 对比 Notebook 才另外读取远端三状态和包内八列表冻结参考，二者都不参与 01 的信号生成。

本次最终快照已通过本地米筐 `rqdatac` 接口更新到 `2026-08-19`，对应最新执行日为 `2026-08-20`；包内八列表参考也按这份最新快照重建。

## 包内内容

- `01_只用远端现货生成八列.ipynb`：调用唯一生产入口，生成最新八列表；
- `02_八列表详细一致性检查.ipynb`：分别对比包内八列表冻结参考和远端三状态；
- `src/run_nonzero_remote.py`：V55/V80 非零退出冻结运行；
- `src/run_zero_transfer_frozen.py`：V38/V57 零段反转冻结运行，只使用冻结阈值，不扫描候选、不使用未来评价标签；
- `src/run_extreme_side.py`：V156/V189 大涨/大跌冻结运行，保留最新形成日预测；
- `src/generate_compact_output.py`：按实际执行日合并八列；
- `src/compare_compact_output.py`：八列表与远端三状态的双重检查；
- `packages/nonzero/1545/`、`packages/extreme/`：原六列冻结引擎；
- `packages/zero_transfer/`：V38/V57 零段反转及现货八状态冻结引擎；
- `expected/local_freeze/最终执行日简表_零段反转冻结参考.csv`：已验证的八列表参考快照；
- `requirements.txt`：固定 pandas 2.x，拒绝 pandas 3.x，避免冻结八状态的累加顺序变化。

包内没有汇报材料、候选扫描结果、成交记录、评价记录、运行缓存或重复套娃目录。

## 最终输出

01 输出：

```text
/home/hzy/cta/最终冻结运行上传包_含零段反转_20260820_1350/runtime_outputs/最终执行日简表.csv
```

列顺序固定为：

```text
实际执行日、三状态、+1反转、-1反转、0转-1、0转+1、大涨、大跌
```

这是在原六列表上新增两列，其他信号的计算和输出方式保持不变。内部运行结果写入：

```text
/home/hzy/cta/最终冻结运行上传包_含零段反转_20260820_1350/runtime_outputs/_engine_outputs/
```

其中 `remote_zero_transfer_predictions.csv` 明确保留 `formation_date`、`execution_date`、两列零段信号，供日期审计；最终简表不包含成交或评价字段。

## 运行方式

部署根目录必须是绝对路径，例如：

```text
/home/hzy/cta/最终冻结运行上传包_含零段反转_20260820_1350
```

也可以显式传入绝对路径：

```bash
python /home/hzy/cta/最终冻结运行上传包_含零段反转_20260820_1350/src/generate_compact_output.py \
  --spot /home/hzy/cta/IC数据更新_最终固化版/现货最终版/CSI500_SPOT_md_eod_raw_最终版.parquet \
  --output /home/hzy/cta/最终冻结运行上传包_含零段反转_20260820_1350/runtime_outputs
```

Notebook 和入口支持 `COMPANY_SPOT_PATH`、`UPLOAD_OUTPUT_DIR`、`FINAL_UPLOAD_PACKAGE_ROOT` 覆盖，但这些值必须是绝对路径。01 不读取本地八列表参考，也不读取远端三状态。

## 冻结规则

零段反转只使用以下已冻结规则：

- `0→-1`：V38，`V38_down_s01_a1_q0.85_c1_H03`，Dev 阈值 `1.2294466376933055`，释放阈值 `0.6645986031220402`；
- `0→+1`：V57，`V57_up_s01_a1_q0.85_c1_H04`，Dev 阈值 `4.671980676328502`，释放阈值 `4.606280193236715`。

运行时只从现货重新生成八状态、三状态和两侧评分，然后直接应用上述冻结阈值及持有规则。不会重新扫描 6,144 条候选，不会按当前 Test 重新选参数，也不会用未来 O2O 标签决定最新一行是否发信号。

## 形成日与执行日口径

所有八列统一遵循：

```text
t 日收盘形成并计算 → 下一实际交易日 t+1 开盘执行
```

因此，最新现货到 18 日时，最终八列表的 19 日行由 18 日收盘及之前数据真实计算得到；19 日只是执行日列，不是默认值占位。若输入还没有下一实际交易行，最后形成日暂按下一个工作日展示，信号数值仍然来自最后一个形成日的冻结规则。每一行都经过 `formation_date < execution_date` 检查，禁止用 0 补齐缺失信号。

零段反转运行还明确记录 `future_o2o_label_used_for_signal=false`。这保证最新形成日可以在下一交易日上午直接使用。

## 02 一致性检查

运行 01 后，再运行 `02_八列表详细一致性检查.ipynb`。它读取：

1. 01 生成的八列表；
2. `expected/local_freeze/最终执行日简表_零段反转冻结参考.csv`；
3. `REMOTE_THREE_STATE_PATH` 指向的远端三状态文件，默认绝对路径为：

```text
/home/hzy/cta/三状态冻结/IC_1545_three_state_and_downside_warning.csv
```

检查结果写入：

```text
/home/hzy/cta/最终冻结运行上传包_含零段反转_20260820_1350/runtime_outputs/八列表逐日对比.csv
/home/hzy/cta/最终冻结运行上传包_含零段反转_20260820_1350/runtime_outputs/三状态逐日对比.csv
/home/hzy/cta/最终冻结运行上传包_含零段反转_20260820_1350/runtime_outputs/八列表一致性结论.json
```

只有 `all_dates_and_signals_match=true` 且 `all_dates_and_three_state_match=true` 时，`success` 才为 `true`。02 只做对比，不回灌生产计算。

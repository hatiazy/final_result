"""绘制 1545 基础三状态与四个反转信号的本地多空策略净值。

本脚本只做冻结信号的本地事后评价，不重新扫描候选、不重新选择参数，
也不使用 Test 集反向筛选。零段反转沿用此前确定的持有期口径：

* 0 转 -1：信号日及其后两个实际交易日，共持有 3 个交易日；
* 0 转 +1：信号日及其后四个实际交易日，共持有 5 个交易日；
* 持有期内基础三状态仍为 0 时保持方向；基础三状态变为非零时跟随基础状态；
* 持有期内出现反方向零段信号，或同日同时出现两个零段信号，调整状态归零。

收益口径：冻结状态已经放在实际执行日，基础本地累计净值沿用此前
“Raw Three-State”结果的严格位置映射，即实际执行日状态（-1/0/+1）乘以
该日收盘到收盘收益 c2c_1d。o2o_h1 仍保留并由执行日开盘价重算，用于核对
信号评价口径，但不直接替代此前本地累计净值的 c2c_1d 口径。净值曲线采用
研究图的加法累计：NAV=1+Σ(状态×当日收益)，不是交易账户的复利净值。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DOWN_HOLD_DAYS = 3
UP_HOLD_DAYS = 5


def _as_int(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)


def build_holding_aware_state(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按四个信号和零段持有期构造调整后三状态。

    返回：调整后三状态、零段持有方向、零段持有结束位置（均按行位置）。
    """

    base = _as_int(frame, "three_state").to_numpy()
    minus_exit = _as_int(frame, "minus_exit_signal").to_numpy()
    plus_exit = _as_int(frame, "plus_exit_signal").to_numpy()
    minus_entry = _as_int(frame, "minus_entry_signal").to_numpy()
    plus_entry = _as_int(frame, "plus_entry_signal").to_numpy()

    adjusted = np.zeros(len(frame), dtype=np.int8)
    hold_direction = np.zeros(len(frame), dtype=np.int8)
    hold_end = np.full(len(frame), -1, dtype=np.int32)
    active_direction = 0
    active_end = -1

    for i, raw_state in enumerate(base):
        raw_state = int(raw_state)

        # 基础状态一旦回到非零，零段持有包立即失效；但非零退出信号
        # 仍按“非零状态退出到 0”的事件日规则执行。
        if raw_state == -1:
            active_direction = 0
            active_end = -1
            adjusted[i] = 0 if minus_exit[i] else -1
        elif raw_state == 1:
            active_direction = 0
            active_end = -1
            adjusted[i] = 0 if plus_exit[i] else 1
        elif raw_state == 0:
            if i > active_end:
                active_direction = 0
                active_end = -1

            down = bool(minus_entry[i])
            up = bool(plus_entry[i])

            if active_direction:
                # 同日两个零段方向同时命中，或者持有期内命中反方向，
                # 按此前确定的冲突规则归零并结束持有。
                opposite = (active_direction == -1 and up) or (active_direction == 1 and down)
                both = down and up
                if opposite or both:
                    adjusted[i] = 0
                    active_direction = 0
                    active_end = -1
                else:
                    # 同方向重复信号可以把持有期向后延长；当前数据中
                    # 基本没有同方向重叠，但这里保留完整的冻结处理逻辑。
                    if active_direction == -1 and down:
                        active_end = max(active_end, i + DOWN_HOLD_DAYS - 1)
                    elif active_direction == 1 and up:
                        active_end = max(active_end, i + UP_HOLD_DAYS - 1)
                    adjusted[i] = active_direction
            else:
                if down and up:
                    adjusted[i] = 0
                elif down:
                    active_direction = -1
                    active_end = i + DOWN_HOLD_DAYS - 1
                    adjusted[i] = -1
                elif up:
                    active_direction = 1
                    active_end = i + UP_HOLD_DAYS - 1
                    adjusted[i] = 1
                else:
                    adjusted[i] = 0
        else:
            raise ValueError(f"三状态存在非法值: {raw_state}")

        if active_direction:
            hold_direction[i] = active_direction
            hold_end[i] = active_end

    return adjusted, hold_direction, hold_end


def _nav(strategy_returns: pd.Series) -> pd.Series:
    # 参考图采用研究净值的加法累计口径，而不是可交易账户的复利口径：
    # NAV_t = 1 + sum_{s<=t}(position_s * return_s)。
    # 第一条可评价日的状态收益也计入该日累计值，因此首日 NAV 通常不等于 1。
    return 1.0 + strategy_returns.astype(float).cumsum()


def _max_drawdown(nav: pd.Series) -> float:
    running_max = nav.cummax()
    return float((nav / running_max - 1.0).min())


def _summary_row(name: str, position: pd.Series, market_return: pd.Series, nav: pd.Series, dates: pd.Series) -> dict[str, object]:
    strategy_return = position * market_return
    active = position.ne(0)
    directional = strategy_return[active]
    years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 1 / 365.25)
    final_nav = float(nav.iloc[-1])
    return {
        "series": name,
        "start_date": dates.iloc[0].strftime("%Y-%m-%d"),
        "end_date": dates.iloc[-1].strftime("%Y-%m-%d"),
        "observations": int(len(dates)),
        "final_nav": final_nav,
        "total_return_pct": (final_nav - 1.0) * 100.0,
        "annualized_return_pct": (final_nav ** (1.0 / years) - 1.0) * 100.0,
        "max_drawdown_pct": _max_drawdown(nav) * 100.0,
        "active_days": int(active.sum()),
        "active_day_share_pct": float(active.mean() * 100.0),
        "directional_hit_rate_pct": float((directional > 0).mean() * 100.0) if len(directional) else np.nan,
        "mean_daily_directional_return_bp": float(directional.mean() * 10000.0) if len(directional) else np.nan,
    }


def _make_plot(frame: pd.DataFrame, figure_path: Path) -> None:
    """用标准 Matplotlib 绘制与参考图一致的线性净值图。"""

    fig, ax = plt.subplots(figsize=(13.0, 5.8), dpi=180)
    ax.plot(frame["date"], frame["base_nav"], color="#2f80ed", linewidth=1.25, label="Raw 1545")
    ax.plot(frame["date"], frame["adjusted_nav"], color="#e74c3c", linewidth=1.25, label="Adj 1545 (+4 Reversal)")
    ax.plot(frame["date"], frame["index_nav"], color="#9aa0a6", linewidth=0.95, label="CSI500 B&H")
    ax.axhline(1.0, color="#303030", linewidth=0.55, linestyle=":")
    ax.set_title("1545 Three-State NAV: Raw vs Adjusted (+4 Reversal) vs Index", fontsize=13, pad=10, weight="bold")
    ax.set_ylabel("NAV")
    ax.set_xlabel("Date")
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.45, alpha=0.7)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.012)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.75)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _load_frame(panel_path: Path, signal_path: Path | None) -> pd.DataFrame:
    """读取本地收益面板，并优先用冻结包内八列表提供状态和信号。"""

    panel = pd.read_csv(panel_path)
    if signal_path is None or not signal_path.exists():
        return panel

    frozen = pd.read_csv(signal_path)
    frozen_date = "实际执行日"
    required_frozen = {frozen_date, "三状态", "+1反转", "-1反转", "0转-1", "0转+1"}
    missing_frozen = sorted(required_frozen.difference(frozen.columns))
    if missing_frozen:
        raise ValueError(f"冻结参考文件缺少必要列: {missing_frozen}")

    frozen = frozen.rename(
        columns={
            frozen_date: "date",
            "三状态": "three_state",
            "+1反转": "plus_exit_signal",
            "-1反转": "minus_exit_signal",
            "0转-1": "minus_entry_signal",
            "0转+1": "plus_entry_signal",
        }
    )
    frozen["date"] = pd.to_datetime(frozen["date"], errors="coerce").dt.normalize()
    # 只把包内冻结的状态/信号覆盖到同日期行情面板；o2o_h1、phase 等
    # 评价字段仍来自本地研究面板。
    signal_columns = [
        "date",
        "three_state",
        "minus_exit_signal",
        "plus_exit_signal",
        "minus_entry_signal",
        "plus_entry_signal",
    ]
    frozen = frozen[signal_columns].dropna(subset=["date"]).drop_duplicates("date", keep="last")
    price_columns = [column for column in panel.columns if column not in signal_columns[1:]]
    panel = panel[price_columns].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    return panel.drop(columns=[column for column in signal_columns[1:] if column in panel.columns], errors="ignore").merge(
        frozen,
        on="date",
        how="inner",
        validate="one_to_one",
    )


def run(panel_path: Path, output_dir: Path, signal_path: Path | None = None) -> tuple[Path, Path, Path]:
    frame = _load_frame(panel_path, signal_path)
    required = {
        "date",
        "three_state",
        "minus_exit_signal",
        "plus_exit_signal",
        "minus_entry_signal",
        "plus_entry_signal",
        "open",
        "c2c_1d",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"panel.csv 缺少必要列: {missing}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    frame["c2c_1d"] = pd.to_numeric(frame["c2c_1d"], errors="coerce")
    frame["o2o_h1"] = pd.to_numeric(frame.get("o2o_h1"), errors="coerce")
    frame = frame.loc[frame["date"].notna() & frame["open"].notna()].copy()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if frame.empty:
        raise ValueError("没有可用于本地策略评价的有效执行日和开盘价")

    frame["execution_date"] = frame["date"]
    # 冻结包里的日期是实际执行日：当日开盘执行信号，收益取当日开盘
    # 到下一实际交易日开盘。这里直接由开盘价重算，避免把形成日标签
    # 错当成执行日收益；同时核对研究面板中的 o2o_h1 是否一致。
    frame["next_execution_date"] = frame["date"].shift(-1)
    frame["next_execution_open"] = frame["open"].shift(-1)
    frame["execution_o2o_return"] = frame["next_execution_open"].div(frame["open"]).sub(1.0)
    alignment_mask = frame["execution_o2o_return"].notna() & frame["o2o_h1"].notna()
    if alignment_mask.any():
        alignment_gap = (frame.loc[alignment_mask, "execution_o2o_return"] - frame.loc[alignment_mask, "o2o_h1"]).abs()
        if float(alignment_gap.max()) > 1e-10:
            raise ValueError(f"执行日开盘收益与 panel.o2o_h1 不一致，最大差值={float(alignment_gap.max())}")

    # 状态机在完整的有价格执行日上运行；最后一个没有收益数据的
    # 执行日仍保留状态审计，但不进入净值评价。
    adjusted, hold_direction, hold_end = build_holding_aware_state(frame)
    base_state = _as_int(frame, "three_state")
    # 历史本地策略图使用执行日状态乘以当日
    # close / prev_close - 1。execution_o2o_return 只作为执行日核对列保留。
    market_return = frame["c2c_1d"].astype(float)
    adjusted_state = pd.Series(adjusted, index=frame.index, dtype="int8")
    base_return = base_state * market_return
    adjusted_return = adjusted_state * market_return
    index_return = market_return.copy()

    frame["adjusted_three_state"] = adjusted_state
    frame["zero_hold_direction"] = hold_direction
    frame["zero_hold_end_date"] = [
        frame.loc[end, "date"].strftime("%Y-%m-%d") if end >= 0 and end < len(frame) else ""
        for end in hold_end
    ]
    frame["base_strategy_return"] = base_return
    frame["adjusted_strategy_return"] = adjusted_return
    frame["index_return"] = index_return
    valid_return = frame["c2c_1d"].notna()
    eval_frame = frame.loc[valid_return].copy().reset_index(drop=True)
    eval_base_state = base_state.loc[valid_return].reset_index(drop=True)
    eval_adjusted_state = adjusted_state.loc[valid_return].reset_index(drop=True)
    eval_market_return = market_return.loc[valid_return].reset_index(drop=True)
    eval_base_return = base_return.loc[valid_return].reset_index(drop=True)
    eval_adjusted_return = adjusted_return.loc[valid_return].reset_index(drop=True)
    eval_index_return = index_return.loc[valid_return].reset_index(drop=True)
    eval_frame["base_nav"] = _nav(eval_base_return)
    eval_frame["adjusted_nav"] = _nav(eval_adjusted_return)
    eval_frame["index_nav"] = _nav(eval_index_return)
    frame["base_nav"] = np.nan
    frame["adjusted_nav"] = np.nan
    frame["index_nav"] = np.nan
    frame.loc[valid_return, "base_nav"] = eval_frame["base_nav"].to_numpy()
    frame.loc[valid_return, "adjusted_nav"] = eval_frame["adjusted_nav"].to_numpy()
    frame.loc[valid_return, "index_nav"] = eval_frame["index_nav"].to_numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "本地策略逐日净值与状态.csv"
    summary_path = output_dir / "本地策略摘要.csv"
    figure_path = output_dir / "1545_Three-State_NAV_Raw_vs_Adjusted_4_Reversal_vs_CSI500.png"

    detail_columns = [
        "date",
        "execution_date",
        "next_execution_date",
        "open",
        "next_execution_open",
        "phase",
        "three_state",
        "minus_exit_signal",
        "plus_exit_signal",
        "minus_entry_signal",
        "plus_entry_signal",
        "adjusted_three_state",
        "zero_hold_direction",
        "zero_hold_end_date",
        "c2c_1d",
        "o2o_h1",
        "execution_o2o_return",
        "base_strategy_return",
        "adjusted_strategy_return",
        "index_return",
        "base_nav",
        "adjusted_nav",
        "index_nav",
    ]
    frame[detail_columns].to_csv(detail_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    summary = pd.DataFrame(
        [
            _summary_row("Raw 1545 Three-State", eval_base_state, eval_market_return, eval_frame["base_nav"], eval_frame["date"]),
            _summary_row("Adjusted 1545 (+4 Reversal)", eval_adjusted_state, eval_market_return, eval_frame["adjusted_nav"], eval_frame["date"]),
            _summary_row("CSI500 Buy & Hold", pd.Series(1, index=eval_frame.index), eval_market_return, eval_frame["index_nav"], eval_frame["date"]),
        ]
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    _make_plot(eval_frame, figure_path)

    return figure_path, detail_path, summary_path


def main() -> None:
    package_root = Path(__file__).resolve().parent
    default_panel = package_root.parent / "01_报告资料与基础上传包" / "统计表" / "panel.csv"
    default_signals = package_root / "expected" / "local_freeze" / "最终执行日简表_零段反转冻结参考.csv"
    parser = argparse.ArgumentParser(description="绘制基础三状态与四信号调整后三状态的本地多空策略净值")
    parser.add_argument("--panel", type=Path, default=default_panel, help="包含状态、四个信号、c2c_1d 和 o2o_h1 的本地 panel.csv")
    parser.add_argument("--signals", type=Path, default=default_signals, help="冻结包内八列表参考；默认优先使用")
    parser.add_argument("--output-dir", type=Path, default=package_root, help="结果输出目录")
    args = parser.parse_args()
    signal_path = args.signals if args.signals.exists() else None
    figure_path, detail_path, summary_path = run(args.panel, args.output_dir, signal_path)
    print(f"signal_source={signal_path if signal_path is not None else args.panel}")
    print(f"figure={figure_path}")
    print(f"detail={detail_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()

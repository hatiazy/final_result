from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


EXTRA_CSS = r"""
    .extreme-view { border-top: 1px solid var(--line); padding-top: 2px; }
    .extreme-view.hide { display: none !important; }
    .extreme-toolbar { display:flex; justify-content:space-between; gap:10px; align-items:center; flex-wrap:wrap; padding:0 0 9px; color:var(--muted); font-size:11px; }
    .extreme-toolbar button { border:1px solid var(--line-strong); border-radius:8px; padding:6px 10px; background:#fff; color:var(--navy-2); font-size:11px; font-weight:700; }
    .extreme-toolbar button:hover { border-color:var(--blue); background:#f2f7fd; }
    .extreme-svg-wrap { overflow-x:auto; overflow-y:hidden; border:1px solid #e5ebf3; border-radius:12px; background:linear-gradient(180deg,#fbfdff,#f8fbfe); cursor:grab; }
    .extreme-svg-wrap.dragging { cursor:grabbing; user-select:none; }
    .extreme-svg-wrap svg { display:block; width:auto; min-width:1120px; height:auto; min-height:320px; }
    .extreme-line { fill:none; stroke:#6f829a; stroke-width:1.55; opacity:.78; }
    .extreme-grid { stroke:#e2eaf2; stroke-width:1; }
    .extreme-axis { stroke:#b9c7d7; stroke-width:1; }
    .extreme-axis-text { fill:#718197; font-size:11px; }
    .extreme-marker-up { fill:#f04438; stroke:#fff; stroke-width:1.5; }
    .extreme-marker-down { fill:#16b364; stroke:#fff; stroke-width:1.5; }
    .extreme-marker-conflict { fill:#7c858f; stroke:#fff; stroke-width:1.5; }
    .extreme-marker-actual { fill:none; stroke:#e8793d; stroke-width:2; }
    .extreme-marker-hit { fill:#fff; stroke:#0f8a43; stroke-width:2.6; }
    .extreme-readout { margin-top:8px; padding:9px 11px; border:1px solid #e2eaf3; border-radius:10px; background:#f7f9fc; color:#40536b; font-size:12px; min-height:38px; }
    .extreme-legend { display:flex; gap:13px; flex-wrap:wrap; padding:9px 0 0; color:var(--muted); font-size:12px; }
    .extreme-legend span { display:inline-flex; align-items:center; gap:6px; }
    .extreme-legend i { width:10px; height:10px; display:inline-block; border-radius:50%; }
    .extreme-legend .up { background:#f04438; }
    .extreme-legend .down { background:#16b364; }
    .extreme-legend .conflict { background:#7c858f; transform:rotate(45deg); border-radius:2px; }
    .extreme-legend .actual { border:2px solid #e8793d; background:transparent; }
    .extreme-legend .hit { border:2px solid #0f8a43; background:#fff; }
    .extreme-tooltip { position:fixed; z-index:80; display:none; max-width:370px; padding:10px 12px; border:1px solid #c9d7e6; border-radius:10px; background:rgba(255,255,255,.97); box-shadow:0 10px 28px rgba(14,41,72,.18); color:#30445e; font-size:12px; line-height:1.6; pointer-events:none; }
    .extreme-section-note { color:var(--muted); font-size:12px; margin:4px 0 0; }
    .extreme-filter { padding:0 18px 15px; }
    .signal-trace-filter { padding:0 22px 15px; grid-template-columns:1fr 1fr 1.8fr; }
    .signal-trace-svg { min-height:320px; }
    .signal-trace-line { fill:none; stroke:#5f7895; stroke-width:2.4; stroke-linecap:round; stroke-linejoin:round; }
    .signal-trace-dot { stroke:#fff; stroke-width:1.4; cursor:crosshair; }
    .signal-trace-dot:hover { stroke:var(--navy); stroke-width:2.5; }
    .signal-trace-marker-line { stroke:#e8793d; stroke-width:1.5; stroke-dasharray:5 5; }
    .signal-trace-marker { fill:#e8793d; stroke:#fff; stroke-width:1.8; cursor:crosshair; }
    .signal-trace-readout { margin-top:8px; padding:9px 11px; border:1px solid #e2eaf3; border-radius:10px; background:#f7f9fc; color:#40536b; font-size:12px; min-height:38px; }
    .signal-trace-legend { padding:0 22px 18px; }
    .signal-trace-note-line { margin:0 22px 18px; }
"""


EXTREME_MARKUP = r"""
      <section class="panel hide" id="extremePanel">
        <div class="panel-head">
          <div><h3 id="extremePanelTitle">大涨大跌</h3><p id="extremePanelNote" class="extreme-section-note"></p></div>
          <div class="panel-note" id="extremePanelTag"></div>
        </div>
        <div id="extremeCombinedView" class="extreme-view hide">
          <div class="panel-head"><div><h3>1 · 全览或左右拖动查看指数价格曲线</h3><p class="extreme-section-note">统计与曲线默认从 2018-01-01 执行日起展示；模型在前一形成日收盘后计算，主日期 t 是下一实际交易日的执行日，实际 O2O 为“t 日开盘 → 下一交易日开盘”。红色向上三角是预测大涨，绿色向下三角是预测大跌，灰色菱形是同日冲突；曲线可横向拖动，悬停读取得分、阈值和实际 O2O。</p></div><div class="panel-note">两个模型独立评分，不互为补集。</div></div>
          <div class="chart-body"><div class="chart-canvas" id="extremeCombinedChart"></div><aside class="detail-pane" id="extremeCombinedDetail">将鼠标移到曲线上查看该日信息。</aside></div>
          <div class="legend extreme-legend"><span><i class="up"></i>预测大涨</span><span><i class="down"></i>预测大跌</span><span><i class="conflict"></i>同日冲突</span></div>
          <div class="comparison-area"><h4>2 · 灰色菱形冲突日</h4><div class="table-wrap"><table class="comparison-table" id="extremeConflictTable"></table></div></div>
          <div class="panel-head"><div><h3>3 · 选择一侧与时间段核对预测</h3><p class="extreme-section-note">实际阈值事件和模型预测同时显示：TP 为预测对，FP 为预测错，FN 为漏报。</p></div></div>
          <div class="filter-controls extreme-filter"><div class="field"><label for="extremePeriodSide">预测方向</label><select id="extremePeriodSide"><option value="up">大涨</option><option value="down">大跌</option></select></div><div class="field"><label for="extremePeriodStart">开始日期</label><input id="extremePeriodStart" type="date"></div><div class="field"><label for="extremePeriodEnd">结束日期</label><input id="extremePeriodEnd" type="date"></div></div>
          <div class="chart-body"><div class="chart-canvas" id="extremePeriodChart"></div><aside class="detail-pane" id="extremePeriodDetail">选择方向和时间段后查看。</aside></div>
          <div class="legend extreme-legend"><span><i class="up"></i>模型预测大涨</span><span><i class="down"></i>模型预测大跌</span><span><i class="actual"></i>实际达到收益率阈值</span><span><i class="hit"></i>预测正确</span></div>
          <div class="comparison-area"><h4>时间段内实际阈值点与预测点</h4><div class="table-wrap"><table class="comparison-table" id="extremePeriodTable"></table></div></div>
        </div>
        <div id="extremeSideView" class="extreme-view hide">
          <div class="panel-head"><div><h3 id="extremeSideChartTitle">独立方向预测</h3><p class="extreme-section-note">统计与曲线默认从 2018-01-01 执行日起展示；模型在前一形成日收盘后计算，圆环是实际达到收益率阈值的点，三角是模型预测点；O2O 统一为执行日开盘到下一交易日开盘，曲线可左右拖动。</p></div><div class="panel-note" id="extremeSideNote"></div></div>
          <div class="chart-body"><div class="chart-canvas" id="extremeSideChart"></div><aside class="detail-pane" id="extremeSideDetail">将鼠标移到曲线上查看该日信息。</aside></div>
          <div class="legend extreme-legend"><span id="extremeSideLegend"></span><span><i class="actual"></i>实际达到收益率阈值</span></div>
          <div class="comparison-area"><h4>预测与实际阈值事件明细</h4><div class="table-wrap"><table class="comparison-table" id="extremeSideTable"></table></div></div>
        </div>
      </section>
"""


SIGNAL_TRACE_MARKUP = r"""
      <section class="panel hide" id="signalTracePanel">
        <div class="panel-head">
          <div><h3>原始三状态段回看</h3><p>模型在前一形成日收盘后计算，并在下一实际交易日开盘执行；页面只显示执行日。先用时间段筛选反转执行日，再选择一个执行点；下方只展示该执行点所在的原始三状态段，不施加退出后的状态修改。曲线画收盘价；H1 O2O 明确按“执行日开盘 → 下一交易日开盘”计算：(O_{t+1}/O_t-1)。</p></div>
          <div class="panel-note" id="signalTraceNote"></div>
        </div>
        <div class="filter-controls signal-trace-filter">
          <div class="field"><label for="signalTraceStart">筛选开始日期</label><input id="signalTraceStart" type="date"></div>
          <div class="field"><label for="signalTraceEnd">筛选结束日期</label><input id="signalTraceEnd" type="date"></div>
          <div class="field"><label for="signalTraceSelect">选择反转信号点</label><select id="signalTraceSelect"></select></div>
        </div>
        <div class="chart-body">
          <div class="chart-canvas"><svg id="signalTraceChart" class="signal-trace-svg" viewBox="0 0 1000 380" role="img" aria-label="原始三状态段收盘价与反转执行点"></svg><div id="signalTraceReadout" class="signal-trace-readout">选择执行点后，鼠标移到曲线上查看执行日收盘、下一交易日收盘，以及 H1 的开盘到开盘口径。</div></div>
          <aside class="detail-pane" id="signalTraceDetail">选择一个反转执行点后，这里会显示执行日开盘、执行日收盘、下一交易日开盘、下一交易日收盘，以及明确的 H1 O2O 日期和公式。</aside>
        </div>
        <div class="legend signal-trace-legend"><span class="legend-item"><span class="legend-swatch state-fill--1"></span>原始 -1</span><span class="legend-item"><span class="legend-swatch state-fill-0"></span>原始 0</span><span class="legend-item"><span class="legend-swatch state-fill-1"></span>原始 +1</span><span class="legend-item"><span class="legend-swatch" style="background:#e8793d"></span>选中的反转执行日</span></div>
        <p class="extreme-section-note signal-trace-note-line">说明：这里统一展示原始 H1 O2O 和四个价格；“退出改善方向化 H1”只是按状态方向对原始 H1 做符号变换，不是新的独立收益口径，因此不单列。</p>
      </section>
"""


SIGNAL_META = r"""    const signalMeta = {
      minus_exit: { label: '负向退出（-1→0）', short: '负向退出', color: '#e8793d', target: 0, from: -1, flag: 'minusExit' },
      plus_exit: { label: '正向退出（+1→0）', short: '正向退出', color: '#8b5cf6', target: 0, from: 1, flag: 'plusExit' },
    };"""


PAGES = r"""    const pages = [
      { key: 'base', kicker: '基础口径 / 三状态', title: '基础三状态', series: 'base', signals: [], tag: '原始执行日标签', description: '先用最基础的 -1 / 0 / +1 标签读完整条指数路径：红色为 +1、绿色为 -1、蓝色为 0。', chartNote: '基础口径不叠加反转信号；重点观察原始段结构、段收益和路径形状。' },
      { key: 'combined', kicker: '加入两个非零退出 / 执行日更新', title: '加入两个非零退出', series: 'combined', signals: Object.keys(signalMeta), tag: '两类非零退出合并口径', description: '只把负向退出（-1→0）和正向退出（+1→0）放在同一条执行日时间轴上；不包含其它转移信号。', chartNote: '模型在前一形成日收盘后计算；退出覆盖只在执行日生效，下一执行日恢复读取基础状态。' },
      { key: 'minus_exit', kicker: '单信号 / 退出', title: '负向退出（-1→0）', series: 'minus_exit', signals: ['minus_exit'], tag: '单信号更新', description: '只把基础 -1 日上触发的负向退出信号改为 0，用于观察退出是否避免后续不利方向波动。', chartNote: '单信号状态由基础三状态和负向退出生效日重建。' },
      { key: 'plus_exit', kicker: '单信号 / 退出', title: '正向退出（+1→0）', series: 'plus_exit', signals: ['plus_exit'], tag: '单信号更新', description: '只把基础 +1 日上触发的正向退出信号改为 0，用于观察退出是否减少正向持仓尾部的反向回撤。', chartNote: '单信号状态由基础三状态和正向退出生效日重建。' },
      { key: 'extreme_combined', kicker: '大涨大跌 / 合并观察', title: '大涨大跌合并', extreme: 'combined', tag: '两个独立二分类器', description: '两个方向模型分别预测收益率尾部事件；同日都预测时保留灰色菱形。' },
      { key: 'extreme_up', kicker: '大涨 / 独立分析', title: '大涨独立分析', extreme: 'up', tag: 'V189', description: '独立查看大涨预测、实际 q90 阈值事件以及预测正确与错误。' },
      { key: 'extreme_down', kicker: '大跌 / 独立分析', title: '大跌独立分析', extreme: 'down', tag: 'V156', description: '独立查看大跌预测、实际 q10 阈值事件以及预测正确与错误。' },
    ];"""


STATE_FOR = r"""    function stateFor(row, series) {
      if (series === 'base') return row.state;
      if (series === 'combined') return row.combined;
      if (series === 'minus_exit') return row.minusExit === 1 && row.state === -1 ? 0 : row.state;
      if (series === 'plus_exit') return row.plusExit === 1 && row.state === 1 ? 0 : row.state;
      return row.state;
    }"""


METRIC_RENDER = r"""    function renderMetrics(view) {
      const summary = view.summary;
      const active = [-1, 1].map((state) => summary.byState[state]);
      const activeReturn = active.map((item) => item.meanReturnBp).filter(finite);
      const stateWinRate = (state) => {
        const item = summary.byState[state];
        return item && finite(item.winRate) ? `${item.winRate.toFixed(1)}%` : '—';
      };
      const stateWinRateNote = (state) => {
        const item = summary.byState[state];
        return item && finite(item.winRate) ? `${item.segments} 个 ${state === -1 ? '-1' : '+1'} 段，按段等权` : '该状态暂无可计算段';
      };
      const cards = [
        ['执行日总数', fmtInt(view.rows.length), `${view.rows[0].date} 至 ${view.rows[view.rows.length - 1].date}`],
        ['当前状态段', fmtInt(summary.totalSegments), `中位时长 ${fmt(summary.medianDuration, 1)} 天`],
        [view.events.length ? '信号执行日' : '非中性执行日', view.events.length ? fmtInt(view.events.length) : fmtInt(summary.activeDays), view.events.length ? '按当前页信号口径' : '状态为 -1 或 +1 的日数'],
        ['段级平均时长', `${fmt(summary.medianDuration, 1)} 天`, '全状态段中位数'],
        ['段级平均方向收益', activeReturn.length ? fmtBp(mean(activeReturn)) : '—', '只计 -1 / +1 且 O2O 完整的段'],
        ['-1 段级胜率', stateWinRate(-1), stateWinRateNote(-1)],
        ['+1 段级胜率', stateWinRate(1), stateWinRateNote(1)],
      ];
      $('metrics').innerHTML = cards.map(([label, value, note]) => `<div class="metric"><div class="metric-label">${label}</div><div class="metric-value">${value}</div><div class="metric-note">${note}</div></div>`).join('');
    }"""


EXTREME_SCRIPT = r"""
    const extremeChartState = {};
    const xHtml = (value) => String(value ?? '').replace(/[&<>\"']/g, (ch) => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '\"':'&quot;', "'":'&#39;' }[ch]));
    const xDateMs = (value) => new Date(`${String(value).slice(0, 10)}T00:00:00`).getTime();
    const xBp = (value) => finite(Number(value)) ? `${Number(value).toFixed(2)} bp` : '—';
    const xPct = (value) => finite(Number(value)) ? `${Number(value).toFixed(2)}%` : '—';
    function showExtremeTooltip(event, content) {
      const tip = $('extremeTooltip');
      tip.innerHTML = content;
      tip.style.display = 'block';
      let left = event.clientX + 14;
      let top = event.clientY + 14;
      if (left + 390 > window.innerWidth) left = event.clientX - 400;
      if (top + 230 > window.innerHeight) top = event.clientY - 240;
      tip.style.left = `${Math.max(8, left)}px`;
      tip.style.top = `${Math.max(8, top)}px`;
    }
    function hideExtremeTooltip() { $('extremeTooltip').style.display = 'none'; }
    function makeExtremeChart(containerId, rows, config = {}) {
      const box = $(containerId);
      if (!box || !rows.length) return;
      const key = config.key || containerId;
      if (!extremeChartState[key]) extremeChartState[key] = { start: 0, end: Math.max(1, rows.length - 1) };
      const state = extremeChartState[key];
      state.start = Math.max(0, Math.min(state.start, Math.max(0, rows.length - 2)));
      state.end = Math.max(state.start + 1, Math.min(state.end, rows.length - 1));
      const shown = rows.slice(state.start, state.end + 1);
      const width = Math.max(1120, Math.min(6400, Math.round(72 + shown.length * 2.25)));
      const height = 360;
      const margin = { left: 62, right: 22, top: 18, bottom: 34 };
      const values = shown.map((row) => Number(row.close)).filter(finite);
      const low = Math.min(...values);
      const high = Math.max(...values);
      const pad = (high - low || 1) * .08;
      const yLow = low - pad;
      const yHigh = high + pad;
      const x = (index) => margin.left + index / Math.max(1, shown.length - 1) * (width - margin.left - margin.right);
      const y = (value) => margin.top + (1 - (Number(value) - yLow) / (yHigh - yLow)) * (height - margin.top - margin.bottom);
      const markers = config.markers || [];
      const toolbar = `<div class="extreme-toolbar"><span>当前区间：${xHtml(shown[0].date)} → ${xHtml(shown[shown.length - 1].date)} · 可左右拖动</span><span><button type="button" data-extreme-action="all">全览（2018+）</button> <button type="button" data-extreme-action="recent">最近3年</button> <button type="button" data-extreme-action="reset">回到2018</button></span></div>`;
      box.innerHTML = `${toolbar}<div class="extreme-svg-wrap"><svg viewBox="0 0 ${width} ${height}" style="width:${width}px;min-width:${width}px" role="img" aria-label="大涨大跌价格曲线"></svg></div><div class="extreme-readout" id="extreme-readout-${key}">将鼠标移到曲线上查看详细信息。</div>`;
      const svg = box.querySelector('svg');
      for (let tick = 0; tick <= 4; tick += 1) {
        const yy = margin.top + tick / 4 * (height - margin.top - margin.bottom);
        const value = yHigh - tick / 4 * (yHigh - yLow);
        el('line', { x1: margin.left, x2: width - margin.right, y1: yy, y2: yy, class: 'extreme-grid' }, svg);
        txt(svg, margin.left - 8, yy + 4, fmt(value, 0), 'extreme-axis-text', 'end');
      }
      el('line', { x1: margin.left, x2: width - margin.right, y1: height - margin.bottom, y2: height - margin.bottom, class: 'extreme-axis' }, svg);
      let path = '';
      shown.forEach((row, index) => { const value = Number(row.close); if (!finite(value)) return; path += `${path ? ' L' : 'M'} ${x(index).toFixed(2)},${y(value).toFixed(2)}`; });
      el('path', { d: path, class: 'extreme-line' }, svg);
      const markerGroup = {};
      markers.forEach((marker) => { (markerGroup[marker.date] ||= []).push(marker.shape); });
      Object.entries(markerGroup).forEach(([date, shapes]) => {
        const globalIndex = rows.findIndex((row) => row.date === date);
        if (globalIndex < state.start || globalIndex > state.end || globalIndex < 0) return;
        const localIndex = globalIndex - state.start;
        const px = x(localIndex);
        const py = y(rows[globalIndex].close);
        shapes.forEach((shape, offset) => {
          const dx = shapes.length > 1 ? (offset - (shapes.length - 1) / 2) * 12 : 0;
          if (shape === 'up') el('path', { d: `M ${px + dx} ${py - 10} L ${px + dx - 8} ${py + 6} L ${px + dx + 8} ${py + 6} Z`, class: 'extreme-marker-up' }, svg);
          if (shape === 'down') el('path', { d: `M ${px + dx} ${py + 10} L ${px + dx - 8} ${py - 6} L ${px + dx + 8} ${py - 6} Z`, class: 'extreme-marker-down' }, svg);
          if (shape === 'conflict') el('path', { d: `M ${px} ${py - 10} L ${px + 10} ${py} L ${px} ${py + 10} L ${px - 10} ${py} Z`, class: 'extreme-marker-conflict' }, svg);
          if (shape === 'actual') el('circle', { cx: px + dx, cy: py, r: 7, class: 'extreme-marker-actual' }, svg);
          if (shape === 'hit') el('circle', { cx: px + dx, cy: py, r: 7, class: 'extreme-marker-hit' }, svg);
        });
      });
      txt(svg, margin.left, height - 10, String(shown[0].date).slice(0, 10), 'extreme-axis-text', 'start');
      txt(svg, width - margin.right, height - 10, String(shown[shown.length - 1].date).slice(0, 10), 'extreme-axis-text', 'end');
      const overlay = el('rect', { x: margin.left, y: margin.top, width: width - margin.left - margin.right, height: height - margin.top - margin.bottom, fill: 'transparent', 'pointer-events': 'all' }, svg);
      const nearest = (event) => {
        const rect = svg.getBoundingClientRect();
        const px = (event.clientX - rect.left) / rect.width * width;
        const ratio = (px - margin.left) / (width - margin.left - margin.right);
        const localIndex = Math.max(0, Math.min(shown.length - 1, Math.round(ratio * (shown.length - 1))));
        return rows[state.start + localIndex];
      };
      const scrollWrap = box.querySelector('.extreme-svg-wrap');
      let drag = null;
      overlay.addEventListener('pointermove', (event) => {
        if (drag) {
          scrollWrap.scrollLeft = drag.scrollLeft - (event.clientX - drag.x);
          event.preventDefault();
          return;
        }
        const row = nearest(event);
        if (!row) return;
        showExtremeTooltip(event, config.tooltip ? config.tooltip(row) : `<b>${xHtml(row.date)}</b><br>收盘 ${fmt(row.close, 2)}`);
        const readout = $(`extreme-readout-${key}`);
        if (readout) readout.innerHTML = config.readout ? config.readout(row) : `${xHtml(row.date)} · 收盘 ${fmt(row.close, 2)}`;
      });
      overlay.addEventListener('pointerleave', hideExtremeTooltip);
      overlay.addEventListener('pointerdown', (event) => {
        if (event.button !== 0) return;
        drag = { x: event.clientX, scrollLeft: scrollWrap.scrollLeft };
        scrollWrap.classList.add('dragging');
        overlay.setPointerCapture(event.pointerId);
      });
      const endDrag = (event) => {
        if (!drag) return;
        drag = null;
        scrollWrap.classList.remove('dragging');
        if (overlay.hasPointerCapture?.(event.pointerId)) overlay.releasePointerCapture(event.pointerId);
      };
      overlay.addEventListener('pointerup', endDrag);
      overlay.addEventListener('pointercancel', endDrag);
      box.querySelectorAll('[data-extreme-action]').forEach((button) => button.addEventListener('click', () => {
        if (button.dataset.extremeAction === 'recent') { const index = rows.findIndex((row) => xDateMs(row.date) >= xDateMs('2023-01-01')); state.start = index < 0 ? Math.max(0, rows.length - 700) : index; state.end = rows.length - 1; }
        else { state.start = 0; state.end = rows.length - 1; }
        makeExtremeChart(containerId, rows, config);
      }));
    }
    function extremeTable(headers, rows) {
      if (!rows.length) return '<div class="empty">当前筛选没有记录。</div>';
      return `<thead><tr>${headers.map((header) => `<th>${xHtml(header.label)}</th>`).join('')}</tr></thead><tbody>${rows.map((row) => `<tr>${headers.map((header) => `<td>${header.render ? header.render(row) : xHtml(row[header.key])}</td>`).join('')}</tr>`).join('')}</tbody>`;
    }
    function extremeMetric(label, value, note) { return `<div class="metric"><div class="metric-label">${xHtml(label)}</div><div class="metric-value">${xHtml(value)}</div><div class="metric-note">${xHtml(note || '')}</div></div>`; }
    function extremePhase(value) { return `<span class="state-pill">${xHtml(value)}</span>`; }
    function extremeCombinedTooltip(row) {
      const up = DATA.extreme.models.up;
      const down = DATA.extreme.models.down;
      const prediction = Number(row.up_predicted) && Number(row.down_predicted) ? '同日冲突' : Number(row.up_predicted) ? '预测大涨' : Number(row.down_predicted) ? '预测大跌' : '无预测';
      return `<b>执行日 ${xHtml(row.date)}</b><br>收盘 ${fmt(row.close, 2)}<br>O2O 日期 ${xHtml(row.date)} 开盘 → ${xHtml(row.target_date)} 开盘<br>大涨得分 / 阈值 ${fmt(row.up_score, 4)} / ${fmt(up.score_threshold, 4)}<br>大跌得分 / 阈值 ${fmt(row.down_score, 4)} / ${fmt(down.score_threshold, 4)}<br>预测 ${prediction}<br>实际 O2O ${xBp(row.target_bp)}`;
    }
    function extremeSideTooltip(side, row) {
      const model = DATA.extreme.models[side];
      const indicators = model.indicator_names.map((name, index) => `${xHtml(name)} ${fmt(row[`indicator_${index + 1}`], 3)}`).join(' · ');
      return `<b>执行日 ${xHtml(row.date)}</b><br>方向 ${side === 'up' ? '大涨' : '大跌'}<br>O2O 日期 ${xHtml(row.date)} 开盘 → ${xHtml(row.target_date)} 开盘<br>得分 / 阈值 ${fmt(row.score, 4)} / ${fmt(row.score_threshold, 4)}<br>超出阈值 ${fmt(row.score_excess, 4)}<br>实际 O2O ${xBp(row.target_bp)}<br>实际阈值事件 ${Number(row.actual_extreme) ? '是' : '否'}<br>${indicators}`;
    }
    function extremeSideReadout(side, row) { return `执行日 ${xHtml(row.date)} · O2O ${xHtml(row.date)} 开盘→${xHtml(row.target_date)} 开盘 · ${side === 'up' ? '大涨' : '大跌'}得分 ${fmt(row.score, 4)} · 阈值 ${fmt(row.score_threshold, 4)} · 实际 O2O ${xBp(row.target_bp)} · ${Number(row.predicted) ? (Number(row.correct) ? '预测正确' : '预测错误') : '未预测'}`; }
    function renderExtremeMetrics(kind) {
      const thresholds = DATA.extreme.thresholds;
      if (kind === 'combined') {
        const up = DATA.extreme.models.up.overall_metrics;
        const down = DATA.extreme.models.down.overall_metrics;
        $('metrics').innerHTML = [
          extremeMetric('统计日期', `${DATA.extreme.display_start_date} 起`, '主日期为执行日'),
          extremeMetric('现货最后日期', DATA.extreme.latest_spot_date || '—', '没有更晚开盘价时不填收益'),
          extremeMetric('实际大跌阈值', xBp(thresholds.q10_bp), 'Development q10'),
          extremeMetric('实际大涨阈值', xBp(thresholds.q90_bp), 'Development q90'),
          extremeMetric('大涨预测 / TP', `${up.n_signal} / ${up.true_positive}`, `实际事件 ${up.n_actual_extreme} 天`),
          extremeMetric('大涨精确率', xPct(up.precision_pct), `召回率 ${xPct(up.recall_pct)}`),
          extremeMetric('大涨 F1 / Lift', `${xPct(up.f1_pct)} / ${fmt(up.lift, 2)}×`, `基础事件率 ${xPct(up.base_rate_pct)} · 整体准确率 ${xPct(up.accuracy_pct)}`),
          extremeMetric('大跌预测 / TP', `${down.n_signal} / ${down.true_positive}`, `实际事件 ${down.n_actual_extreme} 天`),
          extremeMetric('大跌精确率', xPct(down.precision_pct), `召回率 ${xPct(down.recall_pct)}`),
          extremeMetric('大跌 F1 / Lift', `${xPct(down.f1_pct)} / ${fmt(down.lift, 2)}×`, `基础事件率 ${xPct(down.base_rate_pct)} · 整体准确率 ${xPct(down.accuracy_pct)}`),
          extremeMetric('灰色冲突日', `${DATA.extreme.conflict_rows.length} 天`, '2018+ 执行日')
        ].join('');
        return;
      }
      const model = DATA.extreme.models[kind];
      const overall = model.overall_metrics;
      const test = model.phase_metrics.find((row) => row.phase === 'Test');
      $('metrics').innerHTML = [
        extremeMetric('冻结版本', model.version, model.candidate_id),
        extremeMetric('现货最后日期', DATA.extreme.latest_spot_date || '—', '没有更晚开盘价时不填收益'),
        extremeMetric('实际收益率阈值', xBp(model.threshold_target_bp), 'Development 拟合'),
        extremeMetric('模型得分阈值', fmt(model.score_threshold, 4), '超过才发出预测'),
        extremeMetric('2018+ 预测数', `${overall.n_signal} 天`, `实际事件 ${overall.n_actual_extreme} 天`),
        extremeMetric('2018+ 精确率', xPct(overall.precision_pct), `TP ${overall.true_positive} · FP ${overall.false_positive}`),
        extremeMetric('2018+ 召回率', xPct(overall.recall_pct), `FN ${overall.false_negative}`),
        extremeMetric('2018+ F1 / Lift', `${xPct(overall.f1_pct)} / ${fmt(overall.lift, 2)}×`, `实际事件率 ${xPct(overall.base_rate_pct)}`),
        extremeMetric('2018+ 整体准确率', xPct(overall.accuracy_pct), '含未预测的非事件日'),
        extremeMetric('Test 预测 / TP', test ? `${test.n_signal} / ${test.true_positive}` : '—', test ? `精确率 ${xPct(test.precision_pct)} · 召回率 ${xPct(test.recall_pct)}` : '—'),
        extremeMetric('Test 方向准确率', test ? xPct(test.direction_accuracy_pct) : '—', test ? `方向化均值 ${xBp(test.signed_mean_bp)}` : '—')
      ].join('');
    }
    function renderExtremePeriod() {
      const side = $('extremePeriodSide').value;
      const model = DATA.extreme.models[side];
      const all = model.rows;
      const start = $('extremePeriodStart').value || DATA.extreme.display_start_date || all[0].date;
      const end = $('extremePeriodEnd').value || all[all.length - 1].date;
      const rows = all.filter((row) => row.date >= start && row.date <= end);
      const markers = [];
      rows.forEach((row) => { if (Number(row.actual_extreme)) markers.push({ date: row.date, shape: Number(row.predicted) && Number(row.correct) ? 'hit' : 'actual' }); if (Number(row.predicted)) markers.push({ date: row.date, shape: side === 'up' ? 'up' : 'down' }); });
      makeExtremeChart('extremePeriodChart', rows, { key: `extreme-period-${side}`, markers, tooltip: (row) => extremeSideTooltip(side, row), readout: (row) => extremeSideReadout(side, row) });
      $('extremePeriodDetail').innerHTML = `<div class="detail-kicker">时间段核对</div><div class="detail-title">${xHtml(start)} → ${xHtml(end)}</div><div class="detail-grid"><div class="detail-item"><span class="detail-label">方向</span><span class="detail-value">${side === 'up' ? '大涨' : '大跌'}</span></div><div class="detail-item"><span class="detail-label">记录数</span><span class="detail-value">${rows.length}</span></div></div>`;
      const selected = rows.filter((row) => Number(row.predicted) || Number(row.actual_extreme)).slice().reverse();
      $('extremePeriodTable').innerHTML = extremeTable([{ label: '执行日', key: 'date' }, { label: 'O2O 终点', key: 'target_date' }, { label: '阶段', key: 'phase', render: (row) => extremePhase(row.phase) }, { label: '得分 / 阈值', key: 'score', render: (row) => `${fmt(row.score, 4)} / ${fmt(row.score_threshold, 4)}` }, { label: '实际 O2O', key: 'target_bp', render: (row) => xBp(row.target_bp) }, { label: '分类', key: 'class', render: (row) => Number(row.predicted) && Number(row.actual_extreme) ? '<span class="positive">TP · 预测对</span>' : Number(row.predicted) ? '<span class="negative">FP · 预测错</span>' : '<span class="muted">FN · 漏报</span>' }], selected);
    }
    function renderExtremeCombined() {
      const rows = DATA.extreme.combined_rows;
      const markers = rows.filter((row) => row.marker !== 'none').map((row) => ({ date: row.date, shape: row.marker === 'conflict' ? 'conflict' : row.marker }));
      makeExtremeChart('extremeCombinedChart', rows, { key: 'extreme-combined', markers, tooltip: extremeCombinedTooltip, readout: (row) => `执行日 ${xHtml(row.date)} · O2O ${xHtml(row.date)} 开盘→${xHtml(row.target_date)} 开盘 · ${Number(row.conflict) ? '同日冲突' : Number(row.up_predicted) ? '预测大涨' : Number(row.down_predicted) ? '预测大跌' : '无预测'} · 实际 O2O ${xBp(row.target_bp)}` });
      $('extremeCombinedDetail').innerHTML = `<div class="detail-kicker">合并观察</div><div class="detail-title">两个方向独立评分</div><div class="detail-grid"><div class="detail-item"><span class="detail-label">灰色菱形数量</span><span class="detail-value">${DATA.extreme.conflict_rows.length} 天</span></div><div class="detail-item"><span class="detail-label">大涨模型</span><span class="detail-value">${xHtml(DATA.extreme.models.up.version)}</span></div><div class="detail-item"><span class="detail-label">大跌模型</span><span class="detail-value">${xHtml(DATA.extreme.models.down.version)}</span></div><div class="detail-item detail-wide"><span class="detail-label">目标收益</span><span class="detail-value">${xHtml(DATA.extreme.target_description)}</span></div></div>`;
      $('extremeConflictTable').innerHTML = extremeTable([{ label: '执行日', key: 'date' }, { label: 'O2O 终点', key: 'target_date' }, { label: '阶段', key: 'phase', render: (row) => extremePhase(row.phase) }, { label: '实际 O2O', key: 'target_bp', render: (row) => xBp(row.target_bp) }, { label: '大涨得分 / 超出', key: 'up_score', render: (row) => `${fmt(row.up_score, 4)} / ${fmt(row.up_score_excess, 4)}` }, { label: '大跌得分 / 超出', key: 'down_score', render: (row) => `${fmt(row.down_score, 4)} / ${fmt(row.down_score_excess, 4)}` }], DATA.extreme.conflict_rows.slice().reverse());
      const all = DATA.extreme.models.up.rows;
      $('extremePeriodStart').min = DATA.extreme.display_start_date || all[0].date;
      $('extremePeriodEnd').min = DATA.extreme.display_start_date || all[0].date;
      $('extremePeriodStart').value = $('extremePeriodStart').value || DATA.extreme.display_start_date || all[0].date;
      $('extremePeriodEnd').value = $('extremePeriodEnd').value || all[all.length - 1].date;
      $('extremePeriodSide').onchange = renderExtremePeriod;
      $('extremePeriodStart').onchange = renderExtremePeriod;
      $('extremePeriodEnd').onchange = renderExtremePeriod;
      renderExtremePeriod();
    }
    function renderExtremeSide(side) {
      const model = DATA.extreme.models[side];
      const markers = model.rows.filter((row) => Number(row.predicted)).map((row) => ({ date: row.date, shape: side === 'up' ? 'up' : 'down' })).concat(model.rows.filter((row) => Number(row.actual_extreme)).map((row) => ({ date: row.date, shape: Number(row.predicted) && Number(row.correct) ? 'hit' : 'actual' })));
      makeExtremeChart('extremeSideChart', model.rows, { key: `extreme-side-${side}`, markers, tooltip: (row) => extremeSideTooltip(side, row), readout: (row) => extremeSideReadout(side, row) });
      const test = model.phase_metrics.find((row) => row.phase === 'Test');
      $('extremeSideChartTitle').textContent = side === 'up' ? '大涨独立价格—预测总览' : '大跌独立价格—预测总览';
      $('extremeSideNote').textContent = `${model.version} · 2018+ 执行日 · 实际阈值 ${xBp(model.threshold_target_bp)} · Test 预测 ${test ? test.n_signal : '—'} 天`;
      $('extremeSideLegend').innerHTML = `<i class="${side === 'up' ? 'up' : 'down'}"></i>预测${side === 'up' ? '大涨' : '大跌'}`;
      $('extremeSideDetail').innerHTML = `<div class="detail-kicker">独立模型</div><div class="detail-title">${xHtml(model.candidate_id)}</div><div class="detail-grid"><div class="detail-item"><span class="detail-label">得分阈值</span><span class="detail-value">${fmt(model.score_threshold, 4)}</span></div><div class="detail-item"><span class="detail-label">实际收益阈值</span><span class="detail-value">${xBp(model.threshold_target_bp)}</span></div><div class="detail-item detail-wide"><span class="detail-label">判断指标</span><span class="detail-value">${model.indicator_names.map(xHtml).join(' · ')}</span></div></div>`;
      const rows = model.rows.filter((row) => Number(row.predicted) || Number(row.actual_extreme)).slice().reverse();
      $('extremeSideTable').innerHTML = extremeTable([{ label: '执行日', key: 'date' }, { label: 'O2O 终点', key: 'target_date' }, { label: '阶段', key: 'phase', render: (row) => extremePhase(row.phase) }, { label: '得分 / 阈值', key: 'score', render: (row) => `${fmt(row.score, 4)} / ${fmt(row.score_threshold, 4)}` }, { label: '超出阈值', key: 'score_excess', render: (row) => fmt(row.score_excess, 4) }, { label: '实际 O2O', key: 'target_bp', render: (row) => xBp(row.target_bp) }, { label: '结果', key: 'correct', render: (row) => Number(row.predicted) && Number(row.actual_extreme) ? '<span class="positive">TP · 预测对</span>' : Number(row.predicted) ? '<span class="negative">FP · 预测错</span>' : '<span class="muted">FN · 漏报</span>' }], rows);
    }
    function renderExtremePage(kind) {
      const page = pages.find((item) => item.extreme === kind);
      app.activePage = page;
      app.activeView = null;
      document.querySelectorAll('.container > .panel').forEach((panelNode) => panelNode.classList.toggle('hide', panelNode.id !== 'extremePanel'));
      $('extremePanel').classList.remove('hide');
      setText('pageKicker', page.kicker);
      setText('pageTitle', page.title);
      setText('pageDescription', page.description);
      setText('pageTag', page.tag);
      setText('extremePanelTitle', page.title);
      setText('extremePanelNote', page.description);
      setText('extremePanelTag', page.tag);
      renderExtremeMetrics(kind);
      $('extremeCombinedView').classList.toggle('hide', kind !== 'combined');
      $('extremeSideView').classList.toggle('hide', kind === 'combined');
      if (kind === 'combined') renderExtremeCombined();
      else renderExtremeSide(kind);
    }
"""


SIGNAL_TRACE_SCRIPT = r"""
    const signalTraceUi = {};
    function signalTraceNextRow(row) {
      const baseRows = app.views.base?.rows || [];
      const rowIndex = baseRows.findIndex((item) => item.date === row?.date);
      return rowIndex >= 0 ? baseRows[rowIndex + 1] : null;
    }
    function signalTraceDetail(row, event, segment, position) {
      if (!row) return;
      const meta = DATA.eventSummary.find((item) => item.key === event?.key);
      const next = signalTraceNextRow(row);
      const nextDate = next?.date || null;
      const h1 = finite(row.open) && finite(next?.open) ? next.open / row.open - 1 : (finite(row.h1) ? row.h1 : null);
      const closeToClose = finite(row.close) && finite(next?.close) ? next.close / row.close - 1 : null;
      const nextOpenToClose = finite(row.close) && finite(next?.open) ? next.open / row.close - 1 : null;
      const o2oDates = nextDate ? `${row.date} 开盘 → ${nextDate} 开盘` : `${row.date} 开盘 → 下一交易日开盘（缺失）`;
      $('signalTraceReadout').innerHTML = `${xHtml(row.date)} · 原始状态 ${stateLabel(row.state)} · 曲线收盘 ${fmt(row.close, 2)} · H1 O2O（${xHtml(o2oDates)}）${finite(h1) ? fmtPct(h1) : '—'}`;
      $('signalTraceDetail').innerHTML = `
        <div class="detail-kicker">原始三状态 · 鼠标悬停日</div>
        <div class="detail-title">${xHtml(row.date)} · ${stateLabel(row.state)}</div>
        <div class="detail-grid">
          ${detailItem('t 日开盘价', finite(row.open) ? fmt(row.open, 2) : '—')}
          ${detailItem('t 日收盘价（曲线）', finite(row.close) ? fmt(row.close, 2) : '—')}
          ${detailItem('t+1 日开盘价', next && finite(next.open) ? fmt(next.open, 2) : '—')}
          ${detailItem('t+1 日收盘价', next && finite(next.close) ? fmt(next.close, 2) : '—')}
          ${detailItem('H1 O2O 日期', xHtml(o2oDates), 'detail-wide')}
          ${detailItem('H1 O2O 公式', 'Open[t+1] / Open[t] − 1', 'detail-wide')}
          ${detailItem('所在原始段', segment ? `${segment.startDate} → ${segment.endDate}` : '—')}
          ${detailItem('H1 O2O 结果', finite(h1) ? fmtPct(h1) : '—', signClass(h1))}
          ${detailItem('t+1 收盘相对 t 收盘', finite(closeToClose) ? fmtPct(closeToClose) : '—', signClass(closeToClose))}
          ${detailItem('t+1 开盘相对 t 收盘', finite(nextOpenToClose) ? fmtPct(nextOpenToClose) : '—', signClass(nextOpenToClose))}
          ${detailItem('原始状态', stateLabel(row.state))}
          ${detailItem('选中反转执行日', event && event.date === row.date ? `${event.label}` : '不是反转执行日')}
          ${detailItem('段内位置', segment ? `${position + 1} / ${segment.length} 天` : '—')}
          ${event && event.date === row.date ? detailItem('模型', meta?.model || '—', 'detail-wide') : ''}
        </div>`;
    }
    function renderSignalTrace(view) {
      const panelNode = $('signalTracePanel');
      const enabled = view.page.key === 'minus_exit' || view.page.key === 'plus_exit';
      panelNode.classList.toggle('hide', !enabled);
      if (!enabled) return;
      const key = view.page.key;
      const rows = view.rows;
      const ui = signalTraceUi[key] || (signalTraceUi[key] = { start: rows[0].date, end: rows[rows.length - 1].date, eventIndex: null });
      const startInput = $('signalTraceStart');
      const endInput = $('signalTraceEnd');
      const select = $('signalTraceSelect');
      startInput.value = ui.start;
      endInput.value = ui.end;
      const events = view.events.filter((event) => event.date >= ui.start && event.date <= ui.end);
      select.innerHTML = events.length ? events.map((event) => `<option value="${event.index}">${xHtml(event.date)} · ${xHtml(event.label)} · H1 O2O ${finite(event.h1) ? fmtPct(event.h1) : '—'}</option>`).join('') : '<option value="">该时间段没有反转信号</option>';
      let selected = events.find((event) => event.index === ui.eventIndex);
      if (!selected) selected = events[0] || null;
      ui.eventIndex = selected ? selected.index : null;
      if (selected) select.value = String(selected.index);
      $('signalTraceNote').textContent = `${ui.start} → ${ui.end} · 筛选到 ${events.length} 个${view.page.title}信号点`;
      startInput.onchange = () => { ui.start = startInput.value || rows[0].date; if (ui.start > ui.end) ui.end = ui.start; ui.eventIndex = null; renderSignalTrace(view); };
      endInput.onchange = () => { ui.end = endInput.value || rows[rows.length - 1].date; if (ui.end < ui.start) ui.start = ui.end; ui.eventIndex = null; renderSignalTrace(view); };
      select.onchange = () => { ui.eventIndex = select.value === '' ? null : Number(select.value); renderSignalTrace(view); };

      const baseView = app.views.base;
      const baseSegment = selected ? baseView.segmentAt.get(selected.index) : null;
      const chartRows = baseSegment ? baseView.rows.slice(baseSegment.start, baseSegment.end + 1) : baseView.rows.filter((row) => row.date >= ui.start && row.date <= ui.end);
      const svg = $('signalTraceChart');
      clear(svg);
      if (!chartRows.length) {
        txt(svg, 500, 180, '当前时间段没有可展示的原始状态数据', 'axis-text');
        $('signalTraceDetail').innerHTML = '<div class="detail-title">当前时间段没有可展示数据。</div>';
        return;
      }
      const width = 1000;
      const height = 380;
      const margin = { left: 72, right: 20, top: 24, bottom: 48 };
      const scales = chartScales(chartRows, width, height, margin);
      drawAxes(svg, chartRows, scales, width, height, margin);
      const path = linePath(chartRows, scales);
      el('path', { d: path, class: 'signal-trace-line' }, svg);
      chartRows.forEach((row, index) => {
        if (!finite(row.close)) return;
        const dot = el('circle', { cx: scales.x(index), cy: scales.y(row.close), r: 4.2, fill: stateColor[String(row.state)], class: 'signal-trace-dot' }, svg);
        dot.addEventListener('mouseenter', () => signalTraceDetail(row, selected, baseSegment, index));
        dot.addEventListener('focus', () => signalTraceDetail(row, selected, baseSegment, index));
        el('title', {}, dot).appendChild(document.createTextNode(`${row.date} · ${stateLabel(row.state)} · 收盘 ${fmt(row.close, 2)}`));
      });
      if (selected && baseSegment && selected.index >= baseSegment.start && selected.index <= baseSegment.end) {
        const localIndex = selected.index - baseSegment.start;
        const px = scales.x(localIndex);
        const py = scales.y(chartRows[localIndex].close);
        el('line', { x1: px, x2: px, y1: margin.top, y2: height - margin.bottom, class: 'signal-trace-marker-line' }, svg);
        const marker = el('path', { d: `M ${px} ${py - 10} L ${px + 9} ${py} L ${px} ${py + 10} L ${px - 9} ${py} Z`, class: 'signal-trace-marker' }, svg);
        marker.addEventListener('mouseenter', () => signalTraceDetail(chartRows[localIndex], selected, baseSegment, localIndex));
        marker.addEventListener('focus', () => signalTraceDetail(chartRows[localIndex], selected, baseSegment, localIndex));
        el('title', {}, marker).appendChild(document.createTextNode(`${selected.date} · ${selected.label}`));
        signalTraceDetail(chartRows[localIndex], selected, baseSegment, localIndex);
      } else {
        $('signalTraceReadout').textContent = '请选择一个反转执行点；图中将用橙色菱形标记执行日。';
        $('signalTraceDetail').innerHTML = '<div class="detail-kicker">原始三状态段回看</div><div class="detail-title">当前筛选没有可选信号点</div><div class="detail-item"><span class="detail-label">提示</span><span class="detail-value">请扩大时间段，或切换到有事件的日期范围。</span></div>';
      }
    }
"""


def build_html(payload: dict[str, Any], reference_path: Path) -> str:
    template = reference_path.read_text(encoding="utf-8")
    html = re.sub(
        r"window\.__REPORT_DATA__\s*=\s*.*?;\s*</script>",
        "window.__REPORT_DATA__ = __PAYLOAD__;</script>",
        template,
        count=1,
        flags=re.S,
    )
    html = html.replace("<title>三状态与反转信号 · 离线互动汇报</title>", "<title>基础三状态、非零退出与大涨大跌 · 离线互动汇报</title>")
    html = html.replace(
        "把五份汇报、九列生效日数据和 O2O 指数收益放到同一张可探索的时间轴上：先看全周期，再切到任意状态、年份、月份和持仓段。",
        "把基础三状态与两类非零退出放到同一张可探索的时间轴上：先看全周期，再切到任意状态、年份、月份和持仓段。",
    )
    html = html.replace("<strong>规则</strong> <strong>只在生效日更新</strong>", "<strong>规则</strong> <strong>前一形成日收盘后计算，执行日单独生效</strong>")
    html = html.replace("</style>", EXTRA_CSS + "\n  </style>", 1)
    html = html.replace('<section class="metrics" id="metrics"></section>', '<section class="metrics" id="metrics"></section>\n' + EXTREME_MARKUP, 1)
    html = html.replace('<section class="panel filter-panel">', SIGNAL_TRACE_MARKUP + '\n      <section class="panel filter-panel">', 1)
    html = html.replace('</footer>', '<div id="dataStatus" class="small"></div></footer>', 1)
    html = re.sub(r'\n\s*<section class="panel hide" id="analysisPanel">.*?</section>', '', html, count=1, flags=re.S)
    html = html.replace('</body>', '<div id="extremeTooltip" class="extreme-tooltip" role="tooltip"></div>\n</body>', 1)
    html = re.sub(r"    const signalMeta = \{.*?\n    \};", SIGNAL_META, html, count=1, flags=re.S)
    html = re.sub(r"    const pages = \[.*?\n    \];\n\n    const app = \{", PAGES + "\n\n    const app = {", html, count=1, flags=re.S)
    html = re.sub(r"    function stateFor\(row, series\) \{.*?\n    \}", STATE_FOR, html, count=1, flags=re.S)
    html = re.sub(r"    function renderMetrics\(view\) \{.*?\n    \}\n\n    function chartScales", METRIC_RENDER + "\n\n    function chartScales", html, count=1, flags=re.S)
    html = html.replace("    function selectPage(key) {", EXTREME_SCRIPT + "\n" + SIGNAL_TRACE_SCRIPT + "\n    function selectPage(key) {", 1)
    select_page = r"""    function selectPage(key) {
      app.pageKey = key;
      const page = pages.find((item) => item.key === key) || pages[0];
      document.querySelectorAll('.tab').forEach((tab) => { const active = tab.dataset.page === key; tab.classList.toggle('active', active); tab.setAttribute('aria-selected', String(active)); });
      if (page.extreme) { renderExtremePage(page.extreme); return; }
      document.querySelectorAll('.container > .panel').forEach((panelNode) => panelNode.classList.remove('hide'));
      $('extremePanel').classList.add('hide');
      if (!app.views.base) app.views.base = buildView(pages[0]);
      if (!app.views[key]) app.views[key] = buildView(page);
      const view = app.views[key];
      app.activePage = page;
      app.activeView = view;
      renderPageHeader(page, view);
      renderMetrics(view);
      renderMainChart(view);
      renderSignalChart(view);
      renderSignalTrace(view);
      initFilters(view);
      renderLocal(view);
      renderDistributions(view);
      renderEvidence(view);
    }

    function init()"""
    html = re.sub(r"    function selectPage\(key\) \{.*?\n    \}\n\n    function init\(\)", select_page, html, count=1, flags=re.S)
    html = re.sub(r'\n    function renderAnalysis\(view\) \{.*?\n    function drawBars\(svg, values, options = \{', '\n    function drawBars(svg, values, options = {', html, count=1, flags=re.S)
    html = html.replace("pages.forEach((page) => { app.views[page.key] = buildView(page); });", "pages.filter((page) => !page.extreme).forEach((page) => { app.views[page.key] = buildView(page); });", 1)
    html = html.replace("$('sourceList').innerHTML = DATA.sourceFiles.map((file) => `<div>${file}</div>`).join('');", "$('sourceList').innerHTML = DATA.sourceFiles.map((file) => `<div>${file}</div>`).join('');\n      if ($('dataStatus') && DATA.meta?.data_status_note) $('dataStatus').textContent = `数据更新状态：${DATA.meta.data_status_note}`;")
    html = html.replace("四类反转信号发生日", "两类非零退出信号发生日")
    html = html.replace("四信号合并", "两类非零退出合并")
    html = html.replace("四信号", "两类非零退出")
    html = html.replace("四个反转信号", "两类非零退出")
    html = html.replace("最后展示日 2026-08-17 缺少完整未来 O2O 观测时", "末端日期缺少完整未来 O2O 观测时")
    html = html.replace("生效日", "执行日")
    html = html.replace("次日 O2O", "执行日 O2O")
    html = html.replace("收益口径 <strong>执行日 O2O</strong>", "收益口径 <strong>执行日 t 开盘 → 下一实际交易日开盘</strong>")
    html = html.replace("把基础三状态与两类非零退出放到同一条可探索的时间轴上", "把基础三状态与两类非零退出放到同一条可探索的执行日时间轴上")
    embedded = dict(payload["legacy_layout"])
    embedded["extreme"] = payload["extreme"]
    embedded["meta"] = payload.get("meta", {})
    return html.replace("__PAYLOAD__", json.dumps(embedded, ensure_ascii=False, separators=(",", ":")))

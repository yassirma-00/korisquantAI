/* ============================================================
   Page: Risk & Alerts
   ============================================================ */

/* Describe a window by its length instead of by its endpoints.
 *
 * Absolute dates live on the chart's x-axis and nowhere else. Printing them a
 * second time in the surrounding text duplicated the axis when the windows
 * matched it, and misled when they did not: the analytical windows are floored
 * at each model's own minimum, so a 1M selection charts 23 bars while crash
 * risk reads 61 and the bubble 201. A duration answers "how much history is
 * behind this number" without competing with the axis. */
function span(bars) {
  if (bars === null || bars === undefined) return 'unknown length';
  const months = bars / 21;      // ~21 trading days per month
  if (bars < 21) return `${bars} sessions`;
  if (months < 12) return `~${Math.round(months)} months`;
  const years = bars / 252;
  return years < 2 ? `~${years.toFixed(1)} years` : `~${Math.round(years)} years`;
}

/* A score card that can honestly say "unknown".
 *
 * The backend returns null for a score it cannot compute on a short window.
 * Rendering that as 0.00 on a green bar was the dangerous case: it stated the
 * opposite of the truth. Absence of evidence is shown as grey, never as safe. */
function scoreCard(label, block, warnAbove, cautionAbove) {
  const score = block ? block.score : null;
  if (score === null || score === undefined) {
    return `
      <div class="card">
        <div class="stat-label">${label}</div>
        <div class="stat-value text-muted">—</div>
        <div class="meter"><div class="meter-fill" style="width:0"></div></div>
        <div class="text-xs text-muted mt-1">${
  block && block.reason ? block.reason : 'Not enough history in this period'}</div>
      </div>`;
  }
  const colour = score > warnAbove ? 'var(--red)'
    : score > cautionAbove ? 'var(--amber)' : 'var(--green)';

  // A bare "0.41" says nothing about the range it sits in. Percent of a stated
  // 0-100% scale, with the band thresholds spelled out, is self-explanatory.
  const bands = block.scale && block.scale.bands
    ? Object.entries(block.scale.bands).map(([name, range]) =>
      `<span class="${name === block.level ? 'score-band-on' : ''}">${name} ${range}</span>`).join(' · ')
    : '';

  const rows = (block.components || []).map((c) => `
    <div class="score-part">
      <div class="score-part-head">
        <span>${c.name}</span>
        <span class="mono text-muted">${(c.weight * c.value * 100).toFixed(1)} pts
          / ${(c.weight * 100).toFixed(0)}</span>
      </div>
      <div class="score-part-bar"><i style="width:${c.value * 100}%"></i></div>
      <div class="score-part-detail">${c.detail}</div>
    </div>`).join('');

  return `
    <div class="card">
      <div class="stat-label">${label}</div>
      <div class="stat-value ${score > warnAbove ? 'down' : ''}">${(score * 100).toFixed(0)}%</div>
      <div class="meter"><div class="meter-fill"
        style="width:${score * 100}%;background:${colour}"></div></div>
      <div class="text-xs text-muted mt-1">${block.level || '—'} · scale 0–100%</div>
      ${bands ? `<div class="score-bands">${bands}</div>` : ''}
      ${rows ? `<details class="score-detail">
        <summary>How this is calculated</summary>
        <div class="score-parts">${rows}</div>
        <div class="score-part-detail mt-1">Weighted sum of the terms above;
          each is capped so no single input can dominate.</div>
      </details>` : ''}
    </div>`;
}

/* The Overall Risk Score, with every contribution shown.
 *
 * The headline used to be `max(crash_band, bubble_band, anomaly_band)` — an
 * ordinal maximum over three labels, so it could not say how far into a band
 * it sat, ignored absolute volatility entirely, and discarded every input
 * except the loudest one. It is now a weighted mean of eight bounded
 * contributors, and this renders the arithmetic that produced it: each row
 * shows the measured value, the absolute scale it was judged against, and the
 * points it contributed out of its maximum. The rows add up to the score. */
function overallRiskCard(profile) {
  const o = profile && profile.overall;
  if (!o || o.score === null || o.score === undefined) {
    return `
      <div class="card">
        <div class="stat-label">Overall Risk</div>
        <div class="stat-value text-muted">—</div>
        <div class="meter"><div class="meter-fill" style="width:0"></div></div>
        <div class="text-xs text-muted mt-1">${
  (profile && profile.reason) || 'Not enough history in this period'}</div>
      </div>`;
  }
  const pct = o.score * 100;
  const colour = o.level === 'critical' || o.level === 'high' ? 'var(--red)'
    : o.level === 'moderate' ? 'var(--amber)' : 'var(--green)';

  const bands = o.scale
    ? Object.entries(o.scale).map(([name, range]) =>
      `<span class="${name === o.level ? 'score-band-on' : ''}">${name} ${range}</span>`).join(' · ')
    : '';

  const rows = (o.contributions || []).map((c) => {
    if (!c.available) {
      return `
        <div class="score-part score-part-off">
          <div class="score-part-head">
            <span>${c.name}</span><span class="mono text-muted">not measured</span>
          </div>
          <div class="score-part-detail">${c.detail}</div>
        </div>`;
    }
    return `
      <div class="score-part">
        <div class="score-part-head">
          <span>${c.name}</span>
          <span class="mono text-muted">${c.points.toFixed(1)} pts
            / ${c.max_points.toFixed(0)}</span>
        </div>
        <div class="score-part-bar"><i style="width:${c.value * 100}%"></i></div>
        <div class="score-part-detail">${c.detail}
          <span class="text-muted">· scored ${c.scale_low}–${c.scale_high} ${c.unit}</span>
        </div>
      </div>`;
  }).join('');

  return `
    <div class="card">
      <div class="stat-label">Overall Risk</div>
      <div class="stat-value ${o.level === 'high' || o.level === 'critical' ? 'down' : ''}">
        ${pct.toFixed(0)}%</div>
      <div class="meter"><div class="meter-fill"
        style="width:${pct}%;background:${colour}"></div></div>
      <div class="text-xs text-muted mt-1">${ui.riskBadge(o.level)} · weighted score 0–100%</div>
      ${bands ? `<div class="score-bands">${bands}</div>` : ''}
      <div class="text-xs text-muted mt-1">${o.explanation || ''}</div>
      <details class="score-detail">
        <summary>How this is calculated</summary>
        <div class="score-parts">${rows}</div>
        <div class="score-part-detail mt-1">${o.method || ''}
          ${o.weight_redistributed
    ? ' Weights of unmeasured contributors were redistributed across the rest.' : ''}</div>
      </details>
    </div>`;
}

/* The absolute, textbook measures — the ones a risk desk actually quotes.
 * Every figure comes from this symbol's own returns over the selected window. */
function metricsPanel(profile) {
  if (!profile || !profile.available) {
    return `<div class="empty">${(profile && profile.reason) || 'No data for this window'}</div>`;
  }
  const m = profile.metrics;
  const pc = (v, d = 2) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(d)}%`);
  const nb = (v, d = 2) => (v === null || v === undefined ? '—' : Number(v).toFixed(d));
  const row = (label, value, cls = '', title = '') => `
    <div class="signal-row" ${title ? `title="${title}"` : ''}>
      <span class="signal-name">${label}</span>
      <span class="mono ${cls}">${value}</span></div>`;

  const betaText = m.beta === null || m.beta === undefined
    ? `— <span class="text-muted">no overlap</span>`
    : `${nb(m.beta)} <span class="text-muted">vs ${m.benchmark || '?'}</span>`;

  return `
    ${row('Annualised volatility', pc(m.annualised_volatility, 1))}
    ${row('Volatility (last 21d)', pc(m.volatility_21d, 1))}
    ${row('Downside deviation', pc(m.downside_deviation, 1), '',
    'Semi-deviation below the risk-free target — the Sortino denominator')}
    ${row('Daily VaR 95%', pc(m.var_95_daily), 'down')}
    ${row('Daily VaR 99%', pc(m.var_99_daily), 'down')}
    ${row('Daily CVaR 95%', pc(m.cvar_95_daily), 'down',
    'Expected shortfall: the average loss on days that breach VaR')}
    ${row('Daily CVaR 99%', pc(m.cvar_99_daily), 'down')}
    ${row('Maximum drawdown', pc(m.max_drawdown, 1), 'down')}
    ${row('Current drawdown', pc(m.current_drawdown, 1), 'down')}
    ${row('Beta', betaText)}
    ${row('Alpha (annualised)', pc(m.alpha, 2), fmt.cls(m.alpha))}
    ${row('Correlation to benchmark', nb(m.correlation_to_benchmark))}
    ${row('Sharpe ratio', nb(m.sharpe_ratio), fmt.cls(m.sharpe_ratio))}
    ${row('Sortino ratio', nb(m.sortino_ratio), fmt.cls(m.sortino_ratio))}
    ${row('Skewness', nb(m.skewness))}
    ${row('Excess kurtosis', nb(m.excess_kurtosis))}
    <div class="text-xs text-muted mt-2">${m.observations} returns ·
      ${span(m.observations)}${
  m.benchmark_overlap_days ? ` · β on ${m.benchmark_overlap_days} shared days` : ''}</div>
    <div class="text-xs text-muted mt-1">${profile.var_note || ''}</div>`;
}

async function runScan() {
  const symbol = ui.el('kSymbol').value.trim().toUpperCase();
  const period = getTimeRange();
  const lookback = parseInt(ui.el('kLookback').value, 10);
  const btn = ui.el('rScanBtn');
  btn.disabled = true; btn.textContent = 'Scanning…';
  ui.el('activeSymbolBadge').textContent = symbol;
  ui.loading('anomalyChart', 'Detecting anomalies…');
  ui.loading('anomalyList');
  ui.loading('riskCards');
  ui.loading('riskMetrics');
  ui.loading('riskBreakdown');
  ui.loading('varChart');

  try {
    const [scan, history] = await Promise.all([
      api.riskScan(symbol, period, lookback),
      api.history(symbol, period),
    ]);

    const crash = scan.crash_risk || {};
    const bubble = scan.bubble || {};

    // Three different samples feed this page, because the two scores have
    // different data floors (60 vs 200 returns). Stating one window for both
    // was a claim the page could not support on a short selection.
    //
    // These windows are deliberately described in *bars*, not dates. Absolute
    // dates belong on the chart's x-axis and nowhere else; repeating them here
    // duplicated the axis whenever the windows happened to match it, and was
    // actively confusing when they did not — at a 1M selection the chart spans
    // 23 bars while crash risk reads 61 and the bubble 201, so three different
    // date ranges sat above a chart showing a fourth.
    const basis = scan.basis || {};
    const cWin = basis.crash_window || basis.scores_from || {};
    const bWin = basis.bubble_window || basis.scores_from || {};
    const aFrom = basis.anomalies_from || {};
    ui.el('riskBasis').innerHTML = `
      <span><b>Crash Risk &amp; risk metrics</b>: ${cWin.bars ?? '?'} bars
        (${span(cWin.bars)}).</span>
      <span><b>Bubble</b>: ${bWin.bars ?? '?'} bars (${span(bWin.bars)}) —
        it needs 200 to fit a long-run trend.</span>
      <span><b>Anomalies</b>: last ${scan.anomaly_lookback_days} days,
        ${aFrom.warmup_bars ?? '?'} baseline bars before the window.</span>
      ${aFrom.window_truncated || aFrom.warmup_complete === false
    ? `<span class="basis-warn">⚠ ${aFrom.note}</span>` : ''}`;

    ui.el('riskCards').innerHTML = `
      ${overallRiskCard(scan.risk_profile)}
      ${scoreCard('Crash Risk Score', {
    score: crash.crash_risk_score, level: crash.level, scale: crash.scale,
    components: crash.components, reason: crash.recommendation,
  }, 0.55, 0.35)}
      ${scoreCard('Bubble Indicator', {
    score: bubble.bubble_score, level: bubble.level, scale: bubble.scale,
    components: bubble.components, reason: bubble.interpretation,
  }, 0.6, 0.4)}`;

    // ---- price chart with anomaly markers
    const candles = history.candles;
    const anomalyDates = new Set(scan.anomalies.map((a) => a.date));
    const marks = candles.filter((c) => anomalyDates.has(c.date.slice(0, 10)));
    const node = ui.el('anomalyChart');
    node.innerHTML = '';
    Plotly.newPlot(node, [
      { type: 'scatter', mode: 'lines', x: candles.map((c) => c.date), y: candles.map((c) => c.close),
        name: symbol, line: { color: C().accent, width: 1.8 } },
      { type: 'scatter', mode: 'markers', x: marks.map((c) => c.date), y: marks.map((c) => c.close),
        name: 'Anomaly', marker: { color: C().red, size: 9, symbol: 'x', line: { width: 1, color: '#fff' } } },
    ], historyChartLayout({ height: 420, yTitle: 'Price' }), historyChartConfig);

    // ---- breakdown
    // `|| 0` turned a missing figure into a confident "0.00%". A dash is the
    // honest rendering of a value the backend could not compute.
    const asPct = (v) => (v === null || v === undefined ? '—' : fmt.pct(v * 100));

    // The absolute measures a risk desk quotes, computed from this symbol's
    // own returns over the selected window.
    ui.el('riskMetrics').innerHTML = metricsPanel(scan.risk_profile);

    ui.el('riskBreakdown').innerHTML = `
      <div class="signal-row"><span class="signal-name">Volatility regime</span>
        <span class="mono">${fmt.num(crash.volatility_regime, 2)}×</span></div>
      <div class="signal-row"><span class="signal-name">ATR</span>
        <span class="mono">${fmt.num(crash.atr_pct, 2)}%</span></div>
      <div class="signal-row"><span class="signal-name">Down days (last 10)</span>
        <span class="mono">${crash.down_days_last_10 ?? '—'}</span></div>
      <div class="signal-row"><span class="signal-name">Trend deviation</span>
        <span class="mono">${fmt.num(bubble.trend_deviation_sigma, 2)}σ</span></div>
      <div class="signal-row"><span class="signal-name">3-month momentum</span>
        <span class="mono ${fmt.cls(bubble.momentum_3m)}">${asPct(bubble.momentum_3m)}</span></div>
      <div class="signal-row"><span class="signal-name">12-month momentum</span>
        <span class="mono ${fmt.cls(bubble.momentum_12m)}">${asPct(bubble.momentum_12m)}</span></div>
      <div class="signal-row"><span class="signal-name">RSI(14)</span>
        <span class="mono">${fmt.num(bubble.rsi, 1)}</span></div>
      <div class="info-box mt-2 text-xs">${crash.recommendation || ''}</div>
      <div class="text-xs text-muted mt-1">${bubble.interpretation || ''}</div>`;

    // ---- anomaly log
    ui.el('anomalyList').innerHTML = scan.anomalies.length ? scan.anomalies.slice(0, 25).map((a) => `
      <div class="alert-item alert-${a.severity === 'critical' || a.severity === 'high' ? 'critical' : a.severity === 'medium' ? 'warning' : 'info'}">
        <div style="flex:1">
          <div class="alert-title">${a.type.replace(/_/g, ' ')} <span class="badge badge-grey">${a.severity}</span></div>
          <div class="alert-msg">${a.description || ''}</div>
        </div>
      </div>`).join('') : '<div class="empty">No anomalies detected in this window</div>';

    // ---- return distribution
    const closes = candles.map((c) => c.close);
    const returns = closes.slice(1).map((v, i) => (v / closes[i] - 1) * 100);
    // No VaR figure means no VaR line: drawing it at 0% would place the
    // threshold in the middle of the distribution and invent a reading.
    const hasVar = crash.var_95_daily !== null && crash.var_95_daily !== undefined;
    const varLine = hasVar ? crash.var_95_daily * 100 : null;
    const distNode = ui.el('varChart');
    distNode.innerHTML = '';
    Plotly.newPlot(distNode, [{
      type: 'histogram', x: returns, nbinsx: 55, name: 'Daily returns',
      marker: { color: 'rgba(124, 108, 255,.62)', line: { color: C().accent, width: 0.5 } },
    }], ui.plotLayout({
      height: 260, showlegend: false,
      xaxis: { gridcolor: C().grid, title: { text: 'Daily return (%)', font: { size: 10 } } },
      yaxis: { gridcolor: C().grid, title: { text: 'Frequency', font: { size: 10 } } },
      shapes: hasVar ? [{ type: 'line', x0: varLine, x1: varLine, yref: 'paper', y0: 0, y1: 1,
        line: { color: C().red, width: 2, dash: 'dash' } }] : [],
      annotations: hasVar
        ? [{ x: varLine, yref: 'paper', y: 1.03, text: `VaR₉₅ ${varLine.toFixed(2)}%`,
          showarrow: false, font: { color: C().red, size: 10 } }]
        : [{ xref: 'paper', x: 0.5, yref: 'paper', y: 1.03,
          text: 'VaR₉₅ unavailable for this period', showarrow: false,
          font: { color: C().muted || '#888', size: 10 } }],
    }), ui.plotConfig);

    ui.toast(`Risk scan complete — ${scan.overall_risk_level} risk`, 'success');
  } catch (err) {
    ui.error('anomalyChart', err.message);
  } finally {
    btn.disabled = false; btn.textContent = 'Scan';
  }
}

/* ============================================================
   Market Regime Detection
   Replaces the old watchlist alert scan: which environment are we in, on
   what evidence, and what does that imply for positioning.
   ============================================================ */

const REGIME_STYLE = {
  bull_market:     { cls: 'rg-bull',    icon: '▲' },
  recovery:        { cls: 'rg-recover', icon: '◤' },
  low_volatility:  { cls: 'rg-calm',    icon: '≈' },
  sideways:        { cls: 'rg-flat',    icon: '─' },
  high_volatility: { cls: 'rg-vol',     icon: '⇅' },
  bear_market:     { cls: 'rg-bear',    icon: '▼' },
  crash_risk:      { cls: 'rg-crash',   icon: '⚠' },
  unknown:         { cls: 'rg-unknown', icon: '?' },
};

const ACTION_STYLE = {
  BUY: 'act-buy', HOLD: 'act-hold', REDUCE: 'act-reduce',
  HEDGE: 'act-hedge', SELL: 'act-sell',
};

/* A factor bar that reads left (bearish) to right (bullish) from a centre
   line, so the sign is visible without reading the number. */
function factorRow(f) {
  const pct = Math.abs(f.score) * 50;
  const side = f.score >= 0 ? 'left:50%' : `right:50%`;
  const tone = f.score >= 0 ? 'var(--green)' : 'var(--red)';
  const value = f.value === null || f.value === undefined
    ? '' : `${fmt.num(f.value, 2)}${f.unit || ''}`;
  return `
    <div class="rg-factor">
      <div class="rg-factor-head">
        <span>${f.name}</span><span class="mono text-muted">${value}</span>
      </div>
      <div class="rg-factor-track">
        <i style="${side};width:${pct}%;background:${tone}"></i>
      </div>
      <div class="rg-factor-detail">${f.detail}</div>
    </div>`;
}

async function loadRegime() {
  const symbol = ui.el('kSymbol').value.trim().toUpperCase();
  const period = getTimeRange();
  const btn = ui.el('regimeRefreshBtn');
  btn.disabled = true;
  ui.loading('regimeBox', 'Classifying market regime…');

  try {
    const r = await api.marketRegime(symbol, period);
    ui.el('regimeAsOf').textContent = r.as_of ? `classified ${fmt.timeAgo(r.as_of)}` : '—';

    if (r.regime === 'unknown') {
      ui.el('regimeBox').innerHTML = `<div class="empty">${r.reason}</div>`;
      ui.el('regimeTimeline').innerHTML = '';
      ui.el('regimeSpells').innerHTML = '';
      return;
    }

    const style = REGIME_STYLE[r.regime] || REGIME_STYLE.unknown;
    const ranked = Object.entries(r.probabilities).sort((a, b) => b[1] - a[1]);

    ui.el('regimeBox').innerHTML = `
      <div class="rg-top">
        <div class="rg-hero ${style.cls}">
          <div class="rg-hero-icon">${style.icon}</div>
          <div>
            <div class="rg-hero-label">${r.label}</div>
            <div class="rg-hero-sub">${symbol} · ${r.bars_analysed} bars
              (${span(r.bars_analysed)})</div>
          </div>
        </div>
        <div class="rg-metrics">
          <div><span class="stat-label">Probability</span>
            <b>${(r.probability * 100).toFixed(0)}%</b></div>
          <div><span class="stat-label">Confidence</span>
            <b>${(r.confidence * 100).toFixed(0)}%</b></div>
          <div><span class="stat-label">Action</span>
            <b class="rg-action ${ACTION_STYLE[r.action] || ''}">${r.action}</b></div>
        </div>
      </div>

      <div class="rg-probs">
        ${ranked.map(([name, p]) => `
          <div class="rg-prob ${name === r.regime ? 'on' : ''}">
            <div class="rg-prob-head">
              <span>${(REGIME_STYLE[name] || {}).icon || ''}
                ${name.replace(/_/g, ' ')}</span>
              <span class="mono">${(p * 100).toFixed(0)}%</span>
            </div>
            <div class="rg-prob-bar"><i style="width:${p * 100}%"></i></div>
          </div>`).join('')}
      </div>

      <div class="info-box text-xs mt-2">${r.insight}</div>

      <div class="rg-advice">
        <div><b>${r.action}</b> — ${r.action_rationale}</div>
        <div class="text-xs text-muted mt-1">Model reliability in this regime:
          ${r.model_reliability}</div>
      </div>

      <details class="score-detail">
        <summary>Evidence behind the classification (${r.factors.length} factors)</summary>
        <div class="rg-factors">${r.factors.map(factorRow).join('')}</div>
        ${r.sentiment ? `<div class="rg-factor-detail mt-1">News sentiment
          ${r.sentiment.score >= 0 ? '+' : ''}${fmt.num(r.sentiment.score, 3)}
          across ${r.sentiment.articles} headlines (${r.sentiment.source}).</div>`
    : '<div class="rg-factor-detail mt-1">No usable news sentiment for this symbol.</div>'}
        <div class="rg-factor-detail mt-1">${r.confidence_basis}</div>
      </details>

      <div class="rg-links">
        <a href="/portfolio.html">Portfolio</a>
        <a href="/signals.html">AI Recommendations</a>
        <a href="/rl.html">RL Agents</a>
      </div>`;

    renderRegimeTimeline(r);
  } catch (err) {
    ui.error('regimeBox', err.message);
  } finally {
    btn.disabled = false;
  }
}

/* Price coloured by the regime in force at each point, so the classification
   can be judged against what the market actually did next. */
function renderRegimeTimeline(r) {
  const node = ui.el('regimeTimeline');
  if (!r.timeline || r.timeline.length < 2) {
    node.innerHTML = '<div class="empty">Not enough history for a regime timeline</div>';
    ui.el('regimeSpells').innerHTML = '';
    return;
  }
  // Read from the stylesheet so the timeline follows the active theme; the
  // light palette uses darker variants for contrast.
  const token = (name, fallback) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  const colour = {
    bull_market: C().green,
    recovery: token('--regime-recovery', C().accent),
    low_volatility: token('--regime-calm', C().accent),
    sideways: token('--regime-flat', C().grid),
    high_volatility: C().amber,
    bear_market: token('--regime-bear', C().red),
    crash_risk: C().red,
  };
  const traces = [{
    type: 'scatter', mode: 'lines', x: r.timeline.map((p) => p.date),
    y: r.timeline.map((p) => p.close), name: 'Price',
    line: { color: C().grid, width: 1.2 }, hoverinfo: 'skip', showlegend: false,
  }];
  // One trace per regime keeps the legend meaningful and lets a regime be
  // toggled off to see the rest.
  const seen = new Set();
  r.timeline.forEach((p) => {
    if (seen.has(p.regime)) return;
    seen.add(p.regime);
    const pts = r.timeline.filter((q) => q.regime === p.regime);
    traces.push({
      type: 'scatter', mode: 'markers', name: p.label,
      x: pts.map((q) => q.date), y: pts.map((q) => q.close),
      marker: { color: colour[p.regime] || C().accent, size: 7 },
      hovertemplate: `%{x}<br>${p.label}<br>%{y:.2f}<extra></extra>`,
    });
  });
  node.innerHTML = '';
  Plotly.newPlot(node, traces, ui.plotLayout({
    height: 220, yTitle: 'Price',
    legend: { orientation: 'h', y: -0.22, font: { size: 9 } },
  }), ui.plotConfig);

  // Spells routinely span months, and `fmt.timeAgo` falls back to a formatted
  // absolute date past 30 days — which is exactly what this page must not
  // print. Wrapping it was not enough: a spell that ended 98 days ago still
  // rendered "brief, Apr 29, 2026". `ago()` stays relative at any age, so the
  // only dates on this page are the ones on the chart axes.
  const ago = (iso) => {
    const days = Math.max(0, Math.round((Date.now() - new Date(iso)) / 86400000));
    if (days < 1) return 'today';
    if (days < 30) return `${days}d ago`;
    if (days < 365) return `${Math.round(days / 30)} months ago`;
    const years = days / 365;
    return years < 2 ? `${years.toFixed(1)} years ago` : `${Math.round(years)} years ago`;
  };
  const spellAge = (s) => {
    const days = Math.max(1, Math.round(
      (new Date(s.to) - new Date(s.from)) / 86400000));
    const length = days >= 60 ? `${Math.round(days / 30)} months`
      : days >= 14 ? `${Math.round(days / 7)} weeks`
        : `${days} days`;
    // A single-point spell has from === to; say how brief it was.
    return s.from === s.to ? `brief, ${ago(s.to)}` : `${length}, ended ${ago(s.to)}`;
  };

  const spells = (r.transitions || []).slice(-8).reverse();
  ui.el('regimeSpells').innerHTML = spells.map((s) => `
    <span class="rg-spell ${(REGIME_STYLE[s.regime] || {}).cls || ''}">
      ${s.label}<i>${spellAge(s)}</i>
    </span>`).join('');
}

/* ============================================================
   Alert History
   The rule builder was replaced by the AI Confidence Score card on the
   dashboard. The history stays: it serves the automatic scanner alerts
   (news, anomaly, volatility, risk), which are 99% of what it holds.
   ============================================================ */

const PERIOD_LABELS = {
  '1mo': '1 Month', '3mo': '3 Months', '6mo': '6 Months', ytd: 'Year to date',
  '1y': '1 Year', '2y': '2 Years', '3y': '3 Years', '5y': '5 Years',
  '10y': '10 Years', max: 'Max',
};

const SEV_CLASS = { critical: 'badge-red', warning: 'badge-amber', info: 'badge-blue' };

async function loadHistory() {
  const box = ui.el('historyBox');
  try {
    const data = await api.listAlerts(60, {
      q: ui.el('historySearch')?.value.trim() || '',
      severity: ui.el('historySeverity')?.value || '',
    });
    const alerts = data.alerts || [];
    box.innerHTML = alerts.length ? alerts.map((a) => `
      <div class="hist-row">
        <span class="badge ${SEV_CLASS[a.severity] || 'badge-grey'}">${a.severity}</span>
        <div class="hist-main">
          <div class="hist-title">${a.title}
            <span class="text-muted">${a.symbol}</span></div>
          <div class="hist-reason">${a.reason || a.message}</div>
          ${a.triggers && a.triggers.length ? `<div class="hist-triggers">${a.triggers.map((t) => `
            <span class="trig ${t.passed ? 'trig-on' : ''}">${t.description}
              → <b>${t.observed === null || t.observed === undefined ? '—'
    : (typeof t.observed === 'number' ? fmt.num(t.observed, 2) : t.observed)}</b></span>`).join('')}</div>` : ''}
        </div>
        <div class="hist-when">${fmt.timeAgo(a.triggered_at)}
          ${a.period ? `<span class="text-muted">${PERIOD_LABELS[a.period] || a.period}</span>` : ''}</div>
      </div>`).join('')
      : '<div class="empty">Nothing has triggered yet</div>';
  } catch (err) {
    ui.error('historyBox', err.message);
  }
}

/**
 * Scan every watchlist symbol and store whatever fires.
 *
 * Runs on page load: an alerts panel that only fills in after the user
 * remembers to press a button is not an alerts panel.
 */
async function autoScanAlerts() {
  const badge = ui.el('autoScanStatus');
  if (badge) badge.textContent = 'scanning…';
  try {
    const watchlist = getWatchlist();
    const result = await api.scanWatchlist(watchlist, null, true);
    // `alerts` is a dict keyed by symbol, not a list: reading .length on it
    // gave undefined, and the count silently fell back to 0 even when the scan
    // had found (and stored) fourteen alerts.
    const n = result.total_alerts ?? 0;
    const affected = result.symbols_with_alerts ?? 0;
    if (badge) {
      badge.textContent = n
        ? `${n} alert${n === 1 ? '' : 's'} on ${affected}/${result.scanned ?? watchlist.length} symbols`
        : `${result.scanned ?? watchlist.length} symbols · all clear`;
    }
    await loadHistory();
  } catch (err) {
    if (badge) badge.textContent = 'scan failed';
    console.warn('auto scan failed', err);
  }
}

/* Everything on this page is about one symbol over one period; the regime
   panel has to follow both or it silently describes a different instrument
   from the charts above it. */
function refreshAll() {
  runScan();
  loadRegime();
}

document.addEventListener('DOMContentLoaded', () => {
  initTimeRange();
  initSearch((symbol) => { ui.el('kSymbol').value = symbol; refreshAll(); });
  new SymbolPicker('kSymbol', 'kSymbolPanel', () => refreshAll());
  // An alert rule on a mistyped ticker never fires and never says why.
  new SymbolPicker('ruleSymbol', 'ruleSymbolPanel', null, { syncActive: false });
  ui.el('kSymbol').value = getActiveSymbol();
  ui.el('rScanBtn').addEventListener('click', refreshAll);
  // Changing a period and then having to press Scan is a trap: the charts keep
  // showing the old window while the selector claims otherwise.
  ui.el('kLookback').addEventListener('change', runScan);
  ui.el('regimeRefreshBtn').addEventListener('click', loadRegime);
  loadHistory();

  let histTimer = null;
  ui.el('historySearch').addEventListener('input', () => {
    clearTimeout(histTimer);
    histTimer = setTimeout(loadHistory, 220);
  });
  ui.el('historySeverity').addEventListener('change', loadHistory);
  ui.el('historyRefreshBtn').addEventListener('click', loadHistory);

  ui.el('rescanBtn')?.addEventListener('click', autoScanAlerts);
  refreshAll();
  // Scan the whole watchlist on load so the history is populated without the
  // user having to ask for it.
  autoScanAlerts();

  // The global time range drives this page; there is no local
  // period control any more.
  onTimeRangeChange(() => { refreshAll() });
});

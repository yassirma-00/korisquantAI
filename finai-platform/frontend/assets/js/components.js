/* ============================================================
   Reusable render components shared by all pages
   ============================================================ */

/* ------------------------------------------------------------- charts */
function renderCandlestick(containerId, candles, options = {}) {
  const node = document.getElementById(containerId);
  if (!node) return;
  if (!candles || !candles.length) { ui.empty(node, 'No price data'); return; }

  const dates = candles.map((c) => c.date);
  const traces = [{
    type: 'candlestick',
    x: dates,
    open: candles.map((c) => c.open),
    high: candles.map((c) => c.high),
    low: candles.map((c) => c.low),
    close: candles.map((c) => c.close),
    name: options.symbol || 'Price',
    increasing: { line: { color: C().green }, fillcolor: C().green },
    decreasing: { line: { color: C().red }, fillcolor: C().red },
  }];

  const overlays = [
    ['sma_20', 'SMA 20', C().accent], ['sma_50', 'SMA 50', C().amber],
    ['sma_200', 'SMA 200', C().purple], ['ema_12', 'EMA 12', C().cyan],
    ['ema_26', 'EMA 26', C().accent3], ['vwap', 'VWAP', C().text2],
  ];
  overlays.forEach(([key, name, colour]) => {
    if (candles[0] && candles[0][key] !== undefined) {
      traces.push({
        type: 'scatter', mode: 'lines', x: dates, y: candles.map((c) => c[key]),
        name, line: { color: colour, width: 1.4 },
      });
    }
  });

  if (candles[0] && candles[0].bb_upper !== undefined) {
    traces.push({
      type: 'scatter', mode: 'lines', x: dates, y: candles.map((c) => c.bb_upper),
      name: 'BB Upper', line: { color: 'rgba(167, 139, 250,.45)', width: 1, dash: 'dot' },
    });
    traces.push({
      type: 'scatter', mode: 'lines', x: dates, y: candles.map((c) => c.bb_lower),
      name: 'BB Lower', line: { color: 'rgba(167, 139, 250,.45)', width: 1, dash: 'dot' },
      fill: 'tonexty', fillcolor: 'rgba(167, 139, 250,.06)',
    });
  }

  if (options.forecast && options.forecast.length) {
    const fDates = options.forecast.map((f) => f.date);
    const lastDate = dates[dates.length - 1];
    const lastClose = candles[candles.length - 1].close;
    traces.push({
      type: 'scatter', mode: 'lines+markers',
      x: [lastDate, ...fDates], y: [lastClose, ...options.forecast.map((f) => f.price)],
      name: 'AI Forecast', line: { color: C().cyan, width: 2.2, dash: 'dash' },
      marker: { size: 5 },
    });
    traces.push({
      type: 'scatter', mode: 'lines', x: [lastDate, ...fDates],
      y: [lastClose, ...options.forecast.map((f) => f.upper)],
      name: 'Upper 90%', line: { color: 'rgba(34, 211, 238,.28)', width: 0 }, showlegend: false,
    });
    traces.push({
      type: 'scatter', mode: 'lines', x: [lastDate, ...fDates],
      y: [lastClose, ...options.forecast.map((f) => f.lower)],
      name: 'Confidence band', line: { color: 'rgba(34, 211, 238,.28)', width: 0 },
      fill: 'tonexty', fillcolor: 'rgba(34, 211, 238,.11)',
    });
  }

  if (options.trades && options.trades.length) {
    const buys = options.trades.filter((t) => t.action === 'BUY');
    const sells = options.trades.filter((t) => t.action === 'SELL');
    if (buys.length) traces.push({
      type: 'scatter', mode: 'markers', x: buys.map((t) => t.date), y: buys.map((t) => t.price),
      name: 'Agent BUY', marker: { color: C().green, size: 9, symbol: 'triangle-up', line: { color: '#fff', width: 1 } },
    });
    if (sells.length) traces.push({
      type: 'scatter', mode: 'markers', x: sells.map((t) => t.date), y: sells.map((t) => t.price),
      name: 'Agent SELL', marker: { color: C().red, size: 9, symbol: 'triangle-down', line: { color: '#fff', width: 1 } },
    });
  }

  node.innerHTML = '';
  Plotly.newPlot(node, traces, ui.plotLayout({
    xaxis: { gridcolor: C().grid, rangeslider: { visible: false }, type: 'date' },
    yaxis: { gridcolor: C().grid, title: { text: options.yTitle || '', font: { size: 10 } } },
    height: options.height || 420,
  }), ui.plotConfig);
}

function renderVolume(containerId, candles) {
  const node = document.getElementById(containerId);
  if (!node || !candles || !candles.length) return;
  if (!candles.some((c) => c.volume)) { ui.empty(node, 'No volume data for this instrument'); return; }
  const colours = candles.map((c) => (c.close >= c.open ? 'rgba(34, 217, 138,.6)' : 'rgba(255, 77, 106,.6)'));
  node.innerHTML = '';
  Plotly.newPlot(node, [{
    type: 'bar', x: candles.map((c) => c.date), y: candles.map((c) => c.volume),
    marker: { color: colours }, name: 'Volume',
  }], ui.plotLayout({ height: 160, showlegend: false, margin: { l: 52, r: 18, t: 8, b: 30 } }), ui.plotConfig);
}

function renderIndicatorPanel(containerId, candles) {
  const node = document.getElementById(containerId);
  if (!node || !candles || !candles.length) return;
  const dates = candles.map((c) => c.date);
  const traces = [];
  if (candles[0].rsi !== undefined) {
    traces.push({ type: 'scatter', mode: 'lines', x: dates, y: candles.map((c) => c.rsi), name: 'RSI(14)', line: { color: C().purple, width: 1.6 } });
  }
  if (candles[0].stoch_k !== undefined) {
    traces.push({ type: 'scatter', mode: 'lines', x: dates, y: candles.map((c) => c.stoch_k), name: '%K', line: { color: C().cyan, width: 1.2 } });
  }
  if (!traces.length) { ui.empty(node, 'No oscillator data'); return; }
  node.innerHTML = '';
  Plotly.newPlot(node, traces, ui.plotLayout({
    height: 180, yaxis: { gridcolor: C().grid, range: [0, 100] },
    margin: { l: 52, r: 18, t: 8, b: 30 },
    shapes: [
      { type: 'line', x0: dates[0], x1: dates[dates.length - 1], y0: 70, y1: 70, line: { color: 'rgba(255, 77, 106,.45)', dash: 'dash', width: 1 } },
      { type: 'line', x0: dates[0], x1: dates[dates.length - 1], y0: 30, y1: 30, line: { color: 'rgba(34, 217, 138,.45)', dash: 'dash', width: 1 } },
    ],
  }), ui.plotConfig);
}

function renderMACD(containerId, candles) {
  const node = document.getElementById(containerId);
  if (!node || !candles || !candles.length || candles[0].macd === undefined) {
    if (node) ui.empty(node, 'MACD not computed');
    return;
  }
  const dates = candles.map((c) => c.date);
  node.innerHTML = '';
  Plotly.newPlot(node, [
    { type: 'bar', x: dates, y: candles.map((c) => c.macd_hist), name: 'Histogram',
      marker: { color: candles.map((c) => (c.macd_hist >= 0 ? 'rgba(34, 217, 138,.55)' : 'rgba(255, 77, 106,.55)')) } },
    { type: 'scatter', mode: 'lines', x: dates, y: candles.map((c) => c.macd), name: 'MACD', line: { color: C().accent, width: 1.6 } },
    { type: 'scatter', mode: 'lines', x: dates, y: candles.map((c) => c.macd_signal), name: 'Signal', line: { color: C().amber, width: 1.4 } },
  ], ui.plotLayout({ height: 180, margin: { l: 52, r: 18, t: 8, b: 30 } }), ui.plotConfig);
}

function renderLineChart(containerId, series, options = {}) {
  const node = document.getElementById(containerId);
  if (!node) return;
  if (!series || !series.length) { ui.empty(node, options.emptyText || 'No data'); return; }
  const traces = series.map((s) => ({
    type: 'scatter', mode: 'lines', x: s.x, y: s.y, name: s.name,
    line: { color: s.color || C().accent, width: s.width || 2, dash: s.dash },
    fill: s.fill, fillcolor: s.fillcolor,
  }));
  node.innerHTML = '';
  // Time-series charts get zoom, pan, a range slider and unified hover. The
  // helper is defined in timerange.js; pages that do not load it (none today)
  // still get the plain layout rather than throwing.
  const layout = (typeof historyChartLayout === 'function' && options.history !== false)
    ? historyChartLayout : ui.plotLayout;
  const cfg = (typeof historyChartConfig === 'object' && options.history !== false)
    ? historyChartConfig : ui.plotConfig;
  Plotly.newPlot(node, traces, layout({
    height: options.height || 300,
    yaxis: { gridcolor: C().grid, tickformat: options.tickformat, title: { text: options.yTitle || '', font: { size: 10 } } },
    // Not every line chart has a date x-axis: training curves are indexed by
    // epoch, and labelling that axis is what makes the panel readable.
    ...(options.xTitle ? { xaxis: { gridcolor: C().grid, title: { text: options.xTitle, font: { size: 10 } } } } : {}),
    showlegend: options.showlegend !== false,
  }), cfg);
}

/* -------------------------------------------------------------- lists */
function renderQuotesTable(containerId, quotes, onSelect) {
  const node = document.getElementById(containerId);
  if (!node) return;
  if (!quotes || !quotes.length) { ui.empty(node, 'No quotes'); return; }
  node.innerHTML = `
    <table>
      <thead><tr>
        <th>Symbol</th><th class="t-right">Price</th><th class="t-right">Change</th>
        <th class="t-right">%</th><th class="t-right">Volume</th><th>Source</th>
      </tr></thead>
      <tbody>
        ${quotes.map((q) => `
          <tr class="clickable" data-symbol="${q.symbol}">
            <td class="sym-cell">${q.symbol}<div class="text-xs text-muted">${q.name || ''}</div></td>
            <td class="t-right" data-animate="${q.price}" data-format="price"
                data-live-symbol="${q.symbol}" data-live-field="price"
                data-raw-value="${q.price}">${fmt.price(q.price)}</td>
            <td class="t-right ${fmt.cls(q.change)}"
                data-animate="${q.change}" data-format="price"
                data-ref="${q.price}">${fmt.price(q.change, q.price)}</td>
            <td class="t-right ${fmt.cls(q.change_percent)}" data-live-symbol="${q.symbol}"
                data-live-field="change_percent" data-raw-value="${q.change_percent}">
              <span class="arrow">${fmt.arrow(q.change_percent)}</span>
              <span data-animate="${q.change_percent}" data-format="pct">${fmt.pct(q.change_percent)}</span></td>
            <td class="t-right text-muted"${q.volume ? ` data-animate="${q.volume}" data-format="compact"` : ''}>${q.volume ? fmt.compact(q.volume) : '—'}</td>
            <td>${ui.sourceBadge(q.source)}</td>
          </tr>`).join('')}
      </tbody>
    </table>`;
  // Stagger the rows so the table fills in top-to-bottom rather than all at once
  animateContainer(node, { stagger: 45 });
  if (onSelect) {
    node.querySelectorAll('tr[data-symbol]').forEach((row) =>
      row.addEventListener('click', () => onSelect(row.dataset.symbol)));
  }
}

function renderSignals(containerId, signals) {
  const node = document.getElementById(containerId);
  if (!node) return;
  if (!signals || !signals.indicators) { ui.empty(node, 'No signals'); return; }
  const colour = { buy: 'badge-green', sell: 'badge-red', neutral: 'badge-grey' };
  const consensusColour = signals.consensus === 'bullish' ? 'var(--green)'
    : signals.consensus === 'bearish' ? 'var(--red)' : 'var(--text-1)';

  node.innerHTML = `
    <div class="flex items-center justify-between mb-2">
      <div>
        <div class="stat-label">Consensus</div>
        <div style="font-size:19px;font-weight:700;color:${consensusColour};text-transform:capitalize">${signals.consensus}</div>
      </div>
      <div class="text-sm text-muted">
        <span class="up">${signals.buy_votes} buy</span> ·
        <span class="down">${signals.sell_votes} sell</span> ·
        <span>${signals.neutral_votes} neutral</span>
      </div>
    </div>
    <div class="meter mb-2">
      <div class="meter-fill" style="width:${(signals.strength * 100).toFixed(0)}%;background:${consensusColour}"></div>
    </div>
    ${Object.entries(signals.indicators).map(([name, s]) => `
      <div class="signal-row">
        <div>
          <div class="signal-name">${name}</div>
          <div class="signal-note">${s.note}</div>
        </div>
        <div style="text-align:right">
          <span class="badge ${colour[s.signal] || 'badge-grey'}">${s.signal}</span>
          <div class="text-xs mono text-muted mt-1">${s.value ?? '—'}</div>
        </div>
      </div>`).join('')}`;
}

function renderNews(containerId, items, limit = 8) {
  const node = document.getElementById(containerId);
  if (!node) return;
  if (!items || !items.length) { ui.empty(node, 'No recent news'); return; }
  node.innerHTML = items.slice(0, limit).map((n) => {
    const s = n.sentiment || {};
    const impact = n.impact_score || 0;
    return `<div class="news-item">
      <div class="news-title">${n.url ? `<a href="${n.url}" target="_blank" rel="noopener">${n.title}</a>` : n.title}</div>
      <div class="news-meta">
        ${ui.sentimentBadge(s.label)}
        <span class="badge badge-grey">${(n.category || '').replace(/_/g, ' ')}</span>
        <span>${n.source || ''}</span>
        <span>·</span>
        <span>${fmt.timeAgo(n.published_at)}</span>
        ${impact > 0.2 ? `<span class="badge badge-amber">impact ${impact.toFixed(2)}</span>` : ''}
      </div>
    </div>`;
  }).join('');
}

function renderAlerts(containerId, alerts, limit = 12) {
  const node = document.getElementById(containerId);
  if (!node) return;
  if (!alerts || !alerts.length) { ui.empty(node, 'No active alerts — conditions are normal'); return; }
  const icon = { critical: '🔴', warning: '🟠', info: '🔵' };
  node.innerHTML = alerts.slice(0, limit).map((a) => `
    <div class="alert-item alert-${a.severity}">
      <div style="font-size:15px">${icon[a.severity] || '•'}</div>
      <div style="flex:1;min-width:0">
        <div class="alert-title">${a.title}</div>
        <div class="alert-msg">${a.message}</div>
        <div class="alert-time">${a.symbol} · ${(a.alert_type || '').replace(/_/g, ' ')} · ${fmt.timeAgo(a.triggered_at)}</div>
      </div>
    </div>`).join('');
}

function renderXaiBars(containerId, contributions, limit = 10) {
  const node = document.getElementById(containerId);
  if (!node) return;
  if (!contributions || !contributions.length) { ui.empty(node, 'No attribution available'); return; }
  const items = contributions.slice(0, limit);
  const max = Math.max(...items.map((c) => Math.abs(c.contribution))) || 1;
  node.innerHTML = items.map((c) => {
    const width = (Math.abs(c.contribution) / max) * 50;
    const positive = c.contribution > 0;
    return `<div class="xai-bar">
      <div class="xai-label" title="${c.label}">${c.label}</div>
      <div class="xai-track">
        <div class="xai-center"></div>
        <div class="xai-fill ${positive ? 'pos' : 'neg'}" style="width:${width}%"></div>
      </div>
      <div class="xai-value ${positive ? 'up' : 'down'}">${c.contribution > 0 ? '+' : ''}${c.contribution.toFixed(5)}</div>
    </div>`;
  }).join('');
}

function renderMetricsGrid(containerId, metrics, spec) {
  const node = document.getElementById(containerId);
  if (!node) return;
  if (!metrics || !Object.keys(metrics).length) { ui.empty(node, 'No metrics'); return; }
  node.innerHTML = `<div class="grid grid-4">
    ${spec.filter((s) => metrics[s.key] !== undefined && metrics[s.key] !== null).map((s) => {
      const raw = metrics[s.key];
      const value = s.format === 'pctAbs' ? fmt.pct(raw * 100, s.decimals ?? 2, false)
        : s.format === 'pct' ? fmt.pct(raw * 100)
        : s.format === 'pctRaw' ? fmt.pct(raw)
        : s.format === 'money' ? fmt.money(raw)
        : fmt.num(raw, s.decimals ?? 2);
      const cls = s.colour ? fmt.cls(raw * (s.invert ? -1 : 1)) : '';
      const animVal = s.format === 'pct' || s.format === 'pctAbs' ? raw * 100 : raw;
      const animFmt = s.format === 'pctAbs' ? 'pct' : s.format === 'money' ? 'money'
        : s.format === 'pct' ? 'pct' : 'num';
      return `<div class="card animate-in">
        <div class="stat-label">${s.label}</div>
        <div class="stat-value ${cls}" style="font-size:20px"
             data-animate="${animVal}" data-format="${animFmt}"
             data-decimals="${s.decimals ?? 2}"
             data-signed="${s.format === 'pctAbs' ? 'false' : 'true'}">${value}</div>
        ${s.hint ? `<div class="text-xs text-muted mt-1">${s.hint}</div>` : ''}
      </div>`;
    }).join('')}
  </div>`;
  animateContainer(node);
}

const METRIC_SPEC = [
  { key: 'total_return', label: 'Total Return', format: 'pct', colour: true },
  { key: 'annualised_return', label: 'Annual Return', format: 'pct', colour: true },
  { key: 'annualised_volatility', label: 'Volatility', format: 'pctAbs' },
  { key: 'sharpe_ratio', label: 'Sharpe Ratio', decimals: 2, colour: true },
  { key: 'sortino_ratio', label: 'Sortino Ratio', decimals: 2, colour: true },
  { key: 'calmar_ratio', label: 'Calmar Ratio', decimals: 2, colour: true },
  { key: 'max_drawdown', label: 'Max Drawdown', format: 'pct', colour: true },
  { key: 'var_95', label: 'VaR 95% (daily)', format: 'pct', colour: true },
  { key: 'cvar_95', label: 'CVaR 95%', format: 'pct', colour: true },
  { key: 'win_rate', label: 'Win Rate', format: 'pctAbs' },
  { key: 'beta', label: 'Beta', decimals: 2 },
  { key: 'alpha', label: 'Alpha', format: 'pct', colour: true },
];

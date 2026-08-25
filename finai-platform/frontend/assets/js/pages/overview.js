/* ============================================================
   Page: Market Overview
   ============================================================ */

let heatmapClass = '';

/* ======================== AI recommendation + confidence (dashboard pair) */

const CONF_BAND_COLOUR = {
  'very-high': 'var(--green)',
  high: 'var(--green)',
  moderate: 'var(--amber)',
  low: 'var(--red)',
  'very-low': 'var(--red)',
};

const ACTION_COLOUR = {
  STRONG_BUY: 'var(--green)', BUY: 'var(--green)', HOLD: 'var(--text-1)',
  SELL: 'var(--red)', STRONG_SELL: 'var(--red)',
};

/* An SVG ring that animates from empty to its value.
   The dash offset is set after a frame so the browser has something to
   transition *from*; setting it inline renders the final state instantly. */
function confidenceGauge(percent, colour, size = 132) {
  const r = size / 2 - 11;
  const c = 2 * Math.PI * r;
  return `
    <svg class="conf-gauge" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}"
         role="img" aria-label="Confidence ${percent.toFixed(0)} percent">
      <circle class="conf-gauge-bg" cx="${size / 2}" cy="${size / 2}" r="${r}"></circle>
      <circle class="conf-gauge-fg" cx="${size / 2}" cy="${size / 2}" r="${r}"
              stroke="${colour}" stroke-dasharray="${c}" stroke-dashoffset="${c}"
              stroke-linecap="round" data-target="${c * (1 - percent / 100)}"></circle>
      <text class="conf-gauge-value" x="50%" y="47%" text-anchor="middle"
            dominant-baseline="middle" fill="${colour}">${percent.toFixed(0)}%</text>
      <text class="conf-gauge-cap" x="50%" y="65%" text-anchor="middle"
            dominant-baseline="middle">confidence</text>
    </svg>`;
}

function renderVerdict(report) {
  const action = report.action || 'HOLD';
  const colour = ACTION_COLOUR[action] || 'var(--text-1)';
  const signals = (report.recommendation && report.recommendation.signals) || [];
  const label = {
    forecast: 'Deep learning', rl: 'Reinforcement learning',
    technical: 'Technical analysis', sentiment: 'NLP sentiment',
  };
  ui.el('verdictSymbol').textContent = report.symbol || '';
  ui.el('verdictBox').innerHTML = `
    <div class="verdict-main">
      <div class="verdict-action" style="color:${colour}">${action.replace('_', ' ')}</div>
      <div class="verdict-score">composite ${fmt.num(report.composite_score, 2)}
        · ${report.last_price ? fmt.price(report.last_price) : '—'}</div>
    </div>
    <div class="verdict-signals">
      ${signals.map((s) => {
    const on = s.available;
    const dir = s.score > 0.05 ? 'up' : s.score < -0.05 ? 'down' : '';
    return `<div class="verdict-signal ${on ? '' : 'off'}">
          <span class="vs-name">${label[s.source] || s.source}</span>
          <span class="vs-score mono ${dir}">${on ? (s.score >= 0 ? '+' : '') + fmt.num(s.score, 2) : '—'}</span>
        </div>`;
  }).join('')}
    </div>`;
}

function renderConfidence(report) {
  const colour = CONF_BAND_COLOUR[report.band] || 'var(--text-1)';
  const pct = report.percent;
  ui.el('confidenceAsOf').textContent = report.as_of ? fmt.timeAgo(report.as_of) : '';

  ui.el('confidenceBox').innerHTML = `
    <div class="conf-top">
      ${confidenceGauge(pct, colour)}
      <div class="conf-side">
        <span class="conf-badge conf-${report.band}">${report.label}</span>
        <div class="conf-bar" role="progressbar" aria-valuenow="${pct.toFixed(0)}"
             aria-valuemin="0" aria-valuemax="100">
          <i style="background:${colour}" data-width="${pct}%"></i>
        </div>
        <div class="conf-scale">
          ${report.bands.map((b) => `<span class="${b.key === report.band ? 'on' : ''}">${b.label}</span>`).join('')}
        </div>
        <div class="conf-headline">${report.summary}</div>
      </div>
    </div>
    <details class="conf-detail">
      <summary>What drives this score</summary>
      <div class="conf-factors">
        ${report.contributors.map((c) => `
          <div class="conf-factor">
            <div class="cf-head">
              <span>${c.label}</span>
              <span class="mono text-muted">${c.points.toFixed(1)} / ${c.max_points.toFixed(0)}</span>
            </div>
            <div class="cf-bar"><i data-width="${(c.value * 100).toFixed(1)}%"></i></div>
            <div class="cf-detail">${c.detail}</div>
          </div>`).join('')}
      </div>
      <div class="cf-detail mt-1">${report.basis}</div>
    </details>`;

  // Animate on the next frame: the elements must exist with their start value
  // before the transition target is applied, or there is nothing to animate.
  requestAnimationFrame(() => {
    const ring = ui.el('confidenceBox').querySelector('.conf-gauge-fg');
    if (ring) ring.style.strokeDashoffset = ring.dataset.target;
    ui.el('confidenceBox').querySelectorAll('[data-width]').forEach((bar) => {
      bar.style.width = bar.dataset.width;
    });
  });
}

async function loadConfidence() {
  const symbol = getActiveSymbol();
  ui.loading('verdictBox', 'Analysing…');
  ui.loading('confidenceBox', 'Scoring…');
  try {
    const report = await api.aiConfidence(symbol);
    renderVerdict(report);
    renderConfidence(report);
  } catch (err) {
    ui.error('verdictBox', err.message);
    ui.error('confidenceBox', 'Confidence unavailable for this symbol.');
  }
}

/* ============================ intelligence band (forecast / agent / risk) */
async function loadIntelBand() {
  const symbol = getActiveSymbol();
  const node = ui.el('intelBand');
  if (!node) return;
  document.getElementById('signalConflict')?.remove();
  ui.el('benchSymbol').textContent = symbol;
  try {
    requireApi('agentDecision', 'marketRegime', 'strategyBenchmarks');
  } catch {
    return;   // requireApi already surfaced the message
  }

  // Fire everything in parallel; each tile degrades independently so one
  // untrained model never blanks the whole band.
  const [forecast, decision, risk, regime] = await Promise.allSettled([
    api.predict(symbol, 'lstm', 5, getTimeRange(), false),
    api.agentDecision(symbol, 'dueling_dqn', '1y'),
    api.crashRisk(symbol),
    api.marketRegime(symbol),
  ]);

  const tile = (label, value, sub, cls = '', badge = '') => `
    <div class="card animate-in">
      <div class="flex items-center justify-between">
        <div class="stat-label">${label}</div>${badge}
      </div>
      <div class="stat-value ${cls}" style="font-size:20px">${value}</div>
      <div class="text-xs text-muted mt-1">${sub}</div>
    </div>`;

  const tiles = [];

  if (forecast.status === 'fulfilled') {
    const f = forecast.value;
    tiles.push(tile('AI Forecast (5d)', fmt.pct(f.predicted_return * 100),
      `${f.model.toUpperCase()} · target ${fmt.price(f.predicted_price)}`,
      fmt.cls(f.predicted_return),
      `<span class="badge ${f.direction === 'up' ? 'badge-green' : 'badge-red'}">${f.direction}</span>`));
  } else {
    tiles.push(tile('AI Forecast', '—', 'Train a model on the Forecast page', 'text-muted'));
  }

  if (decision.status === 'fulfilled') {
    const d = decision.value;
    const c = d.action === 'BUY' ? 'up' : d.action === 'SELL' ? 'down' : '';
    tiles.push(tile('RL Agent', d.action,
      `${d.algorithm_name} · ${fmt.pct(d.confidence * 100, 0, false)} confidence`, c,
      ui.riskBadge(d.risk?.level)));
    renderAgentPanel(d);
  } else {
    tiles.push(tile('RL Agent', '—', 'Train an agent on the RL page', 'text-muted'));
    ui.empty('agentPanel', 'No trained agent for ' + symbol);
  }

  if (risk.status === 'fulfilled') {
    const r = risk.value;
    tiles.push(tile('Crash Risk', fmt.num(r.crash_risk_score, 2),
      `VaR₉₅ ${fmt.pct((r.var_95_daily || 0) * 100)} · DD ${fmt.pct((r.current_drawdown || 0) * 100, 1)}`,
      r.crash_risk_score > 0.55 ? 'down' : '', ui.riskBadge(r.level)));
  } else {
    tiles.push(tile('Crash Risk', '—', 'unavailable', 'text-muted'));
  }

  if (regime.status === 'fulfilled') {
    const g = regime.value;
    tiles.push(tile('Market Regime', (g.regime || '—').toUpperCase(),
      `vol ratio ${fmt.num(g.volatility_ratio, 2)} · ${g.model_reliability || ''}`.slice(0, 60), ''));
  } else {
    tiles.push(tile('Market Regime', '—', 'unavailable', 'text-muted'));
  }

  node.innerHTML = tiles.join('');
  animateContainer('intelBand');

  // Surface disagreement rather than letting the user reconcile it themselves:
  // a bearish forecast beside a BUY is meaningful information, not a glitch.
  if (forecast.status === 'fulfilled' && decision.status === 'fulfilled') {
    const bearishForecast = forecast.value.predicted_return < 0;
    const bullishAgent = decision.value.action === 'BUY';
    const bullishForecast = forecast.value.predicted_return > 0;
    const bearishAgent = decision.value.action === 'SELL';
    if ((bearishForecast && bullishAgent) || (bullishForecast && bearishAgent)) {
      node.insertAdjacentHTML('afterend', `
        <div class="info-box mb-2" id="signalConflict">
          <strong>Signals disagree.</strong> The price forecast points
          ${bearishForecast ? 'down' : 'up'} while the RL agent says
          ${decision.value.action}. They optimise different objectives — the forecaster
          predicts the next move, the agent maximises risk-adjusted reward after costs
          and may prefer to stay invested through short-term weakness.
          Treat conviction as low until they align.
        </div>`);
    }
  }

  loadBenchmarkChart(symbol);
}

function renderAgentPanel(d) {
  const plan = d.trade_plan || {};
  const colour = d.action === 'BUY' ? 'var(--green)' : d.action === 'SELL' ? 'var(--red)' : 'var(--text-1)';
  ui.el('agentPanel').innerHTML = `
    <div class="t-center mb-2">
      <div style="font-size:30px;font-weight:800;color:${colour}">${d.action}</div>
      <div class="text-xs text-muted">${d.symbol} · ${d.algorithm_name} · ${fmt.pct(d.confidence * 100, 0, false)} conf.</div>
    </div>
    <div class="signal-row"><span class="signal-name">Position size</span>
      <span class="mono">${fmt.pct(plan.position_size_pct ?? 0, 1, false)}</span></div>
    <div class="signal-row"><span class="signal-name">Stop loss</span>
      <span class="mono down">${plan.stop_loss_price ? fmt.price(plan.stop_loss_price) : '—'}</span></div>
    <div class="signal-row"><span class="signal-name">Take profit</span>
      <span class="mono up">${plan.take_profit_price ? fmt.price(plan.take_profit_price) : '—'}</span></div>
    <div class="signal-row"><span class="signal-name">Horizon</span>
      <span class="mono">${d.investment_horizon?.days ?? '—'}d</span></div>
    <div class="text-xs text-muted mt-2">${(d.explanation?.summary || '').slice(0, 190)}</div>
    <a class="btn btn-sm mt-2" href="rl.html" style="width:100%;justify-content:center">Full decision →</a>`;
}

async function loadBenchmarkChart(symbol) {
  const node = ui.el('benchChart');
  ui.loading(node, 'Backtesting reference strategies…');
  try {
    const data = await api.strategyBenchmarks(symbol, { period: getTimeRange() });
    const traces = data.strategies.filter((s) => s.equity_curve?.length).map((s) => ({
      type: 'scatter', mode: 'lines',
      x: s.equity_curve.map((p) => p.date), y: s.equity_curve.map((p) => p.value),
      name: s.label, line: { width: s.is_agent ? 2.6 : 1.6 },
    }));
    node.innerHTML = '';
    Plotly.newPlot(node, traces, historyChartLayout({
      height: 300, yaxis: { gridcolor: C().grid, title: { text: 'Value ($)', font: { size: 10 } } },
    }), historyChartConfig);
  } catch (err) {
    ui.error(node, `Benchmarks unavailable: ${err.message}`);
  }
}


async function loadHealth() {
  try {
    const h = await fetch('/health').then((r) => r.json());
    ui.el('healthBox').innerHTML = `
      <div class="flex items-center gap-1 mb-1">
        <span class="dot ${h.live_providers.length ? 'dot-live' : 'dot-sim'}"></span>
        <strong>${h.status}</strong> v${h.version}
      </div>
      <div>Mode: <strong>${h.data_mode}</strong></div>
      <div>Providers: ${h.live_providers.join(', ') || 'synthetic only'}</div>
      <div>Torch ${h.torch_available ? '✓' : '✗'} · SB3 ${h.sb3_available ? '✓' : '✗'}</div>`;
  } catch {
    ui.el('healthBox').innerHTML = '<span class="down">API unreachable</span>';
  }
}

function indexCard(q) {
  const cls = fmt.cls(q.change_percent);
  // data-animate drives the count-up; data-live-symbol lets the ticker refresh it in place
  return `<div class="card animate-in">
    <div class="flex items-center justify-between">
      <div class="stat-label">${q.name || q.symbol}</div>
      ${ui.sourceBadge(q.source)}
    </div>
    <div class="stat-value" data-animate="${q.price}" data-format="price"
         data-live-symbol="${q.symbol}" data-live-field="price">${fmt.price(q.price)}</div>
    <div class="stat-change ${cls}">${fmt.arrow(q.change_percent)} ${fmt.price(q.change)}
      (<span data-animate="${q.change_percent}" data-format="pct">${fmt.pct(q.change_percent)}</span>)</div>
  </div>`;
}

async function loadOverview() {
  ui.loading('watchlistTable');
  try {
    const data = await api.overview(getWatchlist());

    ui.el('indicesGrid').innerHTML = data.indices.map(indexCard).join('');
    animateContainer('indicesGrid');

    renderQuotesTable('watchlistTable', data.watchlist, (symbol) => {
      setActiveSymbol(symbol);
      window.location.href = 'analysis.html';
    });
    ui.el('watchlistMeta').innerHTML =
      `${data.watchlist.length} instruments <span id="lastUpdated" class="text-muted">· ${fmt.timeAgo(data.as_of)}</span><span class="live-dot" title="auto-refreshing"></span>`;
    // Keep the watchlist ticking without a full page reload
    liveTicker.track(data.watchlist.map((q) => q.symbol)).start();

    const b = data.breadth;
    const pulse = data.sentiment_pulse || {};
    const moodColour = pulse.mood === 'risk-on' ? 'var(--green)'
      : pulse.mood === 'risk-off' ? 'var(--red)' : 'var(--text-1)';
    const advPct = (b.advancing / Math.max(b.advancing + b.declining, 1)) * 100;
    const pulseScore = pulse.score ?? 0;

    ui.el('breadthBox').innerHTML = `
      <div class="mb-2">
        <div class="flex justify-between text-sm mb-1">
          <span class="up"><span data-animate="${b.advancing}" data-format="num" data-decimals="0">${b.advancing}</span> advancing</span>
          <span class="down"><span data-animate="${b.declining}" data-format="num" data-decimals="0">${b.declining}</span> declining</span>
        </div>
        <div class="meter" style="height:9px">
          <div class="meter-fill" style="width:0%;background:var(--green)" data-meter="${advPct}"></div>
        </div>
        <div class="text-xs text-muted mt-1">A/D ratio
          <span data-animate="${b.advance_decline_ratio}" data-format="num">${fmt.num(b.advance_decline_ratio, 2)}</span>
          · avg change
          <span data-animate="${b.average_change_pct}" data-format="pct">${fmt.pct(b.average_change_pct)}</span></div>
      </div>
      <div style="border-top:1px solid var(--border-soft);padding-top:12px">
        <div class="stat-label">News Sentiment</div>
        <div style="font-size:21px;font-weight:700;color:${moodColour};text-transform:uppercase">${pulse.mood || 'n/a'}</div>
        <div class="text-xs text-muted">Aggregate score
          <span data-animate="${pulseScore}" data-format="num" data-decimals="3">${fmt.num(pulseScore, 3)}</span></div>
        ${(pulse.most_bullish || []).length ? `
          <div class="mt-2 text-xs">
            <div class="text-muted mb-1">Most bullish</div>
            ${pulse.most_bullish.map((s) => `<span class="badge badge-green" style="margin-right:4px">${s.symbol} ${fmt.num(s.score, 2)}</span>`).join('')}
          </div>` : ''}
        ${(pulse.most_bearish || []).length ? `
          <div class="mt-1 text-xs">
            <div class="text-muted mb-1">Most bearish</div>
            ${pulse.most_bearish.map((s) => `<span class="badge badge-red" style="margin-right:4px">${s.symbol} ${fmt.num(s.score, 2)}</span>`).join('')}
          </div>` : ''}
        ${!(pulse.most_bullish || []).length && !(pulse.most_bearish || []).length ? `
          <div class="mt-2 text-xs text-muted">No symbol shows meaningful directional sentiment.</div>` : ''}
      </div>`;
    animateContainer('breadthBox');

    const movers = [...data.top_gainers, ...data.top_losers];
    ui.el('moversBox').innerHTML = movers.length ? `
      <table><tbody>
        ${movers.map((q) => `
          <tr class="clickable" onclick="setActiveSymbol('${q.symbol}');window.location.href='analysis.html'">
            <td class="sym-cell">${q.symbol}</td>
            <td class="text-xs text-muted">${(q.name || '').slice(0, 22)}</td>
            <td class="t-right" data-animate="${q.price}" data-format="price"
                data-live-symbol="${q.symbol}" data-live-field="price">${fmt.price(q.price)}</td>
            <td class="t-right ${fmt.cls(q.change_percent)}" style="font-weight:650"
                data-live-symbol="${q.symbol}" data-live-field="change_percent">
              <span class="arrow">${fmt.arrow(q.change_percent)}</span>
              <span data-animate="${q.change_percent}" data-format="pct">${fmt.pct(q.change_percent)}</span></td>
          </tr>`).join('')}
      </tbody></table>` : '<div class="empty">No movers</div>';
    animateContainer('moversBox', { stagger: 45 });
  } catch (err) {
    ui.error('watchlistTable', `Failed to load overview: ${err.message}`);
  }
}

async function loadHeatmap() {
  const node = ui.el('heatmapChart');
  ui.loading(node, 'Building heatmap…');
  try {
    const data = await api.heatmap(heatmapClass, '1mo', 24);
    if (!data.cells.length) { ui.empty(node, 'No data'); return; }
    const cells = data.cells;
    const cols = Math.ceil(Math.sqrt(cells.length));
    const rows = Math.ceil(cells.length / cols);
    const z = [], text = [], hover = [];
    for (let r = 0; r < rows; r += 1) {
      const zRow = [], tRow = [], hRow = [];
      for (let c = 0; c < cols; c += 1) {
        const cell = cells[r * cols + c];
        if (cell) {
          zRow.push(cell.change_pct);
          tRow.push(`<b>${cell.symbol}</b><br>${cell.change_pct > 0 ? '+' : ''}${cell.change_pct}%`);
          hRow.push(`${cell.name}<br>Change: ${cell.change_pct}%<br>Vol: ${cell.volatility_pct}%<br>Price: ${cell.last_price}`);
        } else { zRow.push(null); tRow.push(''); hRow.push(''); }
      }
      z.push(zRow); text.push(tRow); hover.push(hRow);
    }
    node.innerHTML = '';
    Plotly.newPlot(node, [{
      type: 'heatmap', z, text, hovertext: hover, hoverinfo: 'text',
      texttemplate: '%{text}',
      // Labels sit on saturated red/green cells; a fixed high-contrast colour
      // beats a theme colour that would vanish on one side of the scale.
      textfont: { size: 10, color: '#ffffff' },
      colorscale: [[0, C().red], [0.5, C().text2], [1, C().green]],
      zmid: 0, showscale: true,
      colorbar: { thickness: 9, len: 0.85, tickfont: { size: 9 }, title: { text: '%', font: { size: 9 } } },
      xgap: 3, ygap: 3,
    }], ui.plotLayout({
      height: 300, showlegend: false,
      xaxis: { visible: false }, yaxis: { visible: false, autorange: 'reversed' },
      margin: { l: 6, r: 6, t: 6, b: 6 },
    }), ui.plotConfig);
  } catch (err) {
    ui.error(node, `Heatmap failed: ${err.message}`);
  }
}

async function runScreener() {
  const btn = ui.el('screenBtn');
  btn.disabled = true; btn.textContent = 'Analysing…';
  ui.loading('screenerBox', 'Fusing signals across the watchlist…');
  try {
    const data = await api.screen(getWatchlist().slice(0, 8));
    const rows = data.results.filter((r) => !r.error);
    ui.el('screenerBox').innerHTML = rows.length ? `
      <table>
        <thead><tr><th>Symbol</th><th>Action</th><th class="t-right">Score</th>
          <th class="t-right">Conf.</th><th>Risk</th><th class="t-right">Target Wt.</th></tr></thead>
        <tbody>${rows.map((r) => `
          <tr class="clickable" onclick="setActiveSymbol('${r.symbol}');window.location.href='signals.html'">
            <td class="sym-cell">${r.symbol}</td>
            <td>${ui.actionBadge(r.action)}</td>
            <td class="t-right ${fmt.cls(r.score)}">${fmt.num(r.score, 3)}</td>
            <td class="t-right">${fmt.pct(r.confidence * 100, 0, false)}</td>
            <td>${ui.riskBadge(r.risk_level)}</td>
            <td class="t-right">${fmt.pct(r.suggested_weight * 100, 1, false)}</td>
          </tr>`).join('')}</tbody>
      </table>
      <div class="text-xs text-muted mt-2">
        Scores combine technical consensus, news sentiment, and — where models have been trained —
        deep-learning forecasts and the RL agent. Train models on the Forecast/RL pages to strengthen them.
      </div>` : '<div class="empty">Screening returned no results</div>';
  } catch (err) {
    ui.error('screenerBox', `Screening failed: ${err.message}`);
  } finally {
    btn.disabled = false; btn.textContent = 'Run Screen';
  }
}

async function scanAlerts() {
  const btn = ui.el('scanAlertsBtn');
  btn.disabled = true; btn.textContent = 'Scanning…';
  ui.loading('alertsBox', 'Scanning for anomalies…');
  try {
    const data = await api.scanWatchlist(getWatchlist().slice(0, 5), ['price', 'volatility', 'signals', 'risk']);
    const flat = Object.values(data.alerts || {}).flat();
    renderAlerts('alertsBox', flat, 14);
    if (flat.length) ui.toast(`${flat.length} alerts across ${data.symbols_with_alerts} symbols`, 'success');
    else ui.toast('No alerts — market conditions normal', 'success');
  } catch (err) {
    ui.error('alertsBox', `Alert scan failed: ${err.message}`);
  } finally {
    btn.disabled = false; btn.textContent = 'Scan Now';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initTimeRange();
  initSearch((symbol) => { setActiveSymbol(symbol); loadIntelBand(); loadConfidence(); });
  loadHealth();
  loadOverview();
  loadHeatmap();
  loadIntelBand();
  loadConfidence();

  ui.el('refreshBtn').addEventListener('click', () => {
    loadOverview(); loadHeatmap(); loadIntelBand(); loadConfidence();
  });
  ui.el('screenBtn').addEventListener('click', runScreener);
  ui.el('scanAlertsBtn').addEventListener('click', scanAlerts);

  document.querySelectorAll('#heatmapFilters .chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('#heatmapFilters .chip').forEach((c) => c.classList.remove('active'));
      chip.classList.add('active');
      heatmapClass = chip.dataset.class;
      loadHeatmap();
    });
  });

  // One window for the whole dashboard.
  onTimeRangeChange(() => { loadOverview(); loadIntelBand(); loadConfidence(); });
});

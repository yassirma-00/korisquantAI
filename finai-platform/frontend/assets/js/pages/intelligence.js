/* ============================================================
   Instrument Intelligence — full performance dossier + strategy benchmarks
   ============================================================ */

const STRATEGY_COLOURS = {
  buy_and_hold: C().text2,
  ma_crossover_20_50: C().amber,
  momentum_63d: C().purple,
  rsi_mean_reversion: C().cyan,
  rl_agent: C().green,
};

function iValues() {
  return {
    symbol: ui.el('iSymbol').value.trim().toUpperCase(),
    // Was a local #iPeriod dropdown that the global control never touched,
    // so this panel analysed a different window from the charts beside it.
    period: getTimeRange(),
    benchmark: ui.el('iBenchmark').value.trim().toUpperCase(),
    capital: parseFloat(ui.el('iCapital').value) || 100000,
    includeAgent: ui.el('iAgent').value === 'true',
  };
}

async function runIntelligence() {
  const v = iValues();
  const btn = ui.el('iRunBtn');
  btn.disabled = true; btn.textContent = 'Analysing…';
  ui.loading('iKpis', 'Computing full performance dossier…');
  ui.loading('iStrategyChart', 'Backtesting reference strategies…');

  try {
    // Fails loudly with a fix instruction if a stale api.js is cached
    requireApi('portfolioAnalytics', 'strategyBenchmarks');
    const d = await api.portfolioAnalytics(v.symbol, v);
    const m = d.metrics || {};
    const risk = d.risk_exposure || {};

    // ---- KPI tiles ----------------------------------------------------
    const kpi = (label, value, sub, cls = '', anim = null, fmtKind = 'num', dec = 2) => `
      <div class="card animate-in">
        <div class="stat-label">${label}</div>
        <div class="stat-value ${cls}" style="font-size:20px"
          ${anim !== null ? `data-animate="${anim}" data-format="${fmtKind}" data-decimals="${dec}"
             data-signed="${fmtKind === 'pct' && cls ? 'true' : 'false'}"` : ''}>${value}</div>
        <div class="text-xs text-muted mt-1">${sub}</div>
      </div>`;

    ui.el('iKpis').innerHTML = [
      kpi('Total Return', fmt.pct((m.total_return || 0) * 100), `${d.symbol} · ${d.period}`,
          fmt.cls(m.total_return), (m.total_return || 0) * 100, 'pct'),
      kpi('Annualised', fmt.pct((m.annualised_return || 0) * 100), 'CAGR',
          fmt.cls(m.annualised_return), (m.annualised_return || 0) * 100, 'pct'),
      kpi('Volatility', fmt.pct((m.annualised_volatility || 0) * 100, 2, false), 'annualised',
          '', (m.annualised_volatility || 0) * 100, 'pct'),
      kpi('Sharpe', fmt.num(m.sharpe_ratio, 2), 'risk-adjusted return',
          fmt.cls(m.sharpe_ratio), m.sharpe_ratio, 'num'),
      kpi('Sortino', fmt.num(m.sortino_ratio, 2), 'downside-adjusted',
          fmt.cls(m.sortino_ratio), m.sortino_ratio, 'num'),
      kpi('Calmar', fmt.num(m.calmar_ratio, 2), 'return / max drawdown',
          fmt.cls(m.calmar_ratio), m.calmar_ratio, 'num'),
      kpi('Max Drawdown', fmt.pct((m.max_drawdown || 0) * 100), 'peak to trough',
          'down', (m.max_drawdown || 0) * 100, 'pct'),
      kpi('Daily VaR 95%', fmt.pct((m.var_95 || 0) * 100), `CVaR ${fmt.pct((m.cvar_95 || 0) * 100)}`,
          'down', (m.var_95 || 0) * 100, 'pct'),
    ].join('');
    animateContainer('iKpis');

    // ---- risk exposure ------------------------------------------------
    const riskColour = { low: 'var(--green)', moderate: 'var(--amber)',
                         high: 'var(--red)', critical: 'var(--red)' }[risk.level] || 'var(--text-1)';
    ui.el('iRisk').innerHTML = `
      <div class="t-center mb-2">
        <div class="stat-label">Exposure Level</div>
        <div style="font-size:26px;font-weight:800;color:${riskColour};text-transform:uppercase">${risk.level || '—'}</div>
        <div class="meter mt-1"><div class="meter-fill" style="width:0%;background:${riskColour}"
             data-meter="${(risk.score || 0) * 100}"></div></div>
        <div class="text-xs text-muted mt-1">score ${fmt.num(risk.score, 2)} / 1.00</div>
      </div>
      <div class="signal-row"><span class="signal-name">Volatility</span>
        <span class="mono">${fmt.pct((risk.annualised_volatility || 0) * 100, 1, false)}</span></div>
      <div class="signal-row"><span class="signal-name">Max drawdown</span>
        <span class="mono down">${fmt.pct((risk.max_drawdown || 0) * 100, 1)}</span></div>
      <div class="signal-row"><span class="signal-name">Daily VaR 95%</span>
        <span class="mono down">${fmt.pct((risk.daily_var_95 || 0) * 100, 2)}</span></div>
      <div class="signal-row"><span class="signal-name">Downside deviation</span>
        <span class="mono">${fmt.pct((risk.downside_deviation || 0) * 100, 1, false)}</span></div>
      <div class="signal-row"><span class="signal-name">Tail ratio</span>
        <span class="mono">${fmt.num(risk.tail_ratio, 2)}</span></div>
      <div class="text-xs text-muted mt-2">${risk.interpretation || ''}</div>`;
    animateContainer('iRisk');

    renderStrategies(d.strategy_comparison);
    renderRolling(d);
    renderMonthly(d);
    renderDrawdowns(d.drawdown_episodes || []);
    ui.toast(`${d.symbol}: ${d.strategy_comparison?.best_by_sharpe?.replace(/_/g, ' ')} leads on Sharpe`, 'success');
  } catch (err) {
    ui.error('iKpis', `Analysis failed: ${err.message}`);
    ui.empty('iStrategyChart', '');
  } finally {
    btn.disabled = false; btn.textContent = 'Analyse';
  }
}

function renderStrategies(comparison) {
  if (!comparison) { ui.empty('iStrategyChart', 'No comparison available'); return; }
  const node = ui.el('iStrategyChart');
  const traces = comparison.strategies
    .filter((s) => s.equity_curve?.length)
    .map((s) => ({
      type: 'scatter', mode: 'lines',
      x: s.equity_curve.map((p) => p.date), y: s.equity_curve.map((p) => p.value),
      name: s.label,
      line: { color: STRATEGY_COLOURS[s.strategy] || C().accent,
              width: s.is_agent ? 2.6 : 1.7, dash: s.is_agent ? undefined : undefined },
    }));
  node.innerHTML = '';
  if (!traces.length) { ui.empty(node, 'No equity curves'); return; }
  Plotly.newPlot(node, traces, ui.plotLayout({
    height: 420, yaxis: { gridcolor: C().grid, title: { text: 'Portfolio value ($)', font: { size: 10 } } },
  }), ui.plotConfig);

  const rows = [...comparison.strategies].sort((a, b) => (b.sharpe_ratio ?? -99) - (a.sharpe_ratio ?? -99));
  ui.el('iStrategyTable').innerHTML = `
    <table>
      <thead><tr><th>Strategy</th><th class="t-right">Final Value</th><th class="t-right">Return</th>
        <th class="t-right">Ann.</th><th class="t-right">Vol</th><th class="t-right">Sharpe</th>
        <th class="t-right">Sortino</th><th class="t-right">Calmar</th>
        <th class="t-right">Max DD</th><th class="t-right">Trades</th></tr></thead>
      <tbody>${rows.map((s, i) => `
        <tr${s.is_agent ? ' style="background:var(--accent-soft)"' : ''}>
          <td class="sym-cell">${i === 0 ? '🏆 ' : ''}${s.label}</td>
          <td class="t-right">${fmt.money(s.final_value)}</td>
          <td class="t-right ${fmt.cls(s.total_return)}">${fmt.pct((s.total_return || 0) * 100)}</td>
          <td class="t-right ${fmt.cls(s.annualised_return)}">${fmt.pct((s.annualised_return || 0) * 100)}</td>
          <td class="t-right text-muted">${fmt.pct((s.volatility || 0) * 100, 1, false)}</td>
          <td class="t-right ${fmt.cls(s.sharpe_ratio)}" style="font-weight:650">${fmt.num(s.sharpe_ratio, 2)}</td>
          <td class="t-right">${fmt.num(s.sortino_ratio, 2)}</td>
          <td class="t-right">${fmt.num(s.calmar_ratio, 2)}</td>
          <td class="t-right down">${fmt.pct((s.max_drawdown || 0) * 100, 1)}</td>
          <td class="t-right text-muted">${s.n_trades ?? '—'}</td>
        </tr>`).join('')}</tbody>
    </table>
    ${comparison.verdict ? `
      <div class="${comparison.verdict.agent_beats_buy_and_hold ? 'info-box' : 'error-box'} mt-2">
        <strong>Agent vs Buy &amp; Hold:</strong> ${fmt.pct((comparison.verdict.alpha_vs_buy_and_hold || 0) * 100)} alpha.
        ${comparison.verdict.note}
      </div>` : ''}
    <div class="text-xs text-muted mt-2">
      All strategies pay identical costs (${fmt.pct((comparison.cost_model?.transaction_cost || 0) * 100, 2, false)}
      per trade + slippage). Signals act on the next bar, so no look-ahead.
    </div>`;
}

function renderRolling(d) {
  const rolling = d.rolling_sharpe || [];
  // A rolling statistic needs several windows to be a curve. With one or two
  // points Plotly falls back to a millisecond-scale time axis ("23:59:59.999"),
  // which looks like a rendering fault rather than "the window is too short".
  if (rolling.length < 5) {
    ui.empty('iRollingChart',
      'Rolling Sharpe needs a longer window — select 3M or more.');
    return;
  }
  renderLineChart('iRollingChart', [{
    x: rolling.map((p) => p.date), y: rolling.map((p) => p.value),
    name: 'Rolling Sharpe', color: C().accent,
  }], { height: 260, showlegend: false, yTitle: 'Sharpe' });
}

function renderMonthly(d) {
  const monthly = d.monthly_returns || [];
  if (!monthly.length) { ui.empty('iMonthlyChart', 'Not enough history'); return; }
  const node = ui.el('iMonthlyChart');
  node.innerHTML = '';
  Plotly.newPlot(node, [{
    type: 'bar', x: monthly.map((p) => p.month), y: monthly.map((p) => p.return * 100),
    marker: { color: monthly.map((p) => (p.return >= 0 ? 'rgba(34, 217, 138,.75)' : 'rgba(255, 77, 106,.75)')) },
    name: 'Monthly return',
  }], ui.plotLayout({
    height: 260, showlegend: false,
    yaxis: { gridcolor: C().grid, title: { text: '%', font: { size: 10 } } },
  }), ui.plotConfig);
}

function renderDrawdowns(episodes) {
  // The Start / Trough / Recovered columns were calendar dates. What actually
  // informs a decision is how deep each episode went, how long it lasted, and
  // whether it is over — those survive; the dates do not. Ranking is preserved
  // by ordering, so "the worst one" is still the first row.
  ui.el('iDrawdowns').innerHTML = episodes.length ? `
    <table>
      <thead><tr><th>#</th><th>Status</th>
        <th class="t-right">Depth</th><th class="t-right">Duration</th></tr></thead>
      <tbody>${episodes.map((e, i) => `
        <tr><td class="text-muted">${i + 1}</td>
          <td>${e.recovered ? '<span class="badge badge-green">recovered</span>'
            : '<span class="badge badge-amber">ongoing</span>'}</td>
          <td class="t-right down" style="font-weight:650">${fmt.pct(e.depth * 100, 2)}</td>
          <td class="t-right text-muted">${e.duration_days !== null && e.duration_days !== undefined
            ? e.duration_days + 'd' : '—'}</td>
        </tr>`).join('')}</tbody>
    </table>
    <div class="text-xs text-muted mt-2">
      Recovery time matters as much as depth: a 30% drawdown needs a 43% gain to break even.
    </div>` : '<div class="empty">No drawdown beyond -5% in this period</div>';
}

document.addEventListener('DOMContentLoaded', () => {
  if (!ui.el('iSymbol')) return;          // only on the portfolio page
  ui.el('iSymbol').value = getActiveSymbol();
  new SymbolPicker('iSymbol', 'iSymbolPanel', () => runIntelligence());
  // The benchmark was a bare text box: a typo there silently compared the
  // strategy against nothing. syncActive is off — picking a benchmark must not
  // retarget the whole page to it.
  new SymbolPicker('iBenchmark', 'iBenchmarkPanel', null, { syncActive: false });
  ui.el('iRunBtn').addEventListener('click', runIntelligence);
  runIntelligence();

  // Follow the global range like every other panel on this page.
  onTimeRangeChange(() => runIntelligence());
});

/* ============================================================
   Page: Technical Analysis
   ============================================================ */

/* The window comes from the global control now; the old per-page chip row
   offered a different subset from every other page. */
let currentPeriod = getTimeRange();

async function loadAnalysis(symbol) {
  symbol = symbol || getActiveSymbol();
  ui.el('activeSymbolBadge').textContent = symbol;
  ui.el('chartTitle').textContent = `${symbol} — Price & Overlays`;
  ui.loading('priceChart', 'Loading market data…');
  ui.loading('signalsBox');

  try {
    const data = await api.symbolDashboard(symbol, currentPeriod,
      'sma,ema,rsi,macd,bbands,atr,adx,stoch,mfi,vwap');
    const { quote, candles, signals, statistics, profile } = data;

    ui.el('sourceBadge').innerHTML = ui.sourceBadge(profile.data_source, profile.is_live);

    const c = fmt.cls(quote.change_percent);
    ui.el('statCards').innerHTML = `
      <div class="card animate-in">
        <div class="stat-label">${profile.name}</div>
        <div class="stat-value" data-animate="${quote.price}" data-format="price"
             data-live-symbol="${quote.symbol}" data-live-field="price">${fmt.price(quote.price)}</div>
        <div class="stat-change ${c}">${fmt.arrow(quote.change_percent)} ${fmt.price(quote.change)} (${fmt.pct(quote.change_percent)})</div>
      </div>
      <div class="card animate-in">
        <div class="stat-label">Annualised Volatility</div>
        <div class="stat-value" data-animate="${(statistics.annualised_volatility || 0) * 100}"
             data-format="pct" data-decimals="1" data-signed="false">${fmt.pct((statistics.annualised_volatility || 0) * 100, 1, false)}</div>
        <div class="text-xs text-muted">Sharpe ${fmt.num(statistics.sharpe_ratio, 2)}</div>
      </div>
      <div class="card animate-in">
        <div class="stat-label">Max Drawdown</div>
        <div class="stat-value down" data-animate="${(statistics.max_drawdown || 0) * 100}"
             data-format="pct" data-decimals="1">${fmt.pct((statistics.max_drawdown || 0) * 100, 1)}</div>
        <div class="text-xs text-muted">Current ${fmt.pct((statistics.current_drawdown || 0) * 100, 1)}</div>
      </div>
      <div class="card animate-in">
        <div class="stat-label">Risk Level</div>
        <div style="margin-top:6px">${ui.riskBadge(data.risk.overall_risk_level)}</div>
        <div class="text-xs text-muted mt-1">${data.risk.n_anomalies} anomalies · crash ${fmt.num(data.risk.crash_risk?.crash_risk_score, 2)}</div>
      </div>`;

    animateContainer('statCards');
    liveTicker.track([symbol]).start();
    renderCandlestick('priceChart', candles, { symbol, height: 420 });
    renderVolume('volumeChart', candles);
    renderIndicatorPanel('rsiChart', candles);
    renderMACD('macdChart', candles);
    renderSignals('signalsBox', signals);
    renderMetricsGrid('statsGrid', statistics, METRIC_SPEC);

    const news = await api.news(symbol, 10);
    renderNews('newsBox', news.news, 10);
  } catch (err) {
    ui.error('priceChart', `Failed to load ${symbol}: ${err.message}`);
  }
}

/* The correlation basket, owned by the shared picker rather than a
   comma-separated string that accepted typos in silence. */
let corrPicker = null;

function renderCorrChips() {
  const picked = corrPicker ? corrPicker.selected : [];
  const box = ui.el('corrSymbolsSummary');
  if (!box) return;
  box.innerHTML = picked.map((s) => `
    <span class="sp-chip" data-drop="${s}">${s}<i aria-hidden="true">×</i></span>`).join('');
  box.querySelectorAll('[data-drop]').forEach((chip) => {
    chip.style.cursor = 'pointer';
    chip.addEventListener('click', () => corrPicker.deselect(chip.dataset.drop));
  });
}

async function loadCorrelation() {
  const symbols = corrPicker ? [...corrPicker.selected] : [];
  if (symbols.length < 2) {
    ui.toast('Select at least 2 assets to correlate', 'error');
    return;
  }
  const node = ui.el('corrChart');
  ui.loading(node, 'Computing correlations…');
  try {
    const data = await api.correlation(symbols, currentPeriod);
    node.innerHTML = '';
    Plotly.newPlot(node, [{
      type: 'heatmap', z: data.matrix, x: data.symbols, y: data.symbols,
      colorscale: [[0, C().red], [0.5, C().bg3], [1, C().green]], zmid: 0, zmin: -1, zmax: 1,
      text: data.matrix.map((row) => row.map((v) => v.toFixed(2))),
      texttemplate: '%{text}', textfont: { size: 10 },
      colorbar: { thickness: 9, len: 0.9, tickfont: { size: 9 } }, xgap: 2, ygap: 2,
    }], ui.plotLayout({
      height: 300, showlegend: false, margin: { l: 78, r: 10, t: 10, b: 62 },
      xaxis: { tickangle: -40 }, yaxis: { autorange: 'reversed' },
    }), ui.plotConfig);
    ui.toast(`Average correlation ${data.average_correlation}`, 'success');
  } catch (err) {
    ui.error(node, `Correlation failed: ${err.message}`);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initTimeRange();
  initSearch((symbol) => loadAnalysis(symbol));
  loadAnalysis();
  corrPicker = new SymbolPicker('corrSymbols', 'corrSymbolsPanel', null, {
    multi: true,
    selected: ['AAPL', 'MSFT', 'NVDA', 'SPY', 'BTC-USD'],
    onChange: renderCorrChips,
  });
  renderCorrChips();
  ui.el('corrBtn').addEventListener('click', loadCorrelation);
  loadCorrelation();
  onTimeRangeChange((key) => {
    currentPeriod = key;
    loadAnalysis();
    loadCorrelation();
  });
});

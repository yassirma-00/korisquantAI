/* ============================================================
   Page: Explainable AI
   ============================================================ */

async function explain() {
  const symbol = ui.el('xSymbol').value.trim().toUpperCase();
  const period = getTimeRange();
  const horizon = parseInt(ui.el('xHorizon').value, 10);
  const btn = ui.el('explainBtn');
  btn.disabled = true; btn.textContent = 'Explaining…';
  ui.el('activeSymbolBadge').textContent = symbol;
  ui.loading('xShapBars', 'Computing Shapley values…');
  ui.loading('xLimeBars', 'Fitting local surrogate…');
  ui.loading('xGlobalChart', 'Ranking features…');

  try {
    const data = await api.explain(symbol, 'shap,lime,global', horizon);

    if (data.shap) {
      ui.el('shapMethod').textContent = data.shap.method;
      renderXaiBars('xShapBars', data.shap.feature_importance, 10);
      ui.el('xShapNarrative').textContent = data.shap.narrative;
    }
    if (data.lime) {
      ui.el('limeR2').textContent = `local R² ${fmt.num(data.lime.details?.local_r2, 3)}`;
      renderXaiBars('xLimeBars', data.lime.feature_importance, 10);
      ui.el('xLimeNarrative').textContent = data.lime.narrative;
    }
    if (data.global) {
      const items = data.global.permutation_importance.slice(0, 12).reverse();
      const node = ui.el('xGlobalChart');
      node.innerHTML = '';
      Plotly.newPlot(node, [{
        type: 'bar', orientation: 'h',
        y: items.map((d) => d.label), x: items.map((d) => d.importance),
        error_x: { type: 'data', array: items.map((d) => d.std), color: 'rgba(168,184,208,.45)', thickness: 1 },
        marker: { color: C().accent },
      }], ui.plotLayout({
        height: 340, showlegend: false, margin: { l: 150, r: 20, t: 10, b: 34 },
        xaxis: { gridcolor: C().grid, title: { text: 'permutation importance', font: { size: 10 } } },
        yaxis: { gridcolor: 'rgba(0,0,0,0)' }, hovermode: 'closest',
      }), ui.plotConfig);
      ui.el('xGlobalNarrative').textContent = data.global.narrative;
    }
    ui.toast('Explanation generated', 'success');
  } catch (err) {
    ui.error('xShapBars', err.message);
    ui.el('xLimeBars').innerHTML = '';
    ui.el('xGlobalChart').innerHTML = '';
  } finally {
    btn.disabled = false; btn.textContent = 'Explain';
  }
}

async function runCounterfactual() {
  const symbol = ui.el('xSymbol').value.trim().toUpperCase();
  const btn = ui.el('cfBtn');
  btn.disabled = true; btn.textContent = '…';
  ui.loading('xCounterfactual', 'Searching for the minimal flip…');
  try {
    const cf = await api.counterfactual(symbol);
    ui.el('xCounterfactual').innerHTML = `
      <div class="signal-row">
        <span class="signal-name">Current forecast</span>
        <span class="${cf.original_direction === 'up' ? 'up' : 'down'}" style="font-weight:650">
          ${cf.original_direction.toUpperCase()} (${fmt.pct(cf.original_prediction * 100, 3)})</span>
      </div>
      ${cf.counterfactuals.length ? `
        <table class="mt-1">
          <thead><tr><th>Feature</th><th class="t-right">Now</th><th class="t-right">Needs</th><th class="t-right">Δσ</th></tr></thead>
          <tbody>${cf.counterfactuals.map((c) => `
            <tr><td>${c.label}</td>
              <td class="t-right mono">${fmt.num(c.current_value, 4)}</td>
              <td class="t-right mono">${fmt.num(c.required_value, 4)}</td>
              <td class="t-right ${c.change_in_sigma > 0 ? 'up' : 'down'}">${c.change_in_sigma > 0 ? '+' : ''}${fmt.num(c.change_in_sigma, 1)}σ</td>
            </tr>`).join('')}</tbody>
        </table>` : ''}
      <div class="text-sm text-muted mt-2">${cf.narrative}</div>`;
  } catch (err) {
    ui.error('xCounterfactual', err.message);
  } finally {
    btn.disabled = false; btn.textContent = 'Run';
  }
}

async function analyzeText() {
  const text = ui.el('xText').value.trim();
  if (!text) { ui.toast('Enter some text', 'error'); return; }
  const btn = ui.el('analyzeTextBtn');
  btn.disabled = true; btn.textContent = '…';
  ui.loading('xTextResult', 'Analysing…');
  try {
    const r = await api.analyzeText(text);
    const colour = r.label === 'positive' ? 'var(--green)' : r.label === 'negative' ? 'var(--red)' : 'var(--text-1)';
    ui.el('xTextResult').innerHTML = `
      <div class="grid grid-4">
        <div class="card"><div class="stat-label">Sentiment</div>
          <div class="stat-value" style="font-size:20px;color:${colour};text-transform:capitalize">${r.label}</div></div>
        <div class="card"><div class="stat-label">Polarity</div>
          <div class="stat-value ${fmt.cls(r.score)}" style="font-size:20px">${fmt.num(r.score, 3)}</div></div>
        <div class="card"><div class="stat-label">Confidence</div>
          <div class="stat-value" style="font-size:20px">${fmt.pct(r.confidence * 100, 0, false)}</div></div>
        <div class="card"><div class="stat-label">Category</div>
          <div class="stat-value" style="font-size:15px;text-transform:capitalize">${(r.category || '').replace(/_/g, ' ')}</div>
          <div class="text-xs text-muted mt-1">engine: ${r.backend}</div></div>
      </div>
      ${r.keywords?.length ? `<div class="mt-2"><span class="stat-label">Trigger words: </span>
        ${r.keywords.map((k) => `<span class="badge badge-blue" style="margin:2px">${k}</span>`).join('')}</div>` : ''}
      <div class="mt-2">
        <div class="stat-label mb-1">Class probabilities</div>
        ${Object.entries(r.scores).map(([label, p]) => `
          <div class="flex items-center gap-1 mb-1">
            <div style="width:70px;font-size:12px;text-transform:capitalize">${label}</div>
            <div class="meter" style="flex:1;margin:0">
              <div class="meter-fill" style="width:${p * 100}%;background:${
                label === 'positive' ? 'var(--green)' : label === 'negative' ? 'var(--red)' : 'var(--text-2)'}"></div>
            </div>
            <div class="mono text-xs" style="width:48px;text-align:right">${fmt.pct(p * 100, 1, false)}</div>
          </div>`).join('')}
      </div>`;
  } catch (err) {
    ui.error('xTextResult', err.message);
  } finally {
    btn.disabled = false; btn.textContent = 'Analyse';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initTimeRange();
  initSearch((symbol) => { ui.el('xSymbol').value = symbol; explain(); });
  new SymbolPicker('xSymbol', 'xSymbolPanel', () => explain());
  ui.el('xSymbol').value = getActiveSymbol();
  ui.el('activeSymbolBadge').textContent = getActiveSymbol();
  ui.el('explainBtn').addEventListener('click', explain);
  ui.el('cfBtn').addEventListener('click', runCounterfactual);
  ui.el('analyzeTextBtn').addEventListener('click', analyzeText);
  explain();

  // The global time range drives this page; there is no local
  // period control any more.
  onTimeRangeChange(() => { explain() });
});

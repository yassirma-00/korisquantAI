/* ============================================================
   Page: AI Recommendations
   ============================================================ */


/* ==================== populate the model / algorithm selectors ==========
   Both lists are built from the live API rather than hard-coded markup, so a
   new architecture or RL algorithm appears here automatically instead of
   silently going missing from this page.
   ======================================================================== */
let RL_SPECS = [];

const FAMILY_LABEL = {
  value_based: 'Value-based',
  distributional: 'Distributional',
  policy_gradient: 'Policy gradient',
  actor_critic: 'Actor-critic',
};

async function populateForecastModels() {
  const select = ui.el('sModel');
  try {
    const data = await api.forecastModels();
    const current = select.value;
    select.innerHTML = data.models
      .map((m) => `<option value="${m.key}" title="${m.description}">${m.name}</option>`)
      .join('');
    if ([...select.options].some((o) => o.value === current)) select.value = current;
    ui.el('sModelCount').textContent = `(${data.models.length})`;
  } catch (err) {
    console.warn('forecast model list unavailable', err);
  }
}

async function populateRlAlgorithms() {
  const select = ui.el('sRlAlgo');
  try {
    const data = await api.algorithms();
    // Only offer what this install can actually run
    RL_SPECS = data.algorithms.filter((a) => a.available);

    const groups = {};
    RL_SPECS.forEach((a) => (groups[a.family] ||= []).push(a));

    const current = select.value;
    select.innerHTML = Object.entries(groups).map(([family, items]) => `
      <optgroup label="${FAMILY_LABEL[family] || family}">
        ${items.map((a) => {
          // Continuous agents run here as a single-asset allocation, which is
          // worth signalling in the label so the choice is not surprising.
          const suffix = a.action_space === 'continuous' ? ' — allocation' : '';
          return `<option value="${a.key}" title="${a.best_for}">${a.name}${suffix}</option>`;
        }).join('')}
      </optgroup>`).join('');

    if ([...select.options].some((o) => o.value === current)) select.value = current;
    ui.el('sAlgoCount').textContent = `(${RL_SPECS.length})`;
    describeSelection();
  } catch (err) {
    console.warn('RL algorithm list unavailable', err);
  }
}

/** One-line reminder of what the two selected models actually do. */
function describeSelection() {
  const node = ui.el('selectionNote');
  if (!node) return;
  const algo = RL_SPECS.find((a) => a.key === ui.el('sRlAlgo').value);
  const model = ui.el('sModel').selectedOptions[0]?.textContent || '—';
  if (!algo) { node.textContent = ''; return; }
  const continuous = algo.action_space === 'continuous';
  node.innerHTML = `
    <strong>${model}</strong> forecasts the ${ui.el('sHorizon').value}-day return ·
    <strong>${algo.name}</strong> (${FAMILY_LABEL[algo.family] || algo.family})
    ${continuous
      ? 'targets an exposure level for this instrument, which is mapped to BUY/HOLD/SELL'
      : 'emits BUY, HOLD or SELL directly'}.
    Both must be trained for this symbol before they contribute.`;
}

const SOURCE_LABELS = {
  forecast: 'Deep-Learning Forecast',
  rl: 'RL Agent',
  technical: 'Technical Indicators',
  sentiment: 'News Sentiment',
};

/* Which request is the current one. Compared after the await so a slow reply
   for a period the user has already left cannot overwrite a newer result. */
let currentRequestSymbol = null;

async function generate() {
  const symbol = ui.el('sSymbol').value.trim().toUpperCase();
  currentRequestSymbol = symbol;
  const btn = ui.el('genBtn');
  btn.disabled = true; btn.textContent = 'Analysing…';
  ui.el('activeSymbolBadge').textContent = symbol;
  ui.el('recoHero').innerHTML = '<div class="card"><div class="loading"><div class="spinner"></div>Fusing forecast, RL, technical and sentiment signals…</div></div>';

  try {
    // The period was never sent, so `api.recommend` fell back to its '2y'
    // default and every selection — 1M, 6M, 10Y — asked the backend for the
    // same two years. The recommendation looked frozen across periods because
    // it genuinely was the same request. The backend already honours `period`;
    // only the caller was dropping it.
    const period = getTimeRange();
    const reco = await api.recommend(symbol, {
      period,
      model: ui.el('sModel').value,
      rlAlgo: ui.el('sRlAlgo').value,
      horizon: parseInt(ui.el('sHorizon').value, 10),
      includeXai: ui.el('sXai').value === 'true',
    });

    // A slow request for an old period must not overwrite a newer one. Without
    // this, switching 1M -> 5Y quickly can leave the 1M response rendering
    // last and the panel showing a period the user is no longer on.
    if (period !== getTimeRange() || symbol !== currentRequestSymbol) return;

    const colour = reco.action.includes('BUY') ? 'var(--green)'
      : reco.action.includes('SELL') ? 'var(--red)' : 'var(--text-1)';
    // The hero paints a semantic wash and rail behind itself so the call is
    // recognisable before a word is read. Purely presentational: the class is
    // derived from the same `action` string already shown as text.
    const tone = reco.action.includes('BUY') ? 'reco-buy'
      : reco.action.includes('SELL') ? 'reco-sell' : 'reco-hold';

    ui.el('recoHero').innerHTML = `
      <div class="reco-hero ${tone}">
        <div class="reco-gauge-wrap">
          ${ui.gauge(reco.confidence, 116)}
          <div class="reco-gauge-read">
            <div>
              <div class="reco-gauge-pct">${fmt.pct(reco.confidence * 100, 0, false)}</div>
              <div class="reco-gauge-cap">confidence</div>
            </div>
          </div>
        </div>
        <div style="flex:1;min-width:0">
          <div class="reco-instrument">${reco.name} · ${reco.asset_class}</div>
          <div class="reco-action" style="color:${colour}">${reco.action.replace('_', ' ')}</div>
          <div class="reco-badges">
            ${ui.sourceBadge(reco.data_source)}
            ${ui.riskBadge(reco.risk.overall_level)}
            <span class="badge badge-blue">score ${fmt.num(reco.composite_score, 3)}</span>
            <span class="badge badge-grey">agreement ${fmt.pct(reco.signal_agreement * 100, 0, false)}</span>
          </div>
        </div>
        <div class="reco-price">
          <div class="stat-label">Last price</div>
          <div class="stat-value">${fmt.price(reco.last_price)}</div>
          <div class="reco-price-meta">raw ${fmt.num(reco.raw_score, 3)} · risk adj ${fmt.num(reco.risk_adjustment, 3)}</div>
        </div>
      </div>`;

    // ---- contribution chart
    const available = reco.signals.filter((s) => s.available);
    const node = ui.el('signalsChart');
    node.innerHTML = '';
    Plotly.newPlot(node, [{
      type: 'bar', orientation: 'h',
      y: reco.signals.map((s) => SOURCE_LABELS[s.source] || s.source),
      x: reco.signals.map((s) => s.weighted_contribution),
      marker: { color: reco.signals.map((s) => (s.weighted_contribution > 0 ? C().green : s.weighted_contribution < 0 ? C().red : C().border)) },
      text: reco.signals.map((s) => (s.available ? s.weighted_contribution.toFixed(3) : 'unavailable')),
      textposition: 'auto', hoverinfo: 'y+x',
    }], ui.plotLayout({
      height: 220, showlegend: false, margin: { l: 150, r: 20, t: 10, b: 34 },
      xaxis: { gridcolor: C().grid, zerolinecolor: C().accent, title: { text: 'weighted contribution', font: { size: 10 } } },
      yaxis: { gridcolor: 'rgba(0,0,0,0)' }, hovermode: 'closest',
    }), ui.plotConfig);

    ui.el('signalsDetail').innerHTML = `
      <table>
        <thead><tr><th>Signal</th><th class="t-right">Score</th><th class="t-right">Weight</th>
          <th class="t-right">Reliability</th><th>Detail</th></tr></thead>
        <tbody>${reco.signals.map((s) => {
          const d = s.detail || {};
          let detail = '—';
          if (!s.available) detail = `<span class="text-muted text-xs">${d.reason || 'not trained'}</span>`;
          else if (s.source === 'forecast') detail = `${fmt.pct(d.predicted_return * 100)} over ${d.horizon_days}d · ${fmt.num(d.directional_accuracy, 0)}% acc`;
          else if (s.source === 'rl') detail = `${d.action} · Sharpe ${fmt.num(d.agent_sharpe, 2)}`;
          else if (s.source === 'technical') detail = `${d.consensus} · ${d.buy_votes} buy / ${d.sell_votes} sell`;
          else if (s.source === 'sentiment') detail = `${d.label} · ${d.n_articles} articles (${d.backend})`;
          return `<tr>
            <td class="sym-cell">${SOURCE_LABELS[s.source]}</td>
            <td class="t-right ${fmt.cls(s.score)}">${fmt.num(s.score, 3)}</td>
            <td class="t-right text-muted">${fmt.num(s.weight, 2)}</td>
            <td class="t-right">${fmt.pct(s.reliability * 100, 0, false)}</td>
            <td class="text-sm">${detail}</td>
          </tr>`;
        }).join('')}</tbody>
      </table>`;

    // ---- sizing
    const z = reco.position_sizing;
    ui.el('sizingBox').innerHTML = `
      <div class="sizing-headline">
        <div class="stat-label">${z.direction === 'reduce' ? 'Suggested Reduction' : 'Suggested Allocation'}</div>
        <div class="stat-value ${z.direction === 'reduce' ? 'down' : ''}">${
          z.direction === 'reduce'
            ? fmt.pct((z.suggested_trim_fraction || 0) * 100, 0, false)
            : fmt.pct(z.suggested_portfolio_weight * 100, 1, false)}</div>
        <div class="text-xs text-muted">${z.direction === 'reduce'
          ? 'of any existing position (long-only: no shorting)'
          : 'of total portfolio value'}</div>
      </div>
      <div class="signal-row"><span class="signal-name">Conviction</span>
        <span class="mono">${fmt.pct((z.conviction || 0) * 100, 0, false)}</span></div>
      <div class="signal-row"><span class="signal-name">Volatility target</span><span class="mono">${fmt.pct(z.volatility_target * 100, 0, false)}</span></div>
      <div class="signal-row"><span class="signal-name">Asset volatility</span><span class="mono">${fmt.pct(z.asset_volatility * 100, 1, false)}</span></div>
      <div class="signal-row"><span class="signal-name">Stop loss</span><span class="mono down">-${fmt.num(z.stop_loss_pct, 1)}%</span></div>
      <div class="signal-row"><span class="signal-name">Take profit</span><span class="mono up">+${fmt.num(z.take_profit_pct, 1)}%</span></div>
      <div class="signal-row"><span class="signal-name">Max cap</span><span class="mono">${fmt.pct(z.max_weight_cap * 100, 0, false)}</span></div>
      <div class="text-xs text-muted mt-2">${z.rationale}</div>`;

    // ---- narrative
    const summaryHtml = reco.explanation.summary.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    ui.el('narrativeBox').innerHTML = `
      <div class="text-sm" style="line-height:1.65">${summaryHtml}</div>
      <div class="stat-label mt-2 mb-1">Key drivers (by impact)</div>
      ${reco.explanation.key_drivers.map((d) => `
        <div class="signal-row">
          <span class="signal-name">${d.label}</span>
          <span><span class="${fmt.cls(d.score)}" style="font-weight:650">${fmt.num(d.score, 3)}</span>
            <span class="text-xs text-muted"> (rel. ${fmt.pct(d.reliability * 100, 0, false)})</span></span>
        </div>`).join('')}
      ${reco.explanation.missing_signals.length ? `
        <div class="info-box mt-2 text-xs">
          Not yet contributing: ${reco.explanation.missing_signals.join(', ')}.
          Train these models on the Forecast and RL pages to sharpen the recommendation.
        </div>` : ''}
      <div class="text-xs text-muted mt-2">${reco.disclaimer}</div>`;

    // ---- SHAP
    if (reco.xai?.feature_importance) {
      renderXaiBars('shapBox', reco.xai.feature_importance, 10);
      ui.el('shapBox').insertAdjacentHTML('beforeend',
        `<div class="text-xs text-muted mt-2">${reco.xai.narrative}</div>`);
    } else {
      ui.empty('shapBox', 'Set “Include XAI” to Yes and regenerate');
    }

    ui.toast(`${reco.action.replace('_', ' ')} — ${fmt.pct(reco.confidence * 100, 0, false)} confidence`, 'success');
  } catch (err) {
    ui.el('recoHero').innerHTML = `<div class="error-box">Recommendation failed: ${err.message}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = 'Generate';
  }
}

async function screenWatchlist() {
  const btn = ui.el('sScreenBtn');
  btn.disabled = true; btn.textContent = 'Screening…';
  ui.loading('sScreenBox', 'Ranking your watchlist…');
  try {
    const data = await api.screen(getWatchlist().slice(0, 8));
    const rows = data.results.filter((r) => !r.error);
    ui.el('sScreenBox').innerHTML = `
      <table>
        <thead><tr><th>Symbol</th><th>Name</th><th>Action</th><th class="t-right">Score</th>
          <th class="t-right">Confidence</th><th>Risk</th><th class="t-right">Weight</th></tr></thead>
        <tbody>${rows.map((r) => `
          <tr class="clickable" onclick="document.getElementById('sSymbol').value='${r.symbol}';window.scrollTo({top:0,behavior:'smooth'})">
            <td class="sym-cell">${r.symbol}</td>
            <td class="text-xs text-muted">${(r.name || '').slice(0, 26)}</td>
            <td>${ui.actionBadge(r.action)}</td>
            <td class="t-right ${fmt.cls(r.score)}">${fmt.num(r.score, 3)}</td>
            <td class="t-right">${fmt.pct(r.confidence * 100, 0, false)}</td>
            <td>${ui.riskBadge(r.risk_level)}</td>
            <td class="t-right">${fmt.pct(r.suggested_weight * 100, 1, false)}</td>
          </tr>`).join('')}</tbody>
      </table>`;
  } catch (err) {
    ui.error('sScreenBox', err.message);
  } finally {
    btn.disabled = false; btn.textContent = 'Screen Watchlist';
  }
}


document.addEventListener('DOMContentLoaded', async () => {
  // AI Direction Prediction moved to its own page (direction.html) and is no
  // longer rendered here, so nothing on this page loads it any more.
  initTimeRange();
  initSearch((symbol) => { ui.el('sSymbol').value = symbol; generate(); });
  new SymbolPicker('sSymbol', 'sSymbolPanel', () => generate());
  ui.el('sSymbol').value = getActiveSymbol();
  ui.el('activeSymbolBadge').textContent = getActiveSymbol();
  ui.el('genBtn').addEventListener('click', () => generate());
  ui.el('sScreenBtn').addEventListener('click', screenWatchlist);

  // Build both dropdowns before the first run so the defaults are valid
  await Promise.all([populateForecastModels(), populateRlAlgorithms()]);
  ['sModel', 'sRlAlgo', 'sHorizon'].forEach((id) =>
    ui.el(id)?.addEventListener('change', describeSelection));
  describeSelection();

  generate();

  onTimeRangeChange(() => generate());
});

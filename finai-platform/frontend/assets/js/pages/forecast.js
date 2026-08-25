/* ============================================================
   Page: AI Forecasting
   ============================================================ */

let lastTraining = null;

/* Two windows, deliberately named apart.
 *
 * `trainPeriod` is how much history the network fits on — a hyperparameter: a
 * model trained on 5 years is a different model from one trained on 1.
 * `displayPeriod` is what the charts show, and comes from the global control.
 *
 * They were conflated: the chart fetch read the training selector, so picking
 * 1M at the top redrew nothing while quietly requesting 5y. */
function formValues() {
  return {
    symbol: ui.el('fSymbol').value.trim().toUpperCase(),
    model: ui.el('fModel').value,
    trainPeriod: ui.el('fTrainPeriod').value,
    displayPeriod: getTimeRange(),
    horizon: parseInt(ui.el('fHorizon').value, 10),
    lookback: parseInt(ui.el('fLookback').value, 10),
    epochs: parseInt(ui.el('fEpochs').value, 10),
  };
}

/* ---------------------------------------------------------- training curves */

/** Draw the loss curves plus the learning-rate schedule.
 *
 * Shared by the live training run and by the persisted history, so a reloaded
 * page shows exactly what the training session showed. */
function renderTrainingCurves(history, meta = {}) {
  const node = ui.el('lossChart');
  const train = history?.train_loss || [];
  const val = history?.val_loss || [];
  if (!train.length) {
    // An untrained model is a normal state, not a failure. Saying so beats a
    // blank rectangle that reads as broken.
    node.innerHTML = `<div class="empty">No training history available.
      Train the model to generate training curves.</div>`;
    ui.el('lossMeta').textContent = '';
    return;
  }

  const epochs = train.map((_, i) => i + 1);
  const series = [
    { x: epochs, y: train, name: 'Train loss', color: C().accent, width: 2.2 },
    { x: epochs, y: val, name: 'Validation loss', color: C().amber, width: 2.2 },
  ];
  // `history: false` keeps the range slider off: the x axis is epochs, not
  // dates, so a date-range navigator would be meaningless here.
  renderLineChart('lossChart', series,
    { height: 260, yTitle: 'Huber loss', xTitle: 'Epoch', history: false });

  const best = val.length ? Math.min(...val) : null;
  const bestEpoch = best === null ? null : val.indexOf(best) + 1;
  const improvement = train.length > 1
    ? ((train[0] - train[train.length - 1]) / Math.abs(train[0] || 1)) * 100 : 0;
  const lr = history.lr || [];
  const parts = [
    `${train.length} epoch${train.length === 1 ? '' : 's'} run`,
    best !== null ? `best validation ${fmt.num(best, 6)} at epoch ${bestEpoch}` : null,
    `training loss improved ${fmt.num(improvement, 1)}%`,
    lr.length ? `learning rate ${fmt.num(lr[0], 6)} → ${fmt.num(lr[lr.length - 1], 6)}` : null,
    meta.bars_used ? `${meta.bars_used} bars of training history` : null,
    meta.trained_at ? `trained ${fmt.timeAgo(meta.trained_at)}` : null,
  ].filter(Boolean);
  ui.el('lossMeta').textContent = parts.join(' · ');

  // Early stopping is why "8 epochs run" can follow a request for 25; without
  // this the count looks like the trainer ignored the setting.
  if (meta.epochs_requested && train.length < meta.epochs_requested) {
    ui.el('lossMeta').textContent +=
      ` · stopped early (requested ${meta.epochs_requested})`;
  }
}

/** Load the curves recorded for whichever checkpoint the form points at. */
async function loadTrainingCurves() {
  const v = formValues();
  try {
    const data = await api.trainingHistory(v.symbol, v.model, v.horizon);
    if (!data.trained) {
      renderTrainingCurves(null);
      ui.el('lossMeta').textContent = '';
      return;
    }
    renderTrainingCurves(data.history, {
      trained_at: data.trained_at,
      bars_used: data.config?.bars_used,
      epochs_requested: data.config?.epochs_requested,
    });
  } catch (err) {
    ui.error('lossChart', `Training curves unavailable: ${err.message}`);
  }
}

async function trainModel() {
  const v = formValues();
  const btn = ui.el('trainBtn');
  btn.disabled = true; btn.textContent = 'Training…';
  ui.el('trainStatus').innerHTML =
    `<div class="loading"><div class="spinner"></div>Training ${v.model.toUpperCase()} on ${v.symbol} (${v.epochs} epochs). This can take 30–90s…</div>`;
  try {
    const result = await api.trainForecast({
      symbol: v.symbol, model: v.model, period: v.trainPeriod, horizon: v.horizon,
      lookback: v.lookback, epochs: v.epochs,
    });
    lastTraining = result;
    const t = result.metrics.test || {};
    const val = result.metrics.validation || {};
    ui.el('trainStatus').innerHTML = `
      <div class="info-box">
        <strong>${v.model.toUpperCase()} trained on ${result.symbol}</strong> —
        ${result.n_parameters.toLocaleString()} parameters,
        ${result.metrics.epochs_run} epochs in ${result.metrics.train_seconds}s
        (data: ${result.data_source}, ${result.bars_used} bars).
      </div>
      <div class="grid grid-4 mt-2">
        <div class="card"><div class="stat-label">Directional Accuracy</div>
          <div class="stat-value ${t.directional_accuracy > 50 ? 'up' : 'down'}" style="font-size:20px">${fmt.num(t.directional_accuracy, 1)}%</div>
          <div class="text-xs text-muted">out-of-sample</div></div>
        <div class="card"><div class="stat-label">RMSE</div>
          <div class="stat-value" style="font-size:20px">${fmt.num(t.rmse, 5)}</div>
          <div class="text-xs text-muted">return space</div></div>
        <div class="card"><div class="stat-label">R²</div>
          <div class="stat-value ${t.r2 > 0 ? 'up' : 'down'}" style="font-size:20px">${fmt.num(t.r2, 4)}</div>
          <div class="text-xs text-muted">val ${fmt.num(val.r2, 4)}</div></div>
        <div class="card"><div class="stat-label">Test Samples</div>
          <div class="stat-value" style="font-size:20px">${t.n_samples || '—'}</div>
          <div class="text-xs text-muted">chronological split</div></div>
      </div>
      <div class="text-xs text-muted mt-1">
        Note: R² is often near zero or negative on financial returns — that is expected.
        Directional accuracy above 50% is the more meaningful signal.
      </div>`;

    renderTrainingCurves(result.history, {
      trained_at: result.trained_at,
      bars_used: result.bars_used,
      epochs_requested: v.epochs,
    });

    ui.toast(`Model trained — ${fmt.num(t.directional_accuracy, 1)}% directional accuracy`, 'success');
    await predict();
    loadTrained();
  } catch (err) {
    ui.el('trainStatus').innerHTML = `<div class="error-box">Training failed: ${err.message}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = 'Train Model';
  }
}

async function predict() {
  const v = formValues();
  const btn = ui.el('predictBtn');
  btn.disabled = true; btn.textContent = 'Predicting…';
  ui.loading('forecastChart', 'Generating forecast…');
  try {
    const [prediction, history] = await Promise.all([
      // The forecast is produced by the trained model; the chart behind it
      // shows whatever window the user asked for at the top of the page.
      api.predict(v.symbol, v.model, v.horizon, v.displayPeriod, true),
      api.history(v.symbol, v.displayPeriod, '1d', 'sma,bbands'),
    ]);

    ui.el('fSourceBadge').innerHTML = ui.sourceBadge(history.source, history.is_live);
    // Was `.slice(-180)`, which capped every window at ~9 months: selecting
    // 5Y drew the same chart as 1Y. Show what was fetched.
    const candles = history.candles;
    renderCandlestick('forecastChart', candles, {
      symbol: v.symbol, height: 420, forecast: prediction.forecast,
    });

    const up = prediction.direction === 'up';
    const t = prediction.test_metrics || {};
    ui.el('predictionBox').innerHTML = `
      <div class="t-center mb-2">
        ${ui.gauge(prediction.confidence, 116)}
        <div style="margin-top:-72px;font-size:19px;font-weight:700">${fmt.pct(prediction.confidence * 100, 0, false)}</div>
        <div class="text-xs text-muted" style="margin-top:38px">Model confidence</div>
      </div>
      <div class="signal-row"><span class="signal-name">Direction</span>
        <span class="badge ${up ? 'badge-green' : 'badge-red'}">${up ? '▲ UP' : '▼ DOWN'}</span></div>
      <div class="signal-row"><span class="signal-name">Expected return (${prediction.horizon}d)</span>
        <span class="${up ? 'up' : 'down'}" style="font-weight:650">${fmt.pct(prediction.predicted_return * 100)}</span></div>
      <div class="signal-row"><span class="signal-name">Current price</span>
        <span class="mono">${fmt.price(prediction.last_price)}</span></div>
      <div class="signal-row"><span class="signal-name">Target price</span>
        <span class="mono ${up ? 'up' : 'down'}">${fmt.price(prediction.predicted_price)}</span></div>
      <div class="signal-row"><span class="signal-name">Residual σ</span>
        <span class="mono">${fmt.num(prediction.residual_std, 4)}</span></div>
      <div class="signal-row"><span class="signal-name">Directional accuracy</span>
        <span class="mono">${fmt.num(t.directional_accuracy, 1)}%</span></div>
      <div class="text-xs text-muted mt-2">
        Trained ${fmt.timeAgo(prediction.trained_at)} · model ${prediction.model.toUpperCase()}.
        Shaded band is a 90% empirical confidence interval derived from historical residuals.
      </div>`;
  } catch (err) {
    ui.error('forecastChart', `Prediction failed: ${err.message}`);
    ui.el('predictionBox').innerHTML = `<div class="error-box">${err.message}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = 'Predict';
  }
}

async function compareArchitectures() {
  const v = formValues();
  const btn = ui.el('compareBtn');
  btn.disabled = true; btn.textContent = 'Benchmarking…';
  ui.loading('compareBox', 'Training LSTM, GRU and TCN — this takes a minute…');
  try {
    const data = await api.compareModels({
      symbol: v.symbol, models: ['lstm', 'gru', 'tcn'],
      period: v.period, horizon: v.horizon, epochs: Math.min(v.epochs, 15),
    });
    const rows = data.results.filter((r) => !r.error);
    ui.el('compareBox').innerHTML = `
      <table>
        <thead><tr><th>Model</th><th class="t-right">Dir. Acc.</th><th class="t-right">RMSE</th>
          <th class="t-right">R²</th><th class="t-right">Params</th><th class="t-right">Time</th></tr></thead>
        <tbody>${rows.map((r) => `
          <tr${r.model === data.best_model ? ' style="background:var(--accent-soft)"' : ''}>
            <td class="sym-cell">${r.model.toUpperCase()}${r.model === data.best_model ? ' 🏆' : ''}</td>
            <td class="t-right ${r.directional_accuracy > 50 ? 'up' : 'down'}">${fmt.num(r.directional_accuracy, 1)}%</td>
            <td class="t-right mono">${fmt.num(r.rmse, 5)}</td>
            <td class="t-right mono">${fmt.num(r.r2, 4)}</td>
            <td class="t-right text-muted">${(r.n_parameters || 0).toLocaleString()}</td>
            <td class="t-right text-muted">${fmt.num(r.train_seconds, 1)}s</td>
          </tr>`).join('')}</tbody>
      </table>
      <div class="text-xs text-muted mt-2">Ranked by out-of-sample directional accuracy, then RMSE.</div>`;
    ui.toast(`Best architecture: ${(data.best_model || 'n/a').toUpperCase()}`, 'success');
  } catch (err) {
    ui.error('compareBox', `Comparison failed: ${err.message}`);
  } finally {
    btn.disabled = false; btn.textContent = 'Compare Architectures';
  }
}

async function loadTrained() {
  try {
    const data = await api.trainedModels();
    const models = data.models.filter((m) => m.symbol && !m.symbol.includes('__WF'));
    ui.el('trainedBox').innerHTML = models.length ? `
      <table>
        <thead><tr><th>Symbol</th><th>Model</th><th class="t-right">Horizon</th>
          <th class="t-right">Dir. Acc.</th><th class="t-right">RMSE</th><th>Trained</th></tr></thead>
        <tbody>${models.map((m) => `
          <tr class="clickable" onclick="document.getElementById('fSymbol').value='${m.symbol}';document.getElementById('fModel').value='${m.model}';document.getElementById('fHorizon').value='${m.horizon}'">
            <td class="sym-cell">${m.symbol}</td>
            <td><span class="badge badge-blue">${(m.model || '').toUpperCase()}</span></td>
            <td class="t-right">${m.horizon}d</td>
            <td class="t-right ${(m.test_metrics?.directional_accuracy || 0) > 50 ? 'up' : 'down'}">${fmt.num(m.test_metrics?.directional_accuracy, 1)}%</td>
            <td class="t-right mono">${fmt.num(m.test_metrics?.rmse, 5)}</td>
            <td class="text-muted text-xs">${fmt.timeAgo(m.trained_at)}</td>
          </tr>`).join('')}</tbody>
      </table>` : '<div class="empty">No models trained yet</div>';
  } catch (err) {
    ui.error('trainedBox', err.message);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initTimeRange();
  initSearch((symbol) => { ui.el('fSymbol').value = symbol; predict(); loadTrainingCurves(); });
  // Same picker as everywhere else: keyboard-navigable, name-searchable.
  new SymbolPicker('fSymbol', 'fSymbolPanel', () => { predict(); loadTrainingCurves(); });
  ui.el('fSymbol').value = getActiveSymbol();
  ui.el('trainBtn').addEventListener('click', trainModel);
  ui.el('predictBtn').addEventListener('click', predict);
  ui.el('compareBtn').addEventListener('click', compareArchitectures);
  ui.el('refreshTrainedBtn').addEventListener('click', loadTrained);
  loadTrained();
  loadTrainingCurves();
  api.history(getActiveSymbol(), getTimeRange(), '1d', 'sma,bbands')
    .then((h) => {
      ui.el('fSourceBadge').innerHTML = ui.sourceBadge(h.source, h.is_live);
      renderCandlestick('forecastChart', h.candles, { symbol: h.symbol, height: 420 });
    })
    .catch((e) => ui.error('forecastChart', e.message));

  // The global control drives every chart on this page. Training history is a
  // separate, local hyperparameter and deliberately does NOT react to it.
  onTimeRangeChange(() => { predict(); });

  // Switching architecture or horizon points at a different checkpoint, so the
  // curves on screen would otherwise belong to a model you are not looking at.
  ui.el('fModel').addEventListener('change', loadTrainingCurves);
  ui.el('fHorizon').addEventListener('change', loadTrainingCurves);
  ui.el('fSymbol').addEventListener('change', loadTrainingCurves);
});

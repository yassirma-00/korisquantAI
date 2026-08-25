/* ============================================================
   Page: Reinforcement Learning Agent
   ============================================================ */

/* ==================================================== algorithm catalogue */
let ALGORITHMS = [];
let algoFamily = '';

function ratingDots(value) {
  return `<span class="rating-dots">${[1, 2, 3, 4, 5]
    .map((i) => `<span class="rating-dot ${i <= value ? 'on' : ''}"></span>`).join('')}</span>`;
}

function algoCard(a, selected) {
  const cls = [
    'algo-card', `family-${a.family}`,
    a.key === selected ? 'selected' : '',
    a.available ? '' : 'unavailable',
  ].join(' ');
  return `<div class="${cls}" data-algo="${a.key}" title="${a.available ? a.best_for : 'Backend not installed'}">
    <div class="algo-head">
      <div>
        <div class="algo-name">${a.name}</div>
        <div class="text-xs text-muted">${a.full_name}</div>
      </div>
      ${a.available ? `<span class="badge badge-grey">${a.action_space}</span>`
                    : `<span class="badge badge-amber">needs ${a.requires}</span>`}
    </div>
    <div class="algo-desc">${a.description.slice(0, 150)}${a.description.length > 150 ? '…' : ''}</div>
    <div class="algo-ratings">
      <div class="algo-rating"><span class="label">Efficiency</span>${ratingDots(a.performance.sample_efficiency)}</div>
      <div class="algo-rating"><span class="label">Stability</span>${ratingDots(a.performance.stability)}</div>
      <div class="algo-rating"><span class="label">Performance</span>${ratingDots(a.performance.final_performance)}</div>
      <div class="algo-rating"><span class="label">Speed</span>${ratingDots(a.performance.training_speed)}</div>
    </div>
  </div>`;
}

function renderAlgoGrid() {
  const selected = ui.el('rAlgo').value;
  const items = ALGORITHMS.filter((a) => !algoFamily || a.family === algoFamily);
  ui.el('algoGrid').innerHTML = items.map((a) => algoCard(a, selected)).join('');
  ui.el('algoGrid').querySelectorAll('.algo-card').forEach((card) => {
    card.addEventListener('click', () => {
      const key = card.dataset.algo;
      const algo = ALGORITHMS.find((a) => a.key === key);
      if (!algo?.available) {
        ui.toast(`${algo?.name} requires: ${algo?.requires}`, 'error');
        return;
      }
      ui.el('rAlgo').value = key;
      ui.el('algoSelected').textContent = algo.full_name;
      renderAlgoGrid();
      renderAlgoDetail(algo);
      applyActionSpaceMode(algo);
      syncMoeToggle();
    });
  });
}

/** Show the controls that actually apply to the selected action space. */
function applyActionSpaceMode(algo) {
  const continuous = algo?.action_space === 'continuous';
  ui.el('rBasketField')?.classList.toggle('hidden', !continuous);
  ui.el('rSymbolField')?.classList.toggle('hidden', continuous);
  const btn = ui.el('rActionBtn');
  if (btn) {
    // This used to be disabled for continuous agents, because nothing could
    // query a trained basket — five of them sat on disk unusable. Now that
    // /rl/allocation exists the button works for both action spaces; it
    // returns a weight vector instead of BUY/HOLD/SELL.
    btn.disabled = false;
    btn.title = continuous
      ? 'Returns target portfolio weights, with the regime effect per asset.'
      : '';
  }
  const train = ui.el('rTrainBtn');
  if (train) train.textContent = continuous ? 'Train Portfolio Agent' : 'Train Agent';
}

function renderAlgoDetail(a) {
  ui.el('algoDetailTitle').textContent = `${a.name} — ${a.full_name}`;
  ui.el('algoDetail').innerHTML = `
    <div class="flex gap-1 mb-2" style="flex-wrap:wrap">
      <span class="badge badge-blue">${a.family.replace('_', ' ')}</span>
      <span class="badge badge-grey">${a.action_space}</span>
      <span class="badge badge-grey">${a.year}</span>
      ${a.available ? '<span class="badge badge-green">available</span>'
                    : `<span class="badge badge-amber">needs ${a.requires}</span>`}
    </div>
    <div>${a.description}</div>
    <h4>Characteristics</h4><ul>${a.characteristics.map((c) => `<li>${c}</li>`).join('')}</ul>
    <h4>Advantages</h4><ul>${a.advantages.map((c) => `<li class="pro">${c}</li>`).join('')}</ul>
    <h4>Limitations</h4><ul>${a.limitations.map((c) => `<li class="con">${c}</li>`).join('')}</ul>
    <h4>Best for</h4><div>${a.best_for}</div>
    ${a.paper ? `<h4>Reference</h4><div class="text-muted">${a.paper}</div>` : ''}`;
}

async function loadAlgorithms() {
  try {
    requireApi('algorithms', 'compareAlgorithms', 'agentDecision');
    const data = await api.algorithms();
    ALGORITHMS = data.algorithms;
    renderAlgoGrid();
    const current = ALGORITHMS.find((a) => a.key === ui.el('rAlgo').value);
    if (current) {
      ui.el('algoSelected').textContent = current.full_name;
      renderAlgoDetail(current);
      applyActionSpaceMode(current);
    }
    renderAlgoComparison();
  } catch (err) {
    ui.error('algoGrid', `Could not load the algorithm catalogue: ${err.message}`);
  }
}

async function renderAlgoComparison() {
  // The Plotly bundle is loaded from a CDN; on a cold cache it can still be in
  // flight when the catalogue resolves. Wait for it rather than failing silently.
  if (typeof Plotly === 'undefined') {
    for (let i = 0; i < 40 && typeof Plotly === 'undefined'; i += 1) {
      await new Promise((r) => setTimeout(r, 150));
    }
    if (typeof Plotly === 'undefined') {
      ui.empty('algoCompare', 'Chart library unavailable (offline?)');
      return;
    }
  }
  try {
    const data = await api.compareAlgorithms();
    const rows = data.algorithms.filter((r) => r.available);
    const node = ui.el('algoCompare');
    node.innerHTML = '';
    const dims = ['sample_efficiency', 'stability', 'final_performance', 'training_speed'];
    Plotly.newPlot(node, [{
      type: 'heatmap',
      z: dims.map((d) => rows.map((r) => r[d])),
      x: rows.map((r) => r.name),
      y: ['Efficiency', 'Stability', 'Performance', 'Speed'],
      colorscale: [[0, C().bg3], [0.5, C().chart6], [1, C().accent]],
      zmin: 1, zmax: 5, xgap: 2, ygap: 2,
      text: dims.map((d) => rows.map((r) => String(r[d]))),
      texttemplate: '%{text}', textfont: { size: 10 },
      colorbar: { thickness: 8, len: 0.8, tickfont: { size: 9 } },
    }], ui.plotLayout({
      height: 300, showlegend: false, margin: { l: 90, r: 10, t: 10, b: 80 },
      xaxis: { tickangle: -40, tickfont: { size: 9 } },
    }), ui.plotConfig);
  } catch (err) {
    ui.empty('algoCompare', `Comparison unavailable: ${err.message}`);
  }
}


function algoSpec(key) {
  return ALGORITHMS.find((a) => a.key === (key || ui.el('rAlgo').value));
}

/** Continuous-action algorithms allocate across a basket, not one asset. */
function isContinuousOnly(key) {
  return algoSpec(key)?.action_space === 'continuous';
}

function rlValues() {
  return {
    symbol: ui.el('rSymbol').value.trim().toUpperCase(),
    algo: ui.el('rAlgo').value,
    period: ui.el('rPeriod').value,
    episodes: parseInt(ui.el('rEpisodes').value, 10),
    initial_balance: parseFloat(ui.el('rCapital').value),
    transaction_cost: parseFloat(ui.el('rFee').value) / 10000,
    basket: (ui.el('rBasket')?.value || 'AAPL,MSFT,SPY,GC=F')
      .split(',').map((x) => x.trim().toUpperCase()).filter(Boolean),
    // Without this the page could only ever produce regime-blind agents, so
    // the attribution panel kept telling the user to "retrain with
    // regime_aware enabled" through a control that did not exist.
    regime_aware: ui.el('rRegimeAware')?.checked ?? false,
    // The saved profile supplies every training parameter. Sending it is what
    // makes the dashboard selection actually reach the run — without it the
    // service silently fell back to "default" whatever the user picked.
    profile: ui.el('rProfile')?.value || 'default',
  };
}

/** Continuous agents return portfolio weights rather than a BUY/HOLD/SELL. */
function renderPortfolioTraining(meta, v) {
  const perf = meta.test_performance || {};
  const weights = perf.final_weights || {};
  ui.el('rStatus').innerHTML = `
    <div class="info-box">
      <strong>${algoSpec(v.algo)?.name} trained on a ${meta.symbols.length}-asset portfolio</strong>
      — ${meta.symbols.join(', ')}. ${meta.train_bars} training bars, tested on
      ${meta.test_bars} unseen bars.
    </div>
    <div class="grid grid-4 mt-2">
      <div class="card"><div class="stat-label">Return</div>
        <div class="stat-value ${fmt.cls(perf.total_return)}" style="font-size:20px">${fmt.pct((perf.total_return || 0) * 100)}</div></div>
      <div class="card"><div class="stat-label">Sharpe</div>
        <div class="stat-value" style="font-size:20px">${fmt.num(perf.sharpe_ratio, 2)}</div></div>
      <div class="card"><div class="stat-label">Max drawdown</div>
        <div class="stat-value down" style="font-size:20px">${fmt.pct((perf.max_drawdown || 0) * 100)}</div></div>
      <div class="card"><div class="stat-label">vs Equal weight</div>
        <div class="stat-value ${fmt.cls(perf.alpha_vs_equal_weight)}" style="font-size:20px">${fmt.pct((perf.alpha_vs_equal_weight || 0) * 100)}</div></div>
    </div>`;

  const title = ui.el('equityTitle');
  if (title) title.textContent = 'Final Portfolio Allocation';

  const entries = Object.entries(weights);
  if (entries.length) {
    const node = ui.el('equityChart');
    node.innerHTML = '';
    Plotly.newPlot(node, [{
      type: 'bar', x: entries.map(([k]) => k), y: entries.map(([, w]) => w * 100),
      marker: { color: entries.map(([k]) => (k === 'CASH' ? C().text2 : C().accent)) },
      text: entries.map(([, w]) => `${(w * 100).toFixed(1)}%`), textposition: 'auto',
    }], ui.plotLayout({
      height: 420, showlegend: false,
      yaxis: { gridcolor: C().grid, title: { text: 'Allocation (%)', font: { size: 10 } } },
    }), ui.plotConfig);
  }
  ui.empty('actionBox',
    'Continuous agents allocate weights across the basket rather than issuing BUY/HOLD/SELL.');
  ui.empty('perfBox', 'See the allocation above.');
  loadAgents();
  ui.toast(`Portfolio agent trained — ${fmt.pct((perf.total_return || 0) * 100)}`, 'success');
}

function perfTable(perf, baselines) {
  const bh = baselines?.buy_and_hold?.total_return ?? perf.buy_and_hold_return ?? 0;
  const sma = baselines?.sma_crossover || {};
  const alpha = perf.total_return - bh;
  return `
    <div class="grid grid-2 mb-2">
      <div class="card"><div class="stat-label">Agent Return</div>
        <div class="stat-value ${fmt.cls(perf.total_return)}" style="font-size:22px">${fmt.pct(perf.total_return * 100)}</div></div>
      <div class="card"><div class="stat-label">Alpha vs Buy & Hold</div>
        <div class="stat-value ${fmt.cls(alpha)}" style="font-size:22px">${fmt.pct(alpha * 100)}</div></div>
    </div>
    <table>
      <thead><tr><th>Strategy</th><th class="t-right">Return</th><th class="t-right">Sharpe</th>
        <th class="t-right">Max DD</th><th class="t-right">Trades</th></tr></thead>
      <tbody>
        <tr style="background:var(--accent-soft)">
          <td class="sym-cell">RL Agent</td>
          <td class="t-right ${fmt.cls(perf.total_return)}">${fmt.pct(perf.total_return * 100)}</td>
          <td class="t-right">${fmt.num(perf.sharpe_ratio, 2)}</td>
          <td class="t-right down">${fmt.pct(perf.max_drawdown * 100)}</td>
          <td class="t-right">${perf.n_trades}</td>
        </tr>
        <tr><td>Buy & Hold</td>
          <td class="t-right ${fmt.cls(bh)}">${fmt.pct(bh * 100)}</td>
          <td class="t-right text-muted">—</td><td class="t-right text-muted">—</td><td class="t-right">1</td></tr>
        <tr><td>SMA Crossover</td>
          <td class="t-right ${fmt.cls(sma.total_return)}">${sma.total_return !== undefined ? fmt.pct(sma.total_return * 100) : '—'}</td>
          <td class="t-right">${fmt.num(sma.sharpe_ratio, 2)}</td>
          <td class="t-right down">${sma.max_drawdown !== undefined ? fmt.pct(sma.max_drawdown * 100) : '—'}</td>
          <td class="t-right">${sma.n_trades ?? '—'}</td></tr>
        <tr><td>Cash</td><td class="t-right">0.00%</td><td class="t-right">0.00</td>
          <td class="t-right">0.00%</td><td class="t-right">0</td></tr>
      </tbody>
    </table>
    <div class="grid grid-3 mt-2">
      <div><div class="stat-label">Sortino</div><div class="mono">${fmt.num(perf.sortino_ratio, 2)}</div></div>
      <div><div class="stat-label">Calmar</div><div class="mono">${fmt.num(perf.calmar_ratio, 2)}</div></div>
      <div><div class="stat-label">Fees paid</div><div class="mono">${fmt.money(perf.total_transaction_cost)}</div></div>
    </div>
    <div class="text-xs text-muted mt-2">
      All figures are strictly out-of-sample: the agent trained only on data preceding this window.
      An agent that fails to beat Buy &amp; Hold is a genuine, informative result — not a bug.
    </div>`;
}

async function trainAgent() {
  const v = rlValues();
  const btn = ui.el('rTrainBtn');
  btn.disabled = true; btn.textContent = 'Training…';
  // Off-policy continuous agents (SAC/TD3/DDPG) do a gradient update per step,
  // so they run minutes rather than seconds. Say so, and show elapsed time, or
  // the page looks frozen and users reload mid-training.
  const continuous = isContinuousOnly(v.algo);
  const spec = algoSpec(v.algo);
  const estimate = continuous ? '2-5 minutes' : `${Math.max(Math.round(v.episodes / 3), 1)}-${v.episodes * 2}s`;
  const target = continuous ? `a ${v.basket.length}-asset portfolio (${v.basket.join(', ')})` : v.symbol;
  const started = Date.now();
  ui.el('rStatus').innerHTML = `
    <div class="loading" style="justify-content:flex-start">
      <div class="spinner"></div>
      <div>
        <div>Training <strong>${spec?.name || v.algo.toUpperCase()}</strong> on ${target}…</div>
        <div class="text-xs text-muted mt-1">
          Estimated ${estimate} · elapsed <span id="rElapsed">0s</span>
          ${continuous ? ' — off-policy agents update on every step, so this is slow by design.' : ''}
        </div>
      </div>
    </div>`;
  const ticker = setInterval(() => {
    const el = ui.el('rElapsed');
    if (el) el.textContent = `${Math.round((Date.now() - started) / 1000)}s`;
  }, 1000);
  try {
    // SAC/TD3/DDPG output a weight vector, so they need the multi-asset
    // PortfolioEnv. Sending them to the single-asset endpoint is what produced
    // "'sac' is not a valid discrete algorithm".
    const meta = isContinuousOnly(v.algo)
      ? await api.trainPortfolioRL({
          symbols: v.basket, algo: v.algo, period: v.period, profile: v.profile,
          total_timesteps: Math.max(v.episodes * 1500, 5000),
          initial_balance: v.initial_balance, transaction_cost: v.transaction_cost,
          regime_aware: v.regime_aware,
        })
      : await api.trainRL({
          symbol: v.symbol, algo: v.algo, period: v.period, episodes: v.episodes,
          initial_balance: v.initial_balance, transaction_cost: v.transaction_cost,
          profile: v.profile, regime_aware: v.regime_aware,
        });

    if (isContinuousOnly(v.algo)) {
      renderPortfolioTraining(meta, v);
      return;
    }

    ui.el('rStatus').innerHTML = `
      <div class="info-box">
        <strong>${algoSpec(v.algo)?.name || v.algo.toUpperCase()} trained on ${meta.symbol}</strong> —
        ${meta.train_bars} training bars, tested on ${meta.test_bars} unseen bars.
        Data source: ${meta.data_source}.
      </div>`;
    ui.el('perfBox').innerHTML = perfTable(meta.test_performance, meta.baselines);

    const hist = meta.training_history || {};
    const series = [];
    if (hist.final_values?.length) {
      series.push({ x: hist.final_values.map((_, i) => i + 1), y: hist.final_values, name: 'Final portfolio value', color: C().green });
    } else if (hist.episode_rewards?.length) {
      series.push({ x: hist.episode_rewards.map((_, i) => i + 1), y: hist.episode_rewards, name: 'Episode reward', color: C().accent });
    }
    if (series.length) renderLineChart('learningChart', series, { height: 260, yTitle: 'per episode' });
    else ui.empty('learningChart', 'No learning curve for this algorithm');

    await loadBacktest(v.symbol, v.algo);
    await liveAction();
    loadAgents();
    ui.toast(`Agent trained — ${fmt.pct(meta.test_performance.total_return * 100)} out-of-sample`, 'success');
  } catch (err) {
    // Surface the server's guidance rather than a raw validation dump
    let msg = err.message;
    try {
      const detail = err.payload?.detail;
      if (Array.isArray(detail) && detail[0]?.msg) {
        msg = detail[0].msg.replace(/^Value error,\s*/, '');
      }
    } catch { /* keep the original message */ }
    ui.el('rStatus').innerHTML = `<div class="error-box">Training failed: ${msg}</div>`;
  } finally {
    clearInterval(ticker);
    btn.disabled = false;
    btn.textContent = isContinuousOnly(v.algo) ? 'Train Portfolio Agent' : 'Train Agent';
  }
}

/* Algorithms the MoE can actually drive.
 *
 * SB3 policies expose `predict`, not `q_values`, and cannot be fine-tuned
 * through this path — the backend answers 422 for them. Disabling the toggle
 * up front explains why instead of letting the user discover it as an error. */
const MOE_ALGOS = new Set(['dqn', 'double_dqn', 'dueling_dqn', 'c51', 'iqn', 'rainbow']);

/** Keep the MoE toggle honest about what the selected algorithm supports. */
function syncMoeToggle() {
  const box = ui.el('moeToggle');
  const wrap = ui.el('moeToggleWrap');
  if (!box || !wrap) return;
  const algo = ui.el('rAlgo')?.value;
  const ok = MOE_ALGOS.has(algo);
  box.disabled = !ok;
  if (!ok) box.checked = false;
  wrap.title = ok
    ? 'Route each bar to a bull / bear / stress expert and fine-tune the incoming one'
    : `Mixture-of-Experts needs a native discrete agent (${[...MOE_ALGOS].join(', ')}); `
      + `${algo || 'this algorithm'} is a stable-baselines3 policy with no fine-tune path.`;
}

const EXPERT_COLOUR = { bull: 'green', bear: 'red', stress: 'amber', base: 'text2' };

/* Which saved checkpoint this page asks for.
 *
 * The twin selector was removed from the UI on request, so this is always the
 * original agent. The plumbing below it is deliberately left in place: the
 * backend still serves the regime-aware twins under `?variant=regime`, and
 * keeping the call sites parameterised means restoring the control is a
 * one-line change rather than a re-wiring.
 *
 * Reads the element if it is ever put back, so the two cannot drift apart. */
function currentVariant() {
  return ui.el('rVariant')?.value || '';
}

/* Routing trace, adaptation evidence and K-5.
 *
 * Deliberately reports the failures too: switches that could not be adapted,
 * and the strict K-5 reading alongside the flattering one. A panel that showed
 * only the successes would misrepresent how often this mechanism actually
 * fires — on a 2-year window it is usually once. */
function renderMoePanel(moe) {
  const panel = ui.el('moePanel');
  if (!panel) return;
  if (!moe) { panel.classList.add('hidden'); panel.innerHTML = ''; return; }

  const k5 = moe.k5_reaction_delay || {};
  const strict = moe.k5_expert_changes_only || {};
  const acted = moe.bars_acted_by || {};
  const total = Object.values(acted).reduce((a, b) => a + b, 0) || 1;
  const check = moe.adaptation_check || {};

  const segments = Object.entries(acted)
    .sort((a, b) => b[1] - a[1])
    .map(([name, n]) => {
      const colour = C()[EXPERT_COLOUR[name] || 'text2'];
      return { name, n, colour, pct: (n / total) * 100 };
    });

  const bar = segments.map((s) => `<i style="width:${s.pct}%;background:${s.colour}"></i>`).join('');
  const legend = segments.map((s) => `
    <span><i class="moe-swatch" style="background:${s.colour}"></i>
      ${s.name === 'base' ? 'base policy' : `${s.name} expert`} — ${s.n} bars
      (${s.pct.toFixed(0)}%)</span>`).join('');

  const adaptations = (moe.adaptations || []).map((a) => `
    <tr><td><span class="badge badge-grey">${a.expert}</span></td>
        <td class="mono">${a.bars_used}</td>
        <td class="mono">${a.episodes}</td>
        <td class="mono">${a.weight_delta}</td></tr>`).join('');

  // An unadapted switch is a failure to react. Counting them here keeps the
  // headline number from reading better than the mechanism performed.
  const unadapted = k5.unadapted ?? 0;
  const fmtDelay = (v) => (v === null || v === undefined ? '—' : `${v} bar${v === 1 ? '' : 's'}`);

  /* A mean over one observation is not a mean, and printing it bare invites
   * the reader to treat 60 bars as typical when it is a single event. Measured
   * on AAPL/1y: 8 switches, 1 measurable, mean "60 bars". So the sample size
   * travels with the number. */
  const sampleNote = (stats) => {
    const n = stats.measured ?? 0;
    if (!n) return '<div class="text-xs text-muted">no switch could be measured</div>';
    return `<div class="text-xs text-muted">over ${n} of ${stats.n_switches ?? 0} switch${
      (stats.n_switches ?? 0) === 1 ? '' : 'es'}${n === 1 ? ' — single event, not an average' : ''}</div>`;
  };

  panel.innerHTML = `
    <div class="moe-panel">
      <div class="moe-stats">
        <div><div class="moe-stat-label">Regime switches</div>
             <div class="moe-stat-value">${moe.n_switches ?? 0}</div></div>
        <div><div class="moe-stat-label">Real fine-tunes</div>
             <div class="moe-stat-value">${(moe.adaptations || []).length}</div></div>
        <div><div class="moe-stat-label">K-5 reaction (all switches)</div>
             <div class="moe-stat-value">${fmtDelay(k5.mean_reaction_bars)}</div>
             ${sampleNote(k5)}</div>
        <div><div class="moe-stat-label">K-5 reaction (expert changes)</div>
             <div class="moe-stat-value">${fmtDelay(strict.mean_reaction_bars)}</div>
             ${sampleNote(strict)}</div>
        <div><div class="moe-stat-label">Switches with no reaction</div>
             <div class="moe-stat-value" style="color:${unadapted ? C().amber : 'inherit'}">${unadapted}</div>
             <div class="text-xs text-muted">expert never took control</div></div>
      </div>

      <div class="stat-label mb-1">Which policy decided each bar</div>
      <div class="moe-bar">${bar}</div>
      <div class="moe-legend mb-2">${legend}</div>

      ${adaptations ? `
        <div class="stat-label mt-2 mb-1">Fine-tuning actually applied</div>
        <table class="table text-xs">
          <thead><tr><th>Expert</th><th>Bars used</th><th>Episodes</th><th>Max weight Δ</th></tr></thead>
          <tbody>${adaptations}</tbody>
        </table>`
    : `<div class="info-box text-xs">No expert could be fine-tuned on this window —
         every switch was either already held by the acting expert or had fewer than
         ${moe.min_expert_bars} bars of its own regime.</div>`}

      ${check.any_weights_changed === false && (moe.adaptations || []).length === 0 ? '' : `
        <div class="text-xs text-muted mt-2">
          Weights changed: <strong>${check.any_weights_changed ? 'yes' : 'no'}</strong>${
  check.max_weight_delta ? ` · largest parameter change ${check.max_weight_delta}` : ''}
        </div>`}

      ${(moe.notes || []).length ? `
        <ul class="text-xs text-muted mt-2" style="padding-left:1.1em">
          ${moe.notes.map((n) => `<li>${n}</li>`).join('')}
        </ul>` : ''}

      <div class="text-xs text-muted mt-2">${moe.disclaimer || ''}</div>
    </div>`;
  panel.classList.remove('hidden');
}

async function loadBacktest(symbol, algo) {
  const title = ui.el('equityTitle');
  const useMoe = !!ui.el('moeToggle')?.checked && MOE_ALGOS.has(algo);
  if (title) {
    title.textContent = useMoe
      ? 'Equity Curve — Mixture-of-Experts vs Buy & Hold'
      : 'Equity Curve vs Buy & Hold';
  }
  ui.loading('equityChart', useMoe
    ? 'Routing regimes and fine-tuning experts…'
    : 'Running backtest…');
  renderMoePanel(null);
  try {
    const data = await api.rlBacktest(symbol, algo, getTimeRange(),
      { moe: useMoe, variant: currentVariant() });
    const equity = data.equity_curve;
    if (!equity.length) { ui.empty('equityChart', 'No equity curve'); return; }

    const history = await api.history(symbol, getTimeRange());
    const startValue = equity[0].value;
    const closes = history.candles.slice(-equity.length);
    const bhCurve = closes.map((c) => (c.close / closes[0].close) * startValue);

    renderLineChart('equityChart', [
      { x: equity.map((e) => e.date), y: equity.map((e) => e.value),
        name: useMoe ? 'MoE agent' : 'RL Agent', color: C().green, width: 2.2 },
      { x: closes.map((c) => c.date), y: bhCurve, name: 'Buy & Hold', color: C().text2, width: 1.6, dash: 'dash' },
    ], { height: 420, yTitle: 'Portfolio value ($)' });

    renderMoePanel(data.moe);
  } catch (err) {
    ui.error('equityChart', `Backtest failed: ${err.message}`);
    renderMoePanel(null);
  }
}

/* How the detected market regime bore on this specific decision.
 *
 * The backend measures this by counterfactual — it re-queries the agent with
 * the regime block neutralised — so this panel can say "the regime changed
 * nothing" when that is the truth. A template that always asserts influence
 * would read identically whether or not any existed, which is worth nothing to
 * a risk reviewer. */
const INFLUENCE_STYLE = {
  decisive: { cls: 'badge-red', text: 'decisive' },
  contributory: { cls: 'badge-amber', text: 'contributory' },
  negligible: { cls: 'badge-grey', text: 'no effect' },
};

function regimePanel(d) {
  const e = d.regime_explanation;
  if (!e) return '';
  if (!e.available) {
    return `
      <div class="stat-label mt-2 mb-1">Market regime</div>
      <div class="info-box text-xs">${e.reason || 'Not available for this agent.'}</div>`;
  }
  const style = INFLUENCE_STYLE[e.influence] || INFLUENCE_STYLE.negligible;
  const rows = (e.feature_contributions || []).slice(0, 4).map((c) => {
    const mag = Math.abs(c.q_delta);
    const max = Math.max(...e.feature_contributions.map((x) => Math.abs(x.q_delta)), 1e-9);
    const tone = c.q_delta >= 0 ? 'var(--green)' : 'var(--red)';
    return `
      <div class="score-part">
        <div class="score-part-head">
          <span>${c.feature.replace(/^regime_/, '').replace(/_/g, ' ')}</span>
          <span class="mono text-muted">${c.q_delta >= 0 ? '+' : ''}${c.q_delta.toFixed(4)}</span>
        </div>
        <div class="score-part-bar">
          <i style="width:${(mag / max) * 100}%;background:${tone}"></i>
        </div>
        <div class="score-part-detail">observed ${c.value} · neutral ${c.neutral}</div>
      </div>`;
  }).join('');

  return `
    <div class="stat-label mt-2 mb-1">Market regime influence</div>
    <div class="flex gap-1 mb-1" style="flex-wrap:wrap;align-items:center">
      <span class="badge badge-purple">${e.regime_label}</span>
      <span class="badge ${style.cls}">${style.text}</span>
      <span class="text-xs text-muted">${(e.regime_confidence * 100).toFixed(0)}% confidence ·
        risk penalties ×${e.risk_aversion_applied.toFixed(2)}</span>
    </div>
    <div class="text-sm mb-1">${e.summary}</div>
    <div class="text-xs text-muted">Crash probability
      ${(e.crash_probability * 100).toFixed(1)}% · volatility
      ${e.volatility_ratio.toFixed(2)}× its average · drawdown
      ${(e.drawdown * 100).toFixed(1)}%</div>
    ${rows ? `<details class="score-detail">
      <summary>Which regime inputs moved the decision</summary>
      <div class="score-parts">${rows}</div>
      <div class="score-part-detail mt-1">Each row neutralises one regime input
        and reports how far the chosen action's value moved. Positive means the
        input supported this action.</div>
    </details>` : ''}`;
}

/* A basket agent emits a weight vector, so there is no "the action flipped"
 * moment to report. The useful question is which sleeve gained or lost capital
 * and to whom — measured by re-querying the agent with the regime inputs
 * neutralised. */
function renderAllocation(d) {
  const e = d.regime_explanation || {};
  const rows = (d.allocation || []).map((a) => {
    const attr = (e.per_asset || []).find((x) => x.symbol === a.symbol) || {};
    const delta = attr.delta;
    const tone = delta > 0 ? 'up' : delta < 0 ? 'down' : 'text-muted';
    const arrow = delta > 0 ? '▲' : delta < 0 ? '▼' : '■';
    return `
      <tr>
        <td class="sym-cell">${a.symbol}</td>
        <td class="t-right mono">${(a.weight * 100).toFixed(1)}%</td>
        <td class="t-right mono ${tone}">${
  delta === undefined ? '—' : `${arrow} ${(Math.abs(delta) * 100).toFixed(1)}pp`}</td>
        <td class="t-right text-muted">${attr.regime_label || '—'}</td>
      </tr>`;
  }).join('');

  const influence = e.available
    ? `<span class="badge ${e.influence === 'decisive' ? 'badge-red'
      : e.influence === 'contributory' ? 'badge-amber' : 'badge-grey'}">${
  e.influence === 'negligible' ? 'no effect' : e.influence}</span>`
    : '';

  const features = (e.feature_contributions || []).slice(0, 5);
  const maxMoved = Math.max(...features.map((c) => c.capital_moved), 1e-9);

  return `
    <div class="decision-hero mb-2">
      <div>
        <div class="decision-action">REBALANCE</div>
        <div class="text-xs text-muted">${d.symbols.length} assets ·
          ${(d.cash_weight * 100).toFixed(1)}% cash</div>
        <div class="flex gap-1 mt-1" style="flex-wrap:wrap">
          <span class="badge badge-purple">${d.algorithm_name}</span>
          ${influence}
        </div>
      </div>
    </div>

    <table class="mb-2"><thead><tr><th>Asset</th>
      <th class="t-right">Weight</th>
      <th class="t-right">Regime effect</th>
      <th class="t-right">Regime</th></tr></thead>
      <tbody>${rows}
        <tr><td class="sym-cell text-muted">CASH</td>
          <td class="t-right mono">${(d.cash_weight * 100).toFixed(1)}%</td>
          <td class="t-right mono text-muted">${
  e.cash_delta === undefined ? '—'
    : `${e.cash_delta > 0 ? '▲' : '▼'} ${(Math.abs(e.cash_delta) * 100).toFixed(1)}pp`}</td>
          <td class="t-right text-muted">—</td></tr>
      </tbody></table>

    ${e.available ? `
      <div class="stat-label mb-1">Market regime influence</div>
      <div class="text-sm mb-1">${e.summary}</div>
      <div class="text-xs text-muted">${e.capital_moved !== undefined
    ? `${(e.capital_moved * 100).toFixed(1)}% of the book moved · ${e.shift_type}` : ''}
        ${e.all_assets_same_regime
    ? ' · ⚠ every asset shares one regime, so diversification offers little protection'
    : ''}</div>
      ${features.length ? `<details class="score-detail">
        <summary>Which regime inputs moved the book</summary>
        <div class="score-parts">${features.map((c) => `
          <div class="score-part">
            <div class="score-part-head"><span>${c.feature}</span>
              <span class="mono text-muted">${(c.capital_moved * 100).toFixed(2)}%</span></div>
            <div class="score-part-bar">
              <i style="width:${(c.capital_moved / maxMoved) * 100}%"></i></div>
          </div>`).join('')}</div>
        <div class="score-part-detail mt-1">${e.scale_note || ''}</div>
      </details>` : ''}`
    : `<div class="info-box text-xs">${e.reason || ''}</div>`}`;
}

async function liveAction() {
  const v = rlValues();
  const btn = ui.el('rActionBtn');
  btn.disabled = true; btn.textContent = 'Querying…';
  ui.loading('actionBox', 'Consulting the agent…');
  try {
    // A continuous agent trained on a basket allocates across it; asking it for
    // a single BUY/HOLD/SELL would describe a different agent than the one the
    // user trained.
    if (isContinuousOnly(v.algo) && v.basket.length > 1) {
      const alloc = await api.rlAllocation(v.basket, v.algo, '1y');
      ui.el('actionBox').innerHTML = renderAllocation(alloc);
      animateContainer('actionBox');
      return;
    }
    const d = await api.agentDecision(v.symbol, v.algo, '1y', currentVariant());
    const colour = d.action === 'BUY' ? 'var(--green)'
      : d.action === 'SELL' ? 'var(--red)' : 'var(--text-1)';
    const plan = d.trade_plan || {};
    const risk = d.risk || {};
    const hz = d.investment_horizon || {};
    const dist = d.return_distribution;

    ui.el('actionBox').innerHTML = `
      <div class="decision-hero mb-2">
        <div style="position:relative">
          ${ui.gauge(d.confidence, 96)}
          <div style="position:absolute;inset:0;display:grid;place-items:center">
            <div style="text-align:center">
              <div style="font-size:16px;font-weight:700">${fmt.pct(d.confidence * 100, 0, false)}</div>
              <div class="text-xs text-muted">conf.</div>
            </div>
          </div>
        </div>
        <div>
          <div class="decision-action" style="color:${colour}">${d.action}</div>
          <div class="text-xs text-muted">${d.symbol} @ ${fmt.price(d.last_price)}</div>
          <div class="flex gap-1 mt-1" style="flex-wrap:wrap">
            <span class="badge badge-purple">${d.algorithm_name}</span>
            ${ui.riskBadge(risk.level)}
          </div>
        </div>
      </div>

      <div class="decision-grid mb-2">
        <div class="decision-tile"><div class="k">Position size</div>
          <div class="v">${fmt.pct(plan.position_size_pct ?? 0, 1, false)}</div>
          <div class="s">of portfolio</div></div>
        ${plan.reduce_existing_pct !== undefined ? `
          <div class="decision-tile"><div class="k">Reduce by</div>
            <div class="v down">${fmt.pct(plan.reduce_existing_pct, 0, false)}</div>
            <div class="s">of existing position</div></div>` : ''}
        <div class="decision-tile"><div class="k">Stop loss</div>
          <div class="v down">${plan.stop_loss_price ? fmt.price(plan.stop_loss_price) : '—'}</div>
          <div class="s">${plan.stop_loss_pct ? fmt.pct(plan.stop_loss_pct, 2) : 'n/a'}</div></div>
        <div class="decision-tile"><div class="k">Take profit</div>
          <div class="v up">${plan.take_profit_price ? fmt.price(plan.take_profit_price) : '—'}</div>
          <div class="s">${plan.take_profit_pct ? fmt.pct(plan.take_profit_pct, 2) : 'n/a'}</div></div>
        <div class="decision-tile"><div class="k">Risk / reward</div>
          <div class="v">${plan.risk_reward_ratio ? plan.risk_reward_ratio + ' : 1' : '—'}</div>
          <div class="s">reward per unit risk</div></div>
        <div class="decision-tile"><div class="k">Horizon</div>
          <div class="v">${hz.days ?? '—'}d</div>
          <div class="s">${hz.label || ''}</div></div>
        <div class="decision-tile"><div class="k">Risk score</div>
          <div class="v">${fmt.num(risk.score, 2)}</div>
          <div class="s">vol ${fmt.pct((risk.annualised_volatility || 0) * 100, 1, false)}</div></div>
      </div>

      ${dist ? `
        <div class="stat-label mb-1">Return distribution per action (distributional agent)</div>
        <table class="mb-2"><thead><tr><th>Action</th><th class="t-right">Mean</th>
          <th class="t-right">Std</th><th class="t-right">CVaR 5%</th></tr></thead>
        <tbody>${Object.entries(dist).map(([act, x]) => `
          <tr${act === d.action ? ' style="background:var(--accent-soft)"' : ''}>
            <td class="sym-cell">${act}</td>
            <td class="t-right ${fmt.cls(x.mean)}">${fmt.num(x.mean, 4)}</td>
            <td class="t-right text-muted">${fmt.num(x.std, 4)}</td>
            <td class="t-right ${fmt.cls(x.cvar_5pct)}">${fmt.num(x.cvar_5pct, 4)}</td>
          </tr>`).join('')}</tbody></table>` : ''}

      ${Object.keys(d.q_values || {}).length ? `
        <div class="stat-label mb-1">Q-values (relative preference)</div>
        ${(() => {
          const es = Object.entries(d.q_values);
          const vals = es.map(([, q]) => q);
          const lo = Math.min(...vals, 0), hi = Math.max(...vals, 0);
          const range = Math.max(hi - lo, 1e-9);
          const best = es.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
          return es.map(([act, q]) => `
            <div class="xai-bar">
              <div class="xai-label" style="width:52px">${act}</div>
              <div class="xai-track">
                <div style="position:absolute;top:0;bottom:0;left:0;border-radius:3px;
                  width:${Math.max(((q - lo) / range) * 100, 2)}%;background:${act === best
                    ? 'linear-gradient(90deg,rgba(34, 217, 138,.35),var(--green))'
                    : 'linear-gradient(90deg,rgba(115, 127, 163,.25),rgba(115, 127, 163,.6))'}"></div>
              </div>
              <div class="xai-value ${act === best ? 'up' : 'text-muted'}">${fmt.num(q, 3)}</div>
            </div>`).join('');
        })()}` : ''}

      ${regimePanel(d)}

      <div class="stat-label mt-2 mb-1">Explanation</div>
      <div class="text-sm mb-1">${d.explanation?.summary || ''}</div>
      ${(d.explanation?.drivers || []).map((x) =>
        `<div class="driver-item"><span class="driver-bullet">▸</span><span>${x}</span></div>`).join('')}
      <div class="text-xs text-muted mt-2">${d.explanation?.disclaimer || ''}</div>`;

    animateContainer('actionBox');
  } catch (err) {
    ui.el('actionBox').innerHTML = `<div class="error-box">${err.message}</div>
      <div class="text-xs text-muted mt-1">Train an agent for this symbol and algorithm first.</div>`;
  } finally {
    btn.disabled = false; btn.textContent = 'Get Decision';
  }
}

async function loadAgents() {
  try {
    const data = await api.rlAgents();
    ui.el('agentsBox').innerHTML = data.agents.length ? `
      <table>
        <thead><tr><th>Symbol</th><th>Algorithm</th><th class="t-right">Return</th>
          <th class="t-right">Sharpe</th><th class="t-right">Max DD</th><th class="t-right">vs B&H</th><th>Trained</th></tr></thead>
        <tbody>${data.agents.map((a) => {
          const p = a.test_performance || {};
          return `<tr class="clickable"${a.stale ? ' style="opacity:.55" title="' + (a.stale_reason || '') + '"' : ''}
              onclick="document.getElementById('rSymbol').value='${a.symbol}';document.getElementById('rAlgo').value='${a.algo}'">
            <td class="sym-cell">${a.symbol}
              ${a.stale ? '<span class="badge badge-amber" style="margin-left:6px">stale</span>' : ''}</td>
            <td><span class="badge badge-purple">${(a.algo || '').toUpperCase()}</span></td>
            <td class="t-right ${fmt.cls(p.total_return)}">${p.total_return !== undefined ? fmt.pct(p.total_return * 100) : '—'}</td>
            <td class="t-right">${fmt.num(p.sharpe_ratio, 2)}</td>
            <td class="t-right down">${p.max_drawdown !== undefined ? fmt.pct(p.max_drawdown * 100) : '—'}</td>
            <td class="t-right ${fmt.cls(p.alpha_vs_buy_hold)}">${p.alpha_vs_buy_hold !== undefined ? fmt.pct(p.alpha_vs_buy_hold * 100) : '—'}</td>
            <td class="text-xs text-muted">${fmt.timeAgo(a.trained_at)}</td>
          </tr>`;
        }).join('')}</tbody>
      </table>` : '<div class="empty">No agents trained yet</div>';
  } catch (err) {
    ui.error('agentsBox', err.message);
  }
}

/* The profile list is loaded from the platform, never typed by the user: the
   YAML files are provisioned automatically at startup, so whatever exists on
   disk is what appears here. */
async function loadProfiles() {
  const select = ui.el('rProfile');
  if (!select) return;
  try {
    const data = await api.hpCatalogue();
    select.innerHTML = (data.profiles || []).map((p) =>
      `<option value="${p.key}">${p.name}</option>`).join('');
    select.value = 'default';
  } catch (err) {
    // A failed profile list must not block training: the backend falls back to
    // "default", which is exactly what the untouched selector already says.
    console.warn('profile list unavailable, using default', err);
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  loadProfiles();
  initTimeRange();
  initSearch((symbol) => { ui.el('rSymbol').value = symbol; });
  ui.el('rSymbol').value = getActiveSymbol();

  // Grouped, searchable symbol picker that also accepts any custom ticker
  new SymbolPicker('rSymbol', 'rSymbolPanel', () => {
    ui.el('rStatus').innerHTML = '';
    ui.empty('actionBox', 'Click “Get Decision” for this symbol');
  });

  loadAlgorithms();
  document.querySelectorAll('#algoFilters .chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('#algoFilters .chip').forEach((c) => c.classList.remove('active'));
      chip.classList.add('active');
      algoFamily = chip.dataset.family;
      renderAlgoGrid();
    });
  });
  ui.el('rTrainBtn').addEventListener('click', trainAgent);
  ui.el('rActionBtn').addEventListener('click', liveAction);
  ui.el('refreshAgentsBtn').addEventListener('click', loadAgents);
  loadAgents();

  // Flipping the MoE toggle re-runs the backtest for the agent on screen; it
  // must not silently leave a stale curve labelled with the new mode.
  ui.el('moeToggle')?.addEventListener('change', () => {
    const symbol = ui.el('rSymbol')?.value.trim().toUpperCase();
    const algo = ui.el('rAlgo')?.value;
    if (symbol && algo) loadBacktest(symbol, algo);
  });
  syncMoeToggle();

  try {
    const algos = await api.rlAlgorithms();
    ui.el('sb3Status').innerHTML = algos.stable_baselines3_available
      ? '<span class="badge badge-green">Stable-Baselines3 available</span>'
      : '<span class="badge badge-amber">Native agents only</span>';
  } catch { /* ignore */ }

  ui.empty('equityChart', 'Train an agent to see its equity curve');
  ui.empty('learningChart', 'Learning progress appears after training');

  // Training length stays a local hyperparameter (#rPeriod); the global
  // range refreshes the agent list and its displayed results.
  onTimeRangeChange(() => loadAgents());
});

/* ============================================================
   Page: Training Monitor

   Read-only over what training already recorded. Every number here comes from
   an agent's metadata sidecar — nothing is recomputed in the browser, so a
   figure on this page is by construction the figure that run produced.
   ============================================================ */

let RUNS = [];
let PROGRESS = null;
let CHECKPOINTS = [];
const SELECTED = new Set();     // checkpoint episodes picked for comparison

/* Which metrics can be charted, and where each series comes from.
   `source: 'training'` is per-episode on the TRAINING window;
   `source: 'eval'` is periodic on the HELD-OUT window. They are never merged
   into one line: the gap between them is the overfitting signal. */
const METRICS = [
  { key: 'reward', label: 'Reward', source: 'both', evalKey: 'total_return',
    evalLabel: 'Eval return', fmt: 'num', evalFmt: 'pct' },
  { key: 'loss', label: 'Loss', source: 'training', fmt: 'num' },
  { key: 'sharpe_ratio', label: 'Sharpe', source: 'both',
    evalKey: 'sharpe_ratio', evalLabel: 'Eval Sharpe', fmt: 'num' },
  { key: null, label: 'Sortino', source: 'eval', evalKey: 'sortino_ratio', fmt: 'num' },
  { key: null, label: 'Max drawdown', source: 'eval', evalKey: 'max_drawdown', fmt: 'pct' },
  { key: null, label: 'Volatility', source: 'eval', evalKey: 'annualised_volatility', fmt: 'pct' },
  { key: null, label: 'VaR 95%', source: 'eval', evalKey: 'var_95', fmt: 'pct' },
  { key: null, label: 'CVaR 95%', source: 'eval', evalKey: 'cvar_95', fmt: 'pct' },
  { key: 'portfolio_value', label: 'Portfolio value', source: 'both',
    evalKey: 'final_value', evalLabel: 'Eval value', fmt: 'money' },
];

let activeMetric = 'Reward';

const asPct = (v) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(2)}%`);
const asNum = (v, d = 3) => (v === null || v === undefined ? '—' : Number(v).toFixed(d));

/* ------------------------------------------------------------ summary card */
function summaryCard(label, value, sub, cls = '') {
  return `
    <div class="card">
      <div class="stat-label">${label}</div>
      <div class="stat-value ${cls}">${value}</div>
      <div class="text-xs text-muted mt-1">${sub || ''}</div>
    </div>`;
}

function renderSummary(s) {
  const best = s.best_evaluation;
  const latest = s.latest_evaluation;
  // Elapsed wall-clock time is not recorded anywhere, so it is not invented.
  // What is known is how long the evaluations themselves took.
  const evalTime = s.eval_seconds !== null && s.eval_seconds !== undefined
    ? `${s.eval_seconds}s in evaluation` : 'not measured';

  ui.el('tmSummary').innerHTML = [
    summaryCard('Algorithm', (s.algo || '—').toUpperCase(),
      `${s.symbol} · profile ${s.profile || '—'}`),
    summaryCard('Episodes', s.total_episodes ?? '—',
      `${s.train_bars ?? '?'} train / ${s.test_bars ?? '?'} test bars`),
    summaryCard('Best evaluation',
      best ? asPct(best.total_return) : '—',
      best ? `episode ${best.episode} · ${evalTime}` : 'no evaluations recorded',
      best && best.total_return > 0 ? 'up' : best ? 'down' : ''),
    summaryCard('Latest evaluation',
      latest ? asPct(latest.total_return) : '—',
      latest ? `episode ${latest.episode}` : `eval_freq = ${s.eval_freq}`,
      latest && latest.total_return > 0 ? 'up' : latest ? 'down' : ''),
    summaryCard('Checkpoints',
      `${s.checkpoints_on_disk ?? 0}<span class="text-muted" style="font-size:14px"> / ${s.n_checkpoints ?? 0}</span>`,
      `on disk / recorded · every ${s.checkpoint_interval || '—'} ep`),
    summaryCard('Final test return',
      asPct(s.final_performance?.total_return),
      `B&H ${asPct(s.baselines?.buy_and_hold?.total_return)}`,
      (s.final_performance?.total_return ?? 0) > 0 ? 'up' : 'down'),
    summaryCard('Experiment', `<span class="mono" style="font-size:12px">${
      (s.experiment_id || '—').replace('exp_', '')}</span>`,
    `seed ${s.seed ?? '—'} · v${(s.model_version || '—').slice(0, 8)}`),
    summaryCard('Trained', s.trained_at ? fmt.timeAgo(s.trained_at) : '—',
      s.monitoring_enabled ? `eval every ${s.eval_freq} ep` : 'monitoring was off'),
  ].join('');
}

/* ------------------------------------------------------------- main chart */
function renderChart() {
  const spec = METRICS.find((m) => m.label === activeMetric) || METRICS[0];
  const node = ui.el('tmChart');
  const traces = [];

  if (spec.key && (spec.source === 'training' || spec.source === 'both')) {
    const pts = PROGRESS.training.filter((p) => p[spec.key] !== null);
    if (pts.length) {
      traces.push({
        type: 'scatter', mode: 'lines', name: `Training ${spec.label.toLowerCase()}`,
        x: pts.map((p) => p.episode), y: pts.map((p) => p[spec.key]),
        line: { color: C().accent, width: 1.9 },
      });
    }
  }

  if (spec.evalKey && (spec.source === 'eval' || spec.source === 'both')) {
    const pts = (PROGRESS.evaluations || []).filter((e) => e[spec.evalKey] !== null
      && e[spec.evalKey] !== undefined);
    if (pts.length) {
      // Evaluations are discrete events, so markers — a smooth line between
      // two points 5 episodes apart would imply readings that never existed.
      traces.push({
        type: 'scatter', mode: 'lines+markers',
        name: spec.evalLabel || `Eval ${spec.label.toLowerCase()}`,
        x: pts.map((e) => e.episode), y: pts.map((e) => e[spec.evalKey]),
        line: { color: C().green, width: 1.6, dash: 'dot' },
        marker: { size: 9, color: C().green, symbol: 'diamond' },
      });
    }
  }

  node.innerHTML = '';
  if (!traces.length) {
    node.innerHTML = `<div class="empty">No data recorded for ${spec.label} on this run</div>`;
    ui.el('tmChartNote').textContent = '';
    return;
  }

  // Checkpoint events as vertical markers on the episode axis.
  const shapes = (PROGRESS.checkpoints || []).map((c) => ({
    type: 'line', x0: c.episode, x1: c.episode, yref: 'paper', y0: 0, y1: 1,
    line: { color: C().amber, width: 1, dash: 'dash' },
  }));
  const annotations = (PROGRESS.checkpoints || []).map((c) => ({
    x: c.episode, yref: 'paper', y: 1.04, text: '⬤', showarrow: false,
    font: { color: C().amber, size: 8 },
    hovertext: `checkpoint @ episode ${c.episode}`,
  }));

  Plotly.newPlot(node, traces, ui.plotLayout({
    height: 400,
    xaxis: { gridcolor: C().grid, title: { text: 'Episode', font: { size: 10 } } },
    yaxis: { gridcolor: C().grid, title: { text: spec.label, font: { size: 10 } } },
    shapes,
    annotations,
    legend: { orientation: 'h', y: -0.18, font: { size: 10 } },
  }), ui.plotConfig);

  const parts = [];
  if (spec.source === 'both') {
    parts.push('Solid line = training window · dotted = held-out evaluation. '
      + 'A widening gap is the overfitting signal.');
  } else if (spec.source === 'eval') {
    parts.push('Measured on the held-out window at each evaluation.');
  }
  if ((PROGRESS.checkpoints || []).length) {
    parts.push(`Amber markers: ${PROGRESS.checkpoints.length} checkpoint(s) `
      + `every ${PROGRESS.checkpoint_interval} episodes.`);
  }
  ui.el('tmChartNote').textContent = parts.join(' ');
}

function renderTabs() {
  const available = PROGRESS.available_series || {};
  const usable = METRICS.filter((m) => {
    const trainOk = m.key && available[m.key];
    const evalOk = m.evalKey && available[`eval_${m.evalKey}`];
    return trainOk || evalOk;
  });
  if (!usable.some((m) => m.label === activeMetric)) {
    activeMetric = usable.length ? usable[0].label : 'Reward';
  }
  ui.el('tmMetricTabs').innerHTML = usable.map((m) =>
    `<span class="chip ${m.label === activeMetric ? 'active' : ''}"
       data-metric="${m.label}">${m.label}</span>`).join('');
  ui.el('tmMetricTabs').querySelectorAll('.chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      activeMetric = chip.dataset.metric;
      renderTabs();
      renderChart();
    });
  });
}

/* ----------------------------------------------------- evaluation history */
function renderEvaluations() {
  const evals = PROGRESS.evaluations || [];
  ui.el('tmEvalMeta').textContent = evals.length
    ? `${evals.length} evaluation(s) · every ${PROGRESS.eval_freq} episodes`
    : `eval_freq = ${PROGRESS.eval_freq}`;

  ui.el('tmEvalTable').innerHTML = evals.length ? `
    <table><thead><tr>
      <th>Episode</th><th class="t-right">Return</th><th class="t-right">Sharpe</th>
      <th class="t-right">Sortino</th><th class="t-right">Max DD</th>
      <th class="t-right">Vol</th><th class="t-right">VaR₉₅</th><th class="t-right">CVaR₉₅</th>
    </tr></thead><tbody>
    ${evals.map((e) => `
      <tr>
        <td class="mono">${e.episode}</td>
        <td class="t-right mono ${fmt.cls(e.total_return)}">${asPct(e.total_return)}</td>
        <td class="t-right mono ${fmt.cls(e.sharpe_ratio)}">${asNum(e.sharpe_ratio, 2)}</td>
        <td class="t-right mono ${fmt.cls(e.sortino_ratio)}">${asNum(e.sortino_ratio, 2)}</td>
        <td class="t-right mono down">${asPct(e.max_drawdown)}</td>
        <td class="t-right mono">${asPct(e.annualised_volatility)}</td>
        <td class="t-right mono down">${asPct(e.var_95)}</td>
        <td class="t-right mono down">${asPct(e.cvar_95)}</td>
      </tr>`).join('')}
    </tbody></table>`
    : `<div class="empty">No periodic evaluations were recorded.<br>
        <span class="text-xs">Set <b>eval_freq</b> in a hyperparameter profile and
        retrain to populate this.</span></div>`;
}

function renderRiskChart() {
  const evals = (PROGRESS.evaluations || []).filter((e) => e.var_95 !== null
    && e.var_95 !== undefined);
  const node = ui.el('tmRiskChart');
  node.innerHTML = '';
  if (!evals.length) {
    node.innerHTML = '<div class="empty">No tail-risk data recorded</div>';
    return;
  }
  const x = evals.map((e) => e.episode);
  Plotly.newPlot(node, [
    { type: 'bar', name: 'VaR 95%', x, y: evals.map((e) => e.var_95 * 100),
      marker: { color: 'rgba(245, 158, 66,.75)' } },
    { type: 'bar', name: 'CVaR 95%', x, y: evals.map((e) => e.cvar_95 * 100),
      marker: { color: 'rgba(255, 92, 92,.75)' } },
    { type: 'scatter', mode: 'lines+markers', name: 'Max drawdown', yaxis: 'y2',
      x, y: evals.map((e) => (e.max_drawdown ?? 0) * 100),
      line: { color: C().accent, width: 1.6 } },
  ], ui.plotLayout({
    height: 260, barmode: 'group',
    xaxis: { gridcolor: C().grid, title: { text: 'Episode', font: { size: 10 } } },
    yaxis: { gridcolor: C().grid, title: { text: 'Daily tail loss (%)', font: { size: 10 } } },
    yaxis2: { overlaying: 'y', side: 'right', title: { text: 'Drawdown (%)', font: { size: 10 } },
      showgrid: false },
    legend: { orientation: 'h', y: -0.25, font: { size: 9 } },
  }), ui.plotConfig);
}

/* ------------------------------------------------------ checkpoint manager */
function renderCheckpoints() {
  const rows = CHECKPOINTS;
  ui.el('tmRetention').textContent = rows.length
    ? `${rows.filter((c) => c.exists).length} on disk · retention keeps the last ${RETENTION}`
    : '';

  ui.el('tmCheckpoints').innerHTML = rows.length ? `
    <table><thead><tr>
      <th style="width:34px"></th><th>Episode</th><th>Symbol</th><th>Algorithm</th>
      <th>Created</th><th class="t-right">Step</th><th>Profile</th>
      <th class="t-right">Seed</th><th>Version</th><th class="t-right">Size</th><th></th>
    </tr></thead><tbody>
    ${rows.map((c) => `
      <tr class="${c.exists ? '' : 'tm-gone'}">
        <td>${c.exists ? `<input type="checkbox" class="tm-pick"
          data-symbol="${c.symbol}" data-algo="${c.algo}" data-episode="${c.episode}">` : ''}</td>
        <td class="mono">${c.episode}</td>
        <td class="sym-cell">${c.symbol}</td>
        <td>${(c.algo || '').toUpperCase()}</td>
        <td class="text-xs text-muted">${c.created_at ? fmt.timeAgo(c.created_at)
    : '<span title="Saved before creation time was recorded">not recorded</span>'}</td>
        <td class="t-right mono">${c.training_step ?? '—'}</td>
        <td>${c.profile || '—'}</td>
        <td class="t-right mono">${c.seed ?? '—'}</td>
        <td class="mono text-xs">${(c.model_version || '—').slice(0, 10)}</td>
        <td class="t-right mono text-xs">${c.bytes ? `${Math.round(c.bytes / 1024)} KB` : '—'}</td>
        <td class="t-right">
          ${c.exists ? `
            <button class="btn btn-sm tm-restore" data-symbol="${c.symbol}"
              data-algo="${c.algo}" data-episode="${c.episode}"
              title="Overwrite the live agent with this snapshot">Restore</button>
            <button class="btn btn-sm tm-delete" data-symbol="${c.symbol}"
              data-algo="${c.algo}" data-episode="${c.episode}">Delete</button>`
    : '<span class="text-xs text-muted">pruned</span>'}
        </td>
      </tr>`).join('')}
    </tbody></table>`
    : '<div class="empty">No checkpoints recorded yet</div>';

  ui.el('tmCheckpoints').querySelectorAll('.tm-pick').forEach((box) => {
    box.addEventListener('change', () => {
      const id = `${box.dataset.symbol}|${box.dataset.algo}|${box.dataset.episode}`;
      if (box.checked) SELECTED.add(id); else SELECTED.delete(id);
      ui.el('tmCompareBtn').disabled = SELECTED.size !== 2;
    });
  });
  ui.el('tmCheckpoints').querySelectorAll('.tm-restore').forEach((btn) => {
    btn.addEventListener('click', () => restoreCheckpoint(btn.dataset));
  });
  ui.el('tmCheckpoints').querySelectorAll('.tm-delete').forEach((btn) => {
    btn.addEventListener('click', () => deleteCheckpoint(btn.dataset));
  });
}

async function restoreCheckpoint(d) {
  if (!confirm(`Overwrite the live ${d.algo.toUpperCase()} agent for ${d.symbol} `
    + `with the episode ${d.episode} snapshot?\n\nThe current model is backed up first.`)) return;
  try {
    const r = await api.tmRestoreCheckpoint(d.symbol, d.algo, parseInt(d.episode, 10));
    ui.toast(`Restored episode ${r.episode} (backup: ${r.backup || 'none'})`, 'success');
    ui.el('tmNotice').hidden = false;
    ui.el('tmNotice').innerHTML = `<b>Live agent replaced.</b> ${r.warning}`;
  } catch (err) { ui.toast(err.message, 'error'); }
}

async function deleteCheckpoint(d) {
  if (!confirm(`Delete the episode ${d.episode} checkpoint for ${d.symbol}?`)) return;
  try {
    await api.tmDeleteCheckpoint(d.symbol, d.algo, parseInt(d.episode, 10));
    ui.toast(`Checkpoint at episode ${d.episode} deleted`, 'success');
    await loadCheckpoints();
  } catch (err) { ui.toast(err.message, 'error'); }
}

async function compareSelected() {
  const [a, b] = [...SELECTED].map((id) => {
    const [symbol, algo, episode] = id.split('|');
    return { symbol, algo, episode: parseInt(episode, 10) };
  });
  try {
    const r = await api.tmCompareCheckpoints(a, b);
    ui.el('tmCompare').innerHTML = `
      <div class="stat-label mt-2 mb-1">Comparison</div>
      ${r.comparable ? `
        <table><thead><tr><th>Metric</th>
          <th class="t-right">Episode ${r.left.episode}</th>
          <th class="t-right">Episode ${r.right.episode}</th>
          <th class="t-right">Δ</th></tr></thead><tbody>
        ${r.metrics.map((m) => `
          <tr><td>${m.metric.replace(/_/g, ' ')}</td>
            <td class="t-right mono">${asNum(m.left, 4)}</td>
            <td class="t-right mono">${asNum(m.right, 4)}</td>
            <td class="t-right mono ${fmt.cls(m.delta)}">${
  m.delta === null ? '—' : (m.delta > 0 ? '+' : '') + asNum(m.delta, 4)}</td></tr>`).join('')}
        </tbody></table>`
    : `<div class="info-box text-xs">${r.note}</div>`}
      ${r.hyperparameter_differences.length ? `
        <div class="stat-label mt-2 mb-1">Hyperparameter differences</div>
        ${r.hyperparameter_differences.slice(0, 10).map((d) => `
          <div class="signal-row"><span class="signal-name">${d.parameter}</span>
            <span class="mono text-xs">${d.left} → ${d.right}</span></div>`).join('')}`
    : '<div class="text-xs text-muted mt-1">Identical hyperparameters.</div>'}`;
  } catch (err) { ui.toast(err.message, 'error'); }
}

/* ------------------------------------------------------------------ load */
let RETENTION = 5;

async function loadCheckpoints() {
  const data = await api.tmCheckpoints();
  CHECKPOINTS = data.checkpoints || [];
  RETENTION = data.retention?.max_checkpoints ?? 5;
  SELECTED.clear();
  ui.el('tmCompareBtn').disabled = true;
  renderCheckpoints();
}

async function loadRun() {
  const value = ui.el('tmRun').value;
  if (!value) return;
  const [symbol, algo] = value.split('|');
  ui.el('tmRunBadge').textContent = `${symbol} · ${algo.toUpperCase()}`;
  ui.loading('tmChart', 'Loading training record…');
  try {
    const [progress, sum] = await Promise.all([
      api.tmProgress(symbol, algo), api.tmSummary(symbol, algo),
    ]);
    PROGRESS = progress;
    renderSummary(sum);
    renderTabs();
    renderChart();
    renderEvaluations();
    renderRiskChart();

    const notice = ui.el('tmNotice');
    if (!progress.monitoring_enabled || !(progress.evaluations || []).length) {
      notice.hidden = false;
      notice.innerHTML = `<b>Limited data for this run.</b> ${progress.note}`;
    } else {
      notice.hidden = true;
    }
  } catch (err) {
    ui.error('tmChart', err.message);
  }
}


/* ============================================================
   Training Intelligence — fleet view

   The monitor above answers "what happened inside this run". This answers
   "which of my models needs attention", across every symbol and algorithm.
   Both read the same backend records; neither recomputes a metric.
   ============================================================ */

let FLEET = null;

const STATUS_STYLE = {
  converged: 'badge-green',
  improving: 'badge-blue',
  plateaued: 'badge-amber',
  overfitting: 'badge-red',
  unstable: 'badge-red',
  insufficient_data: 'badge-grey',
};

/* Plotly draws into SVG and cannot resolve CSS custom properties: passing
   `var(--green)` produced solid black bars. C() reads the computed palette, so
   the charts follow the active theme; gradeColour() keeps one source for both
   charts and HTML. */
function gradeColour(grade) {
  const palette = C();
  return {
    excellent: palette.green, good: palette.accent,
    fair: palette.amber, poor: palette.red,
  // No literal fallback: every colour must come from the theme, and a
  // hard-coded hex here would be invisible in one of the two palettes.
  }[grade] || palette.muted || palette.grid;
}

const pc = (v, d = 2) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(d)}%`);
const nb = (v, d = 2) => (v === null || v === undefined ? '—' : Number(v).toFixed(d));

function globalCard(label, value, sub, cls = '') {
  return `<div class="card">
      <div class="stat-label">${label}</div>
      <div class="stat-value ${cls}">${value}</div>
      <div class="text-xs text-muted mt-1">${sub || ''}</div>
    </div>`;
}

function renderGlobal(data) {
  const g = data.global;
  const counts = Object.entries(g.status_counts || {})
    .map(([k, v]) => `${v} ${k.replace(/_/g, ' ')}`).join(' · ');
  ui.el('tiGlobal').innerHTML = [
    globalCard('Trained models', g.trained_models,
      `${g.symbols_covered} symbols · ${g.algorithms_used} algorithms`),
    globalCard('Fleet health', g.mean_health === null ? '—' : `${g.mean_health}%`,
      'mean health score across scored runs',
      (g.mean_health ?? 0) >= 62 ? 'up' : (g.mean_health ?? 0) >= 45 ? '' : 'down'),
    globalCard('Needs attention', (g.needs_attention || []).length,
      'overfitting or unstable',
      (g.needs_attention || []).length ? 'down' : 'up'),
    globalCard('Status mix', `${data.count}`, counts || 'no runs'),
  ].join('');

  const attention = ui.el('tiAttention');
  if ((g.needs_attention || []).length) {
    attention.hidden = false;
    attention.innerHTML = `<b>${g.needs_attention.length} model(s) need attention:</b> `
      + g.needs_attention.map((a) =>
        `${a.symbol}/${String(a.algo).toUpperCase()} — ${a.status} → <b>${a.action}</b>`)
        .join(' · ');
  } else {
    attention.hidden = true;
  }
}

function renderFleet(runs) {
  ui.el('tiCount').textContent = `${runs.length} model(s)`;
  ui.el('tiFleet').innerHTML = runs.length ? `
    <table><thead><tr>
      <th>Symbol</th><th>Algorithm</th><th>Status</th>
      <th class="t-right">Health</th><th class="t-right">Return</th>
      <th class="t-right">Sharpe</th><th class="t-right">Max DD</th>
      <th class="t-right">Episodes</th><th>Recommendation</th><th></th>
    </tr></thead><tbody>
    ${runs.map((r) => `
      <tr>
        <td class="sym-cell">${r.symbol}</td>
        <td>${String(r.algo || '').toUpperCase()}
          ${r.regime_aware ? '<span class="badge badge-purple" title="Regime-aware agent">A</span>' : ''}</td>
        <td><span class="badge ${STATUS_STYLE[r.status] || 'badge-grey'}">${r.status_label}</span></td>
        <td class="t-right mono" style="color:${gradeColour(r.health?.grade)}">
          ${r.health?.percent ?? '—'}</td>
        <td class="t-right mono ${fmt.cls(r.metrics.total_return)}">${pc(r.metrics.total_return)}</td>
        <td class="t-right mono">${nb(r.metrics.sharpe_ratio)}</td>
        <td class="t-right mono down">${pc(r.metrics.max_drawdown, 1)}</td>
        <td class="t-right mono">${r.episodes}</td>
        <td class="text-xs">${r.recommendation.action}</td>
        <td class="t-right"><button class="btn btn-sm ti-open"
          data-symbol="${r.symbol}" data-algo="${r.algo}">Inspect</button></td>
      </tr>`).join('')}
    </tbody></table>`
    : '<div class="empty">No models match these filters</div>';

  ui.el('tiFleet').querySelectorAll('.ti-open').forEach((btn) => {
    btn.addEventListener('click', () => openDetail(btn.dataset.symbol, btn.dataset.algo));
  });
}

function renderRanking(data) {
  const rows = data.overall_ranking || [];
  ui.el('tiRanking').innerHTML = rows.length ? rows.map((r) => `
    <div class="signal-row">
      <span class="signal-name"><b>#${r.rank}</b> ${r.symbol}
        <span class="text-muted">${String(r.algo).toUpperCase()}</span></span>
      <span class="mono" style="color:${gradeColour(r.grade)}">${r.health}%</span>
    </div>`).join('')
    : '<div class="empty">Nothing scored yet</div>';
}

function renderHealthChart(runs) {
  const node = ui.el('tiHealthChart');
  const scored = runs.filter((r) => r.health?.percent !== undefined && r.health?.percent !== null);
  node.innerHTML = '';
  if (!scored.length) { node.innerHTML = '<div class="empty">No scored models</div>'; return; }
  const sorted = [...scored].sort((a, b) => a.health.percent - b.health.percent);
  Plotly.newPlot(node, [{
    type: 'bar', orientation: 'h',
    x: sorted.map((r) => r.health.percent),
    y: sorted.map((r) => `${r.symbol} ${String(r.algo).toUpperCase()}`),
    marker: { color: sorted.map((r) => gradeColour(r.health.grade)) },
    hovertemplate: '%{y}<br>health %{x}/100<extra></extra>',
  }], ui.plotLayout({
    height: Math.max(240, sorted.length * 26 + 60),
    xaxis: { gridcolor: C().grid, range: [0, 100], title: { text: 'Health score', font: { size: 10 } } },
    yaxis: { gridcolor: C().grid, automargin: true, tickfont: { size: 9 } },
    showlegend: false,
  }), ui.plotConfig);
}

function renderScatter(runs) {
  const node = ui.el('tiScatter');
  const pts = runs.filter((r) => r.metrics.total_return !== null
    && r.metrics.max_drawdown !== null);
  node.innerHTML = '';
  if (!pts.length) { node.innerHTML = '<div class="empty">No performance data</div>'; return; }
  Plotly.newPlot(node, [{
    type: 'scatter', mode: 'markers+text',
    x: pts.map((r) => Math.abs(r.metrics.max_drawdown) * 100),
    y: pts.map((r) => r.metrics.total_return * 100),
    text: pts.map((r) => String(r.algo).toUpperCase()),
    textposition: 'top center', textfont: { size: 8 },
    marker: {
      size: pts.map((r) => 8 + (r.health?.percent ?? 40) / 8),
      color: pts.map((r) => gradeColour(r.health?.grade)),
      line: { width: 1, color: 'rgba(0,0,0,.25)' },
    },
    hovertemplate: '%{text}<br>return %{y:.2f}%<br>drawdown %{x:.2f}%<extra></extra>',
  }], ui.plotLayout({
    height: 300,
    xaxis: { gridcolor: C().grid, title: { text: 'Max drawdown (%)', font: { size: 10 } } },
    yaxis: { gridcolor: C().grid, title: { text: 'Total return (%)', font: { size: 10 } }, zeroline: true },
    showlegend: false,
  }), ui.plotConfig);
}

function renderLeaderboard(data) {
  const boards = data.leaderboard || [];
  // A symbol with one trained algorithm has no ranking to show; rendering an
  // empty table with only headers looked like a loading failure. Say what is
  // actually known instead: the single model and what it would take to rank.
  ui.el('tiLeaderboard').innerHTML = boards.length ? boards.map((b) => `
    <div class="mb-2">
      <div class="stat-label mb-1">${b.symbol}
        <span class="badge badge-green">${String(b.best.algo).toUpperCase()} — ${b.best.health}%</span></div>
      ${b.ranking.length < 2 ? `<div class="text-xs text-muted">Only one algorithm
        trained on ${b.symbol} (${String(b.best.algo).toUpperCase()},
        return ${pc(b.best.total_return)}). Train a second to rank them.</div>`
    : `<table><thead><tr><th>#</th><th>Algorithm</th>
        <th class="t-right">Health</th><th class="t-right">Return</th>
        <th class="t-right">Sharpe</th></tr></thead><tbody>
        ${b.ranking.map((r) => `<tr>
          <td class="mono">${r.rank}</td>
          <td>${String(r.algo).toUpperCase()}</td>
          <td class="t-right mono">${r.health}</td>
          <td class="t-right mono ${fmt.cls(r.total_return)}">${pc(r.total_return)}</td>
          <td class="t-right mono">${nb(r.sharpe_ratio)}</td></tr>`).join('')}
      </tbody></table>`}
    </div>`).join('')
    : '<div class="empty">No scored models yet</div>';
}

function renderAdaptive(data) {
  const a = data.adaptive_vs_legacy;
  const row = (label, s) => `
    <div class="signal-row"><span class="signal-name">${label}</span>
      <span class="mono">${s.runs ? `${s.runs} runs · health ${s.mean_health}% · return ${pc(s.mean_return)}`
    : 'no runs'}</span></div>`;
  ui.el('tiAdaptive').innerHTML = row('Adaptive (regime-aware)', a.adaptive)
    + row('Legacy', a.legacy)
    + `<div class="text-xs text-muted mt-2">${a.note}</div>`;
}

function renderSeeds(data) {
  const s = data.seed_statistics;
  ui.el('tiSeeds').innerHTML = s.available
    ? `<div class="signal-row"><span class="signal-name">Mean return</span>
         <span class="mono">${pc(s.mean_return)} ± ${pc(s.std_return)}</span></div>
       <div class="text-xs text-muted mt-1">Across seeds ${s.seeds.join(', ')} (${s.runs} runs).</div>`
    : `<div class="info-box text-xs">${s.reason}</div>`;
}

/* Clicking a model loads the existing per-run monitor into the detail panel:
   the fleet view says which run to look at, the monitor says what happened
   inside it. */
async function openDetail(symbol, algo) {
  ui.el('tiDetailCard').hidden = false;
  ui.el('tiDetailTitle').textContent = `${symbol} · ${String(algo).toUpperCase()}`;
  ui.el('tiReport').href = `/api/v1/training/report/${encodeURIComponent(symbol)}`
    + `?algo=${encodeURIComponent(algo)}`;

  const run = (FLEET?.runs || []).find((r) => r.symbol === symbol && r.algo === algo);
  if (run) {
    const rows = (run.health?.contributions || []).map((c) => `
      <div class="score-part">
        <div class="score-part-head"><span>${c.name}</span>
          <span class="mono text-muted">${c.available
    ? `${c.points} pts / ${c.max_points}` : 'not measured'}</span></div>
        ${c.available ? `<div class="score-part-bar">
          <i style="width:${(c.value * 100).toFixed(0)}%"></i></div>` : ''}
        <div class="score-part-detail">${c.detail}</div>
      </div>`).join('');

    ui.el('tiDetail').innerHTML = `
      <div class="flex gap-1 mb-1" style="flex-wrap:wrap;align-items:center">
        <span class="badge ${STATUS_STYLE[run.status]}">${run.status_label}</span>
        <span class="badge badge-grey">${(run.status_confidence * 100).toFixed(0)}% confidence</span>
        <span class="badge badge-purple">health ${run.health?.percent ?? '—'}%</span>
        ${run.regime_aware ? '<span class="badge badge-blue">regime-aware</span>' : ''}
        <span class="text-xs text-muted">profile ${run.profile || '—'} ·
          seed ${run.seed ?? '—'} · v${(run.model_version || '—').slice(0, 8)}</span>
      </div>
      <div class="info-box text-sm mb-1"><b>${run.recommendation.action}.</b>
        ${run.recommendation.rationale}</div>
      <div class="text-xs text-muted mb-2">${run.evidence.map((e) => `• ${e}`).join('<br>')}</div>
      <div class="grid grid-4 mb-2">
        <div><span class="stat-label">Turnover</span>
          <div class="mono">${nb(run.metrics.turnover)}x</div>
          <div class="text-xs text-muted">${run.metrics.n_trades ?? '—'} trades</div></div>
        <div><span class="stat-label">Episode win rate</span>
          <div class="mono">${pc(run.metrics.episode_win_rate, 0)}</div>
          <div class="text-xs text-muted">not per trade</div></div>
        <div><span class="stat-label">VaR₉₅ / CVaR₉₅</span>
          <div class="mono down">${pc(run.metrics.var_95)} / ${pc(run.metrics.cvar_95)}</div>
          <div class="text-xs text-muted">last evaluation</div></div>
        <div><span class="stat-label">Training duration</span>
          <div class="mono text-muted">not recorded</div>
          <div class="text-xs text-muted">loop is never timed</div></div>
      </div>
      <details class="score-detail" open>
        <summary>How the health score is calculated</summary>
        <div class="score-parts">${rows}</div>
        <div class="score-part-detail mt-1">${run.health?.explanation || ''}</div>
      </details>`;
  }

  // Reuse the existing monitor verbatim for the curves and checkpoints.
  const select = ui.el('tmRun');
  select.innerHTML = `<option value="${symbol}|${algo}">${symbol}</option>`;
  select.value = `${symbol}|${algo}`;
  await loadRun();
  await loadCheckpoints();
  ui.el('tiDetailCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function loadFleet() {
  const params = {
    symbol: ui.el('tiSymbol').value,
    algo: ui.el('tiAlgo').value,
    status: ui.el('tiStatus').value,
    search: ui.el('tiSearch').value.trim(),
  };
  ui.loading('tiFleet', 'Analysing training records…');
  try {
    FLEET = await api.tiIntelligence(params);
    ui.el('tiScope').textContent = `${FLEET.count} / ${FLEET.total_runs} models`;
    renderGlobal(FLEET);
    renderFleet(FLEET.runs);
    renderRanking(FLEET);
    renderHealthChart(FLEET.runs);
    renderScatter(FLEET.runs);
    renderLeaderboard(FLEET);
    renderAdaptive(FLEET);
    renderSeeds(FLEET);

    // Populate the facets once, from what actually exists on disk.
    if (!ui.el('tiSymbol').dataset.filled) {
      ui.el('tiSymbol').innerHTML = '<option value="">All symbols</option>'
        + FLEET.facets.symbols.map((s) => `<option value="${s}">${s}</option>`).join('');
      ui.el('tiAlgo').innerHTML = '<option value="">All algorithms</option>'
        + FLEET.facets.algorithms.map((a) => `<option value="${a}">${a.toUpperCase()}</option>`).join('');
      ui.el('tiStatus').innerHTML = '<option value="">All statuses</option>'
        + FLEET.facets.statuses.map((s) =>
          `<option value="${s}">${s.replace(/_/g, ' ')}</option>`).join('');
      ui.el('tiSymbol').dataset.filled = '1';
    }
  } catch (err) {
    ui.error('tiFleet', err.message);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  ['tiSymbol', 'tiAlgo', 'tiStatus'].forEach((id) =>
    ui.el(id).addEventListener('change', loadFleet));
  let timer = null;
  ui.el('tiSearch').addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(loadFleet, 250);
  });
  ui.el('tiRefresh').addEventListener('click', loadFleet);
  ui.el('tiCloseDetail').addEventListener('click', () => {
    ui.el('tiDetailCard').hidden = true;
  });
  ui.el('tmCompareBtn')?.addEventListener('click', compareSelected);
  loadFleet();
});

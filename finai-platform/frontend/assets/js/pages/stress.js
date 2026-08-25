/* AI Stress Testing Engine — page script.
 *
 * Renders what the backend measured and nothing else. Every figure on screen
 * comes from GET /quant/stress-engine/{symbols}, which computes VaR, CVaR,
 * volatility, drawdown and the Euler risk decomposition from realised returns.
 *
 * The one rule this file follows without exception: a null from the API is
 * rendered as N/A with its stated reason. It is never replaced by a zero, a
 * dash that looks like data, or a plausible-looking default.
 */

let SCENARIOS = [];
let BUSY = false;

function setState(text, kind) {
  const el = ui.el('stState');
  el.textContent = text;
  el.className = `badge badge-${kind || 'blue'}`;
}

function note(html) {
  const box = ui.el('stScenarioNote');
  if (!html) { box.hidden = true; box.innerHTML = ''; return; }
  box.hidden = false;
  box.innerHTML = html;
}

/** Render a measured percentage, or N/A when the backend could not measure it. */
function pctOrNa(value, decimals = 2) {
  return (value === null || value === undefined)
    ? '<span class="dir-na">N/A</span>'
    : `${fmt.num(value, decimals)}%`;
}

function moneyOrNa(value) {
  return (value === null || value === undefined)
    ? '<span class="dir-na">N/A</span>' : fmt.money(value);
}

/* ------------------------------------------------------------- rendering */
function renderHero(d) {
  const r = d.resilience || {};
  const score = r.score;
  const tone = score === null || score === undefined ? 'var(--text-3)'
    : score >= 70 ? 'var(--green)'
      : score >= 40 ? 'var(--amber)' : 'var(--red)';
  const verdict = score === null || score === undefined ? 'NOT MEASURABLE'
    : score >= 70 ? 'RESILIENT'
      : score >= 40 ? 'STRAINED' : 'FRAGILE';

  ui.el('stHero').innerHTML = `
    <div class="dir-hero">
      <div class="dir-verdict">
        <div>
          <div class="dir-label" style="color:${tone}">${verdict}</div>
          <div class="dir-sub">${d.scenario_label} \u00b7 ${d.symbols.length}
            ${d.symbols.length === 1 ? 'asset' : 'assets'} \u00b7 ${d.observations} observations</div>
        </div>
      </div>
      <div class="dir-metrics">
        <div class="dir-metric">
          <div class="k">Resilience Score</div>
          <div class="v" style="color:${tone}">${score === null || score === undefined
    ? 'N/A' : `${fmt.num(score, 0)}/100`}</div>
          <div class="s">${score === null || score === undefined
    ? (r.reason || 'not measurable') : '100 = unchanged by the scenario'}</div>
        </div>
        <div class="dir-metric">
          <div class="k">Stressed CVaR</div>
          <div class="v">${pctOrNa(d.after?.cvar_pct)}</div>
          <div class="s">expected loss beyond VaR</div>
        </div>
        <div class="dir-metric">
          <div class="k">Additional Loss</div>
          <div class="v">${moneyOrNa(d.portfolio_loss?.additional_cvar_money)}</div>
          <div class="s">vs. the unstressed book</div>
        </div>
        <div class="dir-metric">
          <div class="k">Worst Drawdown</div>
          <div class="v">${pctOrNa(d.after?.max_drawdown_pct)}</div>
          <div class="s">peak-to-trough under stress</div>
        </div>
      </div>
    </div>
    <div class="info-box text-xs mt-2">
      <strong>${d.scenario_label}</strong> \u2014 ${d.scenario_description}
      Basis: ${d.scenario_basis}.
    </div>`;
}

function renderCompare(d) {
  const rows = [
    ['Value at Risk', 'var_pct', `${fmt.pct(d.confidence * 100, 0, false)} daily VaR`],
    ['Conditional VaR', 'cvar_pct', 'mean loss in the tail beyond VaR'],
    ['Volatility', 'volatility_pct', 'annualised standard deviation'],
    ['Max Drawdown', 'max_drawdown_pct', 'largest peak-to-trough decline'],
    ['Worst Day', 'worst_day_pct', 'single worst observed session'],
  ];
  ui.el('stCompare').innerHTML = `
    <div class="table-scroll">
      <table class="table">
        <thead><tr><th>Measure</th><th class="t-right">Before</th>
          <th class="t-right">After</th><th class="t-right">Change</th>
          <th>Meaning</th></tr></thead>
        <tbody>${rows.map(([label, key, hint]) => {
    const b = d.before?.[key];
    const a = d.after?.[key];
    const delta = (b === null || b === undefined || a === null || a === undefined)
      ? null : a - b;
    // Every measure here is a loss magnitude: bigger is worse.
    const cls = delta === null ? '' : (delta > 0 ? 'down' : delta < 0 ? 'up' : '');
    return `<tr>
            <td>${label}</td>
            <td class="t-right mono">${pctOrNa(b)}</td>
            <td class="t-right mono"><strong>${pctOrNa(a)}</strong></td>
            <td class="t-right mono ${cls}">${delta === null ? '<span class="dir-na">N/A</span>'
      : `${delta > 0 ? '+' : ''}${fmt.num(delta, 2)} pts`}</td>
            <td class="text-xs text-muted">${hint}</td>
          </tr>`;
  }).join('')}</tbody>
      </table>
    </div>
    ${d.before?.reason || d.after?.reason ? `
      <div class="text-xs text-muted mt-1">
        ${d.before?.reason || d.after?.reason}
      </div>` : ''}`;
}

function renderLoss(d) {
  const l = d.portfolio_loss || {};
  ui.el('stLoss').innerHTML = `
    <div class="kv">
      <div class="kv-row"><span>Position value</span>
        <span class="mono">${fmt.money(d.position_value)}</span></div>
      <div class="kv-row"><span>Loss at VaR</span>
        <span class="mono">${moneyOrNa(l.var_money)}</span></div>
      <div class="kv-row"><span>Loss at CVaR</span>
        <span class="mono">${moneyOrNa(l.cvar_money)}</span></div>
      <div class="kv-row"><span>Drawdown exposure</span>
        <span class="mono">${moneyOrNa(l.drawdown_money)}</span></div>
      <div class="kv-row"><span>Additional loss vs. base</span>
        <span class="mono">${moneyOrNa(l.additional_cvar_money)}</span></div>
    </div>
    <div class="text-xs text-muted mt-2">
      Money figures scale the measured percentage by the position value entered
      above; they are not a separate estimate.
    </div>`;
}

function renderContrib(d) {
  const assets = d.assets || [];
  if (assets.length < 2) {
    ui.el('stContrib').innerHTML = `<div class="empty">
      Risk contribution decomposes a portfolio's volatility across its holdings.
      With a single asset it is 100% by definition &mdash; add more symbols
      (comma-separated) to see the split.</div>`;
    return;
  }
  ui.el('stContrib').innerHTML = assets.map((a) => {
    const rc = a.stressed_risk_contribution_pct;
    // A missing decomposition is stated, not drawn as an empty bar at 0% —
    // that would read as "this asset carries no risk", which is a different
    // claim from "this could not be measured".
    if (rc === null || rc === undefined) {
      return `
        <div class="dir-factor mb-1">
          <div class="dir-factor-head">
            <span>${a.symbol} <span class="text-xs text-muted">${fmt.num(a.weight_pct, 1)}% of capital</span></span>
            <span class="mono dir-na">N/A</span>
          </div>
          <div class="text-xs text-muted">risk contribution could not be measured</div>
        </div>`;
    }
    const concentrated = rc > a.weight_pct * 1.5;
    return `
      <div class="dir-factor mb-1">
        <div class="dir-factor-head">
          <span>${a.symbol} <span class="text-xs text-muted">${fmt.num(a.weight_pct, 1)}% of capital</span></span>
          <span class="mono ${concentrated ? 'down' : ''}">${fmt.num(rc, 1)}%</span>
        </div>
        <div class="dp-bar"><div class="dp-bar-fill" style="width:${Math.max(0, Math.min(100, rc)).toFixed(1)}%"></div></div>
        <div class="text-xs text-muted">${concentrated
      ? 'carries more risk than capital \u2014 concentration'
      : 'risk broadly in line with its weight'}</div>
      </div>`;
  }).join('');
}

function renderAssets(d) {
  const assets = d.assets || [];
  ui.el('stAssets').innerHTML = `
    <div class="table-scroll">
      <table class="table">
        <thead><tr><th>Asset</th><th class="t-right">Weight</th>
          <th class="t-right">VaR before</th><th class="t-right">VaR after</th>
          <th class="t-right">CVaR change</th><th class="t-right">Vol change</th>
          <th class="t-right">Share of loss</th></tr></thead>
        <tbody>${assets.map((a) => `
          <tr>
            <td><strong>${a.symbol}</strong></td>
            <td class="t-right mono">${fmt.num(a.weight_pct, 1)}%</td>
            <td class="t-right mono">${pctOrNa(a.before?.var_pct)}</td>
            <td class="t-right mono">${pctOrNa(a.after?.var_pct)}</td>
            <td class="t-right mono ${a.cvar_delta_pct > 0 ? 'down' : ''}">
              ${a.cvar_delta_pct === null || a.cvar_delta_pct === undefined
    ? '<span class="dir-na">N/A</span>'
    : `${a.cvar_delta_pct > 0 ? '+' : ''}${fmt.num(a.cvar_delta_pct, 2)} pts`}</td>
            <td class="t-right mono ${a.volatility_delta_pct > 0 ? 'down' : ''}">
              ${a.volatility_delta_pct === null || a.volatility_delta_pct === undefined
    ? '<span class="dir-na">N/A</span>'
    : `${a.volatility_delta_pct > 0 ? '+' : ''}${fmt.num(a.volatility_delta_pct, 2)} pts`}</td>
            <td class="t-right mono">${a.loss_contribution_pct === null
    || a.loss_contribution_pct === undefined
    ? '<span class="dir-na">N/A</span>' : `${fmt.num(a.loss_contribution_pct, 1)}%`}</td>
          </tr>`).join('')}</tbody>
      </table>
    </div>
    ${d.correlation ? `
      <div class="text-xs text-muted mt-2">
        Average pairwise correlation ${fmt.num(d.correlation.average_correlation, 3)}
        before the scenario${d.stressed_correlation
    ? `, ${fmt.num(d.stressed_correlation.average_correlation, 3)} after` : ''}.
        ${d.correlation.highest_pair?.pair
    ? `Tightest pair: ${d.correlation.highest_pair.pair} at ${d.correlation.highest_pair.correlation}.`
    : ''}
      </div>` : ''}`;
}

function renderNarrative(d) {
  ui.el('stVuln').innerHTML = (d.vulnerabilities || []).map((v) => `
    <div class="dir-summary mb-1">${v}</div>`).join('')
    || '<div class="empty">No vulnerabilities reported.</div>';

  ui.el('stMitig').innerHTML = (d.mitigations || []).map((m) => `
    <div class="dir-summary mb-1">${m}</div>`).join('')
    || '<div class="empty">No recommendations reported.</div>';
}

function renderAll(d) {
  ui.el('stEmpty').hidden = true;
  ui.el('stResult').hidden = false;
  ui.el('stMeta').textContent =
    `${d.scenario_label} \u00b7 ${d.period} window \u00b7 ${fmt.pct(d.confidence * 100, 0, false)} confidence`;
  renderHero(d);
  renderCompare(d);
  renderLoss(d);
  renderContrib(d);
  renderAssets(d);
  renderNarrative(d);
}

/* -------------------------------------------------------------- workflow */
function inputs() {
  const symbols = ui.el('stSymbols').value
    .split(',').map((s) => s.trim().toUpperCase()).filter(Boolean);
  const weightsRaw = ui.el('stWeights').value.trim();
  const weights = weightsRaw
    ? weightsRaw.split(',').map((w) => w.trim()).filter(Boolean) : [];
  return {
    symbols,
    weights,
    scenario: ui.el('stScenario').value,
    period: ui.el('stPeriod').value,
    confidence: parseFloat(ui.el('stConfidence').value),
    positionValue: parseFloat(ui.el('stValue').value) || 100000,
    volMultiplier: parseFloat(ui.el('stVolMult').value) || 2,
  };
}

async function run() {
  if (BUSY) return;
  const cfg = inputs();
  if (!cfg.symbols.length) { note('<strong>Enter at least one symbol.</strong>'); return; }
  if (cfg.weights.length && cfg.weights.length !== cfg.symbols.length) {
    note(`<strong>${cfg.weights.length} weights for ${cfg.symbols.length} symbols.</strong>
          Leave the field empty for an equal-weighted book.`);
    return;
  }

  BUSY = true;
  ui.el('stRun').disabled = true;
  setState('Running\u2026', 'amber');
  note('');
  try {
    const d = await api.stressEngine(cfg.symbols, cfg);
    renderAll(d);
    const score = d.resilience?.score;
    setState(score === null || score === undefined ? 'Complete' : `Resilience ${fmt.num(score, 0)}`,
      score === null || score === undefined ? 'grey'
        : score >= 70 ? 'green' : score >= 40 ? 'amber' : 'red');
    if (d.skipped) {
      note(`<strong>Skipped:</strong> ${Object.entries(d.skipped)
        .map(([s, why]) => `${s} (${why})`).join(', ')}`);
    }
  } catch (e) {
    setState('Failed', 'red');
    note(`<strong>Stress test failed.</strong> ${e.message || e}`);
  } finally {
    BUSY = false;
    ui.el('stRun').disabled = false;
  }
}

async function loadScenarios() {
  try {
    const data = await api.stressScenarios();
    SCENARIOS = data.scenarios || [];
    ui.el('stScenario').innerHTML = SCENARIOS
      .map((s) => `<option value="${s.key}">${s.label}</option>`).join('');
    describeScenario();
  } catch (e) { /* keep the static fallback option */ }
}

function describeScenario() {
  const key = ui.el('stScenario').value;
  const found = SCENARIOS.find((s) => s.key === key);
  // The custom scenario is the only one that takes a user parameter.
  ui.el('stCustomWrap').hidden = !(key === 'custom' || key === 'vol_x2');
  if (found) note(`<strong>${found.label}</strong> \u2014 ${found.description} Basis: ${found.basis}`);
}

document.addEventListener('DOMContentLoaded', () => {
  loadScenarios();
  ui.el('stRun').addEventListener('click', run);
  ui.el('stScenario').addEventListener('change', describeScenario);
  ui.el('stSymbols').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') run();
  });
  // Shared picker appends to the basket rather than replacing it, so building
  // a portfolio does not mean retyping the tickers already chosen.
  new SymbolPicker('stSymbols', 'stSymbolsPanel', (symbol) => {
    const field = ui.el('stSymbols');
    const current = field.value.split(',').map((s) => s.trim()).filter(Boolean);
    if (symbol && !current.includes(symbol.toUpperCase())) {
      field.value = [...current, symbol.toUpperCase()].join(',');
    }
    run();
  });
});

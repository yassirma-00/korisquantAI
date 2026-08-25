/* ============================================================
   Page: Hyperparameter Management

   Edits configs/ from the browser. The form is generated from the resolved
   configuration rather than hand-written, so a parameter added to a YAML file
   appears here without touching this file — a hardcoded form would reintroduce
   exactly the coupling this feature exists to remove.
   ============================================================ */

let CATALOGUE = null;
let RESOLVED = null;
const EDITS = {};        // section -> key -> value, only what the user changed

const SECTION_HELP = {
  training: 'Run length, seed and the train/test split.',
  optimizer: 'Learning rate, discount factor, batch size, gradient clipping.',
  network: 'Layer widths and value-based architecture switches.',
  replay: 'Replay buffer size and target-network synchronisation.',
  exploration: 'Epsilon schedule. Inert for agents that use NoisyNet.',
  environment: 'Starting capital, transaction costs, slippage, position sizing.',
  risk: 'Reward-shaping coefficients and regime adaptation.',
  evaluation: 'Evaluation frequency and checkpoint interval.',
  policy_gradient: 'PPO / A2C / TRPO settings. Null means "not used by this algorithm".',
  off_policy: 'SAC / TD3 / DDPG settings.',
  distributional: 'C51 / IQN / Rainbow settings.',
};

const fieldId = (section, key) => `hp__${section}__${key}`;

/* A value's editor depends on its type, not on a hardcoded list of fields. */
function fieldRow(section, key, value) {
  const id = fieldId(section, key);
  const disabled = value === null ? ' disabled' : '';
  let control;
  if (typeof value === 'boolean') {
    control = `<select id="${id}" data-section="${section}" data-key="${key}">
      <option value="true"${value ? ' selected' : ''}>true</option>
      <option value="false"${!value ? ' selected' : ''}>false</option></select>`;
  } else if (Array.isArray(value)) {
    control = `<input type="text" id="${id}" data-section="${section}" data-key="${key}"
      data-kind="list" value="${value.join(', ')}">`;
  } else if (typeof value === 'number') {
    control = `<input type="number" id="${id}" data-section="${section}" data-key="${key}"
      data-kind="number" step="any" value="${value}">`;
  } else {
    control = `<input type="text" id="${id}" data-section="${section}" data-key="${key}"
      value="${value === null ? '' : value}"${disabled}
      placeholder="${value === null ? 'not used by this algorithm' : ''}">`;
  }
  return `
    <div class="hp-field">
      <label for="${id}">${key.replace(/_/g, ' ')}</label>
      ${control}
    </div>`;
}

function renderForm(resolved) {
  const params = resolved.params || {};
  const order = (CATALOGUE?.sections || Object.keys(params))
    .filter((s) => s !== 'meta' && params[s] && Object.keys(params[s]).length);

  ui.el('hpForm').innerHTML = order.map((section) => `
    <details class="score-detail" ${['training', 'optimizer', 'risk'].includes(section) ? 'open' : ''}>
      <summary>${section.replace(/_/g, ' ')}</summary>
      <div class="score-part-detail mb-1">${SECTION_HELP[section] || ''}</div>
      <div class="hp-grid">
        ${Object.entries(params[section]).map(([k, v]) => fieldRow(section, k, v)).join('')}
      </div>
    </details>`).join('');

  ui.el('hpForm').querySelectorAll('input,select').forEach((el) => {
    el.addEventListener('change', () => {
      const { section, key, kind } = el.dataset;
      let value;
      if (el.tagName === 'SELECT') value = el.value === 'true';
      else if (kind === 'number') value = el.value === '' ? null : Number(el.value);
      else if (kind === 'list') {
        value = el.value.split(',').map((x) => parseInt(x.trim(), 10)).filter((x) => !Number.isNaN(x));
      } else value = el.value;
      EDITS[section] = { ...(EDITS[section] || {}), [key]: value };
      renderPreview();
    });
  });
  renderPreview();
}

/* Show the merged result, not the diff: the effective value is the thing a
   user needs to see before starting a run. */
function renderPreview() {
  const merged = JSON.parse(JSON.stringify(RESOLVED?.params || {}));
  Object.entries(EDITS).forEach(([section, kv]) => {
    merged[section] = { ...(merged[section] || {}), ...kv };
  });
  const lines = [];
  Object.entries(merged).forEach(([section, values]) => {
    if (section === 'meta' || !values || !Object.keys(values).length) return;
    lines.push(`${section}:`);
    Object.entries(values).forEach(([k, v]) => {
      const changed = EDITS[section] && k in EDITS[section];
      const shown = Array.isArray(v) ? `[${v.join(', ')}]` : v === null ? 'null' : v;
      lines.push(`  ${k}: ${shown}${changed ? '    # edited' : ''}`);
    });
  });
  ui.el('hpPreview').textContent = lines.join('\n');
  const dirty = Object.keys(EDITS).length > 0;
  ui.el('hpSaveBtn').textContent = dirty ? 'Save changes' : 'Save';
  ui.el('hpSaveBtn').classList.toggle('btn-primary', dirty);
}

async function loadResolved() {
  const algo = ui.el('hpAlgo').value;
  const profile = ui.el('hpProfile').value;
  ui.loading('hpForm', 'Resolving configuration…');
  try {
    RESOLVED = await api.hpResolve(algo, profile);
    ui.el('hpFingerprint').textContent = `fingerprint ${RESOLVED.fingerprint}`;
    ui.el('hpSources').textContent = (RESOLVED.sources || []).join('  →  ');
    ui.el('hpActiveProfile').textContent = profile;

    const meta = (CATALOGUE.profiles || []).find((p) => p.key === profile) || {};
    ui.el('hpProfileDesc').textContent = meta.description || '';
    const builtin = !!meta.builtin;
    ui.el('hpBuiltinNote').hidden = !builtin;
    ui.el('hpSaveBtn').disabled = builtin;
    ui.el('hpSaveBtn').title = builtin
      ? 'Built-in profiles are read-only. Duplicate this one to edit it.' : '';

    Object.keys(EDITS).forEach((k) => delete EDITS[k]);
    renderForm(RESOLVED);
  } catch (err) {
    ui.error('hpForm', err.message);
  }
}

async function loadCatalogue() {
  CATALOGUE = await api.hpCatalogue();
  ui.el('hpProfile').innerHTML = CATALOGUE.profiles
    .map((p) => `<option value="${p.key}">${p.name}${p.builtin ? ' (built-in)' : ''}</option>`).join('');
  ui.el('hpAlgo').innerHTML = CATALOGUE.algorithms
    .map((a) => `<option value="${a.key}">${a.key.toUpperCase()} — ${a.family || ''}</option>`).join('');
  ui.el('hpAlgo').value = 'dueling_dqn';
  await loadResolved();
}

async function saveProfile() {
  const profile = ui.el('hpProfile').value;
  if (!Object.keys(EDITS).length) {
    ui.toast('Nothing changed yet', 'info');
    return;
  }
  try {
    await api.hpSaveProfile(profile, EDITS, '');
    ui.toast(`Profile "${profile}" saved`, 'success');
    Object.keys(EDITS).forEach((k) => delete EDITS[k]);
    await loadResolved();
  } catch (err) {
    ui.toast(err.message, 'error');
  }
}

async function duplicateProfile() {
  const source = ui.el('hpProfile').value;
  const name = prompt(`Duplicate "${source}" as:`, `${source}_copy`);
  if (!name) return;
  try {
    await api.hpDuplicateProfile(source, name);
    CATALOGUE = await api.hpCatalogue();
    ui.el('hpProfile').innerHTML = CATALOGUE.profiles
      .map((p) => `<option value="${p.key}">${p.name}${p.builtin ? ' (built-in)' : ''}</option>`).join('');
    ui.el('hpProfile').value = name.toLowerCase().replace(/ /g, '_');
    await loadResolved();
    ui.toast(`Created "${name}"`, 'success');
  } catch (err) {
    ui.toast(err.message, 'error');
  }
}

function exportProfile() {
  const profile = ui.el('hpProfile').value;
  // Same-origin download; the endpoint sets Content-Disposition.
  window.location.href = `/api/v1/hyperparams/profiles/${encodeURIComponent(profile)}/export`;
}

function importProfile() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.yaml,.yml,text/yaml';
  input.addEventListener('change', async () => {
    const file = input.files?.[0];
    if (!file) return;
    const name = prompt('Import as profile name:', file.name.replace(/\.(ya?ml)$/i, ''));
    if (!name) return;
    try {
      await api.hpImportProfile(name, await file.text());
      CATALOGUE = await api.hpCatalogue();
      ui.el('hpProfile').innerHTML = CATALOGUE.profiles
        .map((p) => `<option value="${p.key}">${p.name}${p.builtin ? ' (built-in)' : ''}</option>`).join('');
      ui.el('hpProfile').value = name.toLowerCase().replace(/ /g, '_');
      await loadResolved();
      ui.toast(`Imported "${name}"`, 'success');
    } catch (err) {
      ui.toast(err.message, 'error');
    }
  });
  input.click();
}

/* Every run records the configuration that produced it, so a result can be
   traced back to its exact parameters rather than to a filename. */
async function loadRuns() {
  try {
    const data = await api.hpExperiments(40);
    const rows = data.experiments || [];
    ui.el('hpRuns').innerHTML = rows.length ? `
      <table><thead><tr>
        <th>Experiment</th><th>Symbol</th><th>Algorithm</th><th>Profile</th>
        <th class="t-right">Seed</th><th>Fingerprint</th>
        <th class="t-right">Test return</th><th>Trained</th>
      </tr></thead><tbody>
      ${rows.map((r) => `
        <tr>
          <td class="mono text-xs">${r.experiment_id
    || '<span class="text-muted">before config tracking</span>'}</td>
          <td class="sym-cell">${r.symbol || '—'}</td>
          <td>${(r.algo || '—').toUpperCase()}</td>
          <td>${r.profile || '<span class="text-muted">—</span>'}</td>
          <td class="t-right mono">${r.seed ?? '—'}</td>
          <td class="mono text-xs">${r.fingerprint || '—'}</td>
          <td class="t-right mono ${fmt.cls(r.test_performance?.total_return)}">${
  r.test_performance?.total_return === undefined ? '—'
    : fmt.pct(r.test_performance.total_return * 100)}</td>
          <td class="text-xs text-muted">${r.trained_at ? fmt.timeAgo(r.trained_at) : '—'}</td>
        </tr>`).join('')}
      </tbody></table>`
      : '<div class="empty">No training runs recorded yet</div>';
  } catch (err) {
    ui.error('hpRuns', err.message);
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  ui.el('hpProfile').addEventListener('change', loadResolved);
  ui.el('hpAlgo').addEventListener('change', loadResolved);
  ui.el('hpSaveBtn').addEventListener('click', saveProfile);
  ui.el('hpNewBtn').addEventListener('click', duplicateProfile);
  ui.el('hpExportBtn').addEventListener('click', exportProfile);
  ui.el('hpImportBtn').addEventListener('click', importProfile);
  ui.el('hpRefreshRuns').addEventListener('click', loadRuns);
  try {
    await loadCatalogue();
  } catch (err) {
    ui.error('hpForm', err.message);
  }
  loadRuns();
});

/* ============================================================
   Smart Configuration — standard mode

   A standard user picks a training profile and gets a summary. The full
   parameter set still exists and is still recorded for reproducibility; it is
   simply not the thing being asked of them. Advanced Mode reveals it.
   ============================================================ */

let RECOMMENDATION = null;

function renderSmartSummary(r) {
  const t = r.estimated_training;
  const q = r.expected_quality;
  const c = r.confidence;
  const tone = (p) => (p >= 80 ? 'up' : p >= 55 ? '' : 'down');

  ui.el('hpSmartSummary').innerHTML = `
    <div class="grid grid-4 mb-2">
      <div class="card"><div class="stat-label">Selected profile</div>
        <div class="stat-value" style="font-size:19px">${r.profile.replace(/_/g, ' ')}</div>
        <div class="text-xs text-muted mt-1">${r.profile_reason}</div></div>
      <div class="card"><div class="stat-label">Estimated training</div>
        <div class="stat-value">${t.human}</div>
        <div class="text-xs text-muted mt-1">${t.steps.toLocaleString()} steps</div></div>
      <div class="card"><div class="stat-label">Setup quality</div>
        <div class="stat-value ${tone(q.percent)}">${q.percent.toFixed(0)}</div>
        <div class="text-xs text-muted mt-1">${q.band}</div></div>
      <div class="card"><div class="stat-label">Confidence</div>
        <div class="stat-value ${tone(c.percent)}">${c.percent.toFixed(0)}%</div>
        <div class="text-xs text-muted mt-1">in the recommendation</div></div>
    </div>

    <div class="info-box text-sm mb-1">${r.summary}</div>

    <details class="score-detail">
      <summary>What the platform looked at (${r.adjustments.length} adjustment(s))</summary>
      <div class="score-parts">
        ${r.adjustments.map((a) => `
          <div class="score-part">
            <div class="score-part-head"><span>${a.path.replace(/\./g, ' · ')}</span>
              <span class="mono text-muted">${JSON.stringify(a.value)}</span></div>
            <div class="score-part-detail">${a.reason}</div>
          </div>`).join('') || '<div class="score-part-detail">No adjustment was '
    + 'needed: the profile defaults already suit this environment.</div>'}
      </div>
      <div class="score-part-detail mt-1">
        Environment: ${r.environment.bars} bars ·
        ${r.environment.regime.replace(/_/g, ' ')}
        (${(r.environment.regime_confidence * 100).toFixed(0)}% confidence) ·
        ${r.environment.n_assets} asset(s) ·
        ${r.environment.cpu_count} CPU core(s)${r.environment.cuda ? ' + CUDA' : ', no GPU'}
      </div>
      <div class="score-part-detail mt-1"><b>Quality means:</b> ${q.meaning}</div>
      <div class="score-part-detail"><b>Timing:</b> ${t.basis}. ${t.caveat}</div>
      <div class="score-part-detail"><b>Method:</b> ${r.method}</div>
      ${(r.notes || []).map((n) => `<div class="score-part-detail basis-warn">⚠ ${n}</div>`).join('')}
    </details>`;
}

async function analyseEnvironment() {
  const btn = ui.el('hpAnalyse');
  btn.disabled = true; btn.textContent = 'Analysing…';
  ui.loading('hpSmartSummary', 'Reading market regime, data size and hardware…');
  try {
    RECOMMENDATION = await api.hpSmartRecommend({
      symbol: ui.el('hpSymbol').value.trim().toUpperCase(),
      algo: ui.el('hpSmartAlgo').value,
      objective: ui.el('hpObjective').value,
    });
    renderSmartSummary(RECOMMENDATION);
    // Keep Advanced Mode in step: inspecting should show what was chosen.
    if (ui.el('hpAdvanced').checked) {
      ui.el('hpProfile').value = RECOMMENDATION.profile;
      await loadResolved();
    }
  } catch (err) {
    ui.error('hpSmartSummary', err.message);
  } finally {
    btn.disabled = false; btn.textContent = 'Analyse & configure';
  }
}

async function setAdvancedMode(on) {
  ui.el('hpAdvancedControls').hidden = !on;
  ui.el('hpAdvancedPanels').hidden = !on;
  try {
    sessionStorage.setItem('korisquant:hpAdvanced', on ? '1' : '0');
  } catch (e) { /* private mode: the toggle still works for this page view */ }

  // Switching Advanced on must show the parameters the summary just described.
  // Without this the panel kept whichever profile was last selected — the
  // summary said "default" while the editor showed "Aggressive", which invites
  // the reader to trust the wrong set of numbers.
  if (on && RECOMMENDATION && ui.el('hpProfile').value !== RECOMMENDATION.profile) {
    ui.el('hpProfile').value = RECOMMENDATION.profile;
    await loadResolved();
  }
}

async function initSmart() {
  const [profiles, catalogue] = await Promise.all([
    api.hpSmartProfiles(), api.hpCatalogue(),
  ]);
  ui.el('hpObjective').innerHTML = profiles.profiles
    .map((p) => `<option value="${p.key}"${p.key === 'balanced' ? ' selected' : ''}>`
      + `${p.label}</option>`).join('');
  ui.el('hpSmartAlgo').innerHTML = catalogue.algorithms
    .map((a) => `<option value="${a.key}">${a.key.toUpperCase()}</option>`).join('');
  ui.el('hpSmartAlgo').value = 'dueling_dqn';

  // A mistyped ticker would silently analyse the wrong instrument, so this
  // input gets the same picker as every other symbol field on the platform.
  new SymbolPicker('hpSymbol', 'hpSymbolPanel', () => analyseEnvironment(),
    { syncActive: false });
  ui.el('hpAnalyse').addEventListener('click', analyseEnvironment);
  ui.el('hpAdvanced').addEventListener('change', (e) => setAdvancedMode(e.target.checked));

  let advanced = false;
  try {
    advanced = sessionStorage.getItem('korisquant:hpAdvanced') === '1';
  } catch (e) { advanced = false; }
  ui.el('hpAdvanced').checked = advanced;
  setAdvancedMode(advanced);

  await analyseEnvironment();
}

document.addEventListener('DOMContentLoaded', () => { initSmart(); });

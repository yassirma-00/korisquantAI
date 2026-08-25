/* ============================================================
   KorisQuant AI API client + shared UI utilities
   ============================================================ */

const API_BASE = `${window.location.origin}/api/v1`;

/* Bumped whenever this client gains methods that page scripts depend on.
   `requireApi()` uses it to turn a stale-cache mismatch into an actionable
   message instead of "x is not a function". */
const API_CLIENT_VERSION = 25;

class APIError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.status = status;
    this.payload = payload;
  }
}

async function request(path, options = {}) {
  const url = path.startsWith('http') ? path : `${API_BASE}${path}`;
  const config = {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  };
  if (config.body && typeof config.body !== 'string') {
    config.body = JSON.stringify(config.body);
  }
  const response = await fetch(url, config);
  let payload = null;
  const text = await response.text();
  try { payload = text ? JSON.parse(text) : null; } catch { payload = { raw: text }; }

  if (!response.ok) {
    // `detail` may be FastAPI's raw list of validation dicts. Stringifying it
    // put `[{"type":"string_too_short","loc":["body","username"]...}]` in front
    // of users, so it is only ever used when it is already a plain string.
    const detail = typeof payload?.detail === 'string' ? payload.detail : null;
    const msg = payload?.message || detail || `HTTP ${response.status}`;
    const error = new APIError(
      typeof msg === 'string' ? msg : 'That request could not be completed.',
      response.status, payload);
    // Individual field problems, so a form can list them instead of running
    // them together into one long sentence.
    error.problems = Array.isArray(payload?.problems) ? payload.problems : null;
    throw error;
  }
  return payload;
}

const api = {
  health: () => request('/../../health'),

  // ------------------------------------------------------------ market
  instruments: (q = '', assetClass = '') => {
    const p = new URLSearchParams();
    if (q) p.set('q', q);
    if (assetClass) p.set('asset_class', assetClass);
    return request(`/market/instruments?${p}`);
  },
  quote: (symbol) => request(`/market/quote/${encodeURIComponent(symbol)}`),
  quotes: (symbols) => request(`/market/quotes?symbols=${encodeURIComponent(symbols.join(','))}`),
  history: (symbol, period = '1y', interval = '1d', indicators = '') => {
    const p = new URLSearchParams({ period, interval });
    if (indicators) p.set('indicators', indicators);
    return request(`/market/history/${encodeURIComponent(symbol)}?${p}`);
  },
  indicators: (symbol, period = '1y', list = 'sma,ema,rsi,macd,bbands,atr,adx,stoch,mfi') =>
    request(`/market/indicators/${encodeURIComponent(symbol)}?period=${period}&indicators=${list}`),
  statistics: (symbol, period = '2y', benchmark = 'SPY') =>
    request(`/market/statistics/${encodeURIComponent(symbol)}?period=${period}&benchmark=${benchmark}`),
  correlation: (symbols, period = '1y') =>
    request(`/market/correlation?symbols=${encodeURIComponent(symbols.join(','))}&period=${period}`),

  // ---------------------------------------------------------- dashboard
  overview: (watchlist) =>
    request(`/dashboard/overview${watchlist ? `?watchlist=${encodeURIComponent(watchlist.join(','))}` : ''}`),
  symbolDashboard: (symbol, period = '1y', indicators = 'sma,ema,rsi,macd,bbands,atr') =>
    request(`/dashboard/symbol/${encodeURIComponent(symbol)}?period=${period}&indicators=${indicators}`),
  heatmap: (assetClass = '', period = '1mo', limit = 24) => {
    const p = new URLSearchParams({ period, limit });
    if (assetClass) p.set('asset_class', assetClass);
    return request(`/dashboard/heatmap?${p}`);
  },

  // ----------------------------------------------------------- forecast
  forecastModels: () => request('/forecast/models'),
  trainForecast: (body) => request('/forecast/train', { method: 'POST', body }),
  trainForecastAsync: (body) => request('/forecast/train/async', { method: 'POST', body }),
  forecastJob: (jobId) => request(`/forecast/jobs/${jobId}`),
  predict: (symbol, model = 'lstm', horizon = 5, period = '2y', autoTrain = true) =>
    request(`/forecast/predict/${encodeURIComponent(symbol)}?model=${model}&horizon=${horizon}&period=${period}&auto_train=${autoTrain}`),
  compareModels: (body) => request('/forecast/compare', { method: 'POST', body }),
  trainedModels: () => request('/forecast/trained'),

  // ----------------------------------------------------------------- RL
  rlAlgorithms: () => request('/rl/algorithms'),
  trainRL: (body) => request('/rl/train', { method: 'POST', body }),
  trainRLAsync: (body) => request('/rl/train/async', { method: 'POST', body }),
  rlJob: (jobId) => request(`/rl/jobs/${jobId}`),
  trainPortfolioRL: (body) => request('/rl/portfolio/train', { method: 'POST', body }),
  rlAction: (symbol, algo = 'dueling_dqn', variant = '') =>
    request(`/rl/action/${encodeURIComponent(symbol)}?algo=${algo}`
      + (variant ? `&variant=${encodeURIComponent(variant)}` : '')),
  rlAllocation: (symbols, algo = 'sac', period = '1y') =>
    request(`/rl/allocation?symbols=${encodeURIComponent(symbols.join(','))}`
      + `&algo=${algo}&period=${period}`),
  // `moe` routes each bar to a regime-specialised expert and fine-tunes the
  // incoming one. Omitted from the query string when false so the default call
  // stays byte-identical to the one this endpoint has always received.
  rlBacktest: (symbol, algo = 'dueling_dqn', period = '1y', opts = {}) => {
    let url = `/rl/backtest/${encodeURIComponent(symbol)}?algo=${algo}&period=${period}`;
    if (opts.variant) url += `&variant=${encodeURIComponent(opts.variant)}`;
    if (opts.moe) {
      url += '&moe=true';
      if (opts.adapt === false) url += '&moe_adapt=false';
    }
    return request(url);
  },
  rlAgents: () => request('/rl/agents'),

  // ------------------------------------------------------- intelligence
  algorithms: (opts = {}) => {
    const p = new URLSearchParams();
    if (opts.actionSpace) p.set('action_space', opts.actionSpace);
    if (opts.family) p.set('family', opts.family);
    if (opts.availableOnly) p.set('available_only', 'true');
    return request(`/intel/algorithms?${p}`);
  },
  algorithmDetail: (key) => request(`/intel/algorithms/${key}`),
  compareAlgorithms: () => request('/intel/algorithms/compare'),
  recommendAlgorithm: (actionSpace = 'discrete', priority = 'balanced') =>
    request(`/intel/algorithms/recommend?action_space=${actionSpace}&priority=${priority}`),
  symbolGroups: (q = '', assetClass = '') => {
    const p = new URLSearchParams();
    if (q) p.set('q', q);
    if (assetClass) p.set('asset_class', assetClass);
    return request(`/intel/symbols?${p}`);
  },
  marketRegime: (symbol, period = '2y') =>
    request(`/quant/regime/${encodeURIComponent(symbol)}?period=${period}`),
  agentDecision: (symbol, algo = 'dueling_dqn', period = '1y', variant = '') =>
    request(`/intel/agent-decision/${encodeURIComponent(symbol)}?algo=${algo}&period=${period}`
      + (variant ? `&variant=${encodeURIComponent(variant)}` : '')),
  availableAgents: (symbol) =>
    request(`/intel/agent-decision/${encodeURIComponent(symbol)}/available`),
  portfolioAnalytics: (symbol, opts = {}) => {
    const p = new URLSearchParams({
      period: opts.period || '2y', benchmark: opts.benchmark || 'SPY',
      capital: opts.capital || 100000,
      include_agent: opts.includeAgent ? 'true' : 'false',
      algo: opts.algo || 'dueling_dqn',
    });
    return request(`/intel/portfolio-analytics/${encodeURIComponent(symbol)}?${p}`);
  },
  strategyBenchmarks: (symbol, opts = {}) => {
    const p = new URLSearchParams({
      period: opts.period || '2y', capital: opts.capital || 100000,
      transaction_cost: opts.cost ?? 0.001,
      include_agent: opts.includeAgent ? 'true' : 'false',
      algo: opts.algo || 'dueling_dqn',
    });
    return request(`/intel/benchmarks/${encodeURIComponent(symbol)}?${p}`);
  },

  // ---------------------------------------------------------- portfolio
  listPortfolios: () => request('/portfolio'),
  createPortfolio: (body) => request('/portfolio', { method: 'POST', body }),
  getPortfolio: (id) => request(`/portfolio/${id}`),
  deletePortfolio: (id) => request(`/portfolio/${id}`, { method: 'DELETE' }),
  trade: (id, body) => request(`/portfolio/${id}/trade`, { method: 'POST', body }),
  transactions: (id, limit = 50) => request(`/portfolio/${id}/transactions?limit=${limit}`),
  analytics: (id, period = '1y', benchmark = 'SPY') =>
    request(`/portfolio/${id}/analytics?period=${period}&benchmark=${benchmark}`),
  rebalance: (id, body) => request(`/portfolio/${id}/rebalance`, { method: 'POST', body }),
  optimise: (body) => request('/portfolio/optimise', { method: 'POST', body }),

  // ------------------------------------------------------ recommendation
  recommend: (symbol, opts = {}) => {
    const p = new URLSearchParams({
      period: opts.period || '2y',
      forecast_model: opts.model || 'lstm',
      horizon: opts.horizon || 5,
      rl_algo: opts.rlAlgo || 'dueling_dqn',
      include_xai: opts.includeXai ?? false,
    });
    return request(`/signals/recommend/${encodeURIComponent(symbol)}?${p}`);
  },
  // AI Direction Prediction: will this instrument rise, fall or go nowhere?
  // Same signal set as `recommend`, reduced to a directional call.
  direction: (symbol, opts = {}) => {
    const p = new URLSearchParams({
      period: opts.period || '1y',
      horizon: opts.horizon || 5,
      forecast_model: opts.model || 'lstm',
      rl_algo: opts.rlAlgo || 'dueling_dqn',
    });
    return request(`/signals/direction/${encodeURIComponent(symbol)}?${p}`);
  },
  screen: (symbols) => request(`/signals/screen?symbols=${encodeURIComponent(symbols.join(','))}`),

  // ------------------------------------------- AI Stress Testing Engine
  stressScenarios: () => request('/quant/stress-engine/scenarios'),
  stressEngine: (symbols, opts = {}) => {
    const p = new URLSearchParams({
      scenario: opts.scenario || 'market_crash',
      period: opts.period || '5y',
      position_value: opts.positionValue ?? 100000,
      confidence: opts.confidence ?? 0.95,
      vol_multiplier: opts.volMultiplier ?? 2,
      shock_pct: opts.shockPct ?? -10,
      liquidity_penalty: opts.liquidityPenalty ?? 1.5,
      correlation_target: opts.correlationTarget ?? 0.9,
    });
    if (opts.weights && opts.weights.length) p.set('weights', opts.weights.join(','));
    return request(`/quant/stress-engine/${encodeURIComponent(symbols.join(','))}?${p}`);
  },

  // --------------------------------------------------------------- news
  news: (symbol, limit = 12) => request(`/news/${encodeURIComponent(symbol)}?limit=${limit}`),
  sentiment: (symbol, limit = 20) => request(`/news/${encodeURIComponent(symbol)}/sentiment?limit=${limit}`),
  marketPulse: (symbols) =>
    request(`/news/market/pulse${symbols ? `?symbols=${encodeURIComponent(symbols.join(','))}` : ''}`),
  analyzeText: (text) => request('/news/analyze', { method: 'POST', body: { text } }),

  // --------------------------------------------------------------- risk
  riskScan: (symbol, period = '2y', lookbackDays = 180) =>
    request(`/risk/scan/${encodeURIComponent(symbol)}`
      + `?period=${period}&lookback_days=${lookbackDays}`),
  tiIntelligence: (params = {}) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => { if (v) q.set(k, v); });
    const qs = q.toString();
    return request(`/training/intelligence${qs ? `?${qs}` : ''}`);
  },
  // ---------------------------------------------- training monitoring
  tmRuns: () => request('/training/runs'),
  tmProgress: (symbol, algo) =>
    request(`/training/progress/${encodeURIComponent(symbol)}?algo=${encodeURIComponent(algo)}`),
  tmSummary: (symbol, algo) =>
    request(`/training/summary/${encodeURIComponent(symbol)}?algo=${encodeURIComponent(algo)}`),
  tmCheckpoints: (symbol, algo) => {
    const q = new URLSearchParams();
    if (symbol) q.set('symbol', symbol);
    if (algo) q.set('algo', algo);
    const qs = q.toString();
    return request(`/training/checkpoints${qs ? `?${qs}` : ''}`);
  },
  tmCompareCheckpoints: (left, right) =>
    request('/training/checkpoints/compare',
      { method: 'POST', body: JSON.stringify({ left, right }) }),
  tmRestoreCheckpoint: (symbol, algo, episode) =>
    request('/training/checkpoints/restore',
      { method: 'POST', body: JSON.stringify({ symbol, algo, episode }) }),
  tmDeleteCheckpoint: (symbol, algo, episode) =>
    request(`/training/checkpoints?symbol=${encodeURIComponent(symbol)}`
      + `&algo=${encodeURIComponent(algo)}&episode=${episode}`, { method: 'DELETE' }),

  hpSmartProfiles: () => request('/hyperparams/smart/profiles'),
  hpSmartRecommend: (params = {}) => {
    const q = new URLSearchParams(params);
    return request(`/hyperparams/smart/recommend?${q.toString()}`);
  },
  // ------------------------------------------------- hyperparameters
  hpCatalogue: () => request('/hyperparams/catalogue'),
  hpResolve: (algo, profile) =>
    request(`/hyperparams/resolve?algo=${encodeURIComponent(algo)}`
      + `&profile=${encodeURIComponent(profile)}`),
  hpSaveProfile: (name, config, description) =>
    request(`/hyperparams/profiles/${encodeURIComponent(name)}`,
      { method: 'POST', body: JSON.stringify({ config, description }) }),
  hpDuplicateProfile: (name, newName) =>
    request(`/hyperparams/profiles/${encodeURIComponent(name)}/duplicate`,
      { method: 'POST', body: JSON.stringify({ new_name: newName }) }),
  hpDeleteProfile: (name) =>
    request(`/hyperparams/profiles/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  hpImportProfile: (name, yamlText) =>
    request(`/hyperparams/profiles/${encodeURIComponent(name)}/import`,
      { method: 'POST', body: JSON.stringify({ yaml: yamlText }) }),
  hpExperiments: (limit = 50) => request(`/hyperparams/experiments?limit=${limit}`),
  timeRanges: () => request('/market/time-ranges'),
  trainingHistory: (symbol, model = 'lstm', horizon = 5) =>
    request(`/forecast/training-history/${encodeURIComponent(symbol)}`
      + `?model=${model}&horizon=${horizon}`),
  crashRisk: (symbol) => request(`/risk/crash/${encodeURIComponent(symbol)}`),
  aiConfidence: (symbol, period = '2y') =>
    request(`/signals/confidence/${encodeURIComponent(symbol)}?period=${period}`),
  bubble: (symbol) => request(`/risk/bubble/${encodeURIComponent(symbol)}`),
  marketRegime: (symbol, period = '2y') =>
    request(`/risk/regime/${encodeURIComponent(symbol)}?period=${period}`),

  // ---------------------------------------------------------------- XAI
  explain: (symbol, methods = 'shap,lime,global', horizon = 5) =>
    request(`/xai/explain/${encodeURIComponent(symbol)}?methods=${methods}&horizon=${horizon}`),
  importance: (symbol, topK = 12) =>
    request(`/xai/importance/${encodeURIComponent(symbol)}?top_k=${topK}`),
  counterfactual: (symbol) => request(`/xai/counterfactual/${encodeURIComponent(symbol)}`),

  // ---------------------------------------------------------- assistant
  // The Hugging Face token lives on the server; the browser only ever sees this
  // endpoint. Never put a provider key in this file.
  chat: (body) => request('/chat', { method: 'POST', body }),
  chatHealth: () => request('/chat/health'),
  chatTools: () => request('/chat/tools'),
  chatSuggestions: (params) => request(`/chat/suggestions?${params || ''}`),

  // --------------------------------------------------------------- auth
  // The session token is returned as an HttpOnly cookie, so it is never read
  // or stored by this client — that is the point of HttpOnly.
  register: (body) => request('/auth/register', { method: 'POST', body }),
  login: (body) => request('/auth/login', { method: 'POST', body }),
  logout: () => request('/auth/logout', { method: 'POST' }),
  me: () => request('/auth/me'),
  authStatus: () => request('/auth/status'),
  authConfig: () => request('/auth/config'),
  verifyEmail: (token) => request(`/auth/verify?token=${encodeURIComponent(token)}`,
    { method: 'POST' }),
  resendVerification: () => request('/auth/verify/resend', { method: 'POST' }),

  // ------------------------------------------------------------- alerts
  scanAlerts: (symbol, checks = '') =>
    request(`/alerts/scan/${encodeURIComponent(symbol)}${checks ? `?checks=${checks}` : ''}`),
  scanWatchlist: (symbols, checks = null, persist = false) =>
    request('/alerts/scan', { method: 'POST', body: { symbols, checks, persist } }),
  listAlerts: (limit = 50, filters = {}) => {
    const p = new URLSearchParams({ limit });
    if (filters.q) p.set('q', filters.q);
    if (filters.severity) p.set('severity', filters.severity);
    if (filters.rule_id) p.set('rule_id', filters.rule_id);
    return request(`/alerts?${p}`);
  },
  createRule: (body) => request('/alerts/rules', { method: 'POST', body }),
  listRules: (filters = {}) => {
    const p = new URLSearchParams();
    Object.entries(filters).forEach(([k, v]) => { if (v) p.set(k, v); });
    const qs = p.toString();
    return request(`/alerts/rules${qs ? `?${qs}` : ''}`);
  },
  updateRule: (id, body) => request(`/alerts/rules/${id}`, { method: 'PATCH', body }),
  duplicateRule: (id) => request(`/alerts/rules/${id}/duplicate`, { method: 'POST' }),
  bulkRules: (ruleIds, action) =>
    request('/alerts/rules/bulk', { method: 'POST', body: { rule_ids: ruleIds, action } }),
  alertMetrics: () => request('/alerts/metrics'),
  alertTemplates: () => request('/alerts/templates'),
  createRuleFromTemplate: (key, symbol) =>
    request(`/alerts/rules/from-template/${encodeURIComponent(key)}`
      + `?symbol=${encodeURIComponent(symbol)}`, { method: 'POST' }),
  deleteRule: (id) => request(`/alerts/rules/${id}`, { method: 'DELETE' }),
  toggleRule: (id) => request(`/alerts/rules/${id}/toggle`, { method: 'POST' }),
  evaluateRules: () => request('/alerts/rules/evaluate', { method: 'POST' }),
};

/**
 * Assert that the loaded api.js actually provides the methods a page needs.
 * A hard refresh (Ctrl/Cmd+Shift+R) is the fix when this fires.
 */
function requireApi(...methods) {
  const missing = methods.filter((m) => typeof api[m] !== 'function');
  if (!missing.length) return true;
  const msg = `Stale cached script detected: api.js is missing ${missing.join(', ')}. `
    + 'Hard-refresh the page (Ctrl+Shift+R, or Cmd+Shift+R on macOS) to load the current version.';
  console.error(msg, { clientVersion: API_CLIENT_VERSION });
  document.querySelectorAll('[data-api-guard]').forEach((el) => {
    el.innerHTML = `<div class="error-box">⚠ ${msg}</div>`;
  });
  throw new Error(msg);
}

/* ============================================================ formatting */
const fmt = {
  num(v, decimals = 2) {
    if (v === null || v === undefined || Number.isNaN(v)) return '—';
    return Number(v).toLocaleString('en-US', {
      minimumFractionDigits: decimals, maximumFractionDigits: decimals,
    });
  },
  price(v, refValue = null) {
    if (v === null || v === undefined || Number.isNaN(v)) return '—';
    // Decimal places follow the magnitude of the instrument, not of the current
    // frame: during a count-up animation the value passes through ~0, and
    // switching to 6 dp mid-flight makes the column jitter. `refValue` pins the
    // precision to the final figure.
    const abs = Math.abs(Number.isFinite(refValue) ? refValue : v);
    const d = abs >= 1 ? 2 : abs >= 0.01 ? 4 : 6;
    return this.num(v, d);
  },
  money(v, currency = 'USD') {
    if (v === null || v === undefined || Number.isNaN(v)) return '—';
    return Number(v).toLocaleString('en-US', {
      style: 'currency', currency, minimumFractionDigits: 2, maximumFractionDigits: 2,
    });
  },
  // `signed` adds an explicit + for changes/returns; magnitudes (a weight, a
  // win rate, a confidence) should pass signed=false so they read naturally.
  pct(v, decimals = 2, signed = true) {
    if (v === null || v === undefined || Number.isNaN(v)) return '—';
    const sign = signed && v > 0 ? '+' : '';
    return `${sign}${this.num(v, decimals)}%`;
  },
  compact(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return '—';
    const abs = Math.abs(v);
    if (abs >= 1e12) return `${(v / 1e12).toFixed(2)}T`;
    if (abs >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
    if (abs >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
    if (abs >= 1e3) return `${(v / 1e3).toFixed(2)}K`;
    return this.num(v, 2);
  },
  date(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? String(iso).slice(0, 10)
      : d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  },
  timeAgo(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '—';
    const mins = Math.floor((Date.now() - d.getTime()) / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return days < 30 ? `${days}d ago` : this.date(iso);
  },
  cls(v) { return v > 0 ? 'up' : v < 0 ? 'down' : 'flat'; },
  arrow(v) { return v > 0 ? '▲' : v < 0 ? '▼' : '■'; },
};

/* ================================================================= UI */
const ui = {
  el(id) { return document.getElementById(id); },

  loading(target, text = 'Loading…') {
    const node = typeof target === 'string' ? this.el(target) : target;
    if (node) node.innerHTML = `<div class="loading"><div class="spinner"></div>${text}</div>`;
  },

  error(target, message) {
    const node = typeof target === 'string' ? this.el(target) : target;
    if (node) node.innerHTML = `<div class="error-box">⚠ ${message}</div>`;
  },

  empty(target, message = 'No data available') {
    const node = typeof target === 'string' ? this.el(target) : target;
    if (node) node.innerHTML = `<div class="empty">${message}</div>`;
  },

  toast(message, type = 'info', duration = 3800) {
    let wrap = document.querySelector('.toast-wrap');
    if (!wrap) {
      wrap = document.createElement('div');
      wrap.className = 'toast-wrap';
      document.body.appendChild(wrap);
    }
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    wrap.appendChild(toast);
    setTimeout(() => toast.remove(), duration);
  },

  sourceBadge(source, isLive) {
    const live = isLive ?? !['synthetic', 'cache:stale'].includes(source);
    const label = source === 'synthetic' ? 'SIMULATED'
      : source === 'cache' ? 'CACHED'
      : source === 'cache:stale' ? 'STALE CACHE'
      : String(source || '').toUpperCase();
    return `<span class="badge ${live ? 'badge-green' : 'badge-amber'}">
      <span class="dot ${live ? 'dot-live' : 'dot-sim'}"></span>${label}</span>`;
  },

  actionBadge(action) {
    const map = {
      STRONG_BUY: 'badge-green', BUY: 'badge-green', HOLD: 'badge-grey',
      SELL: 'badge-red', STRONG_SELL: 'badge-red',
    };
    return `<span class="badge ${map[action] || 'badge-grey'}">${String(action).replace('_', ' ')}</span>`;
  },

  riskBadge(level) {
    const map = { low: 'badge-green', moderate: 'badge-amber', high: 'badge-red', critical: 'badge-red' };
    return `<span class="badge ${map[level] || 'badge-grey'}">${level || 'unknown'}</span>`;
  },

  sentimentBadge(label) {
    const map = { positive: 'badge-green', negative: 'badge-red', neutral: 'badge-grey' };
    return `<span class="badge ${map[label] || 'badge-grey'}">${label || 'n/a'}</span>`;
  },

  gauge(value, size = 116, colour = null) {
    const pct = Math.max(0, Math.min(1, value));
    const r = size / 2 - 9;
    const c = 2 * Math.PI * r;
    const stroke = colour || (pct > 0.66 ? 'var(--green)' : pct > 0.33 ? 'var(--amber)' : 'var(--red)');
    return `<svg class="progress-ring" width="${size}" height="${size}">
      <circle class="bg" cx="${size / 2}" cy="${size / 2}" r="${r}"></circle>
      <circle cx="${size / 2}" cy="${size / 2}" r="${r}" stroke="${stroke}"
        stroke-dasharray="${c}" stroke-dashoffset="${c * (1 - pct)}" stroke-linecap="round"></circle>
    </svg>`;
  },

  plotLayout(overrides = {}) {
    // Colours come from the active theme's CSS variables (see theme.js) so a
    // theme switch repaints charts instead of leaving them in the old palette.
    const c = typeof themeColors === 'function' ? themeColors() : {
      text1: '#b0b9d6', text0: '#f2f4fd', grid: '#1c2338', zero: '#2a3455',
      bg2: '#151b2e', border: '#2a3455', accent: '#7c6cff',
      colorway: ['#7c6cff', '#22d3ee', '#22d98a', '#ffb63d', '#f472b6', '#60a5fa', '#94a3b8'],
    };
    return {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: c.text1, family: 'Inter, sans-serif', size: 11 },
      margin: { l: 54, r: 18, t: 26, b: 38 },
      colorway: c.colorway,
      xaxis: {
        gridcolor: c.grid, zerolinecolor: c.zero, linecolor: c.grid,
        showspikes: true, spikemode: 'across', spikethickness: 1,
        spikecolor: c.accent, spikedash: 'dot',
      },
      yaxis: { gridcolor: c.grid, zerolinecolor: c.zero, linecolor: c.grid },
      hovermode: 'x unified',
      hoverlabel: { bgcolor: c.bg2, bordercolor: c.border, font: { color: c.text0, size: 11 } },
      showlegend: true,
      legend: { orientation: 'h', y: 1.12, x: 0, font: { size: 10, color: c.text1 } },
      ...overrides,
    };
  },

  plotConfig: { responsive: true, displayModeBar: false, displaylogo: false },
};

/* ------------------------------------------------------------- state */
const store = {
  get(key, fallback = null) {
    try { const v = localStorage.getItem(`korisquant:${key}`); return v ? JSON.parse(v) : fallback; }
    catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(`korisquant:${key}`, JSON.stringify(value)); } catch { /* ignore */ }
  },
};

const DEFAULT_WATCHLIST = ['AAPL', 'MSFT', 'NVDA', 'SPY', 'BTC-USD', 'ETH-USD', 'GC=F', 'EURUSD=X'];

function getWatchlist() { return store.get('watchlist', DEFAULT_WATCHLIST); }
function setWatchlist(list) { store.set('watchlist', list); }
function getActiveSymbol() { return store.get('symbol', 'AAPL'); }
function setActiveSymbol(symbol) { store.set('symbol', symbol); }

/* ---------------------------------------------------- global search box */
function initSearch(onSelect) {
  const input = document.getElementById('globalSearch');
  const results = document.getElementById('searchResults');
  if (!input || !results) return;

  let timer = null;
  input.addEventListener('input', () => {
    clearTimeout(timer);
    const q = input.value.trim();
    if (q.length < 1) { results.classList.remove('open'); return; }
    timer = setTimeout(async () => {
      try {
        const data = await api.instruments(q);
        if (!data.instruments.length) { results.classList.remove('open'); return; }
        results.innerHTML = data.instruments.slice(0, 12).map((i) => `
          <div class="search-result" data-symbol="${i.symbol}">
            <div class="sym">${i.symbol} <span class="badge badge-grey" style="font-size:9px">${i.asset_class}</span></div>
            <div class="nm">${i.name}</div>
          </div>`).join('');
        results.classList.add('open');
        results.querySelectorAll('.search-result').forEach((node) => {
          node.addEventListener('click', () => {
            const symbol = node.dataset.symbol;
            input.value = '';
            results.classList.remove('open');
            setActiveSymbol(symbol);
            if (onSelect) onSelect(symbol);
          });
        });
      } catch { results.classList.remove('open'); }
    }, 220);
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const value = input.value.trim().toUpperCase();
      if (value) {
        input.value = '';
        results.classList.remove('open');
        setActiveSymbol(value);
        if (onSelect) onSelect(value);
      }
    }
  });

  document.addEventListener('click', (e) => {
    if (!input.contains(e.target) && !results.contains(e.target)) results.classList.remove('open');
  });
}

function highlightNav() {
  const page = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-item').forEach((item) => {
    const href = item.getAttribute('href');
    if (href && (href === page || (page === '' && href === 'index.html'))) item.classList.add('active');
  });
}

/**
 * Show who is signed in, in the sidebar of every dashboard page.
 *
 * Signing in has to mean something once you are inside the product, otherwise
 * the landing page's account flow looks decorative. Rendered into the existing
 * health box so no page needs new markup.
 */
async function mountSessionStrip() {
  // Anchor to the sidebar, which every dashboard page has. The health box was
  // the original anchor but only exists on the overview page, so seven of the
  // eight pages silently showed nothing.
  const sidebar = document.querySelector('.sidebar');
  if (!sidebar) return;                    // landing / auth screens have none
  let strip = document.getElementById('sessionStrip');
  if (!strip) {
    strip = document.createElement('div');
    strip.id = 'sessionStrip';
    strip.className = 'session-strip';
    const health = document.getElementById('healthBox');
    if (health) sidebar.insertBefore(strip, health);
    else sidebar.appendChild(strip);
  }
  try {
    const status = await api.authStatus();
    if (status.authenticated) {
      strip.innerHTML = `
        <div class="session-user" title="${status.user.email || ''}">
          <span class="session-avatar">${status.user.username.charAt(0).toUpperCase()}</span>
          <span class="session-name">${status.user.username}</span>
        </div>
        <button class="btn btn-sm btn-ghost" id="sessionOut">Log out</button>`;
      document.getElementById('sessionOut').addEventListener('click', async () => {
        try { await api.logout(); } catch { /* nothing to undo */ }
        window.location.href = '/';
      });
    } else {
      strip.innerHTML = `
        <a class="btn btn-sm btn-primary" href="/" style="width:100%;justify-content:center">
          Sign in</a>`;
    }
  } catch {
    strip.innerHTML = '';        // auth unreachable: stay silent, do not alarm
  }
}

document.addEventListener('DOMContentLoaded', highlightNav);
document.addEventListener('DOMContentLoaded', mountSessionStrip);

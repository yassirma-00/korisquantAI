/* ============================================================
   Theme controller — dark / light with system awareness

   Design decisions
   ----------------
   * The initial theme is applied by an inline script in <head> (before
     first paint) to avoid the white flash that plagues theme switchers.
     This file only handles the toggle, persistence and repainting.
   * Charts cannot inherit CSS variables, so `themeColors()` reads the
     computed values and `Plotly.relayout` repaints every open chart.
   * If the user has never chosen explicitly we follow the OS preference
     and keep following it live; an explicit choice pins the theme.
   ============================================================ */

const THEME_KEY = 'korisquant:theme';
const THEME_EXPLICIT_KEY = 'korisquant:theme-explicit';

/* One-time migration of the pre-rebrand storage prefix.
   These keys live in real browsers: switching prefix without carrying the
   values over would silently reset everyone's theme and watchlist, which is a
   visible regression caused purely by a cosmetic rename. Runs before first
   paint, costs nothing once done. */
(function migrateStorageKeys() {
  try {
    if (localStorage.getItem('korisquant:migrated') === '1') return;
    for (let i = localStorage.length - 1; i >= 0; i -= 1) {
      const key = localStorage.key(i);
      if (!key || !key.startsWith('finai:')) continue;
      const next = `korisquant:${key.slice('finai:'.length)}`;
      if (localStorage.getItem(next) === null) {
        localStorage.setItem(next, localStorage.getItem(key));
      }
      localStorage.removeItem(key);
    }
    localStorage.setItem('korisquant:migrated', '1');
  } catch { /* private mode: the app still works, just without persistence */ }
})();

function systemTheme() {
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

function storedTheme() {
  try { return localStorage.getItem(THEME_KEY); } catch { return null; }
}

function isExplicit() {
  try { return localStorage.getItem(THEME_EXPLICIT_KEY) === '1'; } catch { return false; }
}

function currentTheme() {
  return document.documentElement.getAttribute('data-theme') || 'dark';
}

/** Read the live palette from CSS so JS never duplicates colour values. */
function themeColors() {
  const s = getComputedStyle(document.documentElement);
  const v = (name, fallback) => (s.getPropertyValue(name).trim() || fallback);
  return {
    text0: v('--text-0', '#f2f4fd'),
    text1: v('--text-1', '#b0b9d6'),
    text2: v('--text-2', '#737fa3'),
    bg1: v('--bg-1', '#0e1220'),
    bg2: v('--bg-2', '#151b2e'),
    border: v('--border', '#2a3455'),
    grid: v('--chart-grid', '#1c2338'),
    zero: v('--chart-zero', '#2a3455'),
    accent: v('--accent', '#7c6cff'),
    green: v('--green', '#22d98a'),
    red: v('--red', '#ff4d6a'),
    amber: v('--amber', '#ffb63d'),
    purple: v('--purple', '#a78bfa'),
    cyan: v('--cyan', '#22d3ee'),
    accent3: v('--accent-3', '#f472b6'),
    chart6: v('--chart-6', '#60a5fa'),
    colorway: [1, 2, 3, 4, 5, 6, 7].map((i) => v(`--chart-${i}`, '#7c6cff')),
  };
}

/**
 * Shared palette accessor used by every chart in the app.
 * Defined once here so a colour never has to be duplicated per page.
 */
function C() {
  const s = getComputedStyle(document.documentElement);
  const v = (n, f) => (s.getPropertyValue(n).trim() || f);
  return { ...themeColors(), bg3: v('--bg-3', '#1e2740'), bg1: v('--bg-1', '#0e1220') };
}

/** Repaint every rendered Plotly chart with the active palette. */
function repaintCharts() {
  if (typeof Plotly === 'undefined') return;
  const c = themeColors();
  document.querySelectorAll('.js-plotly-plot').forEach((node) => {
    try {
      Plotly.relayout(node, {
        'font.color': c.text1,
        'xaxis.gridcolor': c.grid,
        'xaxis.zerolinecolor': c.zero,
        'xaxis.linecolor': c.grid,
        'yaxis.gridcolor': c.grid,
        'yaxis.zerolinecolor': c.zero,
        'yaxis.linecolor': c.grid,
        'yaxis2.gridcolor': c.grid,
        'hoverlabel.bgcolor': c.bg2,
        'hoverlabel.bordercolor': c.border,
        'hoverlabel.font.color': c.text0,
        'legend.font.color': c.text1,
        colorway: c.colorway,
      });
    } catch {
      /* a chart mid-render can reject relayout; the next draw picks it up */
    }
  });
}

function applyTheme(theme, { animate = true } = {}) {
  const root = document.documentElement;
  if (animate && !window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
    root.classList.add('theme-transition');
    window.setTimeout(() => root.classList.remove('theme-transition'), 360);
  }
  root.setAttribute('data-theme', theme);
  try { localStorage.setItem(THEME_KEY, theme); } catch { /* private mode */ }

  const toggle = document.querySelector('.theme-toggle');
  if (toggle) {
    toggle.setAttribute('aria-checked', String(theme === 'light'));
    toggle.setAttribute('title', theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode');
  }
  // Charts repaint after the CSS variables have settled
  window.setTimeout(repaintCharts, animate ? 60 : 0);
  window.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
}

function toggleTheme() {
  try { localStorage.setItem(THEME_EXPLICIT_KEY, '1'); } catch { /* ignore */ }
  applyTheme(currentTheme() === 'dark' ? 'light' : 'dark');
}

/** Inject the toggle into a topbar. Idempotent. */
function mountThemeToggle(container) {
  const host = typeof container === 'string' ? document.querySelector(container) : container;
  if (!host || host.querySelector('.theme-toggle')) return;

  const btn = document.createElement('button');
  btn.className = 'theme-toggle';
  btn.type = 'button';
  btn.setAttribute('role', 'switch');
  btn.setAttribute('aria-label', 'Toggle colour theme');
  btn.setAttribute('aria-checked', String(currentTheme() === 'light'));
  btn.innerHTML = `
    <span class="theme-toggle-knob"></span>
    <span class="theme-toggle-icon moon" aria-hidden="true">🌙</span>
    <span class="theme-toggle-icon sun" aria-hidden="true">☀️</span>`;
  btn.addEventListener('click', toggleTheme);
  host.appendChild(btn);
}

// Follow the OS while the user has not made an explicit choice.
window.matchMedia?.('(prefers-color-scheme: light)').addEventListener?.('change', (e) => {
  if (!isExplicit()) applyTheme(e.matches ? 'light' : 'dark');
});

document.addEventListener('DOMContentLoaded', () => {
  applyTheme(storedTheme() || systemTheme(), { animate: false });
  mountThemeToggle('.topbar');
});

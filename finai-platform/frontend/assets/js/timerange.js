/* ============================================================
   Global time range — one control, every page.

   Six pages each carried their own period widget offering a different subset
   of periods in a different shape (chips here, a <select> there). Switching
   pages silently changed the window you were looking at, and no two pages
   agreed on what "recent" meant.

   This is a single segmented control, rendered from the server's catalogue so
   the UI can never offer a range the backend rejects. The choice is held in
   sessionStorage: it follows you between pages for the session, and a new
   session starts clean rather than resurrecting a window you set last week.

   Pages subscribe with onTimeRangeChange(fn); the component calls every
   subscriber when the selection changes.
   ============================================================ */

const TIME_RANGE_KEY = 'korisquant:timerange';
const TIME_RANGE_DEFAULT = '1y';

/* Rendered before the catalogue arrives so the control never pops in late.
   Replaced by the server's list on first load. */
let TIME_RANGES = [
  { key: '1d', label: '1D' }, { key: '5d', label: '5D' },
  { key: '1mo', label: '1M' }, { key: '3mo', label: '3M' },
  { key: '6mo', label: '6M' }, { key: 'ytd', label: 'YTD' },
  { key: '1y', label: '1Y' }, { key: '3y', label: '3Y' },
  { key: '5y', label: '5Y' }, { key: '10y', label: '10Y' },
  { key: 'max', label: 'MAX' },
];

const timeRangeSubscribers = [];

function getTimeRange() {
  try {
    return sessionStorage.getItem(TIME_RANGE_KEY) || TIME_RANGE_DEFAULT;
  } catch {
    return TIME_RANGE_DEFAULT;   // private mode: fall back to the default
  }
}

function getTimeRangeMeta(key = getTimeRange()) {
  return TIME_RANGES.find((r) => r.key === key) || { key, label: key.toUpperCase() };
}

function setTimeRange(key, { silent = false } = {}) {
  const range = TIME_RANGES.find((r) => r.key === key);
  if (!range) return;
  try { sessionStorage.setItem(TIME_RANGE_KEY, key); } catch { /* non-fatal */ }
  document.querySelectorAll('.tr-seg').forEach((seg) => {
    seg.querySelectorAll('.tr-btn').forEach((btn) => {
      const on = btn.dataset.range === key;
      btn.classList.toggle('active', on);
      btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
    const marker = seg.querySelector('.tr-marker');
    const active = seg.querySelector('.tr-btn.active');
    // Slide the highlight rather than repainting it: the movement is what
    // makes the control feel like one object instead of thirteen buttons.
    if (marker && active) {
      marker.style.width = `${active.offsetWidth}px`;
      marker.style.transform = `translateX(${active.offsetLeft}px)`;
    }
  });
  if (!silent) timeRangeSubscribers.forEach((fn) => {
    try { fn(key, range); } catch (err) { console.warn('time-range subscriber failed', err); }
  });
}

function onTimeRangeChange(fn) {
  if (typeof fn === 'function') timeRangeSubscribers.push(fn);
}

/** Render the control into every [data-timerange] host on the page. */
function renderTimeRange() {
  const hosts = document.querySelectorAll('[data-timerange]');
  if (!hosts.length) return;
  const current = getTimeRange();
  hosts.forEach((host) => {
    host.innerHTML = `
      <span class="tr-info" tabindex="0" role="img"
            aria-label="Controls the time range displayed by all charts and historical market data on this page."
            data-tip="Controls the time range displayed by all charts and historical market data on this page.">&#9432;</span>
      <div class="tr-seg" role="group" aria-label="Time range">
        <span class="tr-marker" aria-hidden="true"></span>
        ${TIME_RANGES.map((r) => `
          <button type="button" class="tr-btn ${r.key === current ? 'active' : ''}"
                  data-range="${r.key}" aria-pressed="${r.key === current}"
                  title="${r.label}">${r.label}</button>`).join('')}
      </div>`;
    host.querySelectorAll('.tr-btn').forEach((btn) => {
      btn.addEventListener('click', () => setTimeRange(btn.dataset.range));
    });
    // Keyboard: arrows move along the control, as a real segmented control does.
    host.querySelector('.tr-seg').addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      e.preventDefault();
      const index = TIME_RANGES.findIndex((r) => r.key === getTimeRange());
      const step = e.key === 'ArrowRight' ? 1 : -1;
      const next = TIME_RANGES[(index + step + TIME_RANGES.length) % TIME_RANGES.length];
      setTimeRange(next.key);
      host.querySelector(`.tr-btn[data-range="${next.key}"]`)?.focus();
    });
  });
  // Position the marker once layout has settled.
  requestAnimationFrame(() => setTimeRange(current, { silent: true }));
}

async function initTimeRange() {
  try {
    const data = await api.timeRanges();
    if (data && Array.isArray(data.ranges) && data.ranges.length) {
      TIME_RANGES = data.ranges;
    }
  } catch {
    /* Offline or unauthenticated: the built-in list still renders, and every
       key in it is one the backend already accepts. */
  }
  renderTimeRange();
  window.addEventListener('resize', () => setTimeRange(getTimeRange(), { silent: true }));
}

/* Plotly interaction defaults for every history chart: drag to zoom, shift to
   pan, a range slider for coarse navigation, and unified hover so a tooltip
   reports every series at that date rather than whichever line was nearest. */
function historyChartLayout(overrides = {}) {
  const base = ui.plotLayout(overrides);
  return {
    ...base,
    dragmode: 'zoom',
    hovermode: 'x unified',
    xaxis: {
      ...(base.xaxis || {}),
      rangeslider: { visible: true, thickness: 0.07 },
      // Native Plotly range buttons complement the global control: the global
      // one refetches, these zoom what is already drawn.
      rangeselector: {
        buttons: [
          { count: 1, label: '1M', step: 'month', stepmode: 'backward' },
          { count: 6, label: '6M', step: 'month', stepmode: 'backward' },
          { step: 'all', label: 'All' },
        ],
        bgcolor: 'rgba(0,0,0,0)',
        activecolor: C().accent,
        font: { size: 10, color: C().text2 },
        // The legend also sits at y 1.12 anchored left, so leaving these
        // buttons left-aligned drew them straight through the series names —
        // measured: legend "MoE agent" spanned x 383-435 while "6M"/"All" sat
        // at 386-434. Anchoring the buttons to the right of the plot puts them
        // where nothing else is drawn.
        x: 1,
        xanchor: 'right',
        y: 1.12,
      },
      ...(overrides.xaxis || {}),
    },
    transition: { duration: 320, easing: 'cubic-in-out' },
  };
}

const historyChartConfig = {
  ...(typeof ui !== 'undefined' ? ui.plotConfig : {}),
  scrollZoom: true,
  doubleClick: 'reset',
  displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
};

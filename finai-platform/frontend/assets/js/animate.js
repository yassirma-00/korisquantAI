/* ============================================================
   Number animation & live value transitions

   Design notes
   ------------
   * Counts up with an easing curve rather than a linear ramp, which reads as
     "settling on a value" instead of "spinning".
   * Respects `prefers-reduced-motion` — animation is decorative, never the
     only way information is conveyed.
   * Flashes green/red on change so an updated price is noticeable without
     the user watching that cell.
   * Uses requestAnimationFrame and cancels any in-flight animation on the
     same element, so rapid refreshes never queue up or fight each other.
   ============================================================ */

const REDUCED_MOTION = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;

/* Cubic ease-out: fast start, gentle settle. */
function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

const _running = new WeakMap();

/**
 * Animate an element's numeric text from its current value to `target`.
 * `formatter` keeps currency/percent/decimal rendering identical to the
 * static path, so the final frame matches exactly what fmt.* would produce.
 */
function animateValue(el, target, { duration = 850, formatter = (v) => v.toFixed(2), from = null, onDone = null } = {}) {
  if (!el) return;
  const finish = () => { el.textContent = formatter(target); if (onDone) onDone(); };

  if (REDUCED_MOTION || !Number.isFinite(target)) { finish(); return; }

  const prev = _running.get(el);
  if (prev) cancelAnimationFrame(prev);

  const start = Number.isFinite(from) ? from
    : (Number.isFinite(parseFloat(String(el.dataset.rawValue ?? ''))) ? parseFloat(el.dataset.rawValue) : null);

  if (start === null || Math.abs(target - start) < 1e-12) { finish(); el.dataset.rawValue = String(target); return; }

  const t0 = performance.now();
  const step = (now) => {
    const p = Math.min((now - t0) / duration, 1);
    const value = start + (target - start) * easeOutCubic(p);
    el.textContent = formatter(value);
    if (p < 1) {
      _running.set(el, requestAnimationFrame(step));
    } else {
      _running.delete(el);
      finish();
    }
  };
  el.dataset.rawValue = String(target);
  _running.set(el, requestAnimationFrame(step));
}

/** Brief colour pulse to draw the eye to a changed value. */
function flash(el, direction) {
  if (!el || REDUCED_MOTION) return;
  const cls = direction > 0 ? 'flash-up' : direction < 0 ? 'flash-down' : null;
  if (!cls) return;
  // Flash the enclosing row when there is one: a single changed digit inside a
  // dense table is easy to miss, a highlighted row is not.
  const target = el.closest('tr') || el;
  target.classList.remove('flash-up', 'flash-down');
  void target.offsetWidth;         // force reflow so the animation restarts
  target.classList.add(cls);
  setTimeout(() => target.classList.remove(cls), 900);
}

/**
 * Update a metric element: animate the number and flash on direction change.
 * Returns the delta so callers can drive extra UI (arrows, sparklines...).
 */
function updateMetric(el, value, formatter, { duration = 850, doFlash = true } = {}) {
  if (!el) return 0;
  const previous = parseFloat(el.dataset.rawValue ?? 'NaN');
  const delta = Number.isFinite(previous) ? value - previous : 0;
  animateValue(el, value, { duration, formatter });
  if (doFlash && Number.isFinite(previous) && Math.abs(delta) > 1e-12) {
    flash(el, Math.sign(delta));
  }
  return delta;
}

/** Build the formatter described by an element's data-* attributes. */
function formatterFor(el, target = null) {
  const kind = el.dataset.format || 'num';
  const decimals = parseInt(el.dataset.decimals ?? '2', 10);
  const signed = el.dataset.signed !== 'false';
  return {
    num: (v) => fmt.num(v, decimals),
    // pass the target so precision stays fixed while the value counts up
    price: (v) => fmt.price(v, Number.isFinite(parseFloat(el.dataset.ref))
      ? parseFloat(el.dataset.ref) : target),
    money: (v) => fmt.money(v),
    pct: (v) => fmt.pct(v, decimals, signed),
    compact: (v) => fmt.compact(v),
  }[kind] || ((v) => fmt.num(v, decimals));
}

/**
 * Animate every `[data-animate]` element inside a container (after render).
 *
 * `stagger` delays each successive element by N ms, which makes a table read
 * as filling in row by row instead of every cell spinning simultaneously.
 * `duration` is passed through so slower-moving figures can be tuned.
 */
function animateContainer(root, { stagger = 0, duration = 850 } = {}) {
  const scope = typeof root === 'string' ? document.getElementById(root) : root;
  if (!scope) return;
  const els = [...scope.querySelectorAll('[data-animate]')];

  els.forEach((el, i) => {
    const target = parseFloat(el.dataset.animate);
    if (!Number.isFinite(target)) return;
    const formatter = formatterFor(el, target);
    // Count up from zero on first paint; afterwards ease from the previous value
    const from = el.dataset.animated === 'done' ? undefined : 0;
    el.dataset.animated = 'done';

    // A row's arrow/colour reflect the FINAL sign. While counting up from zero a
    // negative target is briefly positive, which would render "▲ -0.00%".
    // Hide the arrow until the value has passed zero to keep the two consistent.
    const row = el.closest('tr');
    const arrow = row?.querySelector('.arrow');
    const signMatters = (el.dataset.format === 'pct' || el.dataset.format === 'price') && target < 0;
    if (arrow && signMatters && !REDUCED_MOTION) arrow.style.visibility = 'hidden';
    const reveal = () => { if (arrow) arrow.style.visibility = ''; };

    if (stagger > 0 && !REDUCED_MOTION) {
      // Hold the start value so the cell doesn't flash its final number first
      el.textContent = formatter(from ?? target);
      setTimeout(() => animateValue(el, target, { formatter, from, duration, onDone: reveal }),
                 i * stagger);
    } else {
      animateValue(el, target, { formatter, from, duration, onDone: reveal });
    }
  });

  // Progress bars: CSS already transitions `width`, so setting the real value
  // on the next frame makes them sweep out from zero.
  scope.querySelectorAll('[data-meter]').forEach((bar) => {
    const pct = parseFloat(bar.dataset.meter);
    if (!Number.isFinite(pct)) return;
    if (REDUCED_MOTION) { bar.style.width = `${pct}%`; return; }
    requestAnimationFrame(() => { bar.style.width = `${pct}%`; });
  });
}

/** Animate an SVG gauge ring from empty to `value` (0..1). */
function animateGauge(container, value, duration = 900) {
  const scope = typeof container === 'string' ? document.getElementById(container) : container;
  if (!scope) return;
  const ring = scope.querySelector('.progress-ring circle:not(.bg)');
  if (!ring) return;
  const circumference = parseFloat(ring.getAttribute('stroke-dasharray') || '0');
  if (!circumference) return;
  if (REDUCED_MOTION) {
    ring.style.strokeDashoffset = String(circumference * (1 - value));
    return;
  }
  ring.style.strokeDashoffset = String(circumference);
  const t0 = performance.now();
  const step = (now) => {
    const p = Math.min((now - t0) / duration, 1);
    ring.style.strokeDashoffset = String(circumference * (1 - value * easeOutCubic(p)));
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ---------------------------------------------------------- live ticker */
/**
 * Poll quotes and animate any element tagged `data-live-symbol`.
 * Pauses while the tab is hidden — no point burning API calls and battery
 * animating something nobody is looking at.
 */
class LiveTicker {
  constructor(intervalMs = 30000) {
    this.intervalMs = intervalMs;
    this.timer = null;
    this.symbols = new Set();
    this.enabled = true;
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) this.pause(); else this.resume();
    });
  }

  track(symbols) {
    symbols.forEach((s) => this.symbols.add(s));
    return this;
  }

  async tick() {
    if (!this.enabled || !this.symbols.size || document.hidden) return;
    try {
      const data = await api.quotes([...this.symbols]);
      data.quotes.forEach((q) => {
        document.querySelectorAll(`[data-live-symbol="${CSS.escape(q.symbol)}"]`).forEach((el) => {
          const field = el.dataset.liveField || 'price';
          const value = q[field];
          if (!Number.isFinite(value)) return;
          const formatter = field.includes('percent')
            ? (v) => fmt.pct(v, 2)
            : (v) => fmt.price(v);

          // The number may live in a child <span> (next to a ▲/▼ arrow), so
          // animate that node rather than blowing away the cell's markup.
          const numberNode = el.querySelector('[data-animate]') || el;
          updateMetric(numberNode, value, formatter, { duration: 600 });

          // Keep the arrow and colour consistent with the new sign
          const arrow = el.querySelector('.arrow');
          if (arrow) arrow.textContent = fmt.arrow(value);
          if (field.includes('percent') || field === 'change') {
            el.classList.remove('up', 'down', 'flat');
            el.classList.add(fmt.cls(value));
          }
        });
      });
      const stamp = document.getElementById('lastUpdated');
      if (stamp) stamp.textContent = `updated ${new Date().toLocaleTimeString()}`;
    } catch {
      /* transient network failure: keep the last good values on screen */
    }
  }

  start() {
    this.stop();
    this.timer = setInterval(() => this.tick(), this.intervalMs);
    return this;
  }

  stop() { if (this.timer) { clearInterval(this.timer); this.timer = null; } }
  pause() { this.enabled = false; }
  resume() { this.enabled = true; this.tick(); }
}

const liveTicker = new LiveTicker(30000);

/* ============================================================
   Button interactions: ripple + automatic busy state
   ============================================================ */

/** Material-style ripple centred on the pointer. */
function attachRipple(btn) {
  if (btn.dataset.ripple === 'on') return;
  btn.dataset.ripple = 'on';
  btn.addEventListener('pointerdown', (e) => {
    if (REDUCED_MOTION || btn.disabled) return;
    const rect = btn.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const span = document.createElement('span');
    span.className = 'ripple';
    span.style.width = span.style.height = `${size}px`;
    span.style.left = `${e.clientX - rect.left - size / 2}px`;
    span.style.top = `${e.clientY - rect.top - size / 2}px`;
    btn.appendChild(span);
    setTimeout(() => span.remove(), 600);
  });
}

/**
 * Show a travelling progress bar while a button's handler is in flight.
 *
 * Page code already disables buttons during long calls, but a disabled button
 * with unchanged text reads as "nothing happened". Observing the disabled
 * attribute lets every button gain feedback without touching page scripts.
 *
 * "Disabled" alone is not "busy", though. Buttons are also disabled to express
 * a *state*: Hyperparameters disables Save on a built-in profile, and the
 * training monitor ships Compare disabled until two checkpoints are ticked.
 * Treating those as in-flight drew a progress bar that animated forever on a
 * button that was not doing anything — the UI claimed to be working when it
 * was idle. Busy therefore requires the disable to follow the user actually
 * pressing *this* button: the observer runs as a microtask after the click
 * handler, so a genuine in-flight disable is always within the window, while a
 * state-driven one (page load, another control's change) is never.
 */
const BUSY_AFTER_PRESS_MS = 1000;

function observeBusyState(btn) {
  if (btn.dataset.busyObserved === 'on') return;
  btn.dataset.busyObserved = 'on';
  let pressedAt = -Infinity;
  btn.addEventListener('click', () => { pressedAt = performance.now(); }, true);
  new MutationObserver(() => {
    const inFlight = btn.disabled
      && (performance.now() - pressedAt) < BUSY_AFTER_PRESS_MS;
    btn.classList.toggle('is-busy', inFlight);
  }).observe(btn, { attributes: true, attributeFilter: ['disabled'] });
}

function enhanceButtons(root = document) {
  root.querySelectorAll('.btn').forEach((btn) => {
    attachRipple(btn);
    observeBusyState(btn);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  enhanceButtons();
  // Buttons rendered later (tables, panels) are picked up automatically.
  new MutationObserver((records) => {
    for (const r of records) {
      r.addedNodes.forEach((n) => {
        if (n.nodeType !== 1) return;
        if (n.classList?.contains('btn')) { attachRipple(n); observeBusyState(n); }
        else enhanceButtons(n);
      });
    }
  }).observe(document.body, { childList: true, subtree: true });
});

/* ============================================================
   KorisQuant AI — landing page behaviour

   Auth talks to /api/v1/auth. The session token is set by the server as
   an HttpOnly cookie, so nothing here ever touches the token itself:
   a token this script could read is a token an XSS could steal.
   ============================================================ */

/* ------------------------------------------------------------ helpers */
function el(id) { return document.getElementById(id); }

function showError(node, message) {
  node.textContent = message;
  node.classList.add('show');
}

function clearError(node) {
  node.textContent = '';
  node.classList.remove('show');
}

/* --------------------------------------------------------- auth modal */
function openAuth(tab = 'login') {
  const modal = el('authModal');
  modal.classList.add('open');
  switchTab(tab);
  setTimeout(() => el(tab === 'login' ? 'loginId' : 'regUser')?.focus(), 120);
}

function closeAuth() {
  el('authModal').classList.remove('open');
  clearError(el('loginError'));
  clearError(el('registerError'));
}

function switchTab(tab) {
  document.querySelectorAll('.lp-tab').forEach((t) =>
    t.classList.toggle('active', t.dataset.tab === tab));
  el('loginForm').classList.toggle('hidden', tab !== 'login');
  el('registerForm').classList.toggle('hidden', tab !== 'register');
}

/** Reflect the signed-in state in the header. */
function renderSession(user) {
  const actions = el('navActions');
  if (!actions) return;
  if (!user) {
    actions.innerHTML = `
      <button class="btn btn-sm" id="navLogin">Log in</button>
      <button class="btn btn-sm btn-primary" id="navRegister">Register</button>`;
    el('navLogin').addEventListener('click', () => openAuth('login'));
    el('navRegister').addEventListener('click', () => openAuth('register'));
    return;
  }
  actions.innerHTML = `
    <span class="lp-user-chip">Signed in as <strong>${user.username}</strong></span>
    <a class="btn btn-sm btn-primary" href="/dashboard">Open dashboard</a>
    <button class="btn btn-sm" id="navLogout">Log out</button>`;
  el('navLogout').addEventListener('click', async () => {
    try { await api.logout(); } catch { /* signing out locally is enough */ }
    renderSession(null);
  });
}

async function loadSession() {
  try {
    const status = await api.authStatus();
    renderSession(status.authenticated ? status.user : null);
  } catch {
    renderSession(null);           // treat an unreachable check as signed out
  }
}

/* --------------------------------------------------------------- forms */
async function submitLogin(event) {
  event.preventDefault();
  const error = el('loginError');
  const button = el('loginSubmit');
  clearError(error);
  button.disabled = true;
  try {
    const result = await api.login({
      identifier: el('loginId').value.trim(),
      password: el('loginPw').value,
    });
    // Straight to the product: signing in is a means, not the destination.
    window.location.href = `/dashboard?welcome=${encodeURIComponent(result.user.username)}`;
  } catch (err) {
    showError(error, err.message || 'Could not sign you in. Please try again.');
    button.disabled = false;
  }
}

async function submitRegister(event) {
  event.preventDefault();
  const error = el('registerError');
  const button = el('registerSubmit');
  clearError(error);

  const password = el('regPw').value;
  // Mirror the server's rules so the common mistakes are caught without a
  // round-trip. The server stays the authority — this is only courtesy.
  const problems = [];
  if (password.length < 8) problems.push('be at least 8 characters long');
  if (!/[A-Za-z]/.test(password)) problems.push('contain at least one letter');
  if (!/[0-9]/.test(password)) problems.push('contain at least one number');
  if (problems.length) {
    showError(error, `Your password must ${problems.join(', ')}.`);
    return;
  }

  button.disabled = true;
  try {
    const result = await api.register({
      username: el('regUser').value.trim(),
      email: el('regEmail').value.trim(),
      password,
      full_name: el('regName').value.trim() || null,
    });
    window.location.href = `/dashboard?welcome=${encodeURIComponent(result.user.username)}`;
  } catch (err) {
    showError(error, err.message || 'Could not create your account. Please try again.');
    button.disabled = false;
  }
}

/* --------------------------------------------------------------- legal */
const LEGAL = {
  privacy: {
    title: 'Privacy policy',
    body: `
      <p><em>Last updated: 2026. KorisQuant AI is educational and research
      software; this policy describes what the software itself does.</em></p>
      <h4>What is stored</h4>
      <ul>
        <li><strong>Account details</strong> — username, email and an optional
          full name. Passwords are stored only as a bcrypt hash and are never
          recoverable, by us or by anyone else.</li>
        <li><strong>Your work</strong> — portfolios, paper trades, alert rules
          and trained models, kept in the application database.</li>
        <li><strong>Preferences</strong> — theme, watchlist and chat transcript,
          stored in your own browser, not on the server.</li>
      </ul>
      <h4>Session cookie</h4>
      <p>Signing in sets one cookie, <code>korisquant_session</code>. It is
      HttpOnly (unreadable by scripts) and used purely to keep you signed in.
      There are no advertising or tracking cookies.</p>
      <h4>Third parties</h4>
      <p>Market data is fetched from public providers such as Yahoo Finance.
      Questions you send to the AI assistant are forwarded to the configured AI
      provider to generate an answer. Nothing else about your account leaves the
      installation.</p>
      <h4>Your control</h4>
      <p>Deleting a portfolio removes its positions and history permanently.
      To have an account removed, contact
      <a href="mailto:contact@korisquant.ai">contact@korisquant.ai</a>.</p>`,
  },
  terms: {
    title: 'Terms of use',
    body: `
      <p><em>Last updated: 2026.</em></p>
      <h4>Not investment advice</h4>
      <p>KorisQuant AI is educational and research software. Every forecast,
      agent action, score and recommendation is a statistical output produced by
      a model — not a recommendation to buy or sell any instrument. Markets are
      non-stationary and past performance never guarantees future results.</p>
      <h4>Paper trading only</h4>
      <p>Portfolios are simulated. No order is ever routed to a broker, exchange
      or venue of any kind.</p>
      <h4>No warranty</h4>
      <p>The software is provided "as is", without warranty of any kind. Market
      data comes from third parties and may be delayed, incomplete or wrong.
      When no live provider responds, a synthetic engine generates plausible but
      entirely fictional prices; anything derived from it is badged
      <em>SIMULATED</em> in the interface.</p>
      <h4>Your responsibility</h4>
      <p>You are responsible for validating any output independently before
      acting on it, and for complying with the laws that apply to you.</p>`,
  },
};

function openLegal(kind) {
  const doc = LEGAL[kind];
  if (!doc) return;
  el('legalTitle').textContent = doc.title;
  el('legalBody').innerHTML = doc.body;
  el('legalModal').classList.add('open');
}

/* ------------------------------------------------------------ counters */
/** Count the hero statistics up once, when they scroll into view. */
function animateCounters() {
  const nodes = document.querySelectorAll('[data-count]');
  if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return;

  const run = (node) => {
    const target = parseInt(node.dataset.count, 10);
    if (!Number.isFinite(target)) return;
    const duration = 900;
    const start = performance.now();
    const step = (now) => {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - (1 - t) ** 3;
      node.textContent = Math.round(target * eased).toString();
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };

  if (!('IntersectionObserver' in window)) { nodes.forEach(run); return; }
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      run(entry.target);
      observer.unobserve(entry.target);     // once only, never on scroll back
    });
  }, { threshold: 0.5 });
  nodes.forEach((n) => observer.observe(n));
}

/* ----------------------------------------------------------------- init */
document.addEventListener('DOMContentLoaded', () => {
  el('lpYear').textContent = new Date().getFullYear();

  el('heroStart').addEventListener('click', () => { window.location.href = '/dashboard'; });
  el('finalStart').addEventListener('click', () => { window.location.href = '/dashboard'; });
  el('finalRegister').addEventListener('click', () => openAuth('register'));
  el('navLogin').addEventListener('click', () => openAuth('login'));
  el('navRegister').addEventListener('click', () => openAuth('register'));

  el('authClose').addEventListener('click', closeAuth);
  el('authModal').addEventListener('click', (e) => {
    if (e.target === el('authModal')) closeAuth();
  });
  document.querySelectorAll('.lp-tab').forEach((tab) =>
    tab.addEventListener('click', () => switchTab(tab.dataset.tab)));
  el('loginForm').addEventListener('submit', submitLogin);
  el('registerForm').addEventListener('submit', submitRegister);

  el('legalClose').addEventListener('click', () => el('legalModal').classList.remove('open'));
  el('legalModal').addEventListener('click', (e) => {
    if (e.target === el('legalModal')) el('legalModal').classList.remove('open');
  });
  el('privacyLink').addEventListener('click', (e) => { e.preventDefault(); openLegal('privacy'); });
  el('termsLink').addEventListener('click', (e) => { e.preventDefault(); openLegal('terms'); });
  document.querySelectorAll('[data-legal]').forEach((link) =>
    link.addEventListener('click', (e) => { e.preventDefault(); openLegal(link.dataset.legal); }));

  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    closeAuth();
    el('legalModal').classList.remove('open');
  });

  const nav = el('lpNav');
  const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 8);
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  animateCounters();
  loadSession();
});

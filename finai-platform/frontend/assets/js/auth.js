/* ============================================================
   KorisQuant AI — authentication screen

   Two views (sign in, register) in one page so switching
   between them costs no navigation. The session token is set by the server
   as an HttpOnly cookie, so nothing here ever handles the token itself:
   a token this script could read is a token an XSS could steal.
   ============================================================ */

const params = new URLSearchParams(window.location.search);

/* Where to go after signing in. Only same-origin *paths* are honoured:
   accepting a full URL would make this an open redirect, letting a crafted
   link send someone to a convincing phishing page after a real login. */
function safeNext() {
  const next = params.get('next');
  if (!next || !next.startsWith('/') || next.startsWith('//')) return '/dashboard';
  if (next.startsWith('/auth.html')) return '/dashboard';   // no redirect loop
  return next;
}

function el(id) { return document.getElementById(id); }

/**
 * Show a message above the form.
 *
 * The text is inserted as *text*, never parsed as HTML. This used to be
 * `innerHTML = message`, and the `?error=` query parameter was passed straight
 * into it — so a crafted link like
 * `/auth.html?error=<img src=x onerror=...>` ran arbitrary script on the sign-in
 * page, which is the worst possible page to own: it is where credentials are
 * typed. Escaping at this single choke point fixes every caller at once.
 *
 * Nothing on this screen needs to inject markup, so there is no HTML-taking
 * variant to reach for by mistake.
 */
function banner(message, kind = 'info', extra = '') {
  const node = el('authBanner');
  node.className = `auth-banner show ${kind}`;
  node.textContent = message;
  if (extra) {
    const code = document.createElement('code');
    code.textContent = extra;          // reset links etc: data, not markup
    node.appendChild(code);
  }
}

function showError(id, message, problems = null) {
  const node = el(id);
  if (problems && problems.length > 1) {
    // Several things are wrong at once: a bulleted list is scannable, one
    // run-on sentence is not.
    const list = document.createElement('ul');
    list.className = 'auth-error-list';
    problems.forEach((problem) => {
      const item = document.createElement('li');
      item.textContent = problem;      // textContent: never parse server text as HTML
      list.appendChild(item);
    });
    node.textContent = '';
    node.appendChild(list);
  } else {
    node.textContent = (problems && problems[0]) || message;
  }
  node.classList.add('show');
}

/** Pull the readable message and per-field problems off an APIError. */
function errorParts(err, fallback) {
  return [err?.message || fallback, err?.problems || null];
}

function clearError(id) {
  const node = el(id);
  node.textContent = '';
  node.classList.remove('show');
}

const VIEWS = { login: 'viewLogin', register: 'viewRegister' };

function showView(name) {
  Object.entries(VIEWS).forEach(([key, id]) =>
    el(id).classList.toggle('hidden', key !== name));
  const titles = { login: 'Sign in', register: 'Create account' };
  document.title = `${titles[name] || 'Sign in'} · KorisQuant AI`;
}

/* ------------------------------------------------------ password rules */
function evaluatePassword(value) {
  const rules = {
    length: value.length >= 8,
    letter: /[A-Za-z]/.test(value),
    digit: /[0-9]/.test(value),
  };
  // Beyond the three required rules, length and variety are what actually
  // resist an offline guess, so they drive the strength read-out.
  let score = Object.values(rules).filter(Boolean).length;
  if (value.length >= 12) score += 1;
  if (/[^A-Za-z0-9]/.test(value)) score += 1;
  return { rules, score, ok: rules.length && rules.letter && rules.digit };
}

function renderStrength(value) {
  const box = el('pwStrength');
  const { rules, score } = evaluatePassword(value);
  box.hidden = value.length === 0;

  document.querySelectorAll('#pwRules li').forEach((item) =>
    item.classList.toggle('met', rules[item.dataset.rule]));

  const levels = [
    { at: 5, cls: 'strong', label: 'Strong password' },
    { at: 4, cls: 'good', label: 'Good password' },
    { at: 3, cls: 'fair', label: 'Acceptable — longer is safer' },
    { at: 0, cls: 'weak', label: 'Too weak' },
  ];
  const level = levels.find((l) => score >= l.at) || levels[levels.length - 1];
  const bar = el('pwBar');
  bar.className = level.cls;
  bar.style.width = `${Math.min(score / 5, 1) * 100}%`;
  el('pwLabel').textContent = level.label;
}

/* ---------------------------------------------------------- submitting */
async function withBusy(button, label, work) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = label;
  try { await work(); } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

async function submitLogin(event) {
  event.preventDefault();
  clearError('loginError');
  await withBusy(el('loginSubmit'), 'Signing in…', async () => {
    try {
      await api.login({
        identifier: el('loginId').value.trim(),
        password: el('loginPw').value,
        remember_me: el('rememberMe').checked,
      });
      window.location.href = safeNext();
    } catch (err) {
      showError('loginError', ...errorParts(err, 'Could not sign you in.'));
    }
  });
}

async function submitRegister(event) {
  event.preventDefault();
  clearError('registerError');

  const password = el('regPw').value;
  const { ok } = evaluatePassword(password);
  if (!ok) {
    showError('registerError',
      'Your password needs at least 8 characters, one letter and one number.');
    return;
  }
  if (password !== el('regPw2').value) {
    showError('registerError', 'The two passwords do not match.');
    return;
  }
  if (!el('regTerms').checked) {
    showError('registerError', 'Please accept the Terms and Privacy Policy.');
    return;
  }

  await withBusy(el('registerSubmit'), 'Creating account…', async () => {
    try {
      const result = await api.register({
        username: el('regUser').value.trim(),
        email: el('regEmail').value.trim(),
        password,
        full_name: el('regName').value.trim() || null,
        remember_me: el('regRemember').checked,
      });
      // With no SMTP configured the server hands back the verification link
      // instead of emailing it. Showing it beats leaving the user waiting for
      // a message that was never sent.
      const link = result.verification?.link;
      if (link) {
        sessionStorage.setItem('korisquant:pending-verification', link);
      }
      window.location.href = safeNext();
    } catch (err) {
      showError('registerError',
        ...errorParts(err, 'Could not create your account.'));
    }
  });
}

/* ---------------------------------------------------------------- init */
document.addEventListener('DOMContentLoaded', async () => {
  // Someone with a live session has no business on this screen.
  try {
    const status = await api.authStatus();
    if (status.authenticated) { window.location.href = safeNext(); return; }
  } catch { /* auth unreachable: let them try to sign in anyway */ }

  const verifyToken = params.get('verify');

  if (params.get('mode') === 'register') {
    showView('register');
  } else {
    showView('login');
  }

  if (params.get('reset')) {
    // Old links may still exist in inboxes. Say what happened instead of
    // showing a sign-in page that silently ignores the token.
    banner('Password reset links are no longer used. Contact the system '
      + 'administrator if you cannot sign in.', 'warn');
  }
  if (params.get('error')) banner(params.get('error'), 'error');
  if (params.get('next')) {
    banner('Please sign in to continue to that page.', 'warn');
  }

  if (verifyToken) {
    try {
      await api.verifyEmail(verifyToken);
      banner('Your email is confirmed. You can sign in now.', 'success');
    } catch (err) {
      banner(err.message || 'This verification link is no longer valid.', 'error');
    }
  }

  el('loginForm').addEventListener('submit', submitLogin);
  el('registerForm').addEventListener('submit', submitRegister);
  el('regPw').addEventListener('input', (e) => renderStrength(e.target.value));


  document.querySelectorAll('[data-goto]').forEach((button) =>
    button.addEventListener('click', () => showView(button.dataset.goto)));

  document.querySelectorAll('.auth-eye').forEach((button) => {
    button.addEventListener('click', () => {
      const input = el(button.dataset.toggle);
      const revealed = input.type === 'text';
      input.type = revealed ? 'password' : 'text';
      button.setAttribute('aria-label', revealed ? 'Show password' : 'Hide password');
      input.focus();
    });
  });
});

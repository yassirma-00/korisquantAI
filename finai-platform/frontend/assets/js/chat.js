/* ============================================================
   KorisQuant AI Assistant — floating chat available on every page

   Design decisions
   ----------------
   * Self-mounting. Every page only has to include this script; the
     launcher and panel are injected on DOMContentLoaded. No page needs
     its own markup, so a new page gets the assistant for free.
   * Page-aware. It sends the current page and the selected symbol, so
     "analyse this stock" resolves without the user retyping a ticker.
   * The transcript lives in the browser (sessionStorage). The backend is
     stateless: a server restart never drops a conversation, and no chat
     history is persisted server-side.
   * The API key is NOT here. The browser calls /api/v1/chat and the
     backend adds the credential. Anything in this file is public.
   ============================================================ */

/* The assistant speaks first. An empty panel makes the user guess what the
   thing can do; a greeting states it in one breath.

   It is rendered as a normal assistant bubble but flagged `greeting`, which
   keeps it out of the transcript sent to the model: it is UI chrome, not a
   turn anyone took. Replaying it would burn tokens on every request and invite
   the model to answer the greeting instead of the question. */
const CHAT_GREETING = "Hello! I'm KorisQuant AI Assistant. How can I help you "
  + 'with your financial analysis or investment decisions today?';

/* Shown whenever the assistant cannot answer and there is nothing the person
   reading it can do about it. Mirrors the backend's USER_FACING_UNAVAILABLE:
   internal URLs and .env keys are operator details, and a user staring at a
   dashboard cannot act on them. The precise cause goes to the server logs. */
const CHAT_UNAVAILABLE = 'Unable to connect to the AI service. Please check your '
  + 'network connection, try again later, or contact the system developer if the '
  + 'problem persists.';

/* Distinct case: nothing is configured at all. Whoever installed this can fix
   it, so the message points at the setup rather than at the network. */
const CHAT_NOT_CONFIGURED = 'The AI assistant has not been set up on this '
  + 'installation. Contact the system developer to enable it.';

const CHAT_STORE_KEY = 'korisquant:chat:transcript';
const CHAT_SEEN_KEY = 'korisquant:chat:seen';
const CHAT_MAX_STORED = 40;

const chatState = {
  open: false,
  busy: false,
  expanded: false,
  messages: [],       // { role, content, tools, meta }
  available: null,    // null = unknown, true/false once /chat/health answers
};

/* ------------------------------------------------------------ markdown */
/**
 * Minimal Markdown -> HTML.
 *
 * Escaping happens FIRST and unconditionally: model output is untrusted
 * input, and a model can be talked into emitting a <script> tag. Every
 * tag rendered below is one this function created itself, so there is no
 * path from model text to live HTML.
 */
function renderMarkdown(src) {
  const escape = (s) => s
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

  // Fenced code blocks are lifted out before inline rules run, so their
  // contents are never reinterpreted as markup.
  const blocks = [];
  let text = escape(src).replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    blocks.push(`<pre><code data-lang="${lang}">${code.replace(/\n$/, '')}</code></pre>`);
    return `\u0000BLOCK${blocks.length - 1}\u0000`;
  });

  const inline = (s) => s
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*\*([^*]+)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

  const lines = text.split('\n');
  const out = [];
  let list = null;          // 'ul' | 'ol' | null
  let paragraph = [];
  let table = null;

  const flushParagraph = () => {
    if (paragraph.length) { out.push(`<p>${inline(paragraph.join(' '))}</p>`); paragraph = []; }
  };
  const flushList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  const flushTable = () => {
    if (!table) return;
    const [head, ...body] = table;
    out.push('<table><thead><tr>'
      + head.map((c) => `<th>${inline(c)}</th>`).join('')
      + '</tr></thead><tbody>'
      + body.map((row) => `<tr>${row.map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`).join('')
      + '</tbody></table>');
    table = null;
  };
  const flushAll = () => { flushParagraph(); flushList(); flushTable(); };

  const cells = (line) => line.replace(/^\||\|$/g, '').split('|').map((c) => c.trim());

  for (const raw of lines) {
    const line = raw.trimEnd();

    if (/^\u0000BLOCK\d+\u0000$/.test(line.trim())) {
      flushAll();
      out.push(line.trim());
      continue;
    }
    if (!line.trim()) { flushAll(); continue; }

    // table: header row, separator, then body rows
    if (line.includes('|') && line.trim().startsWith('|')) {
      if (/^\|[\s:|-]+\|$/.test(line.trim())) continue;   // separator
      flushParagraph(); flushList();
      (table ??= []).push(cells(line.trim()));
      continue;
    }
    flushTable();

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flushAll();
      const level = Math.min(heading[1].length + 2, 6);
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }

    if (/^\s*([-*_])\1{2,}\s*$/.test(line)) { flushAll(); out.push('<hr>'); continue; }

    if (/^\s*&gt;\s?/.test(line)) {
      flushAll();
      out.push(`<blockquote>${inline(line.replace(/^\s*&gt;\s?/, ''))}</blockquote>`);
      continue;
    }

    const ul = line.match(/^\s*[-*+]\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ul || ol) {
      flushParagraph();
      const want = ul ? 'ul' : 'ol';
      if (list !== want) { flushList(); out.push(`<${want}>`); list = want; }
      out.push(`<li>${inline((ul || ol)[1])}</li>`);
      continue;
    }
    flushList();
    paragraph.push(line.trim());
  }
  flushAll();

  return out.join('').replace(/\u0000BLOCK(\d+)\u0000/g, (_, i) => blocks[Number(i)]);
}

/* -------------------------------------------------------------- state */
function currentPageKey() {
  const file = window.location.pathname.split('/').pop() || 'index.html';
  return file.replace('.html', '') || 'index';
}

function loadTranscript() {
  try {
    const raw = sessionStorage.getItem(CHAT_STORE_KEY);
    chatState.messages = raw ? JSON.parse(raw) : [];
  } catch { chatState.messages = []; }
}

function saveTranscript() {
  try {
    sessionStorage.setItem(CHAT_STORE_KEY,
      JSON.stringify(chatState.messages.slice(-CHAT_MAX_STORED)));
  } catch { /* private mode / quota — the panel still works in-memory */ }
}

/* --------------------------------------------------------- rendering */
function scrollThread(smooth = true) {
  const thread = document.getElementById('chatThread');
  if (!thread) return;
  requestAnimationFrame(() => {
    thread.scrollTo({ top: thread.scrollHeight, behavior: smooth ? 'smooth' : 'auto' });
  });
}

function messageNode({ role, content, tools, meta }) {
  const wrap = document.createElement('div');
  wrap.className = `chat-msg ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'chat-msg-avatar';
  avatar.textContent = role === 'user' ? '🧑' : '◈';
  avatar.setAttribute('aria-hidden', 'true');

  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble';
  if (role === 'assistant') {
    bubble.innerHTML = renderMarkdown(content);
  } else if (role === 'error') {
    // Errors are built by this file, but they are still rendered as text nodes
    // rather than HTML — an error message often quotes an upstream payload.
    // Split on blank lines so the remedy reads as its own paragraph.
    content.split(/\n{2,}/).forEach((part, index) => {
      const line = document.createElement('div');
      line.textContent = part;
      if (index > 0) { line.style.marginTop = '6px'; line.style.opacity = '.85'; }
      bubble.appendChild(line);
    });
  } else {
    bubble.textContent = content;      // user text is never parsed as markup
  }

  if (tools?.length) {
    const strip = document.createElement('div');
    strip.className = 'chat-tools';
    // Internal tool names would be noise; show the engine the user recognises.
    const LABELS = {
      get_quote: 'Live quote', get_technical_analysis: 'Technical indicators',
      get_recommendation: 'Recommendation engine', get_forecast: 'DL forecast',
      get_agent_decision: 'RL agent', get_risk_assessment: 'Risk scan',
      get_news_sentiment: 'News sentiment', compare_strategies: 'Strategy backtest',
      get_market_regime: 'Regime detection', get_performance_metrics: 'Performance dossier',
      explain_prediction: 'SHAP explainability', list_rl_algorithms: 'Algorithm catalogue',
      list_trained_models: 'Model registry', search_instruments: 'Instrument search',
      get_platform_status: 'Platform status',
    };
    tools.forEach((t) => {
      const chip = document.createElement('span');
      chip.className = `chat-tool-chip${t.ok ? '' : ' failed'}`;
      chip.textContent = `${t.ok ? '✓' : '⚠'} ${LABELS[t.tool] || t.tool}`;
      if (t.arguments?.symbol) chip.textContent += ` · ${t.arguments.symbol}`;
      strip.appendChild(chip);
    });
    bubble.appendChild(strip);
  }

  if (meta) {
    const line = document.createElement('div');
    line.className = 'chat-meta';
    line.textContent = meta;
    bubble.appendChild(line);
  }

  wrap.append(avatar, bubble);
  return wrap;
}

async function renderWelcome() {
  const thread = document.getElementById('chatThread');
  if (!thread) return;
  // The health probe may have disabled the panel while this was in flight;
  // repainting the welcome screen over that notice would hide the reason.
  if (chatState.available === false) return;

  const symbol = typeof getActiveSymbol === 'function' ? getActiveSymbol() : null;
  let prompts = [];
  try {
    const params = new URLSearchParams({ page: currentPageKey() });
    if (symbol) params.set('symbol', symbol);
    prompts = (await api.chatSuggestions(params)).suggestions || [];
  } catch {
    prompts = ['What can this platform do?', `Analyse ${symbol || 'AAPL'}`,
               'Which models are trained already?'];
  }

  // The greeting is a real assistant bubble, so the panel reads as a
  // conversation that has already started rather than a form to fill in.
  const greeting = messageNode({ role: 'assistant', content: CHAT_GREETING });

  const welcome = document.createElement('div');
  welcome.className = 'chat-welcome';

  const list = document.createElement('div');
  list.className = 'chat-suggestions';
  prompts.forEach((text) => {
    const btn = document.createElement('button');
    btn.className = 'chat-suggestion';
    btn.type = 'button';
    btn.textContent = text;
    btn.addEventListener('click', () => sendMessage(text));
    list.appendChild(btn);
  });
  welcome.appendChild(list);

  // Re-check after the await: health may have resolved to unavailable, or the
  // user may have sent a message, while the suggestions were being fetched.
  if (chatState.available === false || chatState.messages.length) return;
  thread.innerHTML = '';
  thread.append(greeting, welcome);
}

function renderThread() {
  const thread = document.getElementById('chatThread');
  if (!thread) return;
  if (!chatState.messages.length) { renderWelcome(); return; }
  thread.innerHTML = '';
  chatState.messages.forEach((m) => thread.appendChild(messageNode(m)));
  scrollThread(false);
}

/* Progressive status: a 40 s wait with no feedback reads as a hang. */
function showTyping() {
  const thread = document.getElementById('chatThread');
  if (!thread) return null;
  const wrap = document.createElement('div');
  wrap.className = 'chat-msg assistant';
  wrap.id = 'chatTyping';
  wrap.innerHTML = `
    <div class="chat-msg-avatar" aria-hidden="true">◈</div>
    <div class="chat-bubble">
      <div class="chat-typing"><span></span><span></span><span></span></div>
      <div class="chat-progress" id="chatProgress">Thinking…</div>
    </div>`;
  thread.appendChild(wrap);
  scrollThread();

  const stages = [
    [3500, 'Reading platform data…'],
    [9000, 'Running the models…'],
    [18000, 'Still working — complex queries call several engines…'],
    [32000, 'Almost there…'],
  ];
  const timers = stages.map(([delay, text]) => setTimeout(() => {
    const node = document.getElementById('chatProgress');
    if (node) node.textContent = text;
  }, delay));

  return () => { timers.forEach(clearTimeout); wrap.remove(); };
}

/* ---------------------------------------------------------- sending */
async function sendMessage(text) {
  const input = document.getElementById('chatInput');
  const message = (text ?? input?.value ?? '').trim();
  if (!message || chatState.busy) return;

  if (input) { input.value = ''; input.style.height = 'auto'; }
  chatState.busy = true;
  const sendBtn = document.getElementById('chatSend');
  if (sendBtn) sendBtn.disabled = true;

  const thread = document.getElementById('chatThread');
  if (chatState.messages.length === 0 && thread) thread.innerHTML = '';

  chatState.messages.push({ role: 'user', content: message });
  thread?.appendChild(messageNode({ role: 'user', content: message }));
  saveTranscript();
  scrollThread();

  const stopTyping = showTyping();

  try {
    const symbol = typeof getActiveSymbol === 'function' ? getActiveSymbol() : null;
    const response = await api.chat({
      message,
      // Only completed turns are replayed, and only the last few: the backend
      // trims again server-side.
      history: chatState.messages.slice(0, -1).slice(-10)
        .map(({ role, content }) => ({ role, content })),
      page: currentPageKey(),
      symbol,
    });

    stopTyping?.();
    // The model name is an implementation detail: it changes with the fallback
    // chain, means nothing to the user, and advertises the stack. The response
    // time is genuinely useful, so that is all the footer keeps.
    const seconds = response.elapsed_ms ? (response.elapsed_ms / 1000).toFixed(1) : null;
    const entry = {
      role: 'assistant',
      content: response.reply,
      tools: response.tools_used || [],
      meta: seconds ? `Answered in ${seconds}s` : '',
    };
    chatState.messages.push(entry);
    thread?.appendChild(messageNode(entry));
    saveTranscript();
  } catch (err) {
    stopTyping?.();
    // Only the backend's user-facing message is rendered. The technical detail
    // that rides along in development ("Check OLLAMA_BASE_URL...", model names,
    // endpoint URLs) is for the server log, not for someone reading a
    // dashboard — and it would leak the deployment's wiring into the panel.
    const entry = {
      role: 'error',
      content: `⚠ ${err.message || CHAT_UNAVAILABLE}`,
    };
    chatState.messages.push(entry);
    thread?.appendChild(messageNode(entry));
    saveTranscript();
  } finally {
    chatState.busy = false;
    if (sendBtn) sendBtn.disabled = false;
    scrollThread();
    document.getElementById('chatInput')?.focus();
  }
}

/* ------------------------------------------------------- open / close */
function openChat() {
  chatState.open = true;
  document.getElementById('chatPanel')?.classList.add('open');
  document.getElementById('chatFab')?.classList.add('open');
  document.getElementById('chatFab')?.classList.remove('pulse');
  try { localStorage.setItem(CHAT_SEEN_KEY, '1'); } catch { /* ignore */ }
  if (!chatState.messages.length) renderWelcome();
  setTimeout(() => document.getElementById('chatInput')?.focus(), 260);
  scrollThread(false);
}

function closeChat() {
  chatState.open = false;
  document.getElementById('chatPanel')?.classList.remove('open');
  document.getElementById('chatFab')?.classList.remove('open');
  document.getElementById('chatFab')?.focus();
}

function toggleChat() { chatState.open ? closeChat() : openChat(); }

function clearChat() {
  chatState.messages = [];
  saveTranscript();
  renderWelcome();
}

function toggleExpand() {
  chatState.expanded = !chatState.expanded;
  const panel = document.getElementById('chatPanel');
  panel?.classList.toggle('expanded', chatState.expanded);
  const btn = document.getElementById('chatExpand');
  if (btn) {
    btn.textContent = chatState.expanded ? '⤡' : '⤢';
    btn.title = chatState.expanded ? 'Shrink' : 'Expand';
  }
}

/**
 * Put the composer out of action when the assistant provably cannot answer,
 * and replace the thread with the reason. Better an upfront explanation than a
 * message that travels to the server only to come back as an error.
 */
function disableComposer(reason) {
  const input = document.getElementById('chatInput');
  const send = document.getElementById('chatSend');
  const hint = document.querySelector('.chat-hint');
  if (input) { input.disabled = true; input.placeholder = 'Assistant unavailable'; }
  if (send) send.disabled = true;
  if (hint) hint.textContent = reason;
  if (!chatState.messages.length) {
    const thread = document.getElementById('chatThread');
    if (thread) {
      thread.innerHTML = '';
      const box = document.createElement('div');
      box.className = 'chat-welcome';
      const icon = document.createElement('div');
      icon.className = 'chat-welcome-icon';
      icon.textContent = '⚠';
      const title = document.createElement('div');
      title.className = 'chat-welcome-title';
      title.textContent = 'Assistant unavailable';
      const sub = document.createElement('div');
      sub.className = 'chat-welcome-sub';
      sub.textContent = reason;
      box.append(icon, title, sub);
      thread.appendChild(box);
    }
  }
}

/* -------------------------------------------------------------- mount */
function mountChat() {
  if (document.getElementById('chatFab')) return;      // idempotent

  const fab = document.createElement('button');
  fab.id = 'chatFab';
  fab.className = 'chat-fab';
  fab.type = 'button';
  fab.title = 'Ask the AI assistant  (Ctrl+K)';
  fab.setAttribute('aria-label', 'Open the AI assistant');
  // A word, not a symbol. The launcher used to show "◈" — a decorative diamond
  // that says nothing about what the button does; a user had to hover for the
  // tooltip, or click it to find out. The label is paired with a speech-bubble
  // icon so the control reads as a conversation at a glance.
  fab.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    + 'stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" '
    + 'aria-hidden="true" focusable="false">'
    + '<path d="M21 11.5a8.4 8.4 0 0 1-9 8.4 9.9 9.9 0 0 1-2.9-.4L4 21l1.4-4.1'
    + 'A8.2 8.2 0 0 1 3.6 11.5 8.4 8.4 0 0 1 12 3.1a8.4 8.4 0 0 1 9 8.4z"/>'
    + '<path d="M8.6 11.5h.01M12 11.5h.01M15.4 11.5h.01"/>'
    + '</svg><span class="chat-fab-label">Assistant</span>';
  let seen = false;
  try { seen = localStorage.getItem(CHAT_SEEN_KEY) === '1'; } catch { /* ignore */ }
  if (!seen) fab.classList.add('pulse');

  const panel = document.createElement('div');
  panel.id = 'chatPanel';
  panel.className = 'chat-panel';
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-label', 'KorisQuant AI assistant');
  panel.innerHTML = `
    <div class="chat-header">
      <div class="chat-avatar" aria-hidden="true">◈</div>
      <div style="min-width:0">
        <div class="chat-title">KorisQuant AI Assistant</div>
        <div class="chat-status" id="chatStatus">
          <span class="dot dot-live"></span><span>Connected to live platform data</span>
        </div>
      </div>
      <div class="chat-header-actions">
        <button class="chat-icon-btn" id="chatExpand" type="button" title="Expand"
                aria-label="Expand the panel">⤢</button>
        <button class="chat-icon-btn" id="chatClear" type="button" title="New conversation"
                aria-label="Clear the conversation">⟲</button>
        <button class="chat-icon-btn" id="chatClose" type="button" title="Close"
                aria-label="Close the assistant">✕</button>
      </div>
    </div>
    <div class="chat-thread" id="chatThread" role="log" aria-live="polite"></div>
    <div class="chat-composer">
      <div class="chat-input-row">
        <textarea class="chat-input" id="chatInput" rows="1"
                  placeholder="Ask about any instrument, model or metric…"
                  aria-label="Message"></textarea>
        <button class="chat-send" id="chatSend" type="button"
                aria-label="Send">➤</button>
      </div>
      <div class="chat-hint">Educational research output · not investment advice</div>
    </div>`;

  document.body.append(fab, panel);

  fab.addEventListener('click', toggleChat);
  document.getElementById('chatClose').addEventListener('click', closeChat);
  document.getElementById('chatClear').addEventListener('click', clearChat);
  document.getElementById('chatExpand').addEventListener('click', toggleExpand);
  document.getElementById('chatSend').addEventListener('click', () => sendMessage());

  const input = document.getElementById('chatInput');
  input.addEventListener('keydown', (e) => {
    // Enter sends, Shift+Enter breaks the line — the convention users expect.
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 132)}px`;
  });

  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); toggleChat(); }
    if (e.key === 'Escape' && chatState.open) closeChat();
  });

  loadTranscript();
  renderThread();

  // Report a disabled/unreachable assistant in the header rather than letting
  // the user discover it by sending a message into the void.
  api.chatHealth().then((health) => {
    chatState.available = health.available;
    const status = document.getElementById('chatStatus');
    if (!status) return;
    if (health.available) {
      // Naming the model tells the user nothing they can act on and pins the
      // UI to whichever provider happens to be wired in. The tool count is the
      // part that actually describes the capability.
      status.innerHTML = '<span class="dot dot-live"></span>'
        + `<span>Connected · ${health.tool_count} data tools</span>`;
      return;
    }
    const provider = health.provider || {};
    // The status line names the failure for whoever is looking at the screen,
    // but the body text stays generic unless the remedy is something an end
    // user can actually carry out. Config keys and internal URLs are operator
    // concerns: they are logged server-side, not shown here.
    const BLOCKS = {
      daemon: 'AI service unreachable',
      model: 'AI model not installed',
      auth: 'AI service unavailable',
      subscription: 'AI service unavailable',
    };
    const label = BLOCKS[provider.blocked];
    if (label) {
      status.innerHTML = `<span class="dot dot-sim"></span><span>${label}</span>`;
      // user_message is set only when the backend has advice the user can act
      // on (a local daemon they run themselves); otherwise it is the generic
      // service message.
      disableComposer(provider.user_message || CHAT_UNAVAILABLE);
      return;
    }
    status.innerHTML = '<span class="dot dot-sim"></span><span>AI service unavailable</span>';
    // Letting someone type into an assistant that cannot answer only produces a
    // delayed error, so the composer is disabled either way.
    disableComposer(health.configured ? CHAT_UNAVAILABLE : CHAT_NOT_CONFIGURED);
  }).catch(() => {
    const status = document.getElementById('chatStatus');
    if (status) status.innerHTML = '<span class="dot dot-sim"></span><span>Status unknown</span>';
  });
}

document.addEventListener('DOMContentLoaded', mountChat);

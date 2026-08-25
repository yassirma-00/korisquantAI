/* ============================================================
   Symbol picker — grouped, searchable, accepts custom tickers.

   Rendered as a real listbox rather than a <select> so instruments can be
   grouped by asset class with names, exchanges and badges. Any Yahoo ticker is
   accepted even when absent from the curated universe.
   ============================================================ */

const ASSET_ICONS = {
  equity: '▣', etf: '◧', crypto: '◈', index: '▤', forex: '⇄', commodity: '⬢',
};

class SymbolPicker {
  /**
   * @param {string} inputId    text input that shows/accepts the symbol
   * @param {string} panelId    container for the dropdown
   * @param {function} onSelect called with the chosen symbol
   * @param {object}  options
   *   multi      {boolean}  keep the panel open and accumulate selections
   *   selected   {string[]} initial selection when multi
   *   onChange   {function} called with the full array after every change
   *   syncActive {boolean}  update the platform-wide active symbol (default true)
   */
  constructor(inputId, panelId, onSelect, options = {}) {
    this.input = document.getElementById(inputId);
    this.panel = document.getElementById(panelId);
    this.onSelect = onSelect;
    this.groups = [];
    this.filter = '';
    this.assetClass = '';
    this.loaded = false;

    this.multi = Boolean(options.multi);
    this.selected = [...(options.selected || [])];
    this.onChange = options.onChange || null;
    // Multi-select is a basket builder, not a navigation control: it must not
    // retarget every other panel on the page to the last chip clicked.
    this.syncActive = options.syncActive !== undefined
      ? options.syncActive : !this.multi;

    // Index of the keyboard-highlighted row within the rendered list.
    this.activeIndex = -1;
    this.rows = [];

    if (this.input && this.panel) this._bind();
  }

  async load() {
    if (this.loaded) return;
    try {
      const data = await api.symbolGroups();
      this.groups = data.groups || [];
      this.loaded = true;
    } catch (err) {
      this.groups = [];
      console.warn('symbol picker load failed', err);
    }
  }

  _bind() {
    // A combobox that cannot be driven from the keyboard is unusable for
    // anyone not holding a mouse, and screen readers need the roles to
    // announce it as a list rather than a plain text box.
    this.input.setAttribute('role', 'combobox');
    this.input.setAttribute('aria-autocomplete', 'list');
    this.input.setAttribute('aria-expanded', 'false');
    this.input.setAttribute('aria-controls', this.panel.id);
    this.panel.setAttribute('role', 'listbox');

    this.input.addEventListener('focus', async () => {
      await this.load();
      this.render();
      this.open();
    });
    this.input.addEventListener('input', () => {
      this.filter = this.input.value.trim().toLowerCase();
      // Any keystroke invalidates the previous highlight; pre-selecting the
      // best match means Enter accepts what the list already shows first.
      this.activeIndex = this.filter ? 0 : -1;
      this.render();
      this.open();
    });
    this.input.addEventListener('keydown', (e) => this._onKey(e));
    document.addEventListener('click', (e) => {
      if (!this.input.contains(e.target) && !this.panel.contains(e.target)) this.close();
    });
  }

  _onKey(e) {
    const isOpen = this.panel.classList.contains('open');
    switch (e.key) {
      case 'ArrowDown':
      case 'ArrowUp': {
        e.preventDefault();
        if (!isOpen) { this.render(); this.open(); return; }
        if (!this.rows.length) return;
        const step = e.key === 'ArrowDown' ? 1 : -1;
        // Wrap: reaching the end and stopping dead feels broken.
        this.activeIndex = (this.activeIndex + step + this.rows.length) % this.rows.length;
        this._highlight();
        break;
      }
      case 'Home':
      case 'End':
        if (!isOpen || !this.rows.length) return;
        e.preventDefault();
        this.activeIndex = e.key === 'Home' ? 0 : this.rows.length - 1;
        this._highlight();
        break;
      case 'Enter': {
        e.preventDefault();
        // Take the highlighted row if there is one, otherwise accept whatever
        // was typed: any Yahoo ticker is valid, listed or not.
        const row = this.rows[this.activeIndex];
        if (isOpen && row) {
          this.select(row.dataset.symbol);
        } else {
          const custom = this.input.value.trim().toUpperCase();
          if (custom) this.select(custom);
        }
        break;
      }
      case 'Escape':
        this.close();
        this.input.blur();
        break;
      case 'Backspace':
        // With an empty query, backspace removes the last chip — the standard
        // behaviour of every tag input, and faster than aiming at a tiny ×.
        if (this.multi && !this.input.value && this.selected.length) {
          e.preventDefault();
          this.deselect(this.selected[this.selected.length - 1]);
        }
        break;
      default:
        break;
    }
  }

  _highlight() {
    this.rows.forEach((row, i) => {
      const on = i === this.activeIndex;
      row.classList.toggle('sp-active', on);
      row.setAttribute('aria-selected', on ? 'true' : 'false');
      if (on) {
        row.scrollIntoView({ block: 'nearest' });
        this.input.setAttribute('aria-activedescendant', row.id);
      }
    });
  }

  open() {
    this.panel.classList.add('open');
    this.input.setAttribute('aria-expanded', 'true');
  }

  close() {
    this.panel.classList.remove('open');
    this.input.setAttribute('aria-expanded', 'false');
    this.input.removeAttribute('aria-activedescendant');
    this.activeIndex = -1;
  }

  select(symbol) {
    const ticker = String(symbol).trim().toUpperCase();
    if (!ticker) return;

    if (this.multi) {
      if (!this.selected.includes(ticker)) this.selected.push(ticker);
      // Clear the query so the next asset can be typed straight away, and keep
      // the panel open: building a basket is a repeated action.
      this.input.value = '';
      this.filter = '';
      this.activeIndex = -1;
      this.render();
      this.open();
      this.input.focus();
      if (this.onChange) this.onChange([...this.selected]);
      if (this.onSelect) this.onSelect(ticker);
      return;
    }

    this.input.value = ticker;
    this.close();
    if (this.syncActive) setActiveSymbol(ticker);
    if (this.onSelect) this.onSelect(ticker);
  }

  deselect(symbol) {
    this.selected = this.selected.filter((s) => s !== symbol);
    this.render();
    if (this.onChange) this.onChange([...this.selected]);
  }

  setSelected(symbols) {
    this.selected = [...symbols];
    this.render();
  }

  /**
   * Relevance of one instrument for a query. 0 means "do not show".
   *
   * The ranking mirrors what someone typing a ticker actually expects:
   * an exact symbol first, then symbols that start with the query, then a word
   * in the name starting with it. A match buried mid-word only counts once the
   * query is long enough to be deliberate — that single rule is what stops one
   * letter from returning most of the universe.
   */
  static score(instrument, q) {
    const symbol = instrument.symbol.toLowerCase();
    const name = (instrument.name || '').toLowerCase();

    if (symbol === q) return 1000;
    if (symbol.startsWith(q)) return 900 - symbol.length;
    // "BTC-USD" should still be found by typing "usd"
    if (symbol.split(/[-=.^]/).some((part) => part.startsWith(q))) return 800;
    if (name.startsWith(q)) return 700;
    if (name.split(/\s+/).some((word) => word.startsWith(q))) return 600;
    // Substring anywhere: only once the query is specific enough to mean it.
    if (q.length >= 3 && (symbol.includes(q) || name.includes(q))) return 300;
    return 0;
  }

  render() {
    const q = this.filter;
    let html = `
      <div class="sp-filters">
        ${['', 'equity', 'etf', 'crypto', 'index', 'forex', 'commodity'].map((c) => `
          <span class="chip ${this.assetClass === c ? 'active' : ''}" data-class="${c}">
            ${c === '' ? 'All' : (ASSET_ICONS[c] || '') + ' ' + c.charAt(0).toUpperCase() + c.slice(1)}
          </span>`).join('')}
      </div>`;

    let total = 0;
    const body = this.groups
      .filter((g) => !this.assetClass || g.key === this.assetClass)
      .map((g) => {
        // Rank by how well each instrument matches, then keep only the good
        // matches. A plain `includes()` scored a substring anywhere in the
        // *name* as highly as a ticker prefix, so typing one letter returned 26
        // of 32 instruments — a list that is not a search result.
        const items = q
          ? g.instruments
            .map((i) => ({ i, score: SymbolPicker.score(i, q) }))
            .filter((r) => r.score > 0)
            .sort((a, b) => b.score - a.score || a.i.symbol.localeCompare(b.i.symbol))
            .map((r) => r.i)
          : g.instruments;
        if (!items.length) return '';
        total += items.length;
        return `
          <div class="sp-group">
            <div class="sp-group-label">${ASSET_ICONS[g.key] || ''} ${g.label}
              <span class="text-muted">(${items.length})</span></div>
            ${items.map((i) => `
              <div class="sp-item${this.selected.includes(i.symbol) ? ' sp-picked' : ''}"
                   role="option" data-symbol="${i.symbol}">
                <div>
                  <span class="sp-sym">${i.symbol}</span>
                  <span class="sp-name">${i.name}</span>
                </div>
                <span class="sp-exch">${i.exchange || ''}</span>
              </div>`).join('')}
          </div>`;
      }).join('');

    // Always offer the typed value as a custom ticker: the curated universe is
    // a convenience, not a restriction.
    // Offer the typed value as a custom ticker whenever it is not already an
    // exact match: the curated universe is a convenience, not a restriction,
    // and previously the option vanished as soon as anything else matched.
    const upper = q.toUpperCase();
    const exact = this.groups.some((g) =>
      g.instruments.some((i) => i.symbol.toUpperCase() === upper));
    const custom = q && !exact
      ? `<div class="sp-item sp-custom" role="option" data-symbol="${upper}">
           <div><span class="sp-sym">${upper}</span>
           <span class="sp-name">Use as custom ticker</span></div>
           <span class="badge badge-blue">custom</span></div>`
      : '';

    // Selected assets, shown above the list so the basket is always visible.
    const chips = this.multi && this.selected.length
      ? `<div class="sp-chosen">
           ${this.selected.map((s) => `
             <span class="sp-chip" data-remove="${s}">${s}<i aria-hidden="true">×</i></span>`).join('')}
           <span class="sp-chosen-clear" data-clear="1">Clear all</span>
         </div>`
      : '';

    this.panel.innerHTML = chips + html + (body || '') + custom ||
      '<div class="empty">No instruments match</div>';

    this.panel.querySelectorAll('.chip[data-class]').forEach((chip) => {
      chip.addEventListener('click', (e) => {
        e.stopPropagation();
        this.assetClass = chip.dataset.class;
        this.activeIndex = -1;
        this.render();
      });
    });

    this.panel.querySelectorAll('[data-remove]').forEach((chip) => {
      chip.addEventListener('click', (e) => {
        e.stopPropagation();
        this.deselect(chip.dataset.remove);
      });
    });
    const clear = this.panel.querySelector('[data-clear]');
    if (clear) {
      clear.addEventListener('click', (e) => {
        e.stopPropagation();
        this.selected = [];
        this.render();
        if (this.onChange) this.onChange([]);
      });
    }

    // Register the rows once, so arrow keys and mouse hover address the same
    // list and cannot drift apart.
    this.rows = Array.from(this.panel.querySelectorAll('.sp-item'));
    this.rows.forEach((item, i) => {
      item.id = `${this.panel.id}-opt-${i}`;
      item.setAttribute('aria-selected', 'false');
      item.addEventListener('click', () => this.select(item.dataset.symbol));
      item.addEventListener('mousemove', () => {
        if (this.activeIndex === i) return;
        this.activeIndex = i;
        this._highlight();
      });
    });
    if (this.activeIndex >= this.rows.length) this.activeIndex = this.rows.length - 1;
    if (this.activeIndex >= 0) this._highlight();
  }
}

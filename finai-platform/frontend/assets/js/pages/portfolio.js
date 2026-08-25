/* ============================================================
   Page: Portfolio Management
   ============================================================ */

let activePortfolio = null;
let pPeriod = getTimeRange();
/* Cash and current holdings drive the trade ticket's warnings, so they are
   cached here whenever the portfolio reloads instead of being re-fetched. */
let portfolioCash = 0;

async function loadPortfolios() {
  try {
    const data = await api.listPortfolios();
    const select = ui.el('portfolioSelect');
    if (!data.portfolios.length) {
      select.innerHTML = '<option value="">No portfolio</option>';
      ui.el('portfolioStats').innerHTML =
        '<div class="card span-full"><div class="empty">Create a portfolio to get started →</div></div>';
      return;
    }
    select.innerHTML = data.portfolios.map((p) =>
      `<option value="${p.id}">${p.name} (${fmt.money(p.initial_capital)})</option>`).join('');
    activePortfolio = activePortfolio && data.portfolios.some((p) => p.id === activePortfolio)
      ? activePortfolio : data.portfolios[0].id;
    select.value = activePortfolio;
    await loadPortfolio();
  } catch (err) {
    ui.error('portfolioStats', err.message);
  }
}

async function loadPortfolio() {
  if (!activePortfolio) return;
  ui.loading('holdingsTable');
  try {
    const data = await api.analytics(activePortfolio, pPeriod, 'SPY');

    ui.el('portfolioStats').innerHTML = `
      <div class="card"><div class="stat-label">Total Value</div>
        <div class="stat-value">${fmt.money(data.total_value)}</div>
        <div class="stat-change ${fmt.cls(data.total_pnl)}">${fmt.money(data.total_pnl)} (${fmt.pct(data.total_pnl_pct)})</div></div>
      <div class="card"><div class="stat-label">Cash</div>
        <div class="stat-value">${fmt.money(data.cash)}</div>
        <div class="text-xs text-muted">${fmt.pct(data.cash_weight * 100, 1, false)} of portfolio</div></div>
      <div class="card"><div class="stat-label">Invested</div>
        <div class="stat-value">${fmt.money(data.invested_value)}</div>
        <div class="text-xs text-muted">${data.n_positions} positions</div></div>
      <div class="card"><div class="stat-label">Sharpe Ratio</div>
        <div class="stat-value ${fmt.cls(data.metrics?.sharpe_ratio)}">${fmt.num(data.metrics?.sharpe_ratio, 2)}</div>
        <div class="text-xs text-muted">vs ${data.benchmark || 'SPY'} · β ${fmt.num(data.metrics?.beta, 2)}</div></div>`;

    // Feed the trade ticket: buying power and the position being traded.
    portfolioCash = data.cash || 0;
    const bp = ui.el('tBuyingPower');
    if (bp) bp.textContent = `${fmt.money(portfolioCash)} available`;
    ticket.position = (data.holdings || []).find(
      (h) => h.symbol === (ticket.symbol || '').toUpperCase()) || null;
    renderTradePreview();

    // holdings
    ui.el('holdingsTable').innerHTML = data.holdings.length ? `
      <table>
        <thead><tr><th>Symbol</th><th class="t-right">Qty</th><th class="t-right">Avg</th>
          <th class="t-right">Price</th><th class="t-right">Value</th><th class="t-right">P&L</th>
          <th class="t-right">% of total</th><th></th></tr></thead>
        <tbody>${data.holdings.map((h) => `
          <tr class="clickable holding-row" data-symbol="${h.symbol}"
              title="Load ${h.symbol} into the trade ticket">
            <td class="sym-cell">${h.symbol}<div class="text-xs text-muted">${h.asset_class}</div></td>
            <td class="t-right mono">${fmt.num(h.quantity, 4)}</td>
            <td class="t-right mono">${fmt.price(h.average_price)}</td>
            <td class="t-right mono">${fmt.price(h.current_price)}
              <div class="text-xs ${fmt.cls(h.current_price - h.average_price)}">
                ${h.current_price >= h.average_price ? '▲' : '▼'}
                ${fmt.pct(((h.current_price / h.average_price) - 1) * 100)}</div></td>
            <td class="t-right">${fmt.money(h.market_value)}</td>
            <td class="t-right ${fmt.cls(h.unrealised_pnl)}">${fmt.money(h.unrealised_pnl)}
              <div class="text-xs">${fmt.pct(h.unrealised_pnl_pct)}</div></td>
            <td class="t-right">
              <div>${fmt.pct(h.weight * 100, 1, false)}</div>
              <!-- A weight bar reads faster than a number when scanning for
                   concentration, which is the question this column answers. -->
              <div class="weight-bar"><span style="width:${Math.min(h.weight * 100, 100)}%"></span></div>
            </td>
            <td class="t-right">
              <button class="btn btn-sm btn-ghost close-position" data-symbol="${h.symbol}"
                      title="Sell the whole ${h.symbol} position">Close</button>
            </td>
          </tr>`).join('')}
          <tr style="border-top:1px solid var(--border)">
            <td class="text-xs text-muted">Cash</td>
            <td colspan="3"></td>
            <td class="t-right text-muted">${fmt.money(data.cash)}</td>
            <td></td>
            <td class="t-right text-muted">${fmt.pct(data.cash_weight * 100, 1, false)}</td>
            <td></td>
          </tr></tbody>
      </table>
      <div class="text-xs text-muted mt-1">
        Weights are shares of total portfolio value, so positions and cash add up to 100%.
        Click a row to load it into the trade ticket.
      </div>` : '<div class="empty">No holdings yet — use the Trade panel to buy your first position</div>';

    // Clicking a holding loads it into the ticket: the common next action after
    // looking at a position is trading it, and retyping the ticker invites typos.
    ui.el('holdingsTable').querySelectorAll('.holding-row').forEach((row) => {
      row.addEventListener('click', (event) => {
        if (event.target.closest('.close-position')) return;   // handled below
        ui.el('tSymbol').value = row.dataset.symbol;
        setActiveSymbol(row.dataset.symbol);
        ticket.position = data.holdings.find((h) => h.symbol === row.dataset.symbol) || null;
        refreshTradeQuote();
        ui.el('tSymbol').scrollIntoView({ behavior: 'smooth', block: 'center' });
      });
    });

    ui.el('holdingsTable').querySelectorAll('.close-position').forEach((btn) => {
      btn.addEventListener('click', async (event) => {
        event.stopPropagation();
        const symbol = btn.dataset.symbol;
        const holding = data.holdings.find((h) => h.symbol === symbol);
        if (!holding) return;
        btn.disabled = true;
        try {
          // Sell by quantity, not notional: a notional computed from a slightly
          // stale price leaves a dust position behind instead of closing it.
          const r = await api.trade(activePortfolio, {
            symbol, side: 'SELL', quantity: holding.quantity,
          });
          ui.toast(`Closed ${symbol}: sold ${fmt.num(r.quantity, 4)} @ ${fmt.price(r.price)}`,
                   'success');
          await loadPortfolio();
        } catch (err) {
          ui.toast(`Could not close ${symbol}: ${err.message}`, 'error', 6000);
        } finally {
          btn.disabled = false;
        }
      });
    });

    // equity + drawdown
    if (data.equity_curve?.length) {
      renderLineChart('pEquityChart', [{
        x: data.equity_curve.map((e) => e.date), y: data.equity_curve.map((e) => e.value),
        name: 'Portfolio', color: C().green, fill: 'tozeroy', fillcolor: 'rgba(34, 217, 138,.09)',
      }], { height: 280, yTitle: 'Value ($)', showlegend: false });

      renderLineChart('pDrawdownChart', [{
        x: data.drawdown_curve.map((d) => d.date), y: data.drawdown_curve.map((d) => d.drawdown * 100),
        name: 'Drawdown', color: C().red, fill: 'tozeroy', fillcolor: 'rgba(255, 77, 106,.14)',
      }], { height: 150, yTitle: '%', showlegend: false });
    } else {
      ui.empty('pEquityChart', 'Add positions to see the equity curve');
      ui.el('pDrawdownChart').innerHTML = '';
    }

    // allocation donut
    const alloc = Object.entries(data.allocation || {}).filter(([, v]) => v > 0.0001);
    if (alloc.length) {
      const node = ui.el('pAllocChart');
      node.innerHTML = '';
      Plotly.newPlot(node, [{
        type: 'pie', hole: 0.58,
        labels: alloc.map(([k]) => k), values: alloc.map(([, v]) => v),
        marker: { colors: [C().accent, C().green, C().amber, C().purple, C().cyan, C().text2] },
        textinfo: 'label+percent', textfont: { size: 10 },
      }], ui.plotLayout({ height: 260, showlegend: false, margin: { l: 10, r: 10, t: 10, b: 10 } }), ui.plotConfig);
    } else {
      ui.empty('pAllocChart', 'No allocation');
    }

    if (data.concentration) {
      ui.el('pConcentration').innerHTML = `
        Effective assets: <strong>${data.concentration.effective_n_assets}</strong> ·
        HHI ${data.concentration.herfindahl_index} ·
        largest ${data.concentration.largest_position || '—'} (${fmt.pct((data.concentration.largest_weight || 0) * 100, 1, false)})`;
    }

    renderMetricsGrid('pMetrics', data.metrics || {}, METRIC_SPEC);

    const tx = await api.transactions(activePortfolio, 40);
    ui.el('txTable').innerHTML = tx.transactions.length ? `
      <table>
        <thead><tr><th>Symbol</th><th>Side</th><th class="t-right">Qty</th>
          <th class="t-right">Price</th><th class="t-right">Fees</th><th>Source</th></tr></thead>
        <tbody>${tx.transactions.map((t) => `
          <tr><td class="sym-cell">${t.symbol}</td>
            <td><span class="badge ${t.side === 'BUY' ? 'badge-green' : 'badge-red'}">${t.side}</span></td>
            <td class="t-right mono">${fmt.num(t.quantity, 4)}</td>
            <td class="t-right mono">${fmt.price(t.price)}</td>
            <td class="t-right text-muted">${fmt.money(t.fees)}</td>
            <td class="text-xs text-muted">${t.source}</td></tr>`).join('')}</tbody>
      </table>` : '<div class="empty">No trades yet — buys and sells will be logged here</div>';
  } catch (err) {
    ui.error('holdingsTable', err.message);
  }
}

/* ------------------------------------------------------------- trading */
/* Live state for the ticket, so the preview can be recomputed without
   re-fetching the quote on every keystroke. */
const ticket = { symbol: null, quote: null, position: null };

/** Fetch the quote for the ticket symbol and repaint quote + preview. */
async function refreshTradeQuote() {
  const symbol = ui.el('tSymbol').value.trim().toUpperCase();
  if (!symbol) return;
  ticket.symbol = symbol;
  ui.el('tQuote').innerHTML = '<span class="text-muted text-xs">Loading quote…</span>';
  try {
    const q = await api.quote(symbol);
    if (ui.el('tSymbol').value.trim().toUpperCase() !== symbol) return;  // raced
    ticket.quote = q;
    const cls = fmt.cls(q.change_percent);
    ui.el('tQuote').innerHTML = `
      <div class="trade-quote-main">
        <span class="trade-quote-price">${fmt.price(q.price)}</span>
        <span class="${cls} text-sm">${fmt.arrow(q.change_percent)} ${fmt.pct(q.change_percent)}</span>
      </div>
      <div class="text-xs text-muted">${q.name || symbol} · ${ui.sourceBadge(q.source)}</div>`;
  } catch (err) {
    ticket.quote = null;
    ui.el('tQuote').innerHTML = `<span class="down text-xs">⚠ ${err.message}</span>`;
  }
  renderTradePreview();
}

/** Show exactly what the order will do before it is sent. */
function renderTradePreview() {
  const box = ui.el('tPreview');
  if (!box) return;
  const notional = parseFloat(ui.el('tNotional').value);
  const price = ticket.quote?.price;

  // The order buttons carry a subline showing what the click will actually do.
  // It has to be cleared as well as filled, or a stale figure outlives the
  // symbol it belonged to.
  const setSub = (id, text) => { const n = ui.el(id); if (n) n.textContent = text; };
  const symbol = ui.el('tSymbol').value.trim().toUpperCase();

  if (!price || !Number.isFinite(notional) || notional <= 0) {
    box.innerHTML = '';
    setSub('buySub', '—');
    setSub('sellSub', '—');
    return;
  }

  const qty = notional / price;
  const fees = notional * 0.001;                 // matches DEFAULT_FEE_RATE
  const held = ticket.position?.quantity || 0;
  const cash = portfolioCash;

  // Warn before the backend has to reject the order: an error toast after the
  // fact is a worse experience than a disabled-looking preview beforehand.
  const warnings = [];
  if (notional + fees > cash) {
    warnings.push(`Buying needs ${fmt.money(notional + fees)} but only ${fmt.money(cash)} is available`);
  }
  if (held <= 0) {
    warnings.push('No existing position — selling is not possible');
  } else if (qty > held) {
    warnings.push(`Selling ${fmt.num(qty, 4)} exceeds the ${fmt.num(held, 4)} held`);
  }

  setSub('buySub', `${fmt.num(qty, 4)} ${symbol} · ${fmt.money(notional + fees)}`);
  setSub('sellSub', held > 0
    ? `${fmt.num(Math.min(qty, held), 4)} ${symbol} · ${fmt.money(Math.min(qty, held) * price)}`
    : `nothing held in ${symbol}`);

  box.innerHTML = `
    <div class="trade-preview-row"><span>Quantity</span>
      <span class="mono">${fmt.num(qty, 6)}</span></div>
    <div class="trade-preview-row"><span>Est. fee (0.10%)</span>
      <span class="mono">${fmt.money(fees)}</span></div>
    <div class="trade-preview-row"><span>Total cost</span>
      <span class="mono">${fmt.money(notional + fees)}</span></div>
    ${held > 0 ? `<div class="trade-preview-row"><span>Currently held</span>
      <span class="mono">${fmt.num(held, 4)}</span></div>` : ''}
    ${warnings.map((w) => `<div class="trade-warning">⚠ ${w}</div>`).join('')}`;
}

async function doTrade(side) {
  if (!activePortfolio) { ui.toast('Create a portfolio first', 'error'); return; }
  const symbol = ui.el('tSymbol').value.trim().toUpperCase();
  const notional = parseFloat(ui.el('tNotional').value);
  if (!symbol) { ui.toast('Choose a symbol first', 'error'); return; }
  if (!Number.isFinite(notional) || notional <= 0) {
    ui.toast('Enter an amount greater than zero', 'error'); return;
  }

  const buyBtn = ui.el('buyBtn');
  const sellBtn = ui.el('sellBtn');
  [buyBtn, sellBtn].forEach((b) => { if (b) b.disabled = true; });
  try {
    const r = await api.trade(activePortfolio, { symbol, side, notional });
    ui.toast(`${side} ${fmt.num(r.quantity, 4)} ${symbol} @ ${fmt.price(r.price)}`, 'success');
    await loadPortfolio();
    await refreshTradeQuote();
  } catch (err) {
    ui.toast(`Trade failed: ${err.message}`, 'error', 6000);
  } finally {
    [buyBtn, sellBtn].forEach((b) => { if (b) b.disabled = false; });
  }
}

async function createPortfolio() {
  const name = ui.el('npName').value.trim() || `Portfolio ${new Date().toLocaleDateString()}`;
  const capital = parseFloat(ui.el('npCapital').value) || 100000;
  try {
    const p = await api.createPortfolio({ name, initial_capital: capital });
    activePortfolio = p.id;
    ui.el('npName').value = '';
    ui.toast(`Created “${p.name}”`, 'success');
    await loadPortfolios();
  } catch (err) {
    ui.toast(err.message, 'error');
  }
}

async function deleteActivePortfolio() {
  if (!activePortfolio) { ui.toast('No portfolio selected', 'error'); return; }
  const select = ui.el('portfolioSelect');
  const name = select.options[select.selectedIndex]?.textContent || `#${activePortfolio}`;

  // Deleting takes the positions and the whole trade history with it, and there
  // is no undo. A native confirm() is the right weight here: it is blocking,
  // impossible to miss, and states exactly what is about to disappear.
  const held = ui.el('holdingsTable').querySelectorAll('.holding-row').length;
  const warning = held
    ? `\n\nThis portfolio still holds ${held} position${held === 1 ? '' : 's'}.`
    : '';
  if (!window.confirm(
    `Delete ${name}?${warning}\n\nIts positions and transaction history will be `
    + 'permanently removed. This cannot be undone.')) return;

  const btn = ui.el('deletePBtn');
  btn.disabled = true;
  try {
    await api.deletePortfolio(activePortfolio);
    ui.toast(`Deleted ${name}`, 'success');
    activePortfolio = null;          // force loadPortfolios() to pick another
    await loadPortfolios();
  } catch (err) {
    ui.toast(`Could not delete: ${err.message}`, 'error', 6000);
  } finally {
    btn.disabled = false;
  }
}

async function planRebalance(execute = false) {
  if (!activePortfolio) return;
  const btn = execute ? ui.el('executeBtn') : ui.el('planBtn');
  btn.disabled = true;
  ui.loading('rebalanceBox', execute ? 'Executing orders…' : 'Optimising…');
  try {
    const result = await api.rebalance(activePortfolio, {
      objective: ui.el('optObjective').value, period: pPeriod, execute,
    });
    const plan = execute ? result.plan : result;
    ui.el('rebalanceBox').innerHTML = `
      ${execute ? `<div class="info-box">Executed ${result.executed.length} orders${result.failed.length ? `, ${result.failed.length} failed` : ''}.</div>` : ''}
      <div class="text-sm mb-1">Turnover <strong>${fmt.money(plan.total_turnover)}</strong>
        (${fmt.num(plan.turnover_pct, 1)}%) · est. fees ${fmt.money(plan.total_estimated_fees)}</div>
      ${plan.orders.length ? `
        <table>
          <thead><tr><th>Symbol</th><th>Side</th><th class="t-right">Current</th>
            <th class="t-right">Target</th><th class="t-right">Notional</th></tr></thead>
          <tbody>${plan.orders.map((o) => `
            <tr><td class="sym-cell">${o.symbol}</td>
              <td><span class="badge ${o.side === 'BUY' ? 'badge-green' : 'badge-red'}">${o.side}</span></td>
              <td class="t-right">${fmt.pct(o.current_weight * 100, 1, false)}</td>
              <td class="t-right">${fmt.pct(o.target_weight * 100, 1, false)}</td>
              <td class="t-right">${fmt.money(o.notional)}</td></tr>`).join('')}</tbody>
        </table>` : '<div class="empty">Portfolio is already within tolerance</div>'}
      ${plan.optimiser?.sharpe_ratio ? `<div class="text-xs text-muted mt-2">
        Optimiser target: expected return ${fmt.pct(plan.optimiser.expected_annual_return * 100, 1)},
        volatility ${fmt.pct(plan.optimiser.expected_annual_volatility * 100, 1)},
        Sharpe ${fmt.num(plan.optimiser.sharpe_ratio, 2)}. Positions capped at 40% for diversification.</div>` : ''}`;
    if (execute) { ui.toast('Rebalance executed', 'success'); await loadPortfolio(); }
  } catch (err) {
    ui.error('rebalanceBox', err.message);
  } finally {
    btn.disabled = false;
  }
}

/* The frontier basket. A comma-separated string was the whole interface here:
   it accepted "AAPL,, MSFT" and typos silently, and offered no way to see what
   the universe contained. The picker owns the list now. */
let frontierPicker = null;

function renderFrontierChips() {
  const picked = frontierPicker ? frontierPicker.selected : [];
  ui.el('optSymbolsSummary').innerHTML = picked.map((s) => `
    <span class="sp-chip" data-drop="${s}">${s}<i aria-hidden="true">×</i></span>`).join('');
  ui.el('optSymbolsSummary').querySelectorAll('[data-drop]').forEach((chip) => {
    chip.style.cursor = 'pointer';
    chip.addEventListener('click', () => frontierPicker.deselect(chip.dataset.drop));
  });
}

async function computeFrontier() {
  const symbols = frontierPicker ? [...frontierPicker.selected] : [];
  if (symbols.length < 2) {
    ui.toast('Select at least 2 assets to plot a frontier', 'error');
    return;
  }
  const btn = ui.el('frontierBtn');
  btn.disabled = true;
  ui.loading('frontierChart', 'Running optimisation…');
  try {
    const data = await api.optimise({
      symbols, objective: ui.el('optObjective').value, period: '2y',
    });
    const node = ui.el('frontierChart');
    node.innerHTML = '';
    Plotly.newPlot(node, [
      { type: 'scatter', mode: 'lines+markers', x: data.efficient_frontier.map((p) => p.volatility * 100),
        y: data.efficient_frontier.map((p) => p.return * 100), name: 'Efficient frontier',
        line: { color: C().accent, width: 2 }, marker: { size: 4 } },
      { type: 'scatter', mode: 'markers', x: [data.expected_annual_volatility * 100],
        y: [data.expected_annual_return * 100], name: 'Optimal portfolio',
        marker: { color: C().green, size: 15, symbol: 'star', line: { color: '#fff', width: 1 } } },
    ], ui.plotLayout({
      height: 260, hovermode: 'closest',
      xaxis: { gridcolor: C().grid, title: { text: 'Volatility (%)', font: { size: 10 } } },
      yaxis: { gridcolor: C().grid, title: { text: 'Expected return (%)', font: { size: 10 } } },
    }), ui.plotConfig);

    ui.toast(`Optimal Sharpe ${fmt.num(data.sharpe_ratio, 2)} — ${Object.entries(data.weights)
      .filter(([, w]) => w > 0.01).map(([s, w]) => `${s} ${fmt.pct(w * 100, 0, false)}`).join(', ')}`, 'success', 7000);
  } catch (err) {
    ui.error('frontierChart', err.message);
  } finally {
    btn.disabled = false;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initTimeRange();
  ui.el('tSymbol').value = getActiveSymbol();
  loadPortfolios();

  // Same grouped, ranked picker as the analysis panel, so the Trade ticket is
  // no longer a bare text box where a typo becomes a position.
  new SymbolPicker('tSymbol', 'tSymbolPanel', () => {
    ticket.position = null;
    refreshTradeQuote();
    loadPortfolio();          // refresh the held quantity for this symbol
  });
  refreshTradeQuote();

  ui.el('tNotional').addEventListener('input', renderTradePreview);

  // Size the order as a share of buying power rather than typing a figure.
  document.querySelectorAll('#tQuickAmounts .chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const pct = parseFloat(chip.dataset.pct) / 100;
      // Leave room for the fee on a "Max" order, otherwise it always bounces.
      const usable = portfolioCash / (1 + 0.001);
      ui.el('tNotional').value = Math.max(0, Math.floor(usable * pct * 100) / 100);
      renderTradePreview();
    });
  });

  ui.el('portfolioSelect').addEventListener('change', (e) => {
    activePortfolio = parseInt(e.target.value, 10);
    loadPortfolio();
  });
  ui.el('buyBtn').addEventListener('click', () => doTrade('BUY'));
  ui.el('sellBtn').addEventListener('click', () => doTrade('SELL'));
  ui.el('createPBtn').addEventListener('click', createPortfolio);
  ui.el('deletePBtn').addEventListener('click', deleteActivePortfolio);
  ui.el('refreshPBtn').addEventListener('click', loadPortfolio);
  ui.el('planBtn').addEventListener('click', () => planRebalance(false));
  ui.el('executeBtn').addEventListener('click', () => planRebalance(true));
  // Multi-select: the frontier needs a basket, not one symbol. syncActive is
  // off by default in multi mode so adding an asset here does not retarget the
  // Trade ticket and the analysis panels to it.
  frontierPicker = new SymbolPicker('optSymbols', 'optSymbolsPanel', null, {
    multi: true,
    selected: ['AAPL', 'MSFT', 'NVDA', 'SPY', 'GC=F'],
    onChange: renderFrontierChips,
  });
  renderFrontierChips();
  ui.el('frontierBtn').addEventListener('click', computeFrontier);

  onTimeRangeChange((key) => {
    pPeriod = key;
    loadPortfolio();
  });
});

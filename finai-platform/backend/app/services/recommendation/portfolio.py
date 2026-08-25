"""Paper-trading portfolio service: positions, P&L, analytics, rebalancing."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import PortfolioError, SymbolNotFoundError
from app.core.logging import get_logger
from app.db.models import Portfolio, PortfolioSnapshot, Position, Transaction
from app.services.data.market_data import market_data_service
from app.services.risk.metrics import correlation_matrix, full_metrics, optimise_portfolio

logger = get_logger(__name__)

DEFAULT_FEE_RATE = 0.001


class PortfolioService:
    # ------------------------------------------------------------- CRUD
    async def create(self, db: AsyncSession, name: str, initial_capital: float = 100_000.0,
                     description: str | None = None, base_currency: str = "USD",
                     strategy: str = "manual", user_id: int | None = None) -> Portfolio:
        portfolio = Portfolio(
            name=name, description=description, base_currency=base_currency,
            initial_capital=initial_capital, cash=initial_capital,
            strategy=strategy, user_id=user_id,
        )
        db.add(portfolio)
        await db.flush()
        await db.refresh(portfolio)
        logger.info("portfolio created: %s (#%d)", name, portfolio.id)
        return portfolio

    async def get(self, db: AsyncSession, portfolio_id: int) -> Portfolio:
        result = await db.execute(select(Portfolio).where(Portfolio.id == portfolio_id))
        portfolio = result.scalar_one_or_none()
        if portfolio is None:
            # 404, not 400: the request was well-formed, the resource simply is
            # not there. A client cannot distinguish "you sent nonsense" from
            # "it is gone" when both arrive as 400 — which matters most right
            # after a delete, when polling a stale id is expected.
            raise SymbolNotFoundError(f"Portfolio {portfolio_id} not found")
        return portfolio

    async def list_all(self, db: AsyncSession) -> list[Portfolio]:
        result = await db.execute(select(Portfolio).order_by(Portfolio.created_at.desc()))
        return list(result.scalars().all())

    async def delete(self, db: AsyncSession, portfolio_id: int) -> None:
        portfolio = await self.get(db, portfolio_id)
        await db.delete(portfolio)

    async def positions(self, db: AsyncSession, portfolio_id: int) -> list[Position]:
        result = await db.execute(select(Position).where(Position.portfolio_id == portfolio_id))
        return [p for p in result.scalars().all() if abs(p.quantity) > 1e-9]

    async def transactions(self, db: AsyncSession, portfolio_id: int, limit: int = 200) -> list[Transaction]:
        result = await db.execute(
            select(Transaction).where(Transaction.portfolio_id == portfolio_id)
            .order_by(Transaction.executed_at.desc()).limit(limit))
        return list(result.scalars().all())

    # ------------------------------------------------------------ trading
    async def execute_trade(self, db: AsyncSession, portfolio_id: int, symbol: str, side: str,
                            quantity: float | None = None, notional: float | None = None,
                            price: float | None = None, source: str = "manual",
                            notes: str | None = None) -> dict:
        portfolio = await self.get(db, portfolio_id)
        symbol = symbol.upper().strip()
        side = side.upper().strip()
        if side not in ("BUY", "SELL"):
            raise PortfolioError("side must be BUY or SELL")

        quote = market_data_service.get_quote(symbol)

        # The synthetic engine can price *anything*, so a typo used to execute
        # happily at a fabricated $100.00 and sit in the portfolio as a real
        # position. Refuse a symbol that is neither in the curated universe nor
        # priced by a live provider: booking an invented price into an account
        # is the one mistake this module must never make.
        if price is None and quote.get("source") == "synthetic":
            from app.services.data.universe import get_instrument

            if get_instrument(symbol) is None:
                raise PortfolioError(
                    f"'{symbol}' could not be verified: it is not in the instrument "
                    "universe and no live provider returned a price. Check the "
                    "ticker, or pass an explicit price to trade it anyway.")

        exec_price = float(price or quote.get("price") or 0.0)
        if exec_price <= 0:
            raise PortfolioError(f"No valid price for {symbol}")

        if quantity is None:
            if notional is None:
                raise PortfolioError("Provide either quantity or notional")
            quantity = notional / exec_price
        quantity = float(quantity)
        if quantity <= 0:
            raise PortfolioError("quantity must be positive")

        gross = quantity * exec_price
        fees = gross * DEFAULT_FEE_RATE

        result = await db.execute(select(Position).where(
            Position.portfolio_id == portfolio_id, Position.symbol == symbol))
        position = result.scalar_one_or_none()

        if side == "BUY":
            if portfolio.cash < gross + fees:
                raise PortfolioError(
                    f"Insufficient cash: need {gross + fees:,.2f}, have {portfolio.cash:,.2f}")
            portfolio.cash -= gross + fees
            if position is None:
                position = Position(portfolio_id=portfolio_id, symbol=symbol, quantity=quantity,
                                    average_price=exec_price,
                                    asset_class=quote.get("asset_class", "equity"))
                db.add(position)
            else:
                total_cost = position.average_price * position.quantity + gross
                position.quantity += quantity
                position.average_price = total_cost / position.quantity
        else:  # SELL
            # Tolerance has to scale with the position. Clients read a quantity
            # back from the API (rounded to 8 dp) and send it straight back to
            # close out; on a 116-share holding that rounding is ~3.5e-9, which
            # a fixed 1e-9 epsilon rejected as "insufficient position". The
            # order was for the same position, only re-serialised.
            held = position.quantity if position else 0.0
            tolerance = max(1e-9, held * 1e-6)
            if position is None or held < quantity - tolerance:
                raise PortfolioError(
                    f"Insufficient position in {symbol}: holding {held}, selling {quantity}")
            # Never sell more than is actually held: without this the clamp
            # above would let the rounding overshoot drive the position negative.
            quantity = min(quantity, held)
            gross = quantity * exec_price
            fees = gross * DEFAULT_FEE_RATE
            portfolio.cash += gross - fees
            position.quantity -= quantity
            # Close the position on a *relative* remainder, not an absolute one.
            # Selling a quantity read back from the API (rounded to 8 dp) left
            # ~1e-8 of a share behind — above the old 1e-9 floor, so the holding
            # survived at qty 0.0, and could even go slightly negative, showing
            # a phantom short. The tolerance now scales with the position, and a
            # negative residue is always swept.
            residual_is_dust = abs(position.quantity) < max(1e-9, abs(quantity) * 1e-6)
            if residual_is_dust or position.quantity <= 0:
                await db.delete(position)

        transaction = Transaction(
            portfolio_id=portfolio_id, symbol=symbol, side=side, quantity=quantity,
            price=exec_price, fees=fees, source=source, notes=notes)
        db.add(transaction)
        portfolio.updated_at = datetime.now(UTC)
        await db.flush()

        return {
            "portfolio_id": portfolio_id, "symbol": symbol, "side": side,
            "quantity": round(quantity, 8), "price": round(exec_price, 6),
            "gross": round(gross, 2), "fees": round(fees, 2),
            "cash_after": round(portfolio.cash, 2), "source": source,
            "executed_at": transaction.executed_at.isoformat() if transaction.executed_at else None,
        }

    # ----------------------------------------------------------- valuation
    async def valuation(self, db: AsyncSession, portfolio_id: int) -> dict:
        portfolio = await self.get(db, portfolio_id)
        positions = await self.positions(db, portfolio_id)

        holdings, invested = [], 0.0
        if positions:
            quotes = {q["symbol"]: q for q in
                      market_data_service.get_quotes([p.symbol for p in positions])}
            for pos in positions:
                quote = quotes.get(pos.symbol, {})
                price = float(quote.get("price") or pos.average_price)
                market_value = pos.quantity * price
                cost_basis = pos.quantity * pos.average_price
                invested += market_value
                holdings.append({
                    "symbol": pos.symbol, "asset_class": pos.asset_class,
                    "quantity": round(pos.quantity, 8),
                    "average_price": round(pos.average_price, 6),
                    "current_price": round(price, 6),
                    "market_value": round(market_value, 2),
                    "cost_basis": round(cost_basis, 2),
                    "unrealised_pnl": round(market_value - cost_basis, 2),
                    "unrealised_pnl_pct": round((price / pos.average_price - 1) * 100, 3)
                    if pos.average_price else 0.0,
                    "day_change_pct": round(float(quote.get("change_percent") or 0.0), 3),
                })

        total_value = portfolio.cash + invested
        pnl = total_value - portfolio.initial_capital
        for h in holdings:
            h["weight"] = round(h["market_value"] / total_value, 4) if total_value else 0.0

        allocation_by_class: dict[str, float] = {}
        for h in holdings:
            allocation_by_class[h["asset_class"]] = allocation_by_class.get(h["asset_class"], 0.0) + h["market_value"]
        allocation = {k: round(v / total_value, 4) for k, v in allocation_by_class.items()} if total_value else {}
        if total_value:
            allocation["cash"] = round(portfolio.cash / total_value, 4)

        return {
            "portfolio_id": portfolio_id, "name": portfolio.name,
            "base_currency": portfolio.base_currency,
            "initial_capital": round(portfolio.initial_capital, 2),
            "cash": round(portfolio.cash, 2),
            "invested_value": round(invested, 2),
            "total_value": round(total_value, 2),
            "total_pnl": round(pnl, 2),
            "total_pnl_pct": round(pnl / portfolio.initial_capital * 100, 3) if portfolio.initial_capital else 0.0,
            "n_positions": len(holdings),
            "holdings": sorted(holdings, key=lambda h: h["market_value"], reverse=True),
            "allocation": allocation,
            "cash_weight": round(portfolio.cash / total_value, 4) if total_value else 1.0,
            "as_of": datetime.now(UTC).isoformat(),
        }

    # ------------------------------------------------------------ analytics
    async def analytics(self, db: AsyncSession, portfolio_id: int, period: str = "1y",
                        benchmark: str = "SPY") -> dict:
        valuation = await self.valuation(db, portfolio_id)
        holdings = valuation["holdings"]
        if not holdings:
            return {**valuation, "metrics": {}, "message": "Portfolio holds no positions"}

        symbols = [h["symbol"] for h in holdings]
        weights = np.array([h["market_value"] for h in holdings], dtype=float)
        weights = weights / weights.sum()

        try:
            prices = market_data_service.get_price_matrix(symbols, period=period)
        except Exception as exc:
            return {**valuation, "metrics": {}, "error": f"price history unavailable: {exc}"}

        returns = prices.pct_change().dropna()
        aligned_weights = np.array([weights[symbols.index(c)] for c in returns.columns])
        portfolio_returns = returns @ aligned_weights

        # Benchmark alignment is needed for beta/alpha, but it must not decide
        # what the *portfolio's own* return is. A crypto holding trades 7 days a
        # week (365 bars) while SPY trades ~250: intersecting the two silently
        # dropped every weekend, so the headline figures described a different
        # series from the equity curve drawn right next to them. Observed on the
        # demo portfolio: metrics said -0.16% while the curve showed +2.76%.
        #
        # So: portfolio metrics are always computed on the full series, and the
        # benchmark-relative ones (beta, alpha, information ratio) are computed
        # separately on the overlapping dates where the comparison is valid.
        metrics = full_metrics(portfolio_returns)

        bench_returns = None
        try:
            bench = market_data_service.get_history(benchmark, period=period).df["close"].pct_change().dropna()
            bench_returns = bench.reindex(portfolio_returns.index).dropna()
            overlap = portfolio_returns.reindex(bench_returns.index).dropna()
            bench_returns = bench_returns.reindex(overlap.index)
            if len(overlap) > 20:
                relative = full_metrics(overlap, bench_returns)
                for key in ("beta", "alpha", "information_ratio",
                            "correlation_to_benchmark"):
                    if key in relative:
                        metrics[key] = relative[key]
                # Say how much of the history the comparison actually covers, so
                # a beta computed on 250 of 365 days is not read as absolute.
                metrics["benchmark_overlap_days"] = int(len(overlap))
        except Exception as exc:
            logger.debug("benchmark metrics unavailable for %s: %s", benchmark, exc)

        # Anchor the curve at the initial capital. cumprod() alone starts *after*
        # the first return, so the chart opened at 100,377 on a 100,000 account
        # and visually under-reported the total gain by that first day.
        equity = (1 + portfolio_returns).cumprod() * valuation["initial_capital"]
        if len(equity) and isinstance(equity.index, pd.DatetimeIndex):
            first_day = equity.index[0] - pd.Timedelta(days=1)
            equity = pd.concat([
                pd.Series([float(valuation["initial_capital"])], index=[first_day]),
                equity,
            ])
        drawdown = (equity / equity.cummax() - 1)

        corr = correlation_matrix(returns) if len(symbols) > 1 else {}
        cov = returns.cov().values * 252
        from app.services.risk.metrics import risk_contribution
        rc = risk_contribution(aligned_weights, cov)

        return {
            **valuation,
            "benchmark": benchmark,
            "period": period,
            "metrics": metrics,
            "equity_curve": [{"date": str(d.date()), "value": round(float(v), 2)}
                             for d, v in equity.items()],
            "drawdown_curve": [{"date": str(d.date()), "drawdown": round(float(v), 5)}
                               for d, v in drawdown.items()],
            "correlation": corr,
            "risk_contribution": {sym: round(float(r), 4)
                                  for sym, r in zip(returns.columns, rc, strict=False)},
            "concentration": {
                "herfindahl_index": round(float(np.sum(weights ** 2)), 4),
                "effective_n_assets": round(float(1 / np.sum(weights ** 2)), 2),
                "largest_position": holdings[0]["symbol"] if holdings else None,
                "largest_weight": round(float(weights.max()), 4),
            },
        }

    # ----------------------------------------------------------- rebalance
    async def rebalance_plan(self, db: AsyncSession, portfolio_id: int,
                             target_weights: dict[str, float] | None = None,
                             objective: str = "max_sharpe", period: str = "1y",
                             tolerance: float = 0.02) -> dict:
        valuation = await self.valuation(db, portfolio_id)
        total_value = valuation["total_value"]
        current = {h["symbol"]: h["weight"] for h in valuation["holdings"]}

        if target_weights is None:
            symbols = list(current.keys())
            if len(symbols) < 2:
                raise PortfolioError("Need at least 2 positions to optimise, or pass target_weights")
            returns = market_data_service.get_returns_matrix(symbols, period=period)
            optimisation = optimise_portfolio(returns, objective=objective)
            target_weights = optimisation["weights"]
            optimiser_info = {k: v for k, v in optimisation.items() if k != "efficient_frontier"}
        else:
            total = sum(target_weights.values())
            if total > 1.0001:
                target_weights = {k: v / total for k, v in target_weights.items()}
            optimiser_info = {"objective": "user_specified"}

        orders = []
        for symbol in set(list(current.keys()) + list(target_weights.keys())):
            cur_w = current.get(symbol, 0.0)
            tgt_w = target_weights.get(symbol, 0.0)
            drift = tgt_w - cur_w
            if abs(drift) < tolerance:
                continue
            quote = market_data_service.get_quote(symbol)
            price = float(quote.get("price") or 0)
            if price <= 0:
                continue
            notional = drift * total_value
            orders.append({
                "symbol": symbol, "side": "BUY" if drift > 0 else "SELL",
                "current_weight": round(cur_w, 4), "target_weight": round(tgt_w, 4),
                "drift": round(drift, 4), "notional": round(abs(notional), 2),
                "quantity": round(abs(notional) / price, 6), "price": round(price, 6),
                "estimated_fees": round(abs(notional) * DEFAULT_FEE_RATE, 2),
            })

        orders.sort(key=lambda o: (o["side"] == "BUY", -o["notional"]))
        return {
            "portfolio_id": portfolio_id, "total_value": total_value,
            "current_weights": current, "target_weights": target_weights,
            "orders": orders, "n_orders": len(orders),
            "total_turnover": round(sum(o["notional"] for o in orders), 2),
            "total_estimated_fees": round(sum(o["estimated_fees"] for o in orders), 2),
            "turnover_pct": round(sum(o["notional"] for o in orders) / total_value * 100, 2) if total_value else 0.0,
            "optimiser": optimiser_info,
            "tolerance": tolerance,
        }

    async def execute_rebalance(self, db: AsyncSession, portfolio_id: int,
                                target_weights: dict[str, float] | None = None,
                                objective: str = "max_sharpe", period: str = "1y") -> dict:
        plan = await self.rebalance_plan(db, portfolio_id, target_weights, objective, period)
        executed, failed = [], []
        for order in plan["orders"]:      # sells first (sorted), frees cash
            try:
                executed.append(await self.execute_trade(
                    db, portfolio_id, order["symbol"], order["side"],
                    quantity=order["quantity"], source="rebalance",
                    notes=f"rebalance to {order['target_weight']:.1%}"))
            except Exception as exc:
                failed.append({"symbol": order["symbol"], "side": order["side"], "error": str(exc)[:200]})
        return {"plan": plan, "executed": executed, "failed": failed,
                "valuation": await self.valuation(db, portfolio_id)}

    # ----------------------------------------------------------- snapshots
    async def take_snapshot(self, db: AsyncSession, portfolio_id: int) -> PortfolioSnapshot:
        valuation = await self.valuation(db, portfolio_id)
        snapshot = PortfolioSnapshot(
            portfolio_id=portfolio_id, total_value=valuation["total_value"],
            cash=valuation["cash"], invested_value=valuation["invested_value"],
            pnl=valuation["total_pnl"], pnl_pct=valuation["total_pnl_pct"],
            metrics={"allocation": valuation["allocation"], "n_positions": valuation["n_positions"]},
        )
        db.add(snapshot)
        await db.flush()
        return snapshot

    async def snapshots(self, db: AsyncSession, portfolio_id: int, limit: int = 500) -> list[dict]:
        result = await db.execute(
            select(PortfolioSnapshot).where(PortfolioSnapshot.portfolio_id == portfolio_id)
            .order_by(PortfolioSnapshot.taken_at.asc()).limit(limit))
        return [{"date": s.taken_at.isoformat(), "total_value": s.total_value,
                 "cash": s.cash, "pnl": s.pnl, "pnl_pct": s.pnl_pct}
                for s in result.scalars().all()]


portfolio_service = PortfolioService()

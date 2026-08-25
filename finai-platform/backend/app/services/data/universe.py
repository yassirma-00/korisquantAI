"""Reference universe of tradable instruments exposed by the platform."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

AssetClass = Literal["equity", "crypto", "etf", "commodity", "forex", "index"]


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    asset_class: AssetClass
    exchange: str
    currency: str
    base_price: float          # anchor used by the synthetic engine
    annual_drift: float        # expected yearly log-drift
    annual_vol: float          # annualised volatility
    sector: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


UNIVERSE: tuple[Instrument, ...] = (
    # ------------------------------------------------------------ US equity
    Instrument("AAPL", "Apple Inc.", "equity", "NASDAQ", "USD", 214.0, 0.14, 0.26, "Technology"),
    Instrument("MSFT", "Microsoft Corporation", "equity", "NASDAQ", "USD", 432.0, 0.16, 0.24, "Technology"),
    Instrument("NVDA", "NVIDIA Corporation", "equity", "NASDAQ", "USD", 121.0, 0.35, 0.48, "Semiconductors"),
    Instrument("AMZN", "Amazon.com Inc.", "equity", "NASDAQ", "USD", 186.0, 0.15, 0.30, "Consumer Discretionary"),
    Instrument("GOOGL", "Alphabet Inc. Class A", "equity", "NASDAQ", "USD", 178.0, 0.13, 0.27, "Communication"),
    Instrument("META", "Meta Platforms Inc.", "equity", "NASDAQ", "USD", 505.0, 0.18, 0.35, "Communication"),
    Instrument("TSLA", "Tesla Inc.", "equity", "NASDAQ", "USD", 248.0, 0.10, 0.55, "Automotive"),
    Instrument("JPM", "JPMorgan Chase & Co.", "equity", "NYSE", "USD", 205.0, 0.09, 0.22, "Financials"),
    Instrument("XOM", "Exxon Mobil Corporation", "equity", "NYSE", "USD", 115.0, 0.07, 0.25, "Energy"),
    Instrument("JNJ", "Johnson & Johnson", "equity", "NYSE", "USD", 152.0, 0.05, 0.16, "Healthcare"),
    # ------------------------------------------------------------ EU equity
    Instrument("MC.PA", "LVMH Moet Hennessy", "equity", "Euronext Paris", "EUR", 660.0, 0.08, 0.28, "Luxury"),
    Instrument("AIR.PA", "Airbus SE", "equity", "Euronext Paris", "EUR", 148.0, 0.11, 0.27, "Aerospace"),
    Instrument("SAN.PA", "Sanofi S.A.", "equity", "Euronext Paris", "EUR", 92.0, 0.06, 0.19, "Healthcare"),
    # ----------------------------------------------------------------- ETF
    Instrument("SPY", "SPDR S&P 500 ETF Trust", "etf", "NYSE Arca", "USD", 545.0, 0.10, 0.16, "Broad Market"),
    Instrument("QQQ", "Invesco QQQ Trust", "etf", "NASDAQ", "USD", 470.0, 0.13, 0.21, "Technology"),
    Instrument("IWM", "iShares Russell 2000 ETF", "etf", "NYSE Arca", "USD", 215.0, 0.08, 0.23, "Small Cap"),
    Instrument("EEM", "iShares MSCI Emerging Markets", "etf", "NYSE Arca", "USD", 43.0, 0.06, 0.20, "Emerging"),
    # -------------------------------------------------------------- crypto
    Instrument("BTC-USD", "Bitcoin", "crypto", "Crypto", "USD", 63000.0, 0.45, 0.62, "Digital Assets"),
    Instrument("ETH-USD", "Ethereum", "crypto", "Crypto", "USD", 3200.0, 0.40, 0.72, "Digital Assets"),
    Instrument("SOL-USD", "Solana", "crypto", "Crypto", "USD", 148.0, 0.55, 0.95, "Digital Assets"),
    Instrument("BNB-USD", "BNB", "crypto", "Crypto", "USD", 585.0, 0.30, 0.70, "Digital Assets"),
    # ----------------------------------------------------------- commodity
    Instrument("GC=F", "Gold Futures", "commodity", "COMEX", "USD", 2380.0, 0.06, 0.14, "Precious Metals"),
    Instrument("SI=F", "Silver Futures", "commodity", "COMEX", "USD", 29.5, 0.07, 0.28, "Precious Metals"),
    Instrument("CL=F", "Crude Oil WTI Futures", "commodity", "NYMEX", "USD", 78.0, 0.03, 0.35, "Energy"),
    # --------------------------------------------------------------- forex
    Instrument("EURUSD=X", "Euro / US Dollar", "forex", "FX", "USD", 1.085, 0.0, 0.07, "Majors"),
    Instrument("GBPUSD=X", "British Pound / US Dollar", "forex", "FX", "USD", 1.275, 0.0, 0.08, "Majors"),
    Instrument("USDJPY=X", "US Dollar / Japanese Yen", "forex", "FX", "JPY", 156.0, 0.0, 0.09, "Majors"),
    Instrument("USDMAD=X", "US Dollar / Moroccan Dirham", "forex", "FX", "MAD", 9.85, 0.0, 0.05, "Emerging"),
    # -------------------------------------------------------------- index
    Instrument("^GSPC", "S&P 500 Index", "index", "Index", "USD", 5460.0, 0.10, 0.15, "US Large Cap"),
    Instrument("^IXIC", "NASDAQ Composite", "index", "Index", "USD", 17700.0, 0.13, 0.20, "US Tech"),
    Instrument("^FCHI", "CAC 40", "index", "Index", "EUR", 7600.0, 0.07, 0.17, "France"),
    Instrument("^VIX", "CBOE Volatility Index", "index", "Index", "USD", 13.5, 0.0, 0.85, "Volatility"),
)

BY_SYMBOL: dict[str, Instrument] = {i.symbol.upper(): i for i in UNIVERSE}

DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "SPY", "BTC-USD", "ETH-USD", "GC=F", "EURUSD=X"]


def get_instrument(symbol: str) -> Instrument | None:
    return BY_SYMBOL.get(symbol.upper().strip())


def list_instruments(asset_class: str | None = None, query: str | None = None) -> list[Instrument]:
    items = list(UNIVERSE)
    if asset_class:
        items = [i for i in items if i.asset_class == asset_class]
    if query:
        q = query.lower().strip()
        items = [i for i in items if q in i.symbol.lower() or q in i.name.lower()]
    return items


def infer_instrument(symbol: str) -> Instrument:
    """Return a known instrument or synthesise plausible metadata for an unknown one."""
    known = get_instrument(symbol)
    if known:
        return known
    sym = symbol.upper().strip()
    if sym.endswith("-USD"):
        return Instrument(sym, sym.replace("-USD", ""), "crypto", "Crypto", "USD", 100.0, 0.30, 0.80)
    if sym.endswith("=X"):
        return Instrument(sym, sym.replace("=X", ""), "forex", "FX", "USD", 1.0, 0.0, 0.08)
    if sym.endswith("=F"):
        return Instrument(sym, sym.replace("=F", ""), "commodity", "Futures", "USD", 100.0, 0.04, 0.28)
    if sym.startswith("^"):
        return Instrument(sym, sym.lstrip("^"), "index", "Index", "USD", 4000.0, 0.08, 0.16)
    return Instrument(sym, sym, "equity", "Unknown", "USD", 100.0, 0.10, 0.28)

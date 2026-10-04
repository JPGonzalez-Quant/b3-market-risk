"""Download e cache de preços de ações da B3 via Yahoo Finance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Dez ações líquidas do Ibovespa, de setores diferentes.
DEFAULT_TICKERS = [
    "PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3",
    "ABEV3", "WEGE3", "B3SA3", "RENT3", "LREN3",
]


def load_prices(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = "2012-01-01",
    end: str = "2026-10-01",
    cache_dir: str | Path = "data",
) -> pd.DataFrame:
    """Preços de fechamento ajustados (proventos e desdobramentos).

    Os dados ficam em cache em ``cache_dir`` para que a análise seja
    reprodutível sem depender de novo download.
    """
    cache = Path(cache_dir) / f"prices_{start}_{end}.csv"
    if cache.exists():
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
        return prices[tickers]

    import yfinance as yf

    raw = yf.download(
        [f"{t}.SA" for t in tickers],
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )["Close"]
    prices = raw.rename(columns=lambda c: c.removesuffix(".SA"))[tickers]
    prices = prices.dropna(how="all").ffill().dropna()
    # Remove feriados que o Yahoo registra repetindo o preço de todos os ativos.
    prices = prices[prices.diff().ne(0).any(axis=1)]

    cache.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache)
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Retornos logarítmicos diários."""
    return np.log(prices).diff().dropna()


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Retornos aritméticos diários (agregam linearmente numa carteira)."""
    return prices.pct_change().dropna()

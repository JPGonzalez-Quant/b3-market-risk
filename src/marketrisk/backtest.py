"""Backtesting de modelos de VaR: Kupiec, Christoffersen e semáforo de Basileia."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy


def exceptions(returns: pd.Series, var: pd.Series) -> pd.Series:
    """Dias em que a perda realizada superou o VaR previsto."""
    returns, var = returns.align(var, join="inner")
    return returns < -var


def kupiec_pof(n_obs: int, n_exceptions: int, alpha: float = 0.99) -> tuple[float, float]:
    """Teste de proporção de falhas (POF) de Kupiec (1995).

    H0: a frequência de exceções é igual a 1 - alpha.
    Retorna ``(estatística LR, p-valor)``; LR ~ qui-quadrado(1) sob H0.
    """
    p = 1 - alpha
    x, n = n_exceptions, n_obs
    p_hat = x / n
    log_h0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    log_h1 = xlogy(n - x, 1 - p_hat) + xlogy(x, p_hat)
    lr = -2 * (log_h0 - log_h1)
    return lr, stats.chi2.sf(lr, df=1)


def christoffersen_independence(hits: pd.Series | np.ndarray) -> tuple[float, float]:
    """Teste de independência de Christoffersen (1998).

    H0: uma exceção hoje não altera a probabilidade de exceção amanhã
    (exceções não vêm em clusters). Retorna ``(LR, p-valor)``.
    """
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = np.sum((prev == 0) & (curr == 0))
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))

    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    log_h0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_h1 = (
        xlogy(n00, 1 - pi01) + xlogy(n01, pi01)
        + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    )
    lr = -2 * (log_h0 - log_h1)
    return lr, stats.chi2.sf(lr, df=1)


def basel_zone(n_exceptions_250d: int) -> str:
    """Semáforo de Basileia para o VaR 99% em 250 dias úteis."""
    if n_exceptions_250d <= 4:
        return "Verde"
    if n_exceptions_250d <= 9:
        return "Amarela"
    return "Vermelha"


def backtest_summary(
    returns: pd.Series, forecasts: pd.DataFrame, alpha: float = 0.99
) -> pd.DataFrame:
    """Tabela-resumo do backtest de cada modelo de VaR."""
    rows = {}
    for model in forecasts.columns:
        hits = exceptions(returns, forecasts[model])
        n, x = len(hits), int(hits.sum())
        lr_uc, p_uc = kupiec_pof(n, x, alpha)
        lr_ind, p_ind = christoffersen_independence(hits)
        last_250 = int(hits.iloc[-250:].sum())
        rows[model] = {
            "Dias": n,
            "Exceções": x,
            "Esperado": round(n * (1 - alpha), 1),
            "Taxa (%)": round(100 * x / n, 2),
            "Kupiec p-valor": round(p_uc, 3),
            "Christoffersen p-valor": round(p_ind, 3),
            "Exceções (últ. 250d)": last_250,
            "Basileia": basel_zone(last_250),
        }
    return pd.DataFrame(rows).T

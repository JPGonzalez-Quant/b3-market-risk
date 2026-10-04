"""Value at Risk (VaR) e Expected Shortfall (CVaR).

Convenção: VaR e CVaR são reportados como **perdas positivas**, em fração do
valor da carteira. ``alpha`` é o nível de confiança (ex.: 0.99).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------
# Métodos univariados (sobre a série de retornos da carteira)
# --------------------------------------------------------------------------

def historical_var(returns: np.ndarray | pd.Series, alpha: float = 0.99) -> float:
    """VaR por simulação histórica: quantil empírico das perdas."""
    return -np.quantile(np.asarray(returns), 1 - alpha)


def historical_cvar(returns: np.ndarray | pd.Series, alpha: float = 0.99) -> float:
    """CVaR histórico: perda média nos dias piores que o VaR."""
    r = np.asarray(returns)
    cutoff = np.quantile(r, 1 - alpha)
    return -r[r <= cutoff].mean()


def parametric_var(mu: float, sigma: float, alpha: float = 0.99) -> float:
    """VaR paramétrico (delta-normal)."""
    return -(mu + sigma * stats.norm.ppf(1 - alpha))


def parametric_cvar(mu: float, sigma: float, alpha: float = 0.99) -> float:
    """CVaR sob normalidade: E[-R | R <= -VaR] = -mu + sigma * phi(z) / (1 - alpha)."""
    z = stats.norm.ppf(1 - alpha)
    return -mu + sigma * stats.norm.pdf(z) / (1 - alpha)


def cornish_fisher_var(returns: np.ndarray | pd.Series, alpha: float = 0.99) -> float:
    """VaR paramétrico com correção de assimetria e curtose (Cornish-Fisher)."""
    r = np.asarray(returns)
    z = stats.norm.ppf(1 - alpha)
    s = stats.skew(r)
    k = stats.kurtosis(r)  # excesso de curtose
    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3 * z) * k / 24
        - (2 * z**3 - 5 * z) * s**2 / 36
    )
    return -(r.mean() + r.std(ddof=1) * z_cf)


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """Volatilidade EWMA (RiskMetrics): sigma²_t = lam·sigma²_{t-1} + (1-lam)·r²_{t-1}.

    O valor na data t usa apenas informação até t-1, servindo diretamente
    como previsão de volatilidade para o dia t.
    """
    r2 = returns.pow(2).shift(1)
    var = r2.ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var)


def filtered_historical_var(
    returns: np.ndarray, sigma: np.ndarray, sigma_next: float, alpha: float = 0.99
) -> float:
    """Simulação histórica filtrada (Hull & White, 1998; Barone-Adesi et al., 1999).

    Os retornos passados são padronizados pela volatilidade EWMA da época
    (``r_t / sigma_t``) e reescalados pela volatilidade prevista para amanhã.
    Combina caudas empíricas com uma reação rápida a mudanças de regime.
    """
    z = np.asarray(returns) / np.asarray(sigma)
    return historical_var(z, alpha) * sigma_next


# --------------------------------------------------------------------------
# Métodos multivariados (a partir dos retornos dos ativos)
# --------------------------------------------------------------------------

def monte_carlo_var(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    alpha: float = 0.99,
    n_sims: int = 20_000,
    dist: str = "normal",
    df: float = 5.0,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """VaR e CVaR por Monte Carlo com retornos multivariados.

    ``dist="t"`` usa uma t de Student multivariada (caudas mais pesadas),
    reescalada para ter a mesma matriz de covariância ``cov``.
    Retorna ``(var, cvar)``.
    """
    rng = rng or np.random.default_rng()
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_sims, len(weights)))

    if dist == "t":
        # t multivariada = normal / sqrt(chi2/df); fator corrige a variância df/(df-2)
        w = rng.chisquare(df, size=(n_sims, 1)) / df
        z = z / np.sqrt(w) * np.sqrt((df - 2) / df)
    elif dist != "normal":
        raise ValueError(f"dist desconhecida: {dist}")

    sims = mu + z @ chol.T
    port = sims @ weights
    return historical_var(port, alpha), historical_cvar(port, alpha)


def component_var(
    weights: np.ndarray, cov: np.ndarray, alpha: float = 0.99
) -> np.ndarray:
    """Decomposição de Euler do VaR paramétrico entre os ativos.

    Os componentes somam exatamente o VaR da carteira (com média zero),
    mostrando de onde vem o risco.
    """
    z = -stats.norm.ppf(1 - alpha)
    sigma_p = np.sqrt(weights @ cov @ weights)
    marginal = z * (cov @ weights) / sigma_p
    return weights * marginal


# --------------------------------------------------------------------------
# Previsão rolante (fora da amostra)
# --------------------------------------------------------------------------

def rolling_var_forecasts(
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.99,
    window: int = 252,
    n_sims: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Previsões de VaR de 1 dia para cada data, usando só a janela anterior.

    A previsão para o dia t é estimada com os retornos de [t - window, t - 1],
    então pode ser comparada diretamente com o retorno realizado em t.
    """
    rng = np.random.default_rng(seed)
    port = asset_returns @ weights
    ewma_sigma = ewma_volatility(port)
    z = stats.norm.ppf(1 - alpha)

    values = asset_returns.to_numpy()
    sig = ewma_sigma.bfill().to_numpy()
    rows = []
    for t in range(window, len(asset_returns)):
        hist_assets = values[t - window : t]
        hist_port = hist_assets @ weights
        mu, sigma = hist_port.mean(), hist_port.std(ddof=1)
        mc_var, _ = monte_carlo_var(
            weights,
            hist_assets.mean(axis=0),
            np.cov(hist_assets, rowvar=False),
            alpha,
            n_sims,
            rng=rng,
        )
        rows.append(
            {
                "Histórico": historical_var(hist_port, alpha),
                "Paramétrico": parametric_var(mu, sigma, alpha),
                "Cornish-Fisher": cornish_fisher_var(hist_port, alpha),
                "EWMA (λ=0,94)": -z * ewma_sigma.iloc[t],
                "Monte Carlo": mc_var,
                "Hist. Filtrada (FHS)": filtered_historical_var(
                    hist_port, sig[t - window : t], sig[t], alpha
                ),
            }
        )
    return pd.DataFrame(rows, index=asset_returns.index[window:])

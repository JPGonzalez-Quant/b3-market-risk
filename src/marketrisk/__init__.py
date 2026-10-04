"""Ferramentas de risco de mercado: VaR, CVaR e backtesting."""

from .backtest import (
    backtest_summary,
    basel_zone,
    christoffersen_independence,
    exceptions,
    kupiec_pof,
)
from .data import DEFAULT_TICKERS, load_prices, log_returns, simple_returns
from .var import (
    component_var,
    cornish_fisher_var,
    ewma_volatility,
    filtered_historical_var,
    historical_cvar,
    historical_var,
    monte_carlo_var,
    parametric_cvar,
    parametric_var,
    rolling_var_forecasts,
)

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from marketrisk import (
    basel_zone,
    christoffersen_independence,
    component_var,
    cornish_fisher_var,
    filtered_historical_var,
    historical_cvar,
    historical_var,
    kupiec_pof,
    monte_carlo_var,
    parametric_cvar,
    parametric_var,
)

RNG = np.random.default_rng(0)
NORMAL_SAMPLE = RNG.normal(0.0, 0.02, size=500_000)


def test_parametric_var_matches_normal_quantile():
    # VaR 99% de N(0, 2%) = 2,326 * 2%
    assert parametric_var(0.0, 0.02, 0.99) == pytest.approx(0.02 * 2.3263, rel=1e-4)


def test_historical_var_converges_to_parametric():
    assert historical_var(NORMAL_SAMPLE, 0.99) == pytest.approx(
        parametric_var(0.0, 0.02, 0.99), rel=0.01
    )


def test_historical_cvar_converges_to_parametric():
    assert historical_cvar(NORMAL_SAMPLE, 0.975) == pytest.approx(
        parametric_cvar(0.0, 0.02, 0.975), rel=0.01
    )


def test_cvar_is_larger_than_var():
    assert parametric_cvar(0.0, 0.02, 0.99) > parametric_var(0.0, 0.02, 0.99)
    assert historical_cvar(NORMAL_SAMPLE) > historical_var(NORMAL_SAMPLE)


def test_cornish_fisher_reduces_to_normal_without_skew_or_kurtosis():
    assert cornish_fisher_var(NORMAL_SAMPLE, 0.99) == pytest.approx(
        parametric_var(0.0, 0.02, 0.99), rel=0.02
    )


def test_cornish_fisher_penalizes_fat_tails():
    t_sample = stats.t.rvs(df=4, size=200_000, random_state=1) * 0.01
    normal_var = parametric_var(t_sample.mean(), t_sample.std(), 0.99)
    assert cornish_fisher_var(t_sample, 0.99) > normal_var


def test_monte_carlo_matches_parametric_for_portfolio():
    w = np.array([0.5, 0.5])
    cov = np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
    sigma_p = np.sqrt(w @ cov @ w)
    var_mc, cvar_mc = monte_carlo_var(
        w, np.zeros(2), cov, 0.99, n_sims=400_000, rng=np.random.default_rng(3)
    )
    assert var_mc == pytest.approx(parametric_var(0, sigma_p, 0.99), rel=0.02)
    assert cvar_mc == pytest.approx(parametric_cvar(0, sigma_p, 0.99), rel=0.02)


def test_monte_carlo_student_t_has_heavier_tail():
    w = np.array([1.0])
    cov = np.array([[0.0004]])
    var_n, _ = monte_carlo_var(w, np.zeros(1), cov, 0.999, 400_000, rng=np.random.default_rng(4))
    var_t, _ = monte_carlo_var(w, np.zeros(1), cov, 0.999, 400_000, dist="t", rng=np.random.default_rng(4))
    assert var_t > var_n


def test_component_var_sums_to_total():
    w = np.array([0.2, 0.3, 0.5])
    a = RNG.normal(size=(3, 3))
    cov = a @ a.T * 1e-4
    total = parametric_var(0, np.sqrt(w @ cov @ w), 0.99)
    assert component_var(w, cov, 0.99).sum() == pytest.approx(total)


def test_kupiec_accepts_correct_model_and_rejects_bad_one():
    _, p_good = kupiec_pof(1000, 10, 0.99)  # exatamente o esperado
    _, p_bad = kupiec_pof(1000, 30, 0.99)  # 3x o esperado
    assert p_good == pytest.approx(1.0)
    assert p_bad < 0.001


def test_kupiec_handles_zero_exceptions():
    lr, p = kupiec_pof(250, 0, 0.99)
    assert lr == pytest.approx(-2 * 250 * np.log(0.99))
    assert 0 < p < 1


def test_christoffersen_detects_clustering():
    independent = np.zeros(1000, dtype=bool)
    independent[::100] = True
    clustered = np.zeros(1000, dtype=bool)
    clustered[500:510] = True
    assert christoffersen_independence(independent)[1] > 0.5
    assert christoffersen_independence(clustered)[1] < 0.001


@pytest.mark.parametrize(
    "n, zone", [(0, "Verde"), (4, "Verde"), (5, "Amarela"), (9, "Amarela"), (10, "Vermelha")]
)
def test_basel_zone(n, zone):
    assert basel_zone(n) == zone


def test_filtered_historical_var_rescales_to_current_volatility():
    # Metade da amostra com vol 1%, metade com 3%: padronizados, todos viram N(0,1).
    rng = np.random.default_rng(5)
    sigma = np.r_[np.full(100_000, 0.01), np.full(100_000, 0.03)]
    r = rng.standard_normal(200_000) * sigma
    fhs = filtered_historical_var(r, sigma, sigma_next=0.02, alpha=0.99)
    assert fhs == pytest.approx(parametric_var(0, 0.02, 0.99), rel=0.02)

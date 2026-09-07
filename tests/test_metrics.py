"""Pins metric definitions to docs/verify-models.md §C (fev/gluonts conventions)."""
import numpy as np
import pytest

from tbf import metrics as M


def test_seasonal_error_and_mase_m1():
    ctx = np.array([1, 2, 4, 7, 11], float)  # diffs 1,2,3,4 -> se=2.5
    assert M.seasonal_error(ctx, 1) == pytest.approx(2.5)
    y, yhat = np.array([10, 10]), np.array([12, 7])  # mae 2.5
    assert M.mase(y, yhat, ctx, 1) == pytest.approx(1.0)


def test_seasonal_error_m7_and_nan_cases():
    ctx = np.tile(np.arange(7, dtype=float), 3)  # perfectly periodic -> se=0 -> NaN
    assert np.isnan(M.seasonal_error(ctx, 7))
    assert np.isnan(M.mase([1.0], [1.0], ctx, 7))
    assert np.isnan(M.seasonal_error(np.arange(5.0), 7))  # too short


def test_quantile_loss_factor_two():
    y = np.array([1.0]); q = np.array([[0.0]]); levels = (0.9,)
    # y > q: 2*|(1-0)*(0-0.9)| = 1.8
    assert M.quantile_loss(y, q, levels)[0, 0] == pytest.approx(1.8)
    q = np.array([[2.0]])  # y <= q: 2*|(1-2)*(1-0.9)| = 0.2
    assert M.quantile_loss(y, q, levels)[0, 0] == pytest.approx(0.2)


def test_wql_matches_manual():
    y = np.array([2.0, 4.0]); q = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]); levels = (0.1, 0.5, 0.9)
    ql = M.quantile_loss(y, q, levels)
    expected = np.mean(ql.sum(axis=0) / 6.0)
    assert M.wql(y, q, levels) == pytest.approx(expected)


def test_crps_samples_fair_estimator_degenerate():
    y = np.array([1.0, 2.0]); s = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    assert M.crps_samples(y, s) == pytest.approx(0.0)
    s2 = np.array([[0.0, 0.0], [2.0, 4.0]])  # t1 = [1,2]; t2 = |0-2|,|2-0| sum=4 /(2*2*1)=1 -> [1,2]-[1,2]=0
    assert M.crps_samples(y, s2) == pytest.approx(0.0)


def test_seasonal_naive():
    assert list(M.seasonal_naive_forecast(np.arange(10.0), 3, 1)) == [9.0, 9.0, 9.0]
    assert list(M.seasonal_naive_forecast(np.arange(10.0), 9, 7)) == [3, 4, 5, 6, 7, 8, 9, 3, 4]


def test_geo_mean_and_bootstrap():
    assert M.geo_mean([1.0, 4.0]) == pytest.approx(2.0)
    stat, lo, hi = M.bootstrap_ci(np.full(50, 2.0))
    assert stat == pytest.approx(2.0) and lo == pytest.approx(2.0) and hi == pytest.approx(2.0)


def test_paired_bootstrap_sign():
    a = {f"s{i}": 0.8 for i in range(30)}; b = {f"s{i}": 1.0 for i in range(30)}
    d, lo, hi, n = M.paired_bootstrap(a, b)
    assert n == 30 and d == pytest.approx(np.log(0.8)) and hi < 0
    assert M.win_rate(a, b) == 1.0

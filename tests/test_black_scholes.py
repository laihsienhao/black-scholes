import numpy as np
import pytest

from src.models.black_scholes import BlackScholes, implied_vol

S, K, T, R, SIGMA = 100, 100, 1.0, 0.05, 0.2
H = 1e-4


def price(S, K, T, r, sigma, option_type):
    bs = BlackScholes(S, K, T, r, sigma)
    return bs.call_price() if option_type == "call" else bs.put_price()


def test_reference_prices():
    # Hull, Options, Futures and Other Derivatives: S=K=100, T=1, r=5%, σ=20%
    bs = BlackScholes(S, K, T, R, SIGMA)
    assert bs.call_price() == pytest.approx(10.4506, abs=1e-4)
    assert bs.put_price() == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("K", [60, 90, 100, 110, 150])
@pytest.mark.parametrize("T", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("sigma", [0.1, 0.4, 1.0])
def test_put_call_parity(K, T, sigma):
    bs = BlackScholes(S, K, T, R, sigma)
    assert bs.call_price() - bs.put_price() == pytest.approx(S - K * np.exp(-R * T), abs=1e-10)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_greeks_match_finite_differences(option_type):
    bs = BlackScholes(S, K, T, R, SIGMA)
    p = lambda **kw: price(**{"S": S, "K": K, "T": T, "r": R, "sigma": SIGMA,
                              "option_type": option_type, **kw})

    delta = (p(S=S + H) - p(S=S - H)) / (2 * H)
    gamma = (p(S=S + H) - 2 * p() + p(S=S - H)) / H**2
    theta = -(p(T=T + H) - p(T=T - H)) / (2 * H) / 365
    vega  = (p(sigma=SIGMA + H) - p(sigma=SIGMA - H)) / (2 * H) / 100
    rho   = (p(r=R + H) - p(r=R - H)) / (2 * H) / 100

    assert bs.delta(option_type) == pytest.approx(delta, rel=1e-6)
    assert bs.gamma() == pytest.approx(gamma, rel=1e-4)
    assert bs.theta(option_type) == pytest.approx(theta, rel=1e-6)
    assert bs.vega() == pytest.approx(vega, rel=1e-6)
    assert bs.rho(option_type) == pytest.approx(rho, rel=1e-6)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K", [80, 90, 100, 110, 120])
@pytest.mark.parametrize("T", [0.1, 0.5, 2.0])
@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.8])
def test_implied_vol_round_trip(option_type, K, T, sigma):
    market_price = price(S, K, T, R, sigma, option_type)
    intrinsic = max(S - K * np.exp(-R * T), 0) if option_type == "call" else max(K * np.exp(-R * T) - S, 0)
    if market_price - intrinsic < 0.01:
        # under a tick of time value, price is insensitive to σ, so IV is not identifiable
        pytest.skip("time value below one tick")
    assert implied_vol(market_price, S, K, T, R, option_type) == pytest.approx(sigma, abs=1e-4)


def test_implied_vol_rejects_price_below_intrinsic():
    # a call can't be worth less than S - K·e^{-rT}; no σ reproduces this price
    assert implied_vol(20.0, 130, 100, 1.0, R, "call") is None

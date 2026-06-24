import numpy as np
from scipy.stats import norm

def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


class BlackScholes:
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def _d1(self):
        return d1(self.S, self.K, self.T, self.r, self.sigma)

    def _d2(self):
        return d2(self.S, self.K, self.T, self.r, self.sigma)

    # --- Prices ---

    def call_price(self):
        return self.S * norm.cdf(self._d1()) - self.K * np.exp(-self.r * self.T) * norm.cdf(self._d2())

    def put_price(self):
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-self._d2()) - self.S * norm.cdf(-self._d1())

    # --- Greeks ---

    def delta(self, option_type="call"):
        if option_type == "call":
            return norm.cdf(self._d1())
        return norm.cdf(self._d1()) - 1

    def gamma(self):
        return norm.pdf(self._d1()) / (self.S * self.sigma * np.sqrt(self.T))

    def theta(self, option_type="call"):
        term1 = -(self.S * norm.pdf(self._d1()) * self.sigma) / (2 * np.sqrt(self.T))
        if option_type == "call":
            return (term1 - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self._d2())) / 365
        return (term1 + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self._d2())) / 365

    def vega(self):
        # per 1% move in vol
        return self.S * norm.pdf(self._d1()) * np.sqrt(self.T) / 100

    def rho(self, option_type="call"):
        # per 1% move in rate
        if option_type == "call":
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self._d2()) / 100
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self._d2()) / 100
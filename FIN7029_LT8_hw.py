# LT8_HW

import numpy as np
from math import exp, sqrt, log
from scipy.stats import norm


def BlackScholes (S, K, r, T, vol, q, cp):

    if cp.lower() == 'call':
        alpha = 1
    elif cp.lower() == 'put':
        alpha = -1
    else:
        raise ValueError('invalid cp flag')

    F = S * exp((r-q)*T)

    d1 = (log(F / K) + 0.5 * vol ** 2 * T) / (vol * sqrt(T))
    d2 = d1 - vol * sqrt(T)

    price = alpha * exp(-r*T) * (F*norm.cdf(d1*alpha) - K*norm.cdf(d2*alpha))

    return price


# test
S = 100
K = 105
r = 0.05
T = 0.25
vol = 0.3
q = 0.02
cp = 'call'

p = BlackScholes(S, K, r, T, vol, q, cp)
print(f'{p=:.4f}')


# test how vol influence option value
VOL = np.arange(0.1, 0.5, 0.01)
print(VOL)

import matplotlib.pyplot as plt

calls = [BlackScholes(S, K, r, T, vol, q, cp) for vol in VOL]
plt.plot(VOL, calls, label='Call Value')
plt.xlabel('$VOL$')
plt.ylabel(' Value')
plt.legend()
plt.show()

"""
Under Merton Model equity can be viewed as a call option on the assets of a firm.
Since the call option price increase with the volatility, equity holders should prefer increased risk and uncertainty,
since it increases the value of their investment.
"""


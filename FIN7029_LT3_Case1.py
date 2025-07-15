# -*- coding: utf-8 -*-
# LT3 Case1 BlackScholes

# Import packages
import numpy as np
from scipy.stats import norm

# standard normal cumulative distribution
N = norm.cdf

# BSM model
def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)*N(d2)

def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)

# Option inputs
S = 105
K = 100
r = 0.03
T = 1
sigma = 0.2

call_price = BS_CALL(S, K, T, r, sigma)
put_price = BS_PUT(S, K, T, r, sigma)

print("Call Option Price:", call_price)
print("Put Option Price:", put_price)


# test how S influence option value
S = np.arange(60, 140, 0.1)
# print(S)

# Use version matplotlib 3.9.3, do not use 3.10.0
import matplotlib.pyplot as plt

calls = [BS_CALL(s, K, T, r, sigma) for s in S]
puts = [BS_PUT(s, K, T, r, sigma) for s in S]
plt.plot(S, calls, label='Call Value')
plt.plot(S, puts, label='Put Value')
plt.xlabel('$S_0$')
plt.ylabel(' Value')
plt.legend()
plt.show()

# test how other inputs influence option value after class

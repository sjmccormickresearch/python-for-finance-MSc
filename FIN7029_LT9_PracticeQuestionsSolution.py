# -*- coding: utf-8 -*-
# FIN7029 Practice Questions


# Q1
# Q1 a
import numpy as np
from math import exp, sqrt, log
from scipy.stats import norm


def BlackScholes (S, K, r, T, vol, q, cp):

    # a flag variable of call and put options
    if cp.lower() == 'call':
        alpha = 1
    elif cp.lower() == 'put':
        alpha = -1
    else:
        raise ValueError('invalid cp flag')

    # applying BLM
    F = S * exp((r-q)*T)

    d1 = (log(F / K) + 0.5 * vol ** 2 * T) / (vol * sqrt(T))
    d2 = d1 - vol * sqrt(T)

    price = alpha * exp(-r*T) * (F*norm.cdf(d1*alpha) - K*norm.cdf(d2*alpha))

    return price


# Q1 b
# test for call option
S = 5
K = 5.2
r = 0.05
T = 0.25
vol = 0.3
q = 0.02
cp = 'call'

p = BlackScholes(S, K, r, T, vol, q, cp)
print(f'{p=:.4f}')

# test for put option
S = 5
K = 5.2
r = 0.05
T = 0.25
vol = 0.3
q = 0.02
cp = 'put'

p = BlackScholes(S, K, r, T, vol, q, cp)
print(f'{p=:.4f}')

# test for edge cases (no dividend)
S = 5
K = 5.2
r = 0.05
T = 0.25
vol = 0.3
q = 0
cp = 'call'

p = BlackScholes(S, K, r, T, vol, q, cp)
print(f'{p=:.4f}')

# test for invalid input
S = 5
K = 5.2
r = 0.05
T = 0.25
vol = 0.3
q = 0.02
cp = 'Python'

p = BlackScholes(S, K, r, T, vol, q, cp)
print(f'{p=:.4f}')

# Q1 c
# test how strike price influence option value
K = np.arange(3, 8, 0.1)

S = 5
r = 0.05
T = 0.25
vol = 0.3
q = 0.02
cp = 'put'

import matplotlib.pyplot as plt

# Draw the plot
puts = [BlackScholes(S, k, r, T, vol, q, cp) for k in K]
plt.plot(K, puts, label='Put Value')
plt.xlabel('$K$')
plt.ylabel(' Value')
plt.legend()
plt.show()

# Q1 d
# Simulates a discrete random variable
def simulate_discrete_rv(values, probabilities, n=1):

    cumulative_probs = np.cumsum(probabilities)
    uniform_samples = np.random.uniform(0, 1, n)
    simulated_values = [values[np.searchsorted(cumulative_probs, u)] for u in uniform_samples]

    return simulated_values


# Use the function on given information
values = [0.02, 0.03, 0.04]
probabilities = [0.3, 0.4, 0.3]
n_samples = 1000  # Number of samples to generate

simulated_samples = simulate_discrete_rv(values, probabilities, n_samples)
print("Simulated Samples:", simulated_samples)

# Q1 e
E_r = sum(simulated_samples) / n_samples
S = 10
K = 11
T = 2
vol = 0.3
q = 0.03
cp = 'call'

p = BlackScholes(S, K, E_r, T, vol, q, cp)
print(f'{p=:.4f}')

#######################################################
# Q2

import numpy as np
from math import exp, log
from scipy.stats import norm

np.set_printoptions(precision=4, suppress=True)

# %% Merton model
from scipy import optimize


# Write a function that calculates the error to minimise
def mertonerror(guess):

    # Tell function that these values are defined outside the function
    global T, r, D, volE, E

    # Inputs are passed in as an array and must be unpacked
    A, volA = guess

    d1 = (log(A * exp(r * T) / D) + 0.5 * volA ** 2 * T) / (volA * T ** 0.5)
    d2 = d1 - volA * T ** 0.5

    Eguess = A * norm.cdf(d1) - D * exp(-r * T) * norm.cdf(d2)
    Evolguess = (volA * A * norm.cdf(d1)) / Eguess

    return (Eguess - E) ** 2 + (Evolguess - volE) ** 2


# apply the values given
r = 0.03
T = 1
D = 10
volE = 0.5
E = 4


# Initial guess: asset value and vol are the equity value and vol
guess = (E, volE)

# Call optimiser to find inputs which minimise error function
# Finds the values of A and vol A that minimize mertonerror()
res = optimize.minimize(mertonerror, guess)

A, volA = res.x
print(f"{A=:.2f}, {volA=:0.2%}")

# Calculate default probability using Merton model
d1 = (log(A * exp(r * T) / D) + 0.5 * volA ** 2 * T) / (volA * T ** 0.5)
d2 = d1 - volA * T ** 0.5

DP = norm.cdf(-d2)
print(f"Prob of default: {DP:.4f}")

#############################################################
# Q3
# Q3 a
import numpy as np

# Builds a binomial tree for stock prices.
def build_binomial_tree(S, u, d, N):
    tree = np.zeros((N + 1, N + 1))
    tree[0, 0] = S  # Initial price

    for i in range(1, N + 1):
        tree[i, i] = tree[i - 1, i - 1] * u  # Upward move
        for j in range(i):
            tree[i, j] = tree[i - 1, j] * d  # Downward move

    return tree

# Q3 b
# Builds a binomial tree considering a continuous dividend yield.
def build_binomial_tree_with_dividend(S, vol, r, q, T, N):
    dt = T / N
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u

    # Calculate new probablity when there is a dividned yield q
    p = (np.exp((r - q) * dt) - d) / (u - d)

    tree = np.zeros((N + 1, N + 1))
    tree[0, 0] = S

    for i in range(1, N + 1):
        tree[i, i] = tree[i - 1, i - 1] * u
        for j in range(i):
            tree[i, j] = tree[i - 1, j] * d

    return tree

# Q3 c
# Computes the payoff of a European put option at maturity.
def put_option_payoff(tree, K, N):
    payoff = np.zeros((N + 1, N + 1))

    for j in range(N + 1):
        payoff[N, j] = max(K - tree[N, j], 0)

    return payoff


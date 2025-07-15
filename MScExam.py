from turtledemo.forest import doit1

import numpy as np
from math import exp,log,sqrt
from scipy.stats import norm
from scipy import optimize
import matplotlib.pyplot as plt
np.set_printoptions(precision = 4, suppress = True)

# Q1a BlackScholes Model
def blackscholes(S,K,r,T,vol,q,cp):
    if cp.lower() == 'call':
        alpha = 1
    elif cp.lower() == 'put':
        alpha = -1
    else:
        raise ValueError('invalid cp flag')

    F = S * exp((r-q)*T)

    d1 = (log(F/K) + 0.5 * vol ** 2 * T) / (vol * sqrt(T))
    d2 = d1 - (vol * sqrt(T))

    price = alpha * exp(-r*T) * (F * norm.cdf(d1*alpha) - K * norm.cdf(d2*alpha))

    return price

# Q1b - Test plan
# Test under normal conditions
S = 100
K = 105
r = 0.05
T = 0.25
vol = 0.3
q = 0.02
cp = 'call'

p = blackscholes(S,K,r,T,vol,q,cp)
print(f'Option price: {p:.4f}')

# Test for a put option
S = 100
K = 105
r = 0.05
T = 0.25
vol = 0.3
q = 0.02
cp = 'put'

p = blackscholes(S,K,r,T,vol,q,cp)
print(f'Test price for put: {p:.4f}')

# Test for outliers, e.g. dividend = 0
S = 100
K = 105
r = 0.05
T = 0.25
vol = 0.3
q = 0
cp = 'call'

p = blackscholes(S,K,r,T,vol,q,cp)
print(f'Test price with outlier: {p:.4f}\n')

# Test for invalid/unexpected cp flag (commented out as to not crash code)
# S = 100
# K = 105
# r = 0.05
# T = 0.25
# vol = 0.3
# q = 0.02
# cp = 'python'
#
# p = blackscholes(S,K,r,T,vol,q,cp)
# print(p)


# Q1 c - Testing how risk influences the put option value
S = 100
K = 105
r = 0.05
T = 0.25
vol = 0.3
q = 0.02
cp = 'put'

VOL = np.arange(0.1,1,0.1)
puts = [blackscholes(S, K, r, T, vol, q, cp) for vol in VOL]
plt.plot(VOL, puts, label = 'Volatility vs Put Option Value')
plt.xlabel('VOL')
plt.ylabel('Put Value')
plt.legend()
plt.show()


# Q2a & b Binomial Tree and variable calculations
def build_binomial_tree_auto(S,vol,T,N):
    dt = T/N
    u = np.exp(vol*np.sqrt(dt))
    d = 1/u
    p = (np.exp((r-q)*dt)-d) / (u-d)

    tree = np.zeros((N + 1, N + 1))
    tree[0, 0] = S

    for i in range(1, N + 1):
        tree[i, i] = tree[i - 1, i - 1] * u
        for j in range(i):
            tree[i, j] = tree[i - 1, j] * d

    return tree


# Q2c - Payoff at Terminal Nodes (CALL: cp = 1, PUT: cp = -1
def option_payoff(tree,K,N,cp=-1):
    payoff = np.zeros((N+1,N+1))
    for j in range(N+1):
        payoff[N,j] = max(cp * (tree[N,j]-K),0)
    return payoff


# Q2d - Backward Induction, including call/put, European and American Options
def backward_induction(tree,payoff,K,r,q,T,N,cp=-1,american=False):
    dt = T / N
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r-q)*dt)-d) / (u-d)
    discount = np.exp(-r*dt)

    for i in range(N-1,-1,-1):
        for j in range(i+1):
            continuation_value = discount * (p * payoff[i+1, j+1] + (1-p) * payoff[i+1,j])
            if american:
                immediate_exercise_value = max(cp * (tree[i,j]-K),0)
                payoff[i,j] = max(continuation_value, immediate_exercise_value)
            else:
                payoff[i,j] = continuation_value
    return payoff[0,0]

# Q2e
S = 10
K = 9
r = 0.05
T = 0.5
vol = 0.25
q = 0.02
N = 100

# European Option
tree = build_binomial_tree_auto(S,vol,T,N)
payoff = option_payoff(tree,K,N,cp=-1)
option_price_euro = backward_induction(tree,payoff,K,r,q,T,N,cp=-1,american=False)
print(f'European Option Price, {option_price_euro:.4f}')

# American Option
tree = build_binomial_tree_auto(S,vol,T,N)
payoff = option_payoff(tree,K,N,cp=-1)
option_price_american = backward_induction(tree,payoff,K,r,q,T,N,cp=-1,american=True)
print(f'American Option Price, {option_price_american:.4f}\n')



# Q3a - Stochastic Process
def simulate_discrete_rv(values, probabilities, n=1):
    cumulative_probs = np.cumsum(probabilities)
    uniform_samples = np.random.uniform(0,1,n)
    simulated_values = [values[np.searchsorted(cumulative_probs,u)] for u in uniform_samples]

    return simulated_values

values = [0.03,0.04,0.05]
probabilities = [0.2,0.5,0.3]
n_samples = 1000

simulated_samples = simulate_discrete_rv(values,probabilities,n_samples)
print(f'Simulated Samples: {simulated_samples}')

# Test simulated r in option pricing
E_r = sum(simulated_samples)/n_samples
S = 100
K = 105
T = 0.25
vol = 0.3
q = 0.02
cp = 'call'

p = blackscholes(S,K,E_r,T,vol,q,cp)
print(f'Test option pricing using stochastic r: {p:.4f}\n')


# Q3b - Merton Model
def mertonerror(guess):
    global T, r, D, volE, E
    A, volA = guess

    d1 = (log(A * exp(r*T) / D) + 0.5 * volA ** 2 ** T) / (volA * sqrt(T))
    d2 = d1 - (volA * sqrt(T))

    Eguess = A * norm.cdf(d1) - D * exp(-r*T) * norm.cdf(d2)
    Evolguess = (A * volA * norm.cdf(d1)) / Eguess

    return (Eguess - E) ** 2 + (Evolguess - volE) ** 2

T = 1
r = 0.02
D = 20
volE = 0.4
E = 7

guess = (E, volE)

res = optimize.minimize(mertonerror, guess)
A, volA = res.x
print(f'{A=:.4f}, {volA=:0.2%}')

d1 = (log(A * exp(r * T) / D) + 0.5 * volA ** 2 ** T) / (volA * sqrt(T))
d2 = d1 - (volA * sqrt(T))
DTD = d2
PD = norm.cdf(-d2)
print(f'Probability of Default: {PD:0.2%}')

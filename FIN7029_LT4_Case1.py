# LT4_Case1

import numpy as np
import matplotlib.pyplot as plt


# %% Generate a random walk

def simplewalk(N):
    """Generate a path simulating N +/-1 coin tosses"""

    # generate random numbers with values 0 or 1
    # use a simple formula to transform into -1 and +1
    CT = 1 - 2 * np.random.randint(2, size=N)

    # Add a zero at the start to represent the starting point
    CT = np.concatenate(([0], CT))

    # Return x/y arrays (time/position)
    return np.arange(N + 1), CT.cumsum()


# %%
# Generate on random walk over 10 steps
N = 10
T, X = simplewalk(N)

# visualise the random walk
plt.plot(T, X, '-*r')
plt.grid(True)
plt.show()

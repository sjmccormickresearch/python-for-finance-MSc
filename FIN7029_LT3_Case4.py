# LT3 Case 4
# Find Interpolation

def interpnearest (x, y, z):
    """Interpolates using the neatest point. Mid points are linearly interpolated"""

    # for z smaller or larger than all Xs
    if z <= x[0]:
        return y[0]
    elif z >= x[-1]:
        return y[-1]

    # find sub interval
    else:
        i = len([_ for _ in x[1:] if _ < z])
        diff = (x[i+1]-z) - (z - x[i])

        if diff > 0:
            return y[i]
        elif diff < 0:
            return y[i+1]
        else:
            return (y[i] + y[i+1]) / 2


#tests

import numpy as np
import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y = [1,3,2,4,3]

z = np.linspace(x[0]-1, x[-1]+1, (x[-1]-x[0]+2)*10+1)

print(z)

fz = [interpnearest(x, y, _) for _ in z]

plt.plot(x, y, "r", markersize=10)
plt.plot(z, fz, "b.", markersize=4)
plt.show()




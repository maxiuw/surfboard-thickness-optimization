import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sympy as sp
import scipy.integrate as integ
import math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sympy as sp
import scipy.integrate as integrate
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import os, sys
from glob import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
from matplotlib.widgets import Cursor

# variables


it = 1
p = []
for it in range(0, 1800):
    p.append(it)

length = np.array(p)


# obj funtion
def f(x):
    l = 1800
    Ep = 45000
    Gc = 20
    F = 1080
    ep = x[0]
    ec = x[1]
    Ei = (Ep * ep * l * (ec + ep) ** 2) / 2
    Gi = 1 / (Gc * l * (ec + 2 * ep))
    return (((F ** 2 * (l)) / (Ep * ec * l * (ec + ep) ** 2)) + (F ** 2) / (2 * (ec + 2 * ec) * Gc * l)) / 1000


# constrain 1 - mass

def constrain1(x):
    pc = 4.5 * 10 ** (-5)
    pp = 2.1 * 10 ** (-3)
    V = 3.1 * 10 ** 7
    ep = x[0]
    ec = x[1]
    return 4.5-((((pp * V * ((2 * ep + ec) / 50) * ep) / (ec + 2 * ep)) + (
            (pc * V * ((2 * ep + ec) / 50) * ec) / (ec + 2 * ep))) / 1000)


# constr 2

def constrain2(x):
    ep = x[0]
    ec = x[1]
    return (3.13107 * 10 ** 3) * ep * np.sqrt(ep / ec) - 1080


i1 = 10
i2 = 30
ep_sol = []
ec_sol = []
W_sol = []

for i1 in range(10, 30):
    for i2 in range(30, 60):
        # initial guesses
        x_init = []
        x_init.append(i1/10)
        x_init.append(i2)

        # bounds

        b1 = (0, 3)
        b2 = (30, 60)
        bnds = (b1, b2)
        con1 = {'type': 'ineq', 'fun': constrain1}
        con2 = {'type': 'ineq', 'fun': constrain2}
        cons = [con1, con2]
        sol = minimize(f, x0=x_init, method='SLSQP', bounds=bnds, constraints=cons,
                       options={'ftol': 1e-8, 'maxiter': 1e9, 'disp': True})
        ep_sol.append(sol.x[0])
        ec_sol.append(sol.x[1])

        # print(sol)
        # print(sol.x[1])

        # ---------------------------------------
        # integrating
        # ---------------------------------------

        Gc1 = 1
        # F = int(input("Force (1080N for 80kg person) >> "))
        ec1 = sol.x[0]
        ep1 = sol.x[1]
        Ep1 = 45000
        F1 = 1080
        l1 = 1800


        def k(x1):
            return (((F1 ** 2 * (l1 - x1)) / (Ep1 * ec1 * l1 * (ec1 + ep1) ** 2)) + (F1 ** 2) / (
                        2 * (ec1 + 2 * ec1) * Gc1 * l1)) / 1000


        y3 = k(length)
        I1 = integrate.simps(y3, length)
        W_sol.append(I1)

    # -------------------------------------------
    # ploting solutions
    # -------------------------------------------
    index_min = W_sol.index(min(W_sol))
    print(index_min)
    del (ep_sol[index_min])
    del (ec_sol[index_min])
    del (W_sol[index_min])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('For ep inint=' + str(i1/10))
    ax.scatter(ep_sol, ec_sol, W_sol, c='r', marker='o', s=5)
    '''ax.scatter(ep_sol[index_min], ec_sol[index_min], W_sol[index_min], c='b', marker='o', s=20)
    minimum = "Mnimum values  \n Ply thickness={:03.2f} mm " \
              "\n Core thickness={:03.2f} mm\n Value of W={:03.2f} MPa".format(ep_sol[index_min], ec_sol[index_min],
                                                                               W_sol[index_min])
    ax.text(ep_sol[index_min], ec_sol[index_min], W_sol[index_min],
            minimum, color='blue')
            '''
    ax.set_xlabel("ep [mm]")
    ax.set_ylabel("ec [mm]")
    ax.set_zlabel("W [MPa]")

index_min_global = W_sol.index(min(W_sol))-1
minimum_global=[ep_sol[index_min_global],ec_sol[index_min_global],W_sol[index_min_global]]
print(minimum_global)
print(W_sol)
print(ec_sol)
print(ep_sol)
plt.show()

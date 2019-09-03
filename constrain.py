'''
constrains
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import os, sys
from glob import glob
import matplotlib.image as mpimg

epmin = 1# int(input("Minimum val of ep >> "))
epmax = 4#int(input("Max val of ep >> "))
ecmin = 30#int(input("Minimum val of ec >> "))
ecmax = 70#int(input("Max val of ec >> "))

'''
Mass
'''

fig = plt.figure()
ax = fig.add_subplot(2, 3, 1, projection='3d')  # fig.gca(projection='3d')

# data
ep = np.arange(0.001, epmax)
ec = np.arange(ecmin, ecmax)
ec, ep = np.meshgrid(ec, ep)
pc = 4.5 * 10 ** (-5)
pp = 2.1 * 10 ** (-3)
V = 3.1 * 10 ** 7
Ec =30
Ep=45000
l=1800
# mass
M = (((pp * V*((2*ep+ec)/50) * ep) / (ec + 2*ep)) + ((pc * V *((2*ep+ec)/50)* ec) / (ec + 2*ep))) / 1000
F=1080
Gc=2
W=(((F ** 2 * (l)) / (Ep * ec * l * (ec + ep) ** 2)) + (F ** 2) / (2 * (ec + 2 * ec) * Gc * l)) / 1000

# Plot the surface.
surf = ax.plot_surface(ec, ep, M, cmap='hsv', linewidth=0, antialiased=False)

# axis


ax.set_title('Surfboard Mass')
ax.set_xlabel('ec [mm]')
ax.set_ylabel('ep [mm]')
ax.set_zlabel('M[kg]');
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

'''
second subplot - buckling force 
'''

ax = fig.add_subplot(2, 2, 2, projection='3d')
# equation

Z = (1.64 * Ep*l*ep * np.sqrt((ep*Ec) /(Ep*ec)))/1000


# plot

surf = ax.plot_surface(ec, ep, Z, cmap='hsv', linewidth=0, antialiased=False)
'''
3rd plot - W
'''


# axis
ax.set_title('Buckling strength')

# ax.set_zlim3d(0,10)

ax.set_xlabel('ec [mm]')
ax.set_ylabel('ep [mm]')
ax.set_zlabel('F [N]');

fig.colorbar(surf, shrink=0.5, aspect=5)

# figures
# Mass pic
#ax = fig.add_subplot(2, 2, 3)
#img1 = mpimg.imread('mass.png')
#plt.imshow(img1)
# bulk
ax = fig.add_subplot(2, 2, 4)
img2 = mpimg.imread('bulk.png')
plt.imshow(img2)

plt.show()
ec1=int(input("ec: "))
ep1=float(input("ep1: "))

def mass(ec,ep):
    return (((pp * V * ((2 * ep + ec) / 50) * ep) / (ec + 2 * ep)) + ((pc * V * ((2 * ep + ec) / 50) * ec) / (ec + 2 * ep))) / 1000
def bulk(ec,ep):
    return (3.13107 * 10 ** 3) * ep * np.sqrt(ep / ec)

print(mass(ec1,ep1))
print(bulk(ec1,ep1))
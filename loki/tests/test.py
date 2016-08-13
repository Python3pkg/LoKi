
# -*- coding: utf-8 -*-

"""
Testing module for LoKi
"""

# Import needed packages and functions
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from loki import loki
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


########################################################################



'''
Multiple tests
'''

print('Starting Test')

# Choose some coordinates
ra, dec = 10, 40 

# Create an array of densities that span the entire luminosity function
# This creates 10000 random densities by default
densities = loki.densities()

# Pick a random density from the array
density = np.random.choice(densities, size = 1)

# Count the number of stars along the line-of-sight (defaults to an angular volume of 30')
# In "full" mode this also returns the distance bins and stellar counts for each bin
n, nums, dists = loki.count_nstars(ra, dec, rho0=density, full = True)

# Get parameters for all the stars
stars = loki.stars(ra, dec, n, nums, dists)

plt.figure(1)
plt.quiver(stars.ra, stars.dec, stars.pmra, stars.pmdec)
plt.xlabel('R.A. (deg)')
plt.ylabel('Dec. (deg)')

fig = plt.figure(2)
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(stars.X, stars.Y, stars.Z)
ax.set_xlabel('X (pc)')
ax.set_ylabel('Y (pc)')
ax.set_zlabel('Z (pc)')

plt.show()
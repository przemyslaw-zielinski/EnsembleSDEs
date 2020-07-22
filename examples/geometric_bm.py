#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 21 Jul 2020

@author: Przemyslaw Zielinski
"""
# set cwd to supfolder for local import
import sys, os
cwd = os.getcwd()
sys.path.append(cwd + '/..')

import matplotlib.pyplot as plt
import numpy as np
import spaths

file_name = "geometric_bm"

A = np.array(
    [[-1, 0],
     [ 0, -1]]
)
# vol1, vol2 = np.sqrt(2), np.sqrt(3)
# corr = 0.6
# B = np.array(
#     [[vol1, 0],
#      [vol2*corr, vol2*np.sqrt(1-corr**2)]]
# )
B = np.full((2, 2), np.sqrt(2))

def drift(t, x):
    return A @ x

def dispersion(t, x):
    return B @ x

# seed setting
seed = 3579
rng = np.random.default_rng(seed)
rng.integers(10**2, size=10**2)  # warm up of RNG

# solver
em = spaths.EulerMaruyama(rng)

# simulation parameters
dt = .005
nsam = 2
tspan = (0.0, 3.0)

# initial conditions
x0, y0 = [0.5]*nsam, [0.5]*nsam

gbm = spaths.ItoSDE(drift, dispersion)
ens0 = spaths.make_ens(x0, y0)
sol = em.solve(gbm, ens0, tspan, dt)

fig, ax = plt.subplots(figsize=(8,6))
ls = 16
lw = 2

ax.plot(sol.t, sol.x[:,0,1])
ax.plot(sol.t, sol.x[:,0,0])

plt.show()

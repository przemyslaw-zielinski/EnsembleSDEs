#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 6 Feb 2020

@author: Przemyslaw Zielinski
"""

import matplotlib.pyplot as plt
import numpy as np
import sys, os

cwd = os.getcwd()
sys.path.append(cwd + '/..')
import spaths


# seed setting
seed = 3579
rng = np.random.default_rng(seed)
rng.integers(10**3, size=10**3)  # warm up of RNG

# simulation parameters
dt = 1e-2
nsam = 10
tspan = (0.0, 10.0)

eps = 1 / 50

A = np.array(
    [[-1, 1],
     [ 0, -1 /eps]]
)

# initial conditions
x0, y0 = [10.0]*nsam, [10.0]*nsam

def drift(t, u, du):
    du[0] = A[0,0]*u[0] + A[0,1]*u[1]
    du[1] = A[1,0]*u[0] + A[1,1]*u[1]

def dispersion(t, u, du):
    du[0] = 1.0
    du[1] = np.sqrt(1 / eps)

sde = spaths.SDE(drift, dispersion)
ens0 = spaths.make_ens(x0, y0)
sol = spaths.EMSolver(sde, ens0, tspan, dt, rng)

fig, ax = plt.subplots(figsize=(8,6))
ls = 16
lw = 2

ax.plot(sol.t, sol.x[:,3,1])
ax.plot(sol.t, sol.x[:,3,0])

ax.tick_params(
        axis='both',        # changes apply to
        which='major',       # both major and minor ticks are affected
        bottom=True,       # ticks along the bottom edge are on/off
        left=True,
        labelleft=True,
        labelsize=ls)

fig.tight_layout()
fig.savefig(f"figs/ou_process.pdf")
plt.close()

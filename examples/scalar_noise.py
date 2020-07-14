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
seed = 357
rng = np.random.default_rng(seed)
rng.integers(10**3, size=10**3)  # warm up of RNG

# solver
em = spaths.EulerMaruyama(rng)

# simulation parameters
dt = 1e-2
nsam = 10
tspan = (0.0, 1.0)

# initial conditions
x0, y0 = [1.0]*nsam, [2.0]*nsam

def drift(t, u, du):
    du = u

def dispersion(t, u, du):
    du[0, 0] = u[0]
    du[1, 0] = u[1]

sde = spaths.ItoSDE(drift, dispersion, 1)
ens0 = spaths.make_ens(x0, y0)
sol = em.solve(sde, ens0, tspan, dt)

fig, ax = plt.subplots(figsize=(8,6))
ls = 16
lw = 2

tplot = sol.t
splot = sol(tplot)
ax.plot(sol.t, sol.x[:,3,1])
ax.plot(sol.t, sol.x[:,3,0])

fig.tight_layout()
fig.savefig(f"figs/scalar_noise.pdf")
plt.close()

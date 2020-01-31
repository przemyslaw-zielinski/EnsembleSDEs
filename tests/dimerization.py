#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 2020

@author: Przemyslaw Zielinski
"""

import matplotlib.pyplot as plt
import numpy as np
import sys, os
import math

cwd = os.getcwd()
sys.path.append(cwd + '/..')
from src import models
from src import solvers


# seed setting
seed = 357
rng = np.random.default_rng(seed)
rng.integers(10**3, size=10**3)  # warm up of RNG

# simulation parameters
nsam = 100
initime = 0.0
timestep = 10**(-4)
endtime = 10.0

# initial conditions
x0, y0 = [100.0]*nsam, [100.0]*nsam

k1 = 1.0
k2 = 100.0
k3 = 50.0

def drift(t, u, du):
    du[0] = 2*k2*u[1] - 2*k1*u[0]*(u[0]-1) + k3
    du[1] = k1*u[0]*(u[0]-1) - k2*u[1]

def dispersion(t, u, du):
    du[0,0] = -2*np.sqrt(k1*u[0]*(u[0]-1))
    du[0,1] = 2*np.sqrt(k2*u[1])
    du[0,2] = np.sqrt(k3)
    du[1,0] = np.sqrt(k1*u[0]*(u[0]-1))
    du[1,1] = -np.sqrt(k2*u[1])
    du[1,2] = 0

sde = models.SDE(drift, dispersion, (2, 3))
ens0 = models.ensnd(x0, y0)

nsteps = math.ceil((endtime + timestep) / timestep)
tgrid, solution = solvers.sim(initime, ens0, sde, timestep, nsteps, rng)

fig, ax = plt.subplots(figsize=(8,6))
ls = 16
lw = 2

idx = range(0, len(tgrid), 10)
ax.plot(tgrid[idx], solution[idx,10,1], color="C1", alpha=.5)
ax.plot(tgrid[idx], solution[idx,10,0], color="C0", alpha=.5)

# slow_avg = np.average(solution[idx,:,0], axis=1)
# slow_std = np.std(solution[idx,:,0], axis=1)
# ax.plot(tgrid[idx], slow_avg, color="C0", linewidth=lw)
# ax.fill_between(tgrid[idx], slow_avg - slow_std, slow_avg + slow_std,
#                 facecolor="C0", alpha=.2)

ax.tick_params(
        axis='both',        # changes apply to
        which='major',       # both major and minor ticks are affected
        bottom=True,       # ticks along the bottom edge are on/off
        left=True,
        labelleft=True,
        labelsize=ls)

fig.tight_layout()
fig.savefig(f"figs/dimerization.pdf")
plt.close()
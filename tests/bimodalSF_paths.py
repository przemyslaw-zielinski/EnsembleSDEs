#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 16:06:25 2018

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

file_name = "bimodalSF_paths"

# seed setting
seed = 357
rng = np.random.default_rng(seed)
rng.integers(10**3, size=10**3)  # warm up of RNG

# model parameters
xtemp = .005
ytemp = .07
eps = .1

# simulation parameters
nsam = 100
initime = 0.0
timestep = 1.5e-2
endtime = 100.0

# initial conditions
x0, y0 = [.5]*nsam, [-1]*nsam

def drift(t, x, dx):
    dx[0] = -(2*x[0] + x[1])
    dx[1] = (x[1] - x[1]**3)/eps

def dispersion(t, x, dx):
    dx[0] = np.sqrt(2*xtemp)
    dx[1] = np.sqrt(2*ytemp/eps)

sde = models.SDE(drift, dispersion, noise_rate=(2,2))
iniens = models.ensnd(x0, y0)

nsteps = math.ceil((endtime + timestep) / timestep)
tgrid, solution = solvers.sim(initime, iniens, sde, timestep, nsteps, rng)

fig, ax = plt.subplots(figsize=(8,6))
ls = 16
lw = 2

ax.plot(tgrid, solution[:,0,1], color="C1", alpha=.5)
ax.plot(tgrid, solution[:,0,0], color="C0", alpha=.5)

slow_avg = np.average(solution[:,:,0], axis=1)
slow_std = np.std(solution[:,:,0], axis=1)
ax.plot(tgrid, slow_avg, color="C0", linewidth=lw)
ax.fill_between(tgrid, slow_avg - slow_std, slow_avg + slow_std,
                facecolor="C0", alpha=.2)

ax.tick_params(
        axis='both',        # changes apply to
        which='major',       # both major and minor ticks are affected
        bottom=True,       # ticks along the bottom edge are on/off
        left=False,
        labelleft=False,
        labelsize=ls)

fig.tight_layout()
fig.savefig(f"figs/{file_name}.pdf")
plt.close()

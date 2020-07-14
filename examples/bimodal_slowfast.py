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
import spaths

file_name = "bimodal_slowfast"

# seed setting
seed = 3579
rng = np.random.default_rng(seed)
rng.integers(10**3, size=10**3)  # warm up of RNG

# model parameters
xtemp = .005
ytemp = .07
eps = .1

# simulation parameters
dt = 1.5e-2
nsam = 100
tspan = (0.0, 100.0)

# initial conditions
x0, y0 = [.5]*nsam, [-1]*nsam

def drift(t, x, dx):
    dx[0] = -(2*x[0] + x[1])
    dx[1] = (x[1] - x[1]**3)/eps

def dispersion(t, x, dx):
    dx[0] = np.sqrt(2*xtemp)
    dx[1] = np.sqrt(2*ytemp/eps)

sde = spaths.ItoSDE(drift, dispersion)
ens0 = spaths.make_ens(x0, y0).astype(dtype=np.float32)

print(f"{ens0.dtype = }")
em = spaths.EulerMaruyama(rng)
ens_test = em.burst(sde, ens0, (0.0, 3), dt)
print(f"{ens_test.dtype = }")

sol = em.solve(sde, ens0, tspan, dt)
print(sol)

fig, ax = plt.subplots(figsize=(8,6))
ls = 16
lw = 2

print(sol(0).shape)

tplot = sol.t[::4]
sam_idx = rng.integers(nsam)
sam_path = sol.p[sam_idx][::4]
ax.plot(tplot, sam_path, alpha=.5)

splot = sol(tplot)
slow_avg = np.average(splot[:,:,0], axis=1)
slow_std = np.std(splot[:,:,0], axis=1)
ax.plot(tplot, slow_avg, color="C0", linewidth=lw)
ax.fill_between(tplot, slow_avg - slow_std, slow_avg + slow_std,
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

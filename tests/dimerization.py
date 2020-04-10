#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 2020

@author: Przemyslaw Zielinski
"""

from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import math

cwd = os.getcwd()
sys.path.append(cwd + '/..')
import spaths
from spaths.reactions import intermediate, Reaction

# seed setting
seed = 3579
rng = np.random.default_rng(seed)
rng.integers(10**3, size=10**3)  # warm up of RNG

# simulation parameters
dt = 1e-4
nsam = 10
tspan = (0.0, 10.0)

# initial conditions
x0, y0 = [100.0]*nsam, [100.0]*nsam

Y  = intermediate(1)
X  = intermediate(0)
X2 = intermediate(0, 2)

c1 = 1.0
c2 = 100.0
c3 = 50.0

dimerization = Reaction(c1, [X2], [Y])
dissociation = Reaction(c2, [Y], [X2])
production   = Reaction(c3, [], [X])

def drift(t, u, du):
    du[0] = 2*c2*u[1] - 2*c1*u[0]*(u[0]-1) + c3
    du[1] = c1*u[0]*(u[0]-1) - c2*u[1]

def dispersion(t, u, du):
    du[0,0] = -2*np.sqrt(c1*u[0]*(u[0]-1))
    du[0,1] = 2*np.sqrt(c2*u[1])
    du[0,2] = np.sqrt(c3)
    du[1,0] = np.sqrt(c1*u[0]*(u[0]-1))
    du[1,1] = -np.sqrt(c2*u[1])
    du[1,2] = 0

sde = spaths.ItoSDE(drift, dispersion, 3)
cle = spaths.ChemicalLangevin(2, [dimerization, dissociation, production])
ens0 = spaths.make_ens(x0, y0)

start = timer()
sol_sde = spaths.EMSolver(sde, ens0, tspan, dt, rng)
end = timer()
print("sde sim:", end - start)

start = timer()
sol_cle = spaths.EMSolver(cle, ens0, tspan, dt, rng)
end = timer()
print("cle sim:", end - start)

fig, ax = plt.subplots(figsize=(8,6))
ls = 16
lw = 2

tplot = sol_cle.t[::50]
splot = sol_cle(tplot)
ax.plot(tplot, splot[:,2,1], color="C1", alpha=.5)
ax.plot(tplot, splot[:,2,0], color="C0", alpha=.5)

slow_avg = np.average(splot[:,:,0] + 2*splot[:,:,1], axis=1)
ax.axhline(y=c3, color='k', linestyle='--')
ax.plot(tplot[1:], (slow_avg[1:] - 300) / tplot[1:])
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

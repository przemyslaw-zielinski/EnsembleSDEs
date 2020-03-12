"""
Created on Mon 9 Mar 2020

@author: Przemyslaw Zielinski
"""

# set cwd to supfolder for local import
import sys, os
cwd = os.getcwd()
sys.path.append(cwd + '/..')

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import spaths

file_name = "triatomic_molecule"

# model parameters
sep = 10**(-3)
equ_len = 1.
k = 208.
sad_ang = np.pi / 2
del_ang = sad_ang - 1.187
inv_temp = 1.

def V(t, x):  # need to use jnp (instead of np) for autodiff
    xa, xc, yc = x
    dev_a = xa - equ_len
    dev_c = jnp.sqrt(xc**2 + yc**2) - equ_len
    dev_ang = np.pi / 2 - jnp.arctan(xc / yc) - sad_ang
    return .5 * ((dev_a**2 + dev_c**2) / sep + k * (dev_ang**2 - del_ang**2)**2)

# simulation parameters
dt = 1.0 * sep
nsam = 1
tspan = (0.0, 6.0)

# seed setting
seed = 3579
rng = np.random.default_rng(seed)
rng.integers(10**3, size=10**3)  # warm up of RNG

xa0 = np.array([equ_len] * nsam)
xc0 = np.array([0.] * nsam)
yc0 = np.array([equ_len] * nsam)

sde = spaths.OverdampedLangevin(V, inv_temp)
ens0 = spaths.make_ens(xa0, xc0, yc0)
sol = spaths.EMSolver(sde, ens0, tspan, dt, rng)

fig, ax = plt.subplots(figsize=(8,6))
ls = 16
lw = 2

path = sol.p[0]
path_xa, path_xc, path_yc = path.T
path_ang = np.pi / 2 - np.arctan(path_xc / path_yc)

# ax.plot(sol.t[::5], path[::5])  # to plot all coords at once
ax.plot(sol.t[::5], path_xa[::5], label="xa", alpha=.6)
ax.plot(sol.t[::5], path_xc[::5], label="xc", alpha=.6)
ax.plot(sol.t[::5], path_yc[::5], label="yc", alpha=.6)
ax.plot(sol.t[::5], path_ang[::5], label="$\Theta$")

ax.legend()
ax.set_xlabel("time")

fig.tight_layout()
fig.savefig(f"figs/{file_name}.pdf")
plt.close()

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

# solver
em = spaths.EulerMaruyama(rng)

xa0 = np.array([equ_len] * nsam)
xc0 = np.array([0.] * nsam)
yc0 = np.array([equ_len] * nsam)

sde = spaths.OverdampedLangevin(V, inv_temp)
ens0 = spaths.make_ens(xa0, xc0, yc0)
<<<<<<< HEAD:tests/triatomic_molecule.py
sol = spaths.EMSolver(sde, ens0, tspan, dt, rng)
path = sol.p[0]
=======
sol = em.solve(sde, ens0, tspan, dt)
>>>>>>> c2069535610620c06086aaa13fd097622f0038f7:examples/triatomic_molecule.py


step=4
t = sol.t[::step]
path = sol.p[0,::step]
path_xa, path_xc, path_yc = path.T

fig, ax = plt.subplots(figsize=(7,5))

# ax.plot(t, path)  # to plot all coords at once
ax.plot(t, path_xa, label="$x_A$", alpha=.6)
ax.plot(t, path_xc, label="$x_C$", alpha=.6)
ax.plot(t, path_yc, label="$y_C$", alpha=.6)

ax.legend()
ax.set_xlabel("time")

fig.tight_layout()
fig.savefig(f"figs/{file_name}_dofs.pdf")
plt.close(fig)

fig, ax = plt.subplots(nrows=3, figsize=(7,6), sharex=True)

path_ang = np.pi / 2 - np.arctan(path_xc / path_yc)
path_dist = np.sqrt((path_xa-path_xc)**2 + path_yc**2)

ax[0].plot(t, path_xc, label="x_{C}", alpha=.6)
ax[1].plot(t, path_ang, label="$\Theta_{AC}$", alpha=.6)
ax[2].plot(t, path_dist, label="$d_{AC}$", alpha=.6)

ax[0].set_ylabel("$x_{C}$", rotation=0, labelpad=10)
ax[1].set_ylabel("$\Theta_{AC}$", rotation=0, labelpad=15)
ax[2].set_ylabel("$D_{AC}$", rotation=0, labelpad=15)

ax[-1].set_xlabel("time")

fig.tight_layout()
fig.savefig(f"figs/{file_name}_slow.pdf")
plt.close(fig)

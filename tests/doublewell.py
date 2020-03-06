"""
Created on Fri 6 Mar 2020

@author: Przemyslaw Zielinski
"""

from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os

cwd = os.getcwd()
sys.path.append(cwd + '/..')
import spaths

# model parameters
# def V(t, x):
#     return x*x
V = lambda t, x: np.squeeze((1 - (x - 2.0)**2)**2)
inv_temp = 6

# simulation parameters
dt = .1
nsam = 1000
tspan = (0.0, 200.0)

# seed setting
seed = 3579
rng = np.random.default_rng(seed)
rng.integers(10**3, size=10**3)  # warm up of RNG

# initial conditions
x0 = [1.0]*nsam

# from jax import grad, vmap, jit
# print(V(0.0, x0))
# vV = jit(vmap(grad(V, 1), in_axes=(None, 1), out_axes=1))
# t0 = np.full_like(x0, 1.0)
# print(vV(2.0, x0))


sde = spaths.OverdampedLangevin(V, inv_temp)
ens0 = spaths.make_ens(x0)
sol = spaths.EMSolver(sde, ens0, tspan, dt, rng)

insp_times = np.linspace(*tspan, 5)
insp_sol = sol(insp_times)

fig, ax = plt.subplots(figsize=(8,6))
ls = 16
lw = 2

colors = plt.cm.viridis(np.linspace(0, 1, len(insp_times)))
X_plot = np.linspace(0, 4, 200)[:, np.newaxis]
for n, X in enumerate(insp_sol):
    kde = KernelDensity(kernel='gaussian', bandwidth=.2).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(X_plot, np.exp(log_dens), color=colors[n])
# ax.set_ylim([0,4])

fig.tight_layout()
fig.savefig(f"figs/doublewell.pdf")
plt.close()

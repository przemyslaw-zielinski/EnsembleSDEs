"""
Created on Fri 6 Mar 2020

@author: Przemyslaw Zielinski
"""

# set cwd to supfolder for local import
import sys, os
cwd = os.getcwd()
sys.path.append(cwd + '/..')

from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
import spaths

file_name = "doublewell"

# model parameters
V = lambda t, x: (1 - (x - 2.0)**2)**2
inv_temp = 6

# simulation parameters
dt = .1
nsam = 1000
tspan = (0.0, 200.0)

# initial conditions
x0 = [1.0]*nsam

# seed setting
seed = 3579
rng = np.random.default_rng(seed)
rng.integers(10**3, size=10**3)  # warm up of RNG

sde = spaths.OverdampedLangevin(V, inv_temp)
ens0 = spaths.make_ens(x0)
sol = spaths.EMSolver(sde, ens0, tspan, dt, rng)

insp_times = np.linspace(*tspan, 5)
insp_sols = sol(insp_times)

fig, ax = plt.subplots(figsize=(8,6))
ls = 16
lw = 2

colors = plt.cm.viridis(np.linspace(0, 1, len(insp_times)))
X_plot = np.linspace(0, 4, 200)[:, np.newaxis]
for n, (t, X) in enumerate(zip(insp_times, insp_sols)):
    kde = KernelDensity(kernel='gaussian', bandwidth=.2).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(X_plot, np.exp(log_dens), color=colors[n], label=f"time = {t}")

ax.legend()
fig.tight_layout()
fig.savefig(f"figs/{file_name}.pdf")
plt.close()

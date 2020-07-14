"""
Created on Fri 29 May 2020

@author: Przemyslaw Zielinski
"""

# add supfolder to path for local import
import sys, os
sys.path.append(os.getcwd() + '/..')

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import spaths

file_name = "dimer2D_bimodal"

# seed setting
seed = 3579
rng = np.random.RandomState(seed)
rng.randint(10**5, size=10**3)  # warm up of RNG

# solver
em = spaths.EulerMaruyama(rng)

# model parameters
inv_temp = 0.4
barrier_height = 10.0
compact_state = 0.5
loose_state = compact_state + 1.4

import jax.numpy as jnp  # for autodiff of V
def bond_length(x):
    return jnp.sqrt((x[0]-x[2])**2 + (x[1]-x[3])**2)

def V(t, x):  # dimer ends: (x[0], x[1]), (x[2], x[3])
    blx = bond_length(x)
    he = (loose_state - compact_state) / 2.0  # half elongation
    arg = (blx - compact_state - he) / he
    return barrier_height * (1 - arg**2)**2

# simulation parameters
dt = 2*1e-3
tspan = (0.0, 35.0)


sde = spaths.OverdampedLangevin(V, inv_temp)
x0 = np.array([[0.0, 0.0, compact_state, compact_state]])

sol = em.solve(sde, x0, tspan, dt)
bond_lengths = [bond_length(x.T) for x in sol.x]
data = np.squeeze(sol.x)[::5]
# print(f"{data.shape = }")

# plot evolution of dimer length
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(sol.t[::2], bond_lengths[::2], linewidth=1)

fig.savefig(f"figs/{file_name}_path.pdf")
plt.close(fig)

# plot correlations of data coordinates
data_ctr = data - np.mean(data, axis=0)
data_bond_len = np.array([bond_length(d) for d in data])
fig = plt.figure(figsize=(8,8))
gs = GridSpec(3, 3, figure=fig, hspace=0.05, wspace=0.05)

for n, coord in enumerate(data_ctr.T[:-1]):
    for m in range(n+1):
        ax = fig.add_subplot(gs[n, m])
#         ax.set_title(f"{n}, {m-1}")
        ax.scatter(data_ctr[:,m-1], coord, s=0.5)
        ax.set_xlim([-8.0, 8.0])
        ax.set_ylim([-8.0, 8.0])
        ax.set_aspect("equal")
        if n == 2:
            ax.set_xlabel(f"$x_{(m-1) % 4}$")
            # ax.set_xticks([-.25, 0.0, +.25])
        else:
            ax.set_xticklabels([])
        if m == 0:
            ax.set_ylabel(f"$x_{n}$", rotation=0)
            # ax.set_yticks([-.25, 0.0, +.25])
        else:
            ax.set_yticklabels([])
ax = fig.add_subplot(gs[0,2])
H, edges = np.histogram(data_bond_len, bins=30, density=True, range=(0,3))
ax.plot(.5*(edges[1:]+edges[:-1]), H)
ax.set_xlim([0,3])
ax.set_xlabel("bond length")
ax.set_ylabel("freq", rotation=0, labelpad=10)

fig.savefig(f"figs/{file_name}_coords.pdf")
plt.close(fig)

# plot spectral embedding (diff maps) with gaussian kernel
from sklearn import manifold
lap_eig = manifold.SpectralEmbedding(n_components=1, affinity='rbf', gamma=1/.5)
Z = lap_eig.fit_transform(data)
N = np.linalg.norm(Z, axis=0)
Z = Z / N

fig, axs = plt.subplots(ncols=5, figsize=(15,3), sharey=True)

axs[0].set_ylabel("z", rotation=0, labelpad=5)
for n, ax in enumerate(axs[:-1]):
    axs[n].scatter(data.T[n], Z, s=.5)
    axs[n].set_xlabel(f"$x_{n}$")
axs[4].scatter(data_bond_len, Z, s=.5)
axs[4].set_xlabel("bond length")


fig.tight_layout()
fig.savefig(f"figs/{file_name}_spectral.pdf")
plt.close(fig)

"""
Created on Wed 11 Mar 2020

@author: Przemyslaw Zielinski
"""
# set cwd to supfolder for local import
import sys, os
cwd = os.getcwd()
sys.path.append(cwd + '/..')

import matplotlib.pyplot as plt
from spaths import potentials
# import jax.numpy as jnp
import numpy as np
import spaths

file_name = "dimer_in_2Dsolvent"

# seed setting
seed = 4321
rng = np.random.RandomState(seed)
rng.randint(10**5, size=10**3)  # warm up of RNG

# def boxFold(x, box_length=1):
#     return np.remainder(x, box_length)


# model parameters
nparts = 10
box_length = 10
dimer = {0, 1}  # indices of particle forming the dimer
inv_temp = 0.5

# potentials
strength = 0.1
interaction_radius = 1
wca = potentials.WCAPotential(strength, interaction_radius)

barrier_height = 10.0
compact_state = 2**(1/6)*interaction_radius  # the same as wca cutoff distance
loose_state = compact_state + 1.4
ds = potentials.DSPotential(barrier_height, compact_state, loose_state)

# plot potentials
d_plot = np.linspace(0.1, 1.2*loose_state, 200)
wca_plot = [wca(d) for d in d_plot]
ds_plot = [ds(d) for d in d_plot]
fig, ax = plt.subplots(figsize=(8,6))
ls = 16
lw = 2
ax.plot(d_plot, wca_plot)
ax.plot(d_plot, ds_plot)
ax.set_ylim([0, 1.1*barrier_height])

fig.tight_layout()
fig.savefig(f"figs/potentials.pdf")
plt.close(fig)
################

Wmat = [# only for dimer use the ds potential
    [ds if {m, n} == dimer else wca for m in range(nparts)]
    for n in range(nparts)
]
V = potentials.PairwisePotential(Wmat, dim=2, box_length=box_length)

# simulation parameters
dt = 1e-4 / 2.0
nsam = 1
tspan = (0.0, 2.0)

sde = spaths.OverdampedLangevin(V, inv_temp)
ens0 = potentials.initialize_particles(nparts, V, 2*inv_temp, dt, 200, rng)
sol = spaths.EMSolver(sde, ens0, tspan, dt, rng)
# print(sol.x.shape)

fig, ax = plt.subplots(figsize=(8,6))
ls = 16
lw = 2

dimer_distance = [
    potentials.boxDist(x[0,0:2], x[0,2:4], box_length=box_length)
    for x in sol.x
]

step = 100
ax.plot(sol.t[::step], dimer_distance[::step], alpha=.6)


fig.tight_layout()
fig.savefig(f"figs/{file_name}.pdf")
plt.close()

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
seed = 3579
rng = np.random.RandomState(seed)
rng.randint(10**5, size=10**3)  # warm up of RNG


# model parameters
nparts = 16
box_length = 10
dimer = {0, 1}  # indices of particles forming the dimer
inv_temp = 0.5

# potentials
strength = 1.0
interaction_distance = 0.5
wca = potentials.WCAPotential(strength, interaction_distance)

barrier_height = 10.0
compact_state = interaction_distance  # the same as wca cutoff distance
loose_state = compact_state + 1.4
ds = potentials.DSPotential(barrier_height, compact_state, loose_state)

Wmat = [# only for the dimer use the ds potential
    [ds if {m, n} == dimer else wca for m in range(nparts)]
    for n in range(nparts)
]
V = potentials.PairwisePotential(Wmat, dim=2, box_length=box_length)

# simulation parameters
dt = 1e-4
nsam = 1
tspan = (0.0, 1.0)

# gentle initialization of particles
nsteps = 200
ens0 = potentials.initialize_particles(nparts, V, inv_temp, dt, 200, rng)

sde = spaths.OverdampedLangevin(V, inv_temp)
sol = spaths.EMSolver(sde, ens0, tspan, dt, rng)

# plot evolution of dimer length
fig, ax = plt.subplots(figsize=(8,6))
ls = 16
lw = 2

dimer_distance = [
    potentials.box_dist(x[0,0:2], x[0,2:4], box_length=box_length)
    for x in sol.x
]
step = 100
ax.plot(sol.t[::step], dimer_distance[::step], alpha=.6)


fig.tight_layout()
fig.savefig(f"figs/{file_name}.pdf")
plt.close(fig)

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

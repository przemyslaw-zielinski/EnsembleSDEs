"""
Created on Wed 11 Mar 2020

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

file_name = "dimer_in_2Dsolvent"

# seed setting
seed = 4321
rng = np.random.RandomState(seed)
rng.randint(10**5, size=10**3)  # warm up of RNG


class DSPotential():
    '''
    Double-state potential with two local minima at r = compact_state
    and r = loose_state, separated by a barrier of height = barrier_height.
    '''
    def __init__(self, barrier_height, compact_state, loose_state):
        self.bh = barrier_height
        self.cs = compact_state
        self.he = (loose_state - compact_state) / 2.0  # half elongation

    def __call__(self, dist):
        arg = (distance - self.cs - self.he)/self.he
        return self.bh * (1 - arg**2)**2

    def der(self, dist):
        arg = (dist - self.cs - self.he)/self.he
        inner_der = -2 * arg
        return 2*self.bh * (1 - arg**2) * inner_der

# Weeks–Chandler–Andersen potential
class WCAPotential():
    '''
    Weeks-Chandler-Andersen (WCA) potential is the Lenar-Jones potential
    truncated at the minimum potential energy at a distance
        r = 2**(1/6) * interaction_radius
    on the length scale and shifted upward by the amount strength
    on the energy scale such that both the energy and force are zero at
    or beyond the cutoff distance.
    '''
    def __init__(self, strength, interaction_radius):
        self.s = strength
        self.ir = interaction_radius

    def __call__(self, dist):
        # cutoff_dist = np.maximum(dist, 2**(1/6)*self.ir)
        if dist <= 2**(1/6)*self.ir:
            return 4*self.s*((self.ir/dist)**12 - (self.ir/dist)**6) + self.s
        else:
            return 0

    def der(self, distance):
        return -24*self(distance) / distance

def boxDist(p1, p2, box_length=1):
    dp = p1 - p2
    dp -= np.rint(dp / box_length) * box_length
    return np.linalg.norm(dp)

def boxFold(x, box_length=1):
    return np.remainder(x, box_length)


class PairwisePotential():

    def __init__(self, W, dim=1, box_length=1):
        self.W = W
        self.dim = dim
        self.box_length = box_length

    def __call__(self, x):
        npart = len(x) // self.dim
        pidx = [tuple(range(i, i+self.dim)) for i in range(npart)]

        val = 0.0
        for idx, m in enumerate(pidx[:-1]):
            for n in pidx[idx+1:]:
                # breakpoint()
                dist = boxDist(x[m], x[n], box_length=self.box_length)
                val += self.W(dist)

        return val

    def get_particle(self, i, x):
        return x[self.dim*i:self.dim*(i+1)]

    def grad(self, x):  # x.shape = (ndim, nsam)
        # breakpoint()
        npart = len(x) // self.dim

        grad = np.zeros_like(x)
        for i1 in range(npart-1):
            for i2 in range(i1+1, npart):
                p1 = self.get_particle(i1, x)
                p2 = self.get_particle(i2, x)

                dist = boxDist(p1, p2, box_length=self.box_length)
                force = self.W[i1][i2].der(dist) * (p1 - p2) / dist
                # breakpoint()

                grad[self.dim*i1:self.dim*(i1+1)] += force
                grad[self.dim*i2:self.dim*(i2+1)] -= force

        return grad

# model parameters
nparts = 4
box_length = 10
dimer = {0, 1}  # indices of particle forming the dimer
inv_temp = 0.8

# potentials
strength = 0.01
interaction_radius = 0.5
wca = WCAPotential(strength, interaction_radius)

barrier_height = 10.0
compact_state = 1.0
loose_state = 2.5
ds = DSPotential(barrier_height, compact_state, loose_state)

Wmat = [# only for dimer use ds potential
    [ds if {m, n} == dimer else wca for m in range(nparts)]
    for n in range(nparts)
]
V = PairwisePotential(Wmat, dim=2, box_length=box_length)

# simulation parameters
dt = 1e-4
nsam = 1
tspan = (0.0, 25.0)

ens0 = rng.uniform(high=box_length, size=(1, 2*nparts))

def drift(t, x, dx):
    dx[:] = -V.grad(x)

def dispersion(t, x, dx):
    dx[:] = np.sqrt(2 / inv_temp)

sde = spaths.SDE(drift, dispersion)
# ens0 = spaths.make_ens(x0)
sol = spaths.EMSolver(sde, ens0, tspan, dt, rng)
# print(sol.x.shape)

fig, ax = plt.subplots(figsize=(8,6))
ls = 16
lw = 2

# i = 0
# path_x = boxFold(sol.p[2*i], box_length=5)
# path_y = boxFold(sol.p[2*i+1], box_length=5)
# path_xa, path_xc, path_yc = path.T
# path_ang = np.pi / 2 - np.arctan(path_xc / path_yc)
dimer_distance = [
    boxDist(x[0,0:2], x[0,2:4], box_length=box_length)
    for x in sol.x
]
# print(dimer_distance[:10])

# ax.plot(sol.t[::5], path[::5])  # to plot all coords at once
step = 100
ax.plot(sol.t[::step], dimer_distance[::step], alpha=.6)
# ax.plot(sol.t[::5], path_xc[::5], label="xc", alpha=.6)
# ax.plot(sol.t[::5], path_yc[::5], label="yc", alpha=.6)
# ax.plot(sol.t[::5], path_ang[::5], label="$\Theta$")

# ax.legend()
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_aspect("equal")

fig.tight_layout()
fig.savefig(f"figs/{file_name}.pdf")
plt.close()

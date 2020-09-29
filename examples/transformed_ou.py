#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 22 Jul 2020

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
import timeit

file_name = 'transformed_ou'

# model params
tsep = 0.04
temp = 0.05

# underlying OU process coefficients
def drift_ou(t, u, du):
    du[0] = -u[0] + u[1]
    du[1] = -u[1] / tsep

def dispersion_ou(t, u, du):
    du[0,0] = np.sqrt(2*temp)
    du[1,1] = np.sqrt(2*temp/tsep)

sde_ou = spaths.ItoSDE(drift_ou, dispersion_ou, noise_mixing_dim=2)

# transformation and its inverse
fwdF = lambda u: jnp.asarray([u[1], u[0] + u[1]**2])
invF = lambda x: jnp.asarray([x[1] - x[0]**2, x[0]])
F = spaths.SDETransform(fwdF, invF)

sde_Fou = F(sde_ou)

# SDE coefficients corresponding to F
def drift_rqp(t, x, dx):
    dx[0] = -x[0] / tsep
    dx[1] = -x[1] + x[0] + (1 - 2/tsep)*x[0]**2 + 2*temp/tsep

def dispersion_rqp(t, x, dx):
    dx[0,1] = np.sqrt(2*temp/tsep)
    dx[1,0] = np.sqrt(2*temp)
    dx[1,1] = 2*x[0]*np.sqrt(2*temp/tsep)

# corresponding sde
sde_rqp = spaths.ItoSDE(drift_rqp, dispersion_rqp, noise_mixing_dim=2)

# seed setting and solvers
seed = 3579
rng_Fou = np.random.default_rng(seed)
rng_rqp = np.random.default_rng(seed)
rng_Fou.integers(10**4), rng_rqp.integers(10**4)  # warm up of RNG
em_Fou = spaths.EulerMaruyama(rng_Fou)
em_rqp = spaths.EulerMaruyama(rng_rqp)

# simulation params
dt = .1 * tsep
nsam = 4000
x0, y0 = 0.0, 0.0
tspan = (0.0, 5)
tsteps = (0.0, 10)

ens0 = np.array([[x0, y0] * nsam], dtype=np.float32).reshape(-1, 2)
x0 = ens0.T

print(np.allclose(sde_Fou.drif(0, x0), sde_rqp.drif(0, x0)))
print(np.allclose(sde_Fou.disp(0, x0), sde_rqp.disp(0, x0)))

# timing coeff computations
timeit_cfg = {
    'number': 100,
    'repeat': 10,
    'globals': globals()
}

print(np.mean(timeit.repeat(stmt='sde_rqp.drif(0, x0)', **timeit_cfg)))
print(np.mean(timeit.repeat(stmt='sde_Fou.drif(0, x0)', **timeit_cfg)))

sol_rqp = em_rqp.solve(sde_rqp, ens0, tspan, dt)
sol_Fou = em_Fou.solve(sde_Fou, ens0, tspan, dt)
print(np.allclose(sol_rqp.x, sol_Fou.x, atol=1e-6))

timeit_cfg = {
    'number': 1,
    'repeat': 3,
    'globals': globals()
}
print(np.mean(timeit.repeat(stmt='em_rqp.burst(sde_rqp, ens0, tsteps, dt)', **timeit_cfg)))
print(np.mean(timeit.repeat(stmt='em_Fou.burst(sde_Fou, ens0, tsteps, dt)', **timeit_cfg)))

#
# times = np.linspace(*tspan, 60)
# data_Fou = np.squeeze(sol_Fou(times))
#
# from matplotlib.gridspec import GridSpec
#
# fig = plt.figure(figsize=(15, 5))
# gs = GridSpec(1, 3, figure=fig)
#
# ax_t = fig.add_subplot(gs[0, :-1])
# ax_t.plot(times, data_Fou.T[1], color='saddlebrown', alpha=.8, label="Fast coordinate")
# ax_t.plot(times, data_Fou.T[0], color='limegreen', label="Slow coordinate")
#
# ax_t.set_xlim((times[0], times[-1]))
# ax_t.set_ylim([-1, 1])
# ax_t.set_yticks([-1, -.5, 0, .5, 1])
# ax_t.set_xlabel("$t$")
#
# ax_d = fig.add_subplot(gs[0, -1])
# ax_d.scatter(*data_Fou.T, linewidths=0)
# # ax.scatter(*batch.T, s=.5, alpha=0.5)
# # ax.scatter(*point, s=1, c="red")
# slow_man = np.array([[u, 0.0] for u in np.linspace(-1, 1, 50)])
# # ax_d.scatter(*rqp(slow_man).T)
#
# ax_d.set_xlim([-1, 1])
# ax_d.set_xticks([-1, 1])
# ax_d.set_ylim([-1, 1])
# ax_d.set_yticks([-1, 1])
# ax_d.set_aspect("equal")
# ax_d.set_xlabel("$x$", labelpad=-5)
# ax_d.set_ylabel("$y$", rotation=0, labelpad=-8)
#
# plt.show()

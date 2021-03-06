#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 6 Feb 2020

@author: Przemyslaw Zielinski
"""

# set cwd to supfolder for local import
import sys, os
cwd = os.getcwd()
sys.path.append(cwd + '/..')

import matplotlib.pyplot as plt
import numpy as np
import spaths

file_name = "ornstein-uhlenbeck"

# model parameters
eps = 1 / 50  # time scale separation
bet = 20  # inverse temperature

A = np.array(
    [[-1, 1],
     [ 0, -1 / eps]]
)
b = np.array([np.sqrt(2 / bet), np.sqrt(2 / (eps*bet))])
bb = np.diag(b)
B = np.array(
    [[np.sqrt(2 / bet), np.sqrt(2 / bet), np.sqrt(2 / bet)],
     [np.sqrt(2 / (eps*bet)), np.sqrt(2 / (eps*bet)), np.sqrt(2 / (eps*bet))]]
)

# seed setting
seed = 3579
rng = np.random.default_rng(seed)
rng.integers(10**3, size=10**3)  # warm up of RNG

# solver
em = spaths.EulerMaruyama(rng)

# simulation parameters
dt = eps / 2
nsam = 2
tspan = (0.0, 10.0)

# initial conditions
x0, y0 = [2.0]*nsam, [2.0]*nsam

oue = spaths.OrnsteinUhlenbeck(A, B)
ens0 = spaths.make_ens(x0, y0)
sol = em.solve(oue, ens0, tspan, dt)

fig, ax = plt.subplots(figsize=(8,6))
ls = 16
lw = 2

ax.plot(sol.t, sol.x[:,0,1])
ax.plot(sol.t, sol.x[:,0,0])


ax.tick_params(
        axis='both',        # changes apply to
        which='major',       # both major and minor ticks are affected
        bottom=True,       # ticks along the bottom edge are on/off
        left=True,
        labelleft=True,
        labelsize=ls)

fig.tight_layout()
fig.savefig(f"figs/{file_name}.pdf")
plt.close()

# TODO: move to the class for linear systems
def cov_inv(drmat, dimat, tol=10**(-10)):
    '''
    Computes covariance matrix of invariant measure of linear SDE.
    '''
    from numpy import array, sqrt, newaxis, inf, exp, nditer
    from numpy import real, imag, complex64, float64, real_if_close
    from numpy.linalg import eig, inv
    from scipy.integrate import quad
    eigv, simat = eig(drmat)
    inv_simat = inv(simat)
    bmat = (inv_simat @ dimat @ dimat.T @ inv_simat.T).astype(complex64)
    vmat = eigv[:, newaxis] + eigv
    with nditer(bmat, op_flags=['readwrite']) as bit:
        for b, v in zip(bit, vmat.ravel()):
            b_re, *rest = quad(lambda t: real(b * exp(t*v)), 0, inf)
            b_im, *rest = quad(lambda t: imag(b * exp(t*v)), 0, inf)
            b[...] = b_re + b_im*1.j
    return real_if_close(simat @ bmat @ simat.T)

def cov_inv_dt(drif_mat, diff_mat, dt, tol=10**(-10)):
    """
    Computes variance of invariant distribution for EM scheme
    taking into account the bias by timestep dt
    """

    # zeroth summand
    vardt = dt * diff_mat @ diff_mat.T
    # eigenvectors of drift matrix
    v, vv = np.linalg.eig(drif_mat)
    # new diffusion
    diff_mat_vv = np.linalg.inv(vv) @ diff_mat
    diff_mat_vv = diff_mat_vv @ diff_mat_vv.T

    j = 1
    while True:
        diag = np.diag((1 + dt*v)**j)
        new_summ = dt*np.linalg.multi_dot([vv, diag, diff_mat_vv, diag, vv.T])
        vardt += new_summ
        if np.linalg.norm(new_summ) < tol:
            break
        j += 1
    return vardt

ens = sol(tspan[1])
print(f"Norm of mean differences: {np.linalg.norm(np.average(ens, axis=0))}")
print(f"Norm of cov differences: "
      f"{np.linalg.norm(cov_inv_dt(A, B, dt) - np.cov(ens.T))}")
print("Covariance matrix of invariant Gaussian for EM:")
print(cov_inv_dt(A, B, dt))
print("Covariance matrix of invariant Gaussian")
print(cov_inv(A, B))
print(f'{oue.nmd = }')
print(f'{oue.diff(0, ens0) = }')

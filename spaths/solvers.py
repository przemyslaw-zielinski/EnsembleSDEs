#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 2020

@author: Przemyslaw Zielinski
"""
from numpy import asarray, array, vstack, newaxis, sqrt, zeros, einsum
from math import ceil
from .spaths import StochasticPath

def EMSolver(sde, ens0, tspan, dt, rng):
    '''
    Implements Euler method for solving the stochastic equation

        dx = sde.drift(t,x)*dt + sde.disp(t,x)*dw
        x(t0) = ens0, t0 = tspan[0]

    with timestep dt up to final time T = tspan[1].

    The initial ensemble ens0 is
    an array of shape (nsam, ndim) whose each row is treated as a
    deterministic inital condition of dimension ndim.

    The paths of the process are stored in a numpy array of shape
    (nsteps + 1, nsam, ndim) where nsteps is the smallest integer
    bigger than the ratio of T to dt.

    Returns
    -------
    A tuple of two numpy arrays of floats:
        --> tgrid storing the times on which process was approximated
        --> sol of shape (nsteps + 1, nsam, ndim) with
            ens0 included as the first row of this array.
    '''

    ens0 = asarray(ens0)
    t0, T = tspan

    nsteps = ceil(T / dt) + 1
    tgrid = array([t0 + k * dt for k in range(nsteps + 1)]) # timegrid

    if sde.ndim == sde.nigp:
        def mult(drif_vec, dw):
            return drif_vec * dw
    else:
        def mult(drif_mat, dw):
            return einsum('ijk,ik->ij', drif_mat, dw)

    sol = vstack((ens0[newaxis,:], zeros((nsteps,) + ens0.shape)))
    for n, t in enumerate(tgrid[:-1]):
        dw = rng.standard_normal((len(ens0), sde.nigp))
        sol[n+1] = sol[n] + dt*sde.drif(t, sol[n]) \
                 + sqrt(dt)*mult(sde.disp(t, sol[n]), dw)

    return StochasticPath(tgrid, sol)

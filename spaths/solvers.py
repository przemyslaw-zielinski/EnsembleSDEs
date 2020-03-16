#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 2020

@author: Przemyslaw Zielinski
"""
import numpy as np
from .spath import StochasticPath

def EMSolver(sde, ens0, tspan, dt, rng):
    '''
    Implements Euler method for solving the stochastic equation

        dx = sde.drift(t,x)*dt + sde.disp(t,x)*dw, t in (t0, T)
        x(t0) = ens0,

    with timestep dt, where t0 = tspan[0] and T = tspan[1].

    The initial ensemble ens0 is an array of shape (nsam, ndim) whose each row
    is treated as a deterministic inital condition of dimension ndim.

    The paths of the process are stored in an ndarray of shape
    (nsteps + 1, nsam, ndim) where nsteps is the smallest integer
    bigger than the ratio of T to dt.

    Returns
    -------
    Instance of StochasticPath based on grids:
        --> tgrid -- storing the times on which process was approximated
        --> sol   -- of shape (nsteps + 1, nsam, ndim) with
                     ens0 included as the first row of this array.
    '''

    ens0 = np.asarray(ens0)
    tgrid, nsteps = generate_tgrid(tspan, dt)

    if len(sde.nmd) == 0:
        dw_shape = ens0.shape
        def mult(disp_vec, dw):
            return disp_vec * dw
    else:
        dw_shape = (len(ens0),) + sde.nmd
        def mult(disp_mat, dw):
            # sum over noise dimension k; i = nsam, j = ndim
            return np.einsum('ijk,ik->ij', disp_mat, dw)

    sol = np.vstack((ens0[np.newaxis,:], np.zeros((nsteps,) + ens0.shape)))
    for n, t in enumerate(tgrid[:-1]):
        dw = rng.standard_normal(dw_shape)
        sol[n+1] = sol[n] + dt*sde.drif(t, sol[n]) \
                          + np.sqrt(dt)*mult(sde.disp(t, sol[n]), dw)

    return StochasticPath(tgrid, sol)

def generate_tgrid(tspan, dt):
    tgrid = []
    t = tspan[0]
    while True:
        tgrid.append(t)
        if t > tspan[1]:
            break
        t += dt
    return np.array(tgrid), len(tgrid) - 1

def boxFold(x, box_length=1):
    return np.remainder(x, box_length)

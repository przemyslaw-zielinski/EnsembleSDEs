#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 2020

@author: Przemyslaw Zielinski
"""

import numpy as np
from jax import jacfwd, jacrev, jit, vmap

class ItoSDE():

    def __init__(self, drift, dispersion, noise_mixing_dim=0):
        '''
        Ito stochastic differential equation

            dX = A(t, X)dt + B(t, X)dW

        Parameters 'drift' and 'dispersion' have to be functions of
        scalar time t, and arrays x and dx,
        such that dx stores the value of A and B respectively:

        ->  def drift(t, x, dx):
                dx = A(t,x)

        ->  dispersion(t, x, dx):
                dx = B(t, x)
        '''

        self.drift = drift
        self.dispersion = dispersion
        if noise_mixing_dim == 0:
            self.nmd = ()
        else:  # TODO: add case for a scalar noise (nmd=1)
            self.nmd = (noise_mixing_dim,)

    def coeffs(self, t, ens):  # TODO: can we use that in solvers?
        return self.drif(t, ens), self.disp(t, ens)

    def drif(self, t, ens):
        # self.test_dim(ens)
        # breakpoint()
        dx = np.zeros_like(ens)
        self.drift(t, ens.T, dx.T)  # (ndim, nsam)
        return dx

    def disp(self, t, ens):
        # self.test_dim(ens)
        # breakpoint()
        dx = np.zeros(ens.shape + self.nmd, dtype=ens.dtype)
        self.dispersion(t, ens.T, np.moveaxis(dx, 0, -1))  # (ndim, (nmd,) nsam)
        return dx

    def test_dim(self, ens):
        if ens.ndim != 2 or ens.shape[1] != self.ndim:
            raise IndexError(f"Bad ensemble: shape={ens.shape}.")

class SDETransform():

    def __init__(self, function):
        self.f = vmap(function, in_axes=1, out_axes=1)
        self.df = jit(vmap(jacfwd(function), in_axes=1, out_axes=0))
        self.ddf = jit(jacfwd(jacrev(function)))

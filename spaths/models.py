#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 2020

@author: Przemyslaw Zielinski
"""

import numpy
from jax import grad, vmap, jit
from inspect import signature
from .potentials import PairwisePotential

class SDE():

    def __init__(self, drift, dispersion, noise_mixing_dim=0):
        '''
        drift and dispersion have to be functions of scalar time t
        and a number ndim of array coordinates x, y, z, ...
        ->  drift = drift(t, x, dx)
        ->  dispersion = dispersion(t, x, dx)
        ->  drift, dispersion act elementwise on x, y, z, ... and return
            a tuple of arrays storing the coordinates of the result
        '''

        self.drift = drift
        self.dispersion = dispersion
        if noise_mixing_dim == 0:
            self.nmd = ()
        else:  # TODO: add case for a scalar noise (nmd=1)
            self.nmd = (noise_mixing_dim,)

    def drif(self, t, ens):
        # self.test_dim(ens)
        # breakpoint()
        dx = numpy.zeros(ens.shape)
        self.drift(t, ens.T, dx.T)  # (ndim, nsam)
        return dx

    def disp(self, t, ens):
        # self.test_dim(ens)
        dx = numpy.zeros(ens.shape + self.nmd)
        self.dispersion(t, ens.T, numpy.moveaxis(dx, 0, -1))  # (ndim, (nmd,) nsam)
        return dx

    def test_dim(self, ens):
        if ens.ndim != 2 or ens.shape[1] != self.ndim:
            raise IndexError(f"Bad ensemble: shape={ens.shape}.")

class OverdampedLangevin(SDE):
    '''
    dX = -gradV(t,X)dt + sqrt(2*inv_temp**(-1))dW
    '''
    def __init__(self, V, inv_temp):

        if isinstance(V, PairwisePotential):
            self.gradV = V.grad
        else:
            V = squeeze(V)
            self.gradV = jit(vmap(grad(V, 1), in_axes=(None, 1), out_axes=1))
        # settings for vmap
        # in_axes=(None, 1): don't parallelize over the time and parallelize
        #                    over the samples axis
        # out_axes=1: put result along second axis
        self.inv_temp = inv_temp

        super().__init__(self.oLdrift, self.oLdispersion)

    def oLdrift(self, t, u, du):
        du[:] = -self.gradV(t, u)

    def oLdispersion(self, t, u, du):
        du[:] = numpy.sqrt(2 / self.inv_temp)

def make_ens(*coords):
    '''
    Builds appropriate ensemble from iterable of positions coordinates.
    '''
    return numpy.array(coords).T

def squeeze(func):
    def wrapper(*args, **kwargs):
        return numpy.squeeze(func(*args, **kwargs))
    return wrapper

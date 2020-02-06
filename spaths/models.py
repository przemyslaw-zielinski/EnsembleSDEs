#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 2020

@author: Przemyslaw Zielinski
"""

import numpy

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

def make_ens(*coords):
    '''
    Builds appropriate ensemble from iterable of positions coordinates.
    '''
    return numpy.array(coords).T

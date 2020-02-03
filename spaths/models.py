#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 2020

@author: Przemyslaw Zielinski
"""

import numpy

class SDE():

    def __init__(self, drift, dispersion, noise_rate):
        '''
        drift and dispersion have to be functions of scalar time t
        and a number ndim of array coordinates x, y, z, ...
        ->  drift = drift(t, x, dx)
        ->  dispersion = dispersion(t, x, dx)
        ->  drift, dispersion act elementwise on x, y, z, ... and return
            a tuple of arrays storing the coordinates of the result
        '''

        self.ndim = noise_rate[0]
        self.nigp = noise_rate[1]
        if self.ndim == self.nigp:
            self.nrp = (self.ndim,)
        else:
            self.nrp = noise_rate
        self.drift = drift
        self.dispersion = dispersion

    def drif(self, t, ens):
        self.test_dim(ens)
        dx = numpy.zeros((self.ndim, len(ens)))
        self.drift(t, ens.T, dx)
        return dx.T # back to (nsam, ndim) array

    def disp(self, t, ens):
        self.test_dim(ens)
        dx = numpy.zeros(self.nrp + (len(ens),))
        self.dispersion(t, ens.T, dx)
        return numpy.moveaxis(dx, -1, 0) # back to (nsam, ndim, nigp) array

    def test_dim(self, ens):
        if ens.ndim != 2 or ens.shape[1] != self.ndim:
            raise IndexError(f"Bad ensemble: shape={ens.shape}.")

def make_ens(*coords):
    '''
    Builds appropriate ensemble from iterable of positions coordinates.
    '''
    return numpy.array(coords).T

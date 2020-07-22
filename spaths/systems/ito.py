#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 2020

@author: Przemyslaw Zielinski
"""

import numpy as np
import jax.numpy as jnp
from inspect import signature
from jax import jacfwd, hessian, jit, vmap

class ItoSDE():

    def __init__(self, drift, dispersion, noise_mixing_dim=0):
        '''
        Ito stochastic differential equation

            dX = A(t, X)dt + B(t, X)dW

        Parameters 'drift' and 'dispersion' have to be functions of
        scalar time t and arrays x, and optionally of array dx,
        such that dx stores the value of A and B respectively:

        ->  def drift(t, x):
                return A(t, x)
        ->  def drift(t, x, dx):
                dx[:] = A(t, x)

        ->  dispersion(t, x):
                return B(t, x)
        ->  dispersion(t, x, dx):
                dx[:] = B(t, x)

        Here t is float and x.shape = (ndim, nsam).
        The drift should result in an array of shape (ndim, nsam).
        The dispersion should result in an arryay of shape (ndim[, nmd], nsam)

        Parameters
        ----------
            drift, diffusion (callable) : coefficients of the equation
            noise_mixing_dim (int) : if positive, the dimension of noise
                if zero, indicates the diagonal noise
        '''

        self._drif = drift
        self._disp = dispersion

        self.drif = self._ex_drif if is_explicit(drift) else self._im_drif
        self.disp = self._ex_disp if is_explicit(dispersion) else self._im_disp

        # self.dnp = self._diag_dnp if noise_mixing_dim == 0 else self._gene_dnp

        if noise_mixing_dim == 0:  # diagonal noise
            self.nmd = ()
            self.dnp = self._diag_dnp
            self.diff = self._diag_diff
        else:
            self.nmd = (noise_mixing_dim,)
            self.dnp = self._gene_dnp
            self.diff = self._gene_diff

    # various drift and dispersion options that can be chosen in init
    # here ens.shape = (nsam, ndim)
    def _ex_drif(self, t, ens):
        return self._drif(t, ens.T).T

    def _im_drif(self, t, ens):
        dx = np.zeros_like(ens)
        self._drif(t, ens.T, dx.T)  # (ndim, nsam)
        return dx

    def _ex_disp(self, t, ens):
        return np.moveaxis(self._disp(t, ens.T), -1, 0)  # (d[, m], s) -> (s, d[, m])
        # (nsam, ndim[, nmd])

    def _im_disp(self, t, ens):
        dx = np.zeros(ens.shape + self.nmd, dtype=ens.dtype)  # (nsam, ndim[, nmd])
        self._disp(t, ens.T, np.moveaxis(dx, 0, -1))  # (s, d[, m]) -> (d[, m], s)
        # (ndim[, nmd], nsam)
        return dx

    def coeffs(self, t, ens):  # TODO: can we use that in solvers?
        return self.drif(t, ens), self.disp(t, ens)

    # options to compute dispersion noise product
    def _diag_dnp(self, t, ens, dw):  #  for the diagonal noise (nmd = 0)
        return self.disp(t, ens) * dw

    def _gene_dnp(self, t, ens, dw):  # for the general case (nmd >= 1)
        # s - size of ensemble (= nsam)
        # d - dimension of system (= ndim)
        # m - mixing dimension of noise (= nmd)
        return np.einsum('sdm,sm->sd', self.disp(t, ens), dw)

    def _diag_diff(self, t, ens):
        disp = self.disp(t, ens)
        diff = disp * disp
        # multiplies dXd identity matrix by each row of diff via broadcasting
        return np.eye(ens.shape[1]) * diff[:, np.newaxis]

    def _gene_diff(self, t, ens):
        disp = self.disp(t, ens)
        return np.einsum('sij,skj->sik', disp, disp)

    def test_dim(self, ens):
        if ens.ndim != 2 or ens.shape[1] != self.ndim:
            raise IndexError(f"Bad ensemble: shape={ens.shape}.")

    def get_noise_shape(self, ens):
        return ens.shape if self.nmd == () else ens.shape[:1] + self.nmd


def is_explicit(coeff_func):
    sig = signature(coeff_func)
    return len(sig.parameters) == 2

class SDETransform():

    def __init__(self, func, ifunc):
        '''
        function(x) = y with x.shape = (d, b), y.shape = (p, b)
        where d - input dimesion, p - output dimension, b - batch dimension
        (coord major)
        '''

        self.f = jit(func)
        self.g = jit(ifunc)

        self.df = jit(vmap(jacfwd(func), in_axes=1, out_axes=2))
        # we map over the batch dimension b
        # (d, b) -> (p, d, b): array of gradients of components of function
        self.ddf = jit(vmap(hessian(func), in_axes=1, out_axes=3))
        # (d, b) -> (p, d, d, b)

    def __call__(self, sde):
        nmd = sde.nmd[0] if sde.nmd else 0  # TODO: diag noise can change into non-diag
        return ItoSDE(self.func_drif(sde), self.func_disp(sde), noise_mixing_dim=nmd)

    def func_drif(self, sde):

        def drif(t, y):
            x = self.g(y)
            return (
                batch_dot(self.df(x), sde.drif(t, x.T).T) + \
                batch_trace(batch_quad(self.ddf(x), np.moveaxis(sde.disp(t, x.T), 0, -1))) / 2
                )  # (s, d[, m]) -> (d[, m], s)
        return drif

    def func_disp(self, sde):

        def disp(t, y):
            x = self.g(y)
            return batch_mul(self.df(x), np.moveaxis(sde.disp(t, x.T), 0, -1))

        return disp

def batch_dot(bmat, bvec):
    return np.einsum('pdb,db->pb', bmat, bvec)

def batch_mul(bmat1, bmat2):
    return np.einsum('pdb,dmb->pmb', bmat1, bmat2)

def batch_quad(bten, bmat):
    bmatt = np.moveaxis(bmat, 1, 0)
    return np.einsum('mdb,pdcb,cnb->pmnb', bmatt, bten, bmat)

def batch_trace(bten):
    return np.einsum('pmmb->pb', bten)

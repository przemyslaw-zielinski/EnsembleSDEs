#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 2020

@author: Przemyslaw Zielinski
"""

import numpy as np
from .ito import ItoSDE
from ..reactions import intermediate, Reaction
from itertools import zip_longest
from scipy.special import comb


class ChemicalLangevin(ItoSDE):

    def __init__(self, nb_species, ls_reactions):

        self.nb_species = nb_species
        self.nb_reactions = len(ls_reactions)

        ls_rates = []
        ls_idxs  = []
        ls_coeff = []
        for react in ls_reactions:
            ls_rates.append(react.rate)
            ls_idxs.append([subs.species_id for subs in react.substrates])
            ls_coeff.append([subs.coeff for subs in react.substrates])
        self.ar_rates = np.array(ls_rates)[:, np.newaxis]
        self.ar_idxs  = np.array(list(zip_longest(*ls_idxs, fillvalue=nb_species)))
        self.ar_coeff = np.array(list(zip_longest(*ls_coeff, fillvalue=0)))

        self.sm_mat = self.generate_stoichiometric_matrix(ls_reactions)

        super().__init__(self.cl_drift, self.cl_dispersion,
                         noise_mixing_dim=self.nb_reactions)

    def cl_drift(self, t, u, du):
        du[:] = self.sm_mat @ self.propensity(u)

    def cl_dispersion(self, t, u, du):
        du[:] = self.sm_mat[..., np.newaxis] * np.sqrt(self.propensity(u))

    def propensity(self, x):  # x.shape = (nb_species, nb_samples)

        prop = np.ones((self.nb_reactions, x.shape[1]))
        # breakpoint()
        prop = self.ar_rates * prop
        for idxs, coeff in zip(self.ar_idxs, self.ar_coeff):
            active = idxs < self.nb_species
            idxs = idxs[active]
            coeff = coeff[active][:, np.newaxis]
            prop *= comb(x[idxs], coeff)

        return prop

    def generate_stoichiometric_matrix(self, ls_reactions):

        sm_matrix = []
        for react in ls_reactions:
            v = np.zeros(self.nb_species)  # stoichiometric vector
            for species_idx, coeff in react.substrates:
                v[species_idx] = -coeff  # substrts are used up
            for species_idx, coeff in react.products:
                v[species_idx] = +coeff  # products are created
            sm_matrix.append(v)
        return np.array(sm_matrix).T

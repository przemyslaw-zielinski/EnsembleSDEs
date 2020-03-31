#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 2020

@author: Przemyslaw Zielinski
"""

# set cwd to supfolder for local import
import sys, os
cwd = os.getcwd()
sys.path.append(cwd + '/..')

import matplotlib.pyplot as plt
import numpy as np
import spaths
from spaths.reactions import intermediate, Reaction

file_name = "michaelis_menten"

# list_of_species = [species(idx, name=cośtam) for idx in range(N)]
#
# [2 * s for s in species_list]

# seed setting
seed = 357
rng = np.random.default_rng(seed)
rng.integers(10**3, size=10**3)  # warm up of RNG

S = intermediate(0)  # substrate
E = intermediate(1)  # enzyme
ES = intermediate(2) # complex
P = intermediate(3)  # product

nA = 6.023e23  # Avagadro’s number
vol = 1e-15  # volume of system

y0 = np.zeros(4)
y0[0] = np.round(5e-7*nA*vol)  # molecules of substrate S
y0[1] = np.round(2e-7*nA*vol)  # molecules of enzyme E
ens0 = np.tile(y0, (5, 1))
# print(ens0)

c1, c2, c3 = 1e6/(nA*vol), 1e-4, 0.1

binding = Reaction(c1, [S, E], [ES])
dissociation = Reaction(c2, [ES], [E, S])
product = Reaction(c3, [ES], [P, E])

print(binding)
print(dissociation)
print(product)

t_span = (0.0, 50.0)
dt = t_span[1] / 250

cle = spaths.ChemicalLangevin(4, [binding, dissociation, product])
# print(cle.ar_idxs)
# print(cle.ar_coeff)
sol = spaths.EMSolver(cle, ens0, t_span, dt, rng)


fig, ax = plt.subplots(figsize=(8,6))

ax.plot(sol.t, sol.p[1])

fig.tight_layout()
fig.savefig(f"figs/{file_name}.pdf")
plt.close()
# print(cle.ar_idxs)
# print(cle.ar_coeff)
#
# x = np.array(range(10, 30)).reshape(4, 5)
# print(cle.propensity(x))
# print(cle.sm_mat)
# print(cle.sm_mat[...,np.newaxis] * cle.propensity(x))

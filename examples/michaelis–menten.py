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

# solver
em = spaths.EulerMaruyama(rng)

S = intermediate(0)  # substrate
E = intermediate(1)  # enzyme
ES = intermediate(2) # complex
P = intermediate(3)  # product

nA = 6.023e23  # Avagadro’s number
vol = 1e-15  # volume of system

y0 = np.zeros(4)
y0[0] = 3000  # np.round(5e-7*nA*vol)  # molecules of substrate S
y0[1] = 220  # np.round(2e-7*nA*vol)  # molecules of enzyme E
ens0 = np.tile(y0, (5, 1))
# print(ens0)

c1, c2, c3 = 1e6/(nA*vol), 1e-4, 0.1
c1 = 1e-4 # binding rate
c2 = 1.0 # unbinding rate
c3 = 1e-4 # production rate

binding = Reaction(c1, [S, E], [ES])
dissociation = Reaction(c2, [ES], [E, S])
product = Reaction(c3, [ES], [P, E])

print(binding)
print(dissociation)
print(product)

t_span = (0.0, 1.0)
dt = 1e-4  # t_span[1] / 2500

cle = spaths.ChemicalLangevin(4, [binding, dissociation, product])
# print(cle.ar_idxs)
# print(cle.ar_coeff)
sol = em.solve(cle, ens0, t_span, dt)


fig, ax = plt.subplots(figsize=(8,6))

tt = sol.t
ss, ee, cc, pp = sol.p[1].T
# ax.plot(sol.t, sol.p[1])# / sol.p[1, 100])

ax.plot(tt, ss - ss[-1], label="substrate")
ax.plot(tt, ee - ee[-1], "-.", label="enzyme")
ax.plot(tt, cc - cc[-1], "-.", label="complex")
ax.plot(tt, pp - pp[-1], label="product")

ax.legend()

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

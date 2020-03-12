"""
Created on Wed 11 Mar 2020

@author: Przemyslaw Zielinski
"""
 set cwd to supfolder for local import
import sys, os
cwd = os.getcwd()
sys.path.append(cwd + '/..')

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import spaths

file_name = "dimer_in_2Dsolvent"


# double well potential
class DWPotential():
    def __init__(self, barrierHeight, compactState, halfElongation):
        self.bh = barrierHeight
        self.cs = compactState
        self.he = halfElongation
    def __call__(self, distance):
        return self.bh * (1 - ((distance - self.cs - self.he)/self.he)**2)**2
    def der(self, distance):
        val = self.bh * (1 - ((distance - self.cs - self.he)/self.he)**2)
        return -4 * val * (distance - self.cs - self.he) /  self.he**2

# Weeks–Chandler–Andersen potential
class WCAPotential():
    def __init__(self, strength, interactionRadius):
        self.s = strength
        self.ir = interactionRadius
    def __call__(self, distance):
        if distance <= 2**(1/6)*self.ir:
            return 4*self.s*((self.ir/distance)**12-(self.ir/distance)**6)+self.s
        else:
            return 0
    def der(self, distance):
        return -24*self.value(distance) / distance

def V(t, x):

    npart = len(x) // 2
    pidx = [[i,i+1] for i in range(npart)]

    for m in pidx[:-1]:
        for n in pidx[m+1:]:
            dist = boxDist(x[m], x[n])
            if

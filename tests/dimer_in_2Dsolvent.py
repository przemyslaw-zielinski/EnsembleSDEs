"""
Created on Wed 11 Mar 2020

@author: Przemyslaw Zielinski
"""
# set cwd to supfolder for local import
import sys, os
cwd = os.getcwd()
sys.path.append(cwd + '/..')

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import spaths

file_name = "dimer_in_2Dsolvent"

# seed setting
seed = 4321
rng = np.random.RandomState(seed)
rng.randint(10**5, size=10**3)  # warm up of RNG


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
        return -24*self(distance) / distance

def boxDist(p1, p2, boxLength=1):
    dp = p1 - p2
    dp -= np.rint(dp / boxLength) * boxLength
    return np.linalg.norm(dp)

class V():

    def __init__(self, W, box_length=1, dim=1):
        self.W = W
        self.dim = dim
        self.box_length = box_length

    def __call__(self, x):
        npart = len(x) // self.dim
        pidx = [tuple(range(i, i+self.dim)) for i in range(npart)]

        val = 0.0
        for idx, m in enumerate(pidx[:-1]):
            for n in pidx[idx+1:]:
                # breakpoint()
                dist = boxDist(x[m], x[n], boxLength=self.box_length)
                val += self.W(dist)

        return val

    def grad(self, x):
        npart = len(x) // self.dim
        pidx = [tuple(range(i, i+self.dim)) for i in range(npart)]

        grad = np.zeros_like(x)
        for i1, m1 in enumerate(pidx[:-1]):
            for i, m2 in enumerate(pidx[i1+1:]):
                i2 = i1 + i + 1
                # breakpoint()
                dist = boxDist(x[m1], x[m2], boxLength=self.box_length)
                force = self.W[i1][i2].der(dist) * (x[m1] - x[m2]) / dist
                grad[m1] += force
                grad[m2] -= force

        return grad

npart = 2
wca = WCAPotential(.1, 3)
vv = V([[wca]*npart]*npart, box_length=5)

x = rng.uniform(0, 5, size=(npart, 3))
print(f"{x = }")
# print(f"{vv(x) = }")
print(f"{vv.grad(x) = }")

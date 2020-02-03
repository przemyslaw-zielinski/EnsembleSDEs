#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 3 Feb 2020

@author: Przemyslaw Zielinski
"""

import numpy as np

class StochasticPath():

    def __init__(self, tgrid, vgrid):
        '''
        Stores ensemble of trajectories.
        '''
        self.t = tgrid
        self.val = vgrid

    def __call__(self, times):

        times = np.asarray(times)
        scalar_time = False
        if times.ndim == 0:
            times = times[np.newaxis]  # Makes x 1D
            scalar_time = True

        idxs = []
        for t in times:
            diff = self.t - t
            idxs.append((diff>=0).argmax())

        x = self.val[idxs]
        if scalar_time:
            x = np.squeeze(x, axis=0)
        return x

#
# -*- coding: utf-8 -*-
#

import numpy as np
from tqdm import tqdm


class IsingModel:
    def __init__(self, size, beta=1., h=0.0, J=None):
        self.size = size
        if J is None:   # Ferro
            self.Jmat = np.ones((size, size)) - np.eye(size)

        self.beta = beta
        self.s = np.random.binomial(1, 0.5, size=size) * 2 - 1.
        self.energy = self.H(self.s)
        self.acccnt = 0

    def H(self, value, beta=None):
        if beta is None:
            beta = self.beta
        return - 0.5 * beta * value.dot(self.Jmat.dot(value)) / self.size

    def mcstep(self, value=None):
        if value is None:
            value = self.s
        order = np.random.permutation(self.size)
        rvals = np.random.uniform(0., 1., self.size)

        for idx in order:
            oldE = self.energy
            self.s[idx] *= -1
            newE = self.H(self.s)
            delta = np.exp(-(newE - oldE))

            # print('r: %g, delta(new:%f - old:%f): %g' % (rvals[idx], newE, oldE, delta))
            if(rvals[idx] < delta):  # 'accept'
                # print('accept')
                self.energy = newE
                self.acccnt += 1
            else:
                # print('reject')
                self.s[idx] *= -1

    def trace(self, iter, reset=False):
        Es = []
        States = []
        if reset is True:
            self.acccnt = 0

        for it in tqdm(range(iter)):
            self.mcstep()
            Es.append(self.energy)
            States.append(self.s)

        self.logs = {'energy': np.array(Es), 'state': np.array(States)}
        return self.logs


size = 128
J0 = 1.0
Jmat = J0 * (np.ones((size, size)) - np.ones(size))

model = IsingModel(128)

burn = model.trace(2000)
mclog = model.trace(1000, reset=True)

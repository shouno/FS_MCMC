#
# -*- coding: utf-8 -*-
#

import numpy as np
from tqdm import tqdm


class IsingModel:
    def __init__(self, size, beta=1., h=0.0, state=None, J=None):
        self.size = size
        if J is None:   # Ferro
            self.Jmat = np.ones((size, size)) - np.eye(size)
        else:
            self.Jmat = J

        self.beta = beta
        self.h = h
        if state is None:
            self.s = np.random.binomial(1, 0.5, size=size) * 2 - 1.
        else:
            self.s = state

        self.energy = self.H(self.s)
        self.acccnt = 0

    def H(self, value):
        return - 0.5 * self.beta * value.dot(self.Jmat.dot(value)) / self.size

    def mcstep(self, value=None):
        if value is None:
            value = self.s
        order = np.random.permutation(self.size)
        rvals = np.random.uniform(0., 1., self.size)

        for idx in order:
            oldE = self.energy
            self.s[idx] *= -1
            newE = self.H(self.s)
            delta = newE - oldE
            pdelta = np.exp(-delta)

            # print('r: %g, delta(new:%f - old:%f): %g' % (rvals[idx], newE, oldE, delta))
            if(rvals[idx] < pdelta):  # 'accept'
                # print('accept')
                self.energy = newE
                self.acccnt += 1
            else:  # 'reject' restore
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


size = 512
J0 = 1.0
itrs = 50   # default linspacevalue
Jmat = J0 * (np.ones((size, size)) - np.eye(size))

betas = 1. / (np.linspace(1e-12, 2.0, num=itrs) * J0)

mdat = []
stddat = []
edat = []
for beta in betas:
    model = IsingModel(size, beta=beta, J=Jmat)

    burn = model.trace(2000)
    mclog = model.trace(1000, reset=True)
    mdat.append(mclog['state'].mean())
    stddat.append(mclog['state'].std())
    edat.append(mclog['energy'])

mdat = np.array(mdat)
stddat = np.array(stddat)
edat = np.array(edat)

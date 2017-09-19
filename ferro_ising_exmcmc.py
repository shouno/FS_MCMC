#
# -*- coding: utf-8 -*-
#

import numpy as np
from tqdm import tqdm
from math import pow

# 温度交換バージョンをつくってみよう


class IsingModelMC:
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
        if self.beta == 0.:  # 温度∞ (beta=0.0) は全とっかえ
            self.s = np.random.binomial(1, 0.5, size=self.size) * 2 - 1.
            self.acccnt += self.size
            return
        # 有限温度の場合
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


class IsingModelEMC:
    def __init__(self, size, betas=None, h=0.0, state=None, J=None):
        self.size = size
        if betas is None:
            self.nbeta = 16
            self.betas = [pow(1.25, l-self.nbeta+1) for l in range(self.nbeta)]
            self.betas[0] = 0.
        self.MCs = [IsingModelMC(size, beta=beta, h=h, state=state, J=J) for beta in self.betas]
        # betas が偶数なとき
        self.evnset = [(i, i+1, self.MCs[i], self.MCs[i+1]) for i in range(0, self.nbeta-1, 2)]
        self.oddset = [(i, i+1, self.MCs[i], self.MCs[i+1]) for i in range(1, self.nbeta-1, 2)]

    def mcexstep(self, isodd=False):
        for mc in self.MCs:
            mc.mcstep()
        if isodd:
            exset = self.oddset
        else:
            exset = self.evnset

        exlog = np.arange(self.nbeta)
        rvals = np.random.uniform(0., 1., len(exset))
        for (rval, (id1, id2, mc1, mc2)) in zip(rvals, exset):
            r = (mc2.beta - mc1.beta) * (mc2.energy - mc1.energy)
            if rval <= r:  # accept exchange
                (mc1.s, mc2.s) = (mc2.s, mc1.s)
                (mc1.energy, mc2.energy) = (mc2.energy, mc1.energy)
                (exlog[id1], exlog[id2]) = (exlog[id2], exlog[id1])

        return exlog

    def trace(self, iterations, reset=False):
        Es = []
        States = []
        if reset is True:
            for mc in self.MCs:
                mc.acccnt = 0

        exlogs = []
        for it in tqdm(range(iterations)):
            exlogs.append(self.mcexstep(isodd=bool(it % 2)))
            Es.append([mc.energy for mc in self.MCs])
            States.append([mc.s for mc in self.MCs])

        exlogs = np.array(exlogs).reshape((iterations, self.nbeta))
        Es = np.array(Es).reshape((iterations, self.nbeta))
        States = np.array(States).reshape((iterations, self.nbeta, self.size))
        return exlogs, Es, States


size = 128
J0 = 1.0
itrs = 50   # default linspacevalue
Jmat = J0 * (np.ones((size, size)) - np.eye(size))

model = IsingModelEMC(size, J=Jmat)

burn = model.trace(1000)
mclog = model.trace(500)

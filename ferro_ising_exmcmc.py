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
        return - 0.5 * value.dot(self.Jmat.dot(value)) / self.size

    def mcstep(self, value=None):
        if self.beta == 0.:  # 温度∞ (beta=0.0) は全とっかえ
            self.s = np.random.binomial(1, 0.5, size=self.size) * 2 - 1.
            self.energy = self.H(self.s)
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
            pdelta = np.exp(-self.beta * delta)

            # print('r: %g, delta(new:%f - old:%f): %g' % (rvals[idx], newE, oldE, delta))
            if(rvals[idx] < pdelta):  # 'accept'
                # print('accept')
                self.energy = newE
                self.acccnt += 1
            else:  # 'reject' restore
                # print('reject')
                self.s[idx] *= -1

            assert self.energy == self.H(self.s), "Energy is incorrect %f != %f" % (self.energy, self.H(self.s))

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
            self.nbeta = 24
            self.betas = [pow(1.25, l-16+1) for l in range(self.nbeta)]   # 決め打ち
            self.betas[0] = 0.
        self.MCs = [IsingModelMC(size, beta=beta, h=h, state=state, J=J) for beta in self.betas]
        # betas が偶数なとき
        self.evnset = [(i, i+1, self.MCs[i], self.MCs[i+1]) for i in range(0, self.nbeta-1, 2)]
        self.oddset = [(i, i+1, self.MCs[i], self.MCs[i+1]) for i in range(1, self.nbeta-1, 2)]

    def mcexstep(self, isodd=False):
        for i, mc in enumerate(self.MCs):
            assert mc.energy == mc.H(mc.s), "pre: mc[%d]: %f <==> %f" % (i, mc.energy, mc.H(mc.s)) 

        for mc in self.MCs:
            mc.mcstep()

        # exchange process
        if isodd:
            exset = self.oddset
        else:
            exset = self.evnset

        exlog = np.arange(self.nbeta)

        for i, mc in enumerate(self.MCs):
            assert mc.energy == mc.H(mc.s), "post: mc[%d]: %f <==> %f" % (i, mc.energy, mc.H(mc.s)) 

        rvals = np.random.uniform(0., 1., len(exset))
        for (rval, (id1, id2, mc1, mc2)) in zip(rvals, exset):
            r = np.exp((mc2.beta - mc1.beta) * (mc2.energy - mc1.energy))
            if rval <= r:  # accept exchange
                mm1 = mc1.s.mean()
                mm2 = mc2.s.mean()
                me1 = mc1.energy
                me2 = mc2.energy
                (mc1.s, mc2.s) = (mc2.s, mc1.s)
                (mc1.energy, mc2.energy) = (mc2.energy, mc1.energy)
                (exlog[id1], exlog[id2]) = (exlog[id2], exlog[id1])
                assert mm1 == mc2.s.mean(), "m1 is incorrect"
                assert mm2 == mc1.s.mean(), "m2 is incorrect"
                assert me1 == mc2.H(mc2.s), "energy exchange fails in 1"
                assert me2 == mc1.H(mc1.s), "energy exchange fails in 2"

            assert mc1.energy == mc1.H(mc1.s), "mc1[%d] energy:%f != H: %f" % (id1, mc1.energy, mc1.H(mc1.s))
            assert mc2.energy == mc2.H(mc2.s), "mc2[%d] energy:%f != H: %f" % (id2, mc2.energy, mc2.H(mc2.s))

        return exlog

    def trace(self, iterations, reset=False):
        Es = []
        States = []
        exlogs = []
        if reset is True:
            for mc in self.MCs:
                mc.acccnt = 0

        for it in tqdm(range(iterations)):
            exl = self.mcexstep(isodd=bool(it % 2))
            exlogs.append(exl)
            Es.append([mc.energy for mc in self.MCs])
            States.append([mc.s for mc in self.MCs])

        exlogs = np.array(exlogs).reshape((iterations, self.nbeta))
        Es = np.array(Es).reshape((iterations, self.nbeta))
        States = np.array(States).reshape((iterations, self.nbeta, self.size))
        return {'exlogs': exlogs, 'Eslog': Es, 'slog': States}


if __name__ == '__main__':
    size = 128
    J0 = 1.0
    Jmat = J0 * (np.ones((size, size)) - np.eye(size))

    model = IsingModelEMC(size, J=Jmat)

    burn = model.trace(1000)
    mclog = model.trace(1000)

    np.savez('burnlog.npz', Betas=model.betas, exlogs=burn['exlogs'], Eslog=burn['Eslog'], Slog=burn['slog'])
    np.savez('mclog.npz', Betas=model.betas, exlogs=mclog['exlogs'], Eslog=mclog['Eslog'], Slog=mclog['slog'])

"""Ising 模型を EMC で"""
#
# -*- coding: utf-8 -*-
#

import numpy as np
from tqdm import tqdm

# 温度交換バージョンをつくってみよう


class IsingModelMC:
    '''Ising Model with single MC'''
    def __init__(self, sz, beta=1., state=None, h = None, J=None):
        self.size = sz
        if J is None:   # Ferro
            self.Jmat = np.ones((size, size)) - np.eye(size)
        else:
            self.Jmat = J
        if h is None: # no bias
            self.h = 0
        else:
            self.h = h   # bias　はこのあと考えてないので要対応       

        self.beta = beta
        if state is None:
            self.s = np.random.binomial(1, 0.5, size=size) * 2 - 1.
        else:
            self.s = state

        self.energy = self.H(self.s)
        self.acccnt = 0

    def H(self, value):
        '''Calculate Energy'''
        return - 0.5 * value.dot(self.Jmat.dot(value)) / self.size

    def mcstep(self, value=None):
        '''update one MC step'''
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
            if rvals[idx] < pdelta:  # 'accept'
                # print('accept')
                self.energy = newE
                self.acccnt += 1
            else:  # 'reject' restore
                # print('reject')
                self.s[idx] *= -1

    def trace(self, iter, reset=False):
        '''interation multi MC step'''
        Es = []
        States = []
        if reset is True:
            self.acccnt = 0

        for it in tqdm(range(iter)):
            self.mcstep()
            Es.append(self.energy)
            States.append(np.array(self.s))

        self.logs = {'energy': np.array(Es), 'state': np.array(States)}
        return self.logs


class IsingModelEMC:
    '''Ising model with Exchange MC'''
    def __init__(self, sz, betas=None, state=None, J=None):
        self.size = sz
        if betas is None:
            self.nbeta = 24
            self.betas = [pow(1.25, l-16+1) for l in range(self.nbeta)]   # 決め打ち
            self.betas[0] = 0.
        self.MCs = [IsingModelMC(size, beta=beta, state=state, J=J) for beta in self.betas]
        # betas が偶数なとき
        self.evnset = [(i, i+1, self.MCs[i], self.MCs[i+1]) for i in range(0, self.nbeta-1, 2)]
        self.oddset = [(i, i+1, self.MCs[i], self.MCs[i+1]) for i in range(1, self.nbeta-1, 2)]

    def mcexstep(self, isodd=False):
        '''MC step with EX process'''
        for mc in self.MCs:
            mc.mcstep()

        # exchange process
        if isodd:
            exset = self.oddset
        else:
            exset = self.evnset

        exlog = np.arange(self.nbeta)

        rvals = np.random.uniform(0., 1., len(exset))
        for (rval, (id1, id2, mc1, mc2)) in zip(rvals, exset):
            r = np.exp((mc2.beta - mc1.beta) * (mc2.energy - mc1.energy))
            if rval <= r:  # accept exchange
                (mc1.s, mc2.s) = (mc2.s, mc1.s)
                (mc1.energy, mc2.energy) = (mc2.energy, mc1.energy)
                (exlog[id1], exlog[id2]) = (exlog[id2], exlog[id1])

        return exlog

    def trace(self, iterations, reset=False):
        '''Multi MC-EX steps'''
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
            States.append(np.array([mc.s for mc in self.MCs]))

        exlogs = np.array(exlogs).reshape((iterations, self.nbeta))
        Es = np.array(Es).reshape((iterations, self.nbeta))
        States = np.array(States).reshape((iterations, self.nbeta, self.size))

        return {'Exlog': exlogs, 'Elog': Es, 'Slog': States}


if __name__ == '__main__':
    size = 128
    J0 = 1.0
    Jmat = J0 * (np.ones((size, size)) - np.eye(size))

    model = IsingModelEMC(size, J=Jmat)

    burn = model.trace(5000)
    mclog = model.trace(5000)

#    np.savez('burnlog.npz', Betas=model.betas, Exlog=burn['Exlog'], Elog=burn['Elog'], Slog=burn['Slog'])
#    np.savez('mclog.npz', Betas=model.betas, Exlog=mclog['Exlog'], Elog=mclog['Elog'], Slog=mclog['Slog'])

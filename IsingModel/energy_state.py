#
# -*- coding: utf-8 -*-
#

import numpy as np
from tqdm import tqdm

# 全探索バージョン


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

    def trace(self, iter, reset=False):
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


if __name__ == "__main__":
    size = 24
    J0 = 1.0
    Jmat = J0 * (np.ones((size, size)) - np.eye(size))

    model = IsingModelMC(size, J=Jmat)

    Es = []
    for n in tqdm(range(2**size)):
        s1 = np.array([i for i in bin(n)[2:]], dtype=np.float)
        s0 = np.zeros((size - s1.shape[0],))
        s = np.hstack((s0, s1))

        Es.append(model.H(s))

    np.savez('Ising2ExSearch.npz', Es=np.array(Es))

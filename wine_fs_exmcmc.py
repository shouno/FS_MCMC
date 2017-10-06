'''Feature selection for Wine Dataset'''
#
# -*- coding: utf-8 -*-
#

import numpy as np
from numpy.random import binomial, uniform, permutation
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.datasets import load_wine
# from joblib import Parallel, delayed
# from numba import jit


# @jit(nopython=True, cache=True)
def CVError(sval, X, y):
        '''Definition of 'Energy function', that is a CV err
        input sval, dset
        output cverr
        '''
        nsplits = 5
        skf = StratifiedKFold(n_splits=nsplits)
        clsf = LinearSVC(C=1.0)
        cidx = np.array(sval+1, np.bool)    # sval が ±1 で構成されていることを課程
        scrs = []

        if cidx.sum() == 0:
            return 0.5

        for trn, tst in skf.split(X, y):
            clsf.fit(X[trn][:, cidx], y[trn])
            pred = clsf.predict(X[tst][:, cidx])
            scrs.append(1. - np.sum(y[tst] == pred) / (y[tst].shape[0]))
        scrs = np.array(scrs)
        return scrs.mean()


# @jit(nopython=True, cache=True)
def MCstep(sval, energy, beta, X, y):
        '''single MC step for trial value
        input sval, energy, beta, dset
        output sval, energy, accept count
        '''
        size = sval.shape[0]
        acccnt = 0
        if beta == 0.:  # 温度∞ (beta=0.0) は全とっかえ
            sval = binomial(1, 0.5, size=size) * 2 - 1.
            energy = size * CVError(sval, X, y)
            acccnt = size
            return sval, energy, acccnt
        # 有限温度の場合
        order = permutation(size)
        rvals = uniform(0., 1., size)

        for idx in order:
            oldE = energy
            sval[idx] *= -1
            newE = size * CVError(sval, X, y)
            delta = newE - oldE
            pdelta = np.exp(-beta * delta)

            if rvals[idx] < pdelta:
                # 'accept' the state state
                energy = newE
                acccnt += 1
            else:
                # 'reject' restore
                sval[idx] *= -1
        return sval, energy, acccnt


class FSsingleMC:
    '''Feature Selection with Sing MC'''
    def __init__(self, dset, beta=1., nsplits=5, clsf=None):
        # 識別データセット
        self.X = dset['X']
        self.y = dset['y']
        self.size = self.X.shape[1]   # データセットは 列方向に疎性

        self.beta = beta
        self.nsplits = nsplits
        self.s = binomial(1, 0.5, size=self.size) * 2 - 1.

        # 識別器
        if clsf is None:
            self.clsf = LinearSVC(C=1.0)
        else:
            self.clsf = clsf

        # エネルギー関数
        self.energy = self.H(self.s)
        self.acccnt = 0

    # @jit(cache=True)
    def H(self, sval=None):
        '''Definition of 'Energy function', that is a CV err'''
        skf = StratifiedKFold(n_splits=self.nsplits)
        if sval is None:
            sval = self.s
        cidx = np.array(sval+1, np.bool)    # sval が ±1 で構成されていることを課程
        scrs = []
        for trn, tst in skf.split(self.X, self.y):
            self.clsf.fit(self.X[trn][:, cidx], self.y[trn])
            pred = self.clsf.predict(self.X[tst][:, cidx])
            scrs.append(np.sum(self.y[tst] == pred) / (self.y[tst].shape[0]))
        scrs = np.array(scrs)
        return scrs.mean()

    # @jit(cache=True)
    def mcstep(self, value=None):
        '''single MC step for trial value'''
        if self.beta == 0.:  # 温度∞ (beta=0.0) は全とっかえ
            self.s = binomial(1, 0.5, size=self.size) * 2 - 1.
            self.energy = self.H(self.s)
            self.acccnt += self.size
            return
        # 有限温度の場合
        if value is None:
            value = self.s
        order = permutation(self.size)
        rvals = uniform(0., 1., self.size)

        for idx in order:
            oldE = self.energy
            self.s[idx] *= -1
            newE = self.H(self.s)
            delta = newE - oldE
            pdelta = np.exp(-self.beta * delta)

            if rvals[idx] < pdelta:
                # 'accept' the state state
                self.energy = newE
                self.acccnt += 1
            else:
                # 'reject' restore
                self.s[idx] *= -1


class FeatureSelectionEMC:
    '''Feature Selextion using EMC'''
    def __init__(self, dset, betas=None, nsplits=5, clsf=None):
        self.size = dset['X'].shape[1]
        if betas is None:
            self.nbeta = 12
            self.betas = [pow(1.25, l-11+1) for l in range(self.nbeta)]   # 決め打ち
            self.betas[0] = 0.
            self.MCs = [FSsingleMC(dset, beta=beta, nsplits=nsplits, clsf=clsf)
                        for beta in self.betas]
        # betas が偶数なとき
        self.evnset = [(i, i+1, self.MCs[i], self.MCs[i+1]) for i in range(0, self.nbeta-1, 2)]
        self.oddset = [(i, i+1, self.MCs[i], self.MCs[i+1]) for i in range(1, self.nbeta-1, 2)]

    # @jit(cache=True)
    def mcexstep(self, isodd=False):
        '''A exchange MC step'''
        for mc in self.MCs:
            #   mc.mcstep()
            mc.s, mc.energy, mc.acccnt = MCstep(mc.s, mc.energy, mc.beta, mc.X, mc.y)

        # exchange process
        if isodd:
            exset = self.oddset
        else:
            exset = self.evnset

        exlog = np.arange(self.nbeta)

        rvals = uniform(0., 1., len(exset))
        for (rval, (id1, id2, mc1, mc2)) in zip(rvals, exset):
            r = np.exp((mc2.beta - mc1.beta) * (mc2.energy - mc1.energy))
            if rval <= r:  # accept exchange
                (mc1.s, mc2.s) = (mc2.s, mc1.s)
                (mc1.energy, mc2.energy) = (mc2.energy, mc1.energy)
                (exlog[id1], exlog[id2]) = (exlog[id2], exlog[id1])

        return exlog

    def trace(self, iterations, reset=False):
        '''multiple exmc method for iteration times'''
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

    wine = load_wine()
    X = wine['data']
    y = wine['target']

    XX = (X - X.mean(axis=0))/(X.std(axis=0))
    yy = np.array(wine['target'] == 0, dtype=np.float)
    model = FeatureSelectionEMC(dset={'X': XX, 'y': yy})

    burn = model.trace(100)
#   mclog = model.trace(5000)

#   np.savez('burnlog.npz', Betas=model.betas,
#   Exlog=burn['Exlog'], Elog=burn['Elog'], Slog=burn['Slog'])
#   np.savez('mclog.npz', Betas=model.betas,
#   Exlog=mclog['Exlog'], Elog=mclog['Elog'], Slog=mclog['Slog'])

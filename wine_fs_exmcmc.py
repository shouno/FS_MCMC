#
# -*- coding: utf-8 -*-
#
'''Feature selection for Wine Dataset'''

import numpy as np
from numpy.random import binomial, uniform, permutation
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.datasets import load_wine


# クラス内においておくと激オソなので外に引っ張り出しておく
def CVScore(sval, X, y, clsf, skf):
    '''Definition of 'Energy function', that is a CV score'''
    cidx = (sval == 1.0)   # 変数選択リストの作成
    scrs = []

    if cidx.sum() == 0:
        return 0.5

    for trn, tst in skf.split(X, y):
        clsf.fit(X[trn][:, cidx], y[trn])
        pred = clsf.predict(X[tst][:, cidx])
        scrs.append(- np.sum(y[tst] == pred) / (y[tst].shape[0]))

    return np.array(scrs).mean()


# @jit(nopython=True, cache=True)
def MCstep(sval, energy, beta, X, y, clsf, skf):
    '''single MC step for trial value'''

    size = sval.shape[0]
    acccnt = 0
    if beta == 0.:  # 温度∞ (beta=0.0) は全とっかえ
        sval = binomial(1, 0.5, size=size) * 2 - 1.
        energy = size * CVScore(sval, X, y, clsf, skf)
        acccnt = size
        return energy, acccnt

    # 有限温度の場合
    order = permutation(size)
    rvals = uniform(0., 1., size)

    for idx in order:
        oldE = energy
        sval[idx] *= -1
        newE = size * CVScore(sval, X, y, clsf, skf)
        delta = newE - oldE
        pdelta = np.exp(-beta * delta)

        if rvals[idx] < pdelta:
            # 'accept' the state state
            energy = newE
            acccnt += 1
        else:
            # 'reject' restore
            sval[idx] *= -1

    return energy, acccnt


class FSsingleMC:
    '''Feature Selection with Sing MC'''
    def __init__(self, dset, beta=1., nsplits=5):
        # 識別データセット
        self.X = dset['X']
        self.y = dset['y']
        self.size = self.X.shape[1]   # データセットは 列方向に疎性

        self.beta = beta
        self.nsplits = nsplits
        self.s = binomial(1, 0.5, size=self.size) * 2 - 1.

        # 識別器とCV用のクラスを内包しておく
        self.clsf = LinearSVC(C=1.0)
        self.skf = StratifiedKFold(n_splits=nsplits)

        # エネルギー関数
        self.energy = self.size * CVScore(self.s, self.X, self.y, self.clsf, self.skf)
        self.acccnt = 0


class FeatureSelectionEMC:
    '''Feature Selextion using EMC'''
    def __init__(self, dset, betas=None, nsplits=5, clsf=None):
        self.size = dset['X'].shape[1]
        if betas is None:
            self.nbeta = 12
            self.betas = [pow(1.5, l-7+1) for l in range(self.nbeta)]   # 決め打ち
            self.betas[0] = 0.
            self.MCs = [FSsingleMC(dset, beta=beta, nsplits=nsplits) for beta in self.betas]

            self.evnset = [(i, i+1, self.MCs[i], self.MCs[i+1]) for i in range(0, self.nbeta-1, 2)]
            self.oddset = [(i, i+1, self.MCs[i], self.MCs[i+1]) for i in range(1, self.nbeta-1, 2)]

    # @jit(cache=True)
    def mcexstep(self, isodd=False):
        '''A exchange MC step'''
        for mc in self.MCs:
            mc.energy, dummy = MCstep(mc.s, mc.energy, mc.beta, mc.X, mc.y, mc.clsf, mc.skf)
            mc.acccnt += dummy
            # r = [MCstep(mc.s, mc.energy, mc.beta, mc.X, mc.y) for mc in self.MCs]

        exlog = np.arange(self.nbeta)

        # exchange process
        if isodd:
            exset = self.oddset
        else:
            exset = self.evnset

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
        AccRate = np.array([mc.acccnt/(self.size*iterations) for mc in self.MCs])

        return {'Exlog': exlogs, 'Elog': Es, 'Slog': States, 'AccRate': AccRate}


if __name__ == '__main__':
    wine = load_wine()
    X = wine['data']
    y = wine['target']

    XX = (X - X.mean(axis=0))/(X.std(axis=0))
    yy = np.array(wine['target'] == 0, dtype=np.float)
    model = FeatureSelectionEMC(dset={'X': XX, 'y': yy})
    burn = model.trace(1000)
    mclog = model.trace(1000, reset=True)

    np.savez('burnlogB15_12_1000.npz', Betas=model.betas,
             Exlog=burn['Exlog'], Elog=burn['Elog'], Slog=burn['Slog'],
             AccRate=burn['AccRate'])
    np.savez('mclogB15_12_1000.npz', Betas=model.betas,
             Exlog=mclog['Exlog'], Elog=mclog['Elog'], Slog=mclog['Slog'],
             AccRate=mclog['AccRate'])

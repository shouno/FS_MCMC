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
from numba import jit


# クラス内においておくと激オソなので外に引っ張り出しておく
def CVScore(sval, X, y):
        '''Definition of 'Energy function', that is a CV score
        input sval, dset
        output average of minus cvscore
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
            scrs.append(- np.sum(y[tst] == pred) / (y[tst].shape[0]))
        scrs = np.array(scrs)
        return scrs.mean()


@jit(cache=True)
def MCstep(sval, energy, beta, X, y):
        '''single MC step for trial value
        input sval, energy, beta, dset
        output sval, energy, accept count
        '''
        size = sval.shape[0]
        acccnt = 0
        if beta == 0.:  # 温度∞ (beta=0.0) は全とっかえ
            sval = binomial(1, 0.5, size=size) * 2 - 1.
            energy = size * CVScore(sval, X, y)
            acccnt = size
            return energy, acccnt
        # 有限温度の場合
        order = permutation(size)
        rvals = uniform(0., 1., size)

        for idx in order:
            oldE = energy
            sval[idx] *= -1
            newE = size * CVScore(sval, X, y)
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
    def __init__(self, dset, beta=1., nsplits=5, clsf=None):
        # 識別データセット
        self.X = dset['X']
        self.y = dset['y']
        self.size = self.X.shape[1]   # データセットは 列方向に疎性

        self.beta = beta
        self.nsplits = nsplits
        self.s = binomial(1, 0.5, size=self.size) * 2 - 1.

        # エネルギー関数
        self.energy = CVScore(self.s, self.X, self.y)
        self.acccnt = 0


if __name__ == '__main__':

    wine = load_wine()
    X = wine['data']
    y = wine['target']
    size = 13
    XX = (X - X.mean(axis=0))/(X.std(axis=0))
    yy = np.array(wine['target'] == 0, dtype=np.float)

    s = binomial(1, 0.5, size=size) * 2 - 1.
    energy = CVScore(s, XX, yy)
    burnEs = []
    burnEs.append(energy)
    acccnt = 0
    beta = 0.0

    for n in tqdm(range(5000)):
        energy, dummy = MCstep(s, energy, beta, X, y)
        burnEs.append(energy)
        acccnt += dummy

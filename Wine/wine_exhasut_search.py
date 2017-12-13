'''Feature selection for Wine Dataset'''
#
# -*- coding: utf-8 -*-
#

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.datasets import load_wine
# from joblib import Parallel, delayed
# from numba import jit, f8


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

    return np.array(scrs).mean()


if __name__ == '__main__':

    wine = load_wine()
    X = wine['data']
    y = wine['target']
    XX = (X - X.mean(axis=0))/(X.std(axis=0))
    yy = np.array(wine['target'] == 0, dtype=np.float)
    size = 13

    Es = []
    for n in tqdm(range(1, 2**size)):
        s1 = np.array([i for i in bin(n)[2:]], dtype=np.float)
        s0 = np.zeros((size - s1.shape[0],))
        s = np.hstack((s0, s1)) * 2 - 1
        scr = CVScore(s, XX, yy)
        Es.append(scr)

    np.savez('WineExsearch.npz', Es=np.array(Es))

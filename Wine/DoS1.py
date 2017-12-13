#
# -*- coding: utf-8 -*-
#

import numpy as np
import matplotlib.pylab as plt

size = 13   # dim.size of features in 'Wine' (see, wine_fs_exmcmc.py)

ESlogfile = 'WineExsearch.npz'
AESlogfile = 'mclogB15_12_1000.npz'

truees = np.load(ESlogfile)['Es']
aeslog = np.load(AESlogfile)
mces = aeslog['Elog'] / size   # in MCMC calculation, we consider a size effect
mcbetas = aeslog['Betas']

# Here we assume: 
#    truees shape is (#candidates,)
#    mces shape is (#iteations of MC, #MCs)


# setup of histogram
hmin, hmax = -1.0, -0.5
hnums = 40
htics = np.linspace(hmin, hmax, num=hnums+1, endpoint=True)

# number of iterations
mcitrs = mces.shape[0]

eshist = np.histogram(truees, bins=htics)[0]
Hbetas = np.array([np.histogram(mces[:, i], bins=htics)[0] for i in range(len(mcbetas))])

plt.clf()
ext = (hmin, hmax, mcbetas[0], mcbetas[-1])
plt.imshow(Hbetas, extent=ext, aspect=0.01)
plt.xlabel('Energy(negative CVscore)')
plt.ylabel('Betas')
plt.show()


def getDensity(Hs, Zs, Es, betas, N):
    '''Calculation of density for enery tics
    Hs: 2d array for Energies (#betas, #energytics)
    Zs: array of (estimated) partition function for betas
    Es: array of Energy tics
    betas: array of inverse temps.
    N: samples for histograms of H
    '''

    numerator = np.sum(Hs, axis=0)
    print(numerator)
    denominator = np.array([N * np.exp(-b * Es) / Z for (b, Z) in zip(betas, Zs)]).sum(axis=0)
    return numerator/denominator


def getPartition(G, Es, betas):
    '''Calculation of Parition for betas
    input
    G: array of density with Energy Es
    Es: array of Energy tics
    betas: array of inverse temps.
    output
    partition function array Z of betas
    '''
    return np.array([np.sum(G * np.exp(-b * Es)) for b in betas])




Densities = np.ones(hnums) / hnums * mcitrs    # Densities G(E)
EngTics = (htics[:-1] + htics[1:])/2.          # エネルギー軸の刻み E, ヒストグラムの中央点

plt.figure()

for it in range(10):
    plt.plot(EngTics, Densities)
    Partitions = getPartition(Densities, EngTics, mcbetas)
    Densities = getDensity(Hbetas, Partitions, EngTics, mcbetas, mcitrs)

plt.show()

import numpy as np

import ot 
import sklearn.preprocessing

import itertools as it
from util import unique_rows

def metric_bases(distance_matrix, k):
    '''list all sets of k indices that result in unique representations when projecting.
    That is, if k is the metric dimension of the graph whose distance matrix we are looking
    at, the return will be the set of all metric bases of the graph.
    If k is smaller than the metric dimension, the returned set will be empty.
    '''

    bases = list()
    for idx in it.combinations(range(distance_matrix.shape[0]), k):
        projection = distance_matrix[:,idx]
        unique, _ = unique_rows(projection)
        if unique.shape[0] == distance_matrix.shape[0]:
            bases.append(idx)

    return bases


def kernel_dist(gram):
    ''' return the distance matrix corresponding to the gram matrix '''
    dg = np.diag(gram).reshape([gram.shape[0], 1])
    return np.sqrt(dg - 2 * gram + dg.T)


def gromov_wasserstein(C1, C2, verbose=False):

    n_samples = C1.shape[0]
    if (C2.shape[0] != n_samples) or (C2.shape[1] != n_samples) or (C1.shape[1] != n_samples):
        raise ValueError(f'Incompatible shapes: {C1.shape} and {C2.shape}')

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    gw0, log0 = ot.gromov.gromov_wasserstein(
        C1, C2, p, q, 'square_loss', verbose=verbose, log=True)

    # gw, log = ot.gromov.entropic_gromov_wasserstein(
    #     C1, C2, p, q, 'square_loss', epsilon=5e-4, log=True, verbose=True)


    # print('Gromov-Wasserstein distances: ' + str(log0['gw_dist']))
    # print('Entropic Gromov-Wasserstein distances: ' + str(log['gw_dist']))


    # pl.figure(1, (10, 5))

    # pl.subplot(1, 2, 1)
    # pl.imshow(gw0, cmap='jet')
    # pl.title('Gromov Wasserstein')

    # pl.subplot(1, 2, 2)
    # pl.imshow(gw, cmap='jet')
    # pl.title('Entropic Gromov Wasserstein')

    # pl.show()

    return log0['gw_dist']


def compute_meta_distance_matrix(f, mlist):
    ''' compute any *symmetric* distance function between anything.
    note that f(x,x) = 0 and f(x,y) = f(y,x) is assumed

    TODO should be vectorized, if possible '''
    meta_dist = np.zeros([len(mlist), len(mlist)])
    for i, a in enumerate(mlist):
        for j, b in enumerate(mlist):
            if j < i:
                meta_dist[i,j] = f(a,b)
            else:
                break
    
    return meta_dist + meta_dist.T




if __name__ == '__main__':
    '''test stuff'''

    d1 = np.loadtxt('testdata/distmatrix_tree_3.txt')
    d2 = np.loadtxt('testdata/distmatrix_tree_4.txt')

    print(metric_bases(d1, 2))
    print(metric_bases(d2, 2))

    a = np.array([[1,0,1],[0,1,0], [1,0,1]])

    print(a)
    print(kernel_dist(a))

    print(gromov_wasserstein(a, a, verbose=True))
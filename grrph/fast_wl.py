import numpy as np
import scipy.sparse as sparse
import torch
# from torch_sparse import SparseTensor
from torch_geometric.nn import MessagePassing

from importlib import resources as impresources
from . import data


# Weisfeiler-Lehman functions by Till
class WeisfeilerLemanPython:
    conc2lbl = {}
    lbl_counter = 0

    def __init__(self, first_lbl_idx):
        self.lbl_counter = first_lbl_idx
        self.conc2lbl = {}

    # assigns each (vlabel, neighborlabels) a unique label
    def getLbl(self, v_lbl, neigh_lbls):
        neigh_lbls.sort()
        conc = (v_lbl, tuple(neigh_lbls))
        if conc not in self.conc2lbl:
            self.conc2lbl[conc] = self.lbl_counter
            self.lbl_counter += 1
        return self.conc2lbl[conc]


inp_file = (impresources.files(data) / 'logprimes1.npy')
primes = np.load(inp_file)


def compress(labels: np.array):
    '''Weisfeiler Leman Label compression'''
    _, inv = np.unique(labels, return_inverse=True)
    # use the indices to uniq array to select first |uniq| primes
    labels = primes[inv]
    return labels, inv


def wl_direct_igraph(g, n_iter=5):
    '''A Weisfeiler Leman implementation in igraph.

    input: an igraph Graph g and a number of iterations
    output: numpy array giving wl labels encoded as logs of prime numbers
    '''

    import igraph as ig

    oldlbl = primes[0] * np.ones(len(g.vs))
    newlbl = oldlbl # only relevant for n_iter=0

    for i in range(n_iter):
        newlbl = np.pi * oldlbl
        for v in g.vs:
            for w in g.neighbors(v):
                newlbl[v.index] += oldlbl[w] 
        if i < n_iter-1:
            oldlbl, _ = compress(newlbl)

    return newlbl


def wl_direct_scipysparse(a: sparse.csr_matrix, n_iter=5):
    '''A Weisfeiler Leman implementation in scipy sparse.

    input: an adjacency matrix a as scipy.sparse.csr_matrix and a number of iterations
    output: numpy array giving wl labels encoded as logs of prime numbers
    '''

    oldlbl = primes[0] * np.ones(a.shape[0])
    newlbl = oldlbl # only relevant for n_iter=0

    for i in range(n_iter):
        newlbl = np.pi * oldlbl + a @ oldlbl

        if i < n_iter-1:
            oldlbl, _ = compress(newlbl)

    return newlbl


def wllt_direct_scipysparse(a: sparse.csr_matrix, n_iter=5):
    '''An implementation of the Weisfeiler Leman algorithm that allows to 
    reconstruct the labeling tree. 
    That is, for each label, 
    
    input: an adjacency matrix a as scipy.sparse.csr_matrix and a number of iterations
    output: newlbl, newparents
        newlbl: numpy array giving wl labels encoded as logs of prime numbers
        newparents: numpy array giving the indices of parent label in the previous iteration
        '''
    
    oldlbl = primes[0] * np.ones(len(g.vs))
    oldids = np.ones(len(g.vs)) # all labels have the same parent
    
    newlbl = oldlbl # only relevant for n_iter=0
    newparents = oldids # only relevant for n_iter=0

    for i in range(n_iter):
        newlbl = np.pi * oldlbl + a @ oldlbl
        newparents = oldids

        if i < n_iter-1:
            oldlbl, oldids = compress_inv(newlbl)

    return newlbl, newparents


def wl_direct_scipysparse_nocomp(a: sparse.csr_matrix, n_iter=5):
    '''A 'Weisfeiler Leman' implementation in scipy sparse.

    Note that this version skips the compression step of hashing values to log primes.
    This makes this message passing scheme not guaranteed to be injective.

    input: an adjacency matrix a as scipy.sparse.csr_matrix and a number of iterations
    output: numpy array giving wl labels encoded as logs of prime numbers
    '''

    oldlbl = primes[0] * np.ones(len(g.vs))
    newlbl = oldlbl # only relevant for n_iter=0

    for i in range(n_iter):
        newlbl = np.pi * oldlbl + a @ oldlbl

        if i < n_iter-1:
            oldlbl = np.copy(newlbl)

    return newlbl


# class FastWLConv(MessagePassing):
#     '''A Weisfeiler Leman implementation as pytorch geometric message passing layer.
#     '''

#     def __init__(self):
#         self.primes = torch.from_numpy(primes)
#         self.hash = None

#     def compress(self, labels: np.array):
#         '''Weisfeiler Leman Label compression'''
#         _, self.hash = torch.unique(labels, return_inverse=True)
#         # use the indices to uniq array to select first |uniq| primes
#         labels = self.primes[self.hash]
#         return labels

#     @torch.no_grad()
#     def forward(self, x, edge_index):
    
#         adj_t = SparseTensor(row=edge_index[1], col=edge_index[0],
#                                     sparse_sizes=(x.size(0), x.size(0)))
#         x = self.compress(x)
#         x = np.pi * x + adj_t @ x

#         return x



def speed_test():
    import igraph as ig

    with open('twitch.pickle', 'rb') as f:
        g = ig.Graph.Read_Pickle(f)

        a = g.get_adjacency_sparse()

        print(a.shape)

        vlist = np.array([ 42, 123, 11024, 11585, 12280, 34117])

        for k in [10]:
            rk1 = wl_direct_igraph(g, n_iter=k)
            print('igraph unique labels', np.unique(rk1).shape)

            rk2 = wl_direct_scipysparse(a, n_iter=k)
            print('scipy unique labels', np.unique(rk2).shape)

            rk3 = wl_direct_scipysparse_nocomp(a, n_iter=k)
            print('scipy nocompress unique labels', np.unique(rk3).shape)

        # setup Weisfeiler-Lehman relabeling function
        WL = WeisfeilerLehman(1)

        # assign iteration 0 labels, i.e., initial vertex labels
        g.vs['label_0'] = 0

        # relabeling procedure
        for i in range(10):  # h is the number of WL iterations
            labels = np.array(g.vs['label_'+str(i)])
            
            # for each vertex, extract its node label and the node labels of all of its neighbors
            for v in g.vs:
                v_lbl = labels[v.index]
                neigh_lbls = labels[g.neighbors(v)]

                # map the pair (vlabel, neighborlabels) to a new label using an injective function
                new_lbl = WL.getLbl(v_lbl, neigh_lbls)
                g.vs[v.index]['label_'+str(i+1)] = new_lbl
        
        print('wl unique labels', np.unique(g.vs['label_10']).shape)


def convert_primes(infile='primes1.txt', intermediatefile='primes1_clean.txt', outfile='logprimes1.npy'):
    with open(infile, 'r') as f:
        with open(intermediatefile, 'w') as w:
            for line in f:
                if line != '\n':
                    w.write(line)

    logprimes = np.log2(np.loadtxt(intermediatefile, skiprows=1).flatten())
    print(logprimes.shape)
    np.save(outfile, logprimes)


if __name__ == '__main__':
    convert_primes()
    speed_test()
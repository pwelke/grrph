import scipy.sparse
import numpy as np


def disjoint_union(list_o_graphs):
    union = scipy.sparse.block_diag(list_o_graphs)
    indicators = [graph.shape[0] for graph in list_o_graphs]
    return union, indicators


def disjoint_split(union, indicator):
    list_o_graphs = list()
    for i,j in zip(indicator, indicator[1:]):
        graph = union[i:j, i:j]
        list_o_graphs.append(graph)
    return list_o_graphs


def is_symmetric(adj):
    if not scipy.sparse.isspmatrix_csr(adj):
        adj = scipy.sparse.csr_matrix(adj)
    adjt = adj.T.tocsr()
    return len(adj.data) == len(adjt.data) and np.alltrue(adj.data == adjt.data) and np.alltrue(adj.indptr == adjt.indptr) and np.alltrue(adj.indices == adjt.indices)


def unique_rows(X):
    '''returns the number and set of unique rows in X and the index to the first representative of each unique row in X'''
    return np.unique(X, axis=0, return_inverse=True, ) 
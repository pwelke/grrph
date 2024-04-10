import scipy.sparse as sp
import numpy as np
from collections import Counter


class InductiveWL:
    
    def __init__(self, n_iter=5):
        self.n_iter = n_iter
        self.labeldict = [dict() for _ in range(n_iter + 1)]
        self.labels = [None for _ in range(n_iter + 1)]
        self.compressed_labels = [None for _ in range(n_iter + 1)]
        
        # the first n_iter labels will be kept for the unseen node labels at these iterations
        self.unseendict = {i: i for i in range(n_iter + 1)}
        self.current_hash = n_iter


    def fit_hash(self, label, iteration):
        if label in self.labeldict[iteration]:
            return self.labeldict[iteration][label]
        else:
            self.current_hash += 1
            self.labeldict[iteration][label] = self.current_hash
            return self.current_hash


    def transform_hash(self, label, iteration):
        if label in self.labeldict[iteration]:
            return self.labeldict[iteration][label]
        else:
            return self.unseendict[iteration]


    def node_label(self, v, label, adj: sp.csr_matrix):
        neighborlabels = [label[i] for i in adj.indices[adj.indptr[v]:adj.indptr[v+1]]]
        sorted(neighborlabels)
        return tuple([label[v]] + neighborlabels)


    def wl_relabel(self, adj, node_labels=None, hash_function=None):

        if not sp.isspmatrix_csr(adj):
            adj = sp.csr_matrix(adj)

        if node_labels is None:
            node_labels = np.ones(adj.shape[0])
        
        self.labels[0] = node_labels
        self.compressed_labels[0] = [hash_function(label, 0) for label in node_labels]

        for i in range(self.n_iter):
            self.labels[i+1] = [self.node_label(v, self.compressed_labels[i], adj) for v in range(adj.shape[0])]
            self.compressed_labels[i+1] = [hash_function(l, i+1) for l in self.labels[i+1]]

        return self.compressed_labels[-1]


    def fit(self, adj, node_labels=None):
        return self.wl_relabel(adj, node_labels=node_labels, hash_function=self.fit_hash)


    def transform(self, adj, node_labels=None):
        return self.wl_relabel(adj, node_labels=node_labels, hash_function=self.transform_hash)


    def fit_transform(self, adj, node_labels=None):
        return self.wl_relabel(adj, node_labels=node_labels, hash_function=self.fit_hash)


    def get_cumulative_histogram(self):

        histogram = Counter()

        for i in range(self.n_iter+1):
            histogram.update(compressed_labels[i])
        
        return histogram


            




            
    
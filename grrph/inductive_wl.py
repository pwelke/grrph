import scipy.sparse as sp
import numpy as np
from collections import Counter


class InductiveWL:
    '''An implementation of the Weisfeiler Leman (WL) algorithm that can be applied in inductive learning tasks.
    That is, the perfect hash function constructed during the iterations of WL on some graph G can be reused to 
    label nodes of a graph H afterwards. In case H contains WL labels that were not seen during 'fitting' on G, 
    two alternative ways of handling are possible.
    
        - Encode unseen labels explicitly as 'unseen': (use iwl.transform(G) )
            If a novel label is seen in H, we compress it to an artificial label which encodes 'unseen label at iteration i.
            Due to implementation, the first n_iter + 1 labels are reserved for these special labels.
            Benefits of this method include that the dimensionality of histogram representations of any transformed
            graphs stays constant. Drawbacks include that for higher iterations, there will likely be many unseen labels, 
            which may increase the similarity of transformed graphs H1 and H2 due to many shared 'unseen' labels. 

            Example:
                from scipy.sparse import csr_matrix
                import numpy as np
                from grrph.inductive_wl import InductiveWL

                G = csr_matrix(np.array([[0,1,1],[0,0,0],[0,0,0]]))
                H = csr_matrix(np.array([[0,1,1,0],[0,0,0,0],[0,0,0,0],[0,0,1,0]]))

                iwl = InductiveWL()

                print(iwl.fit(G))
                print(iwl.transform(H))

            Output:
            [15, 16, 16]
            [15, 16, 16, 5]
        
            This indicates that the last node in H got the compressed label 5. indicating that its 5-neighborhood was not encountered
            during fitting on G.

        - Continue assigning new labels to neighborhoods unseen during fitting on G. (use iwl.fit(H))
            This is identical to applying a transductive WL implementation on the disjoint union of G and H.
            It has the drawback that histogram dimensionality will likely change (increase) and you have to deal with the consequences.
            For example, the histograms of G and H will have different numbers of dimensions and are not straightforwardly compatible.

            Example:
                from scipy.sparse import csr_matrix
                import numpy as np
                from grrph.inductive_wl import InductiveWL

                G = csr_matrix(np.array([[0,1,1],[0,0,0],[0,0,0]]))
                H = csr_matrix(np.array([[0,1,1,0],[0,0,0,0],[0,0,0,0],[0,0,1,0]]))

                iwl = InductiveWL()

                print(iwl.fit(G))
                print(iwl.fit(H))

            Output:
                [15, 16, 16]
                [15, 16, 16, 21]

    '''
    
    def __init__(self, n_iter=5):
        # stores the number of WL iterations to apply
        self.n_iter = n_iter
        # for each iteration, we create a perfect hash function via a dictionary {neighborhood: compressed_label}
        self.labeldict = [dict() for _ in range(n_iter + 1)]
        # for each iteration, this stores the uncompressed labels of all nodes in the last graph processed via .fit or .transform
        self.labels = [None for _ in range(n_iter + 1)]        
        # for each iteration, this stores the compressed labels of all nodes in the last graph processed via .fit or .transform
        self.compressed_labels = [None for _ in range(n_iter + 1)]
        
        # the first n_iter labels will be kept for the unseen node labels at these iterations
        self.unseendict = {i: i for i in range(n_iter + 1)}
        self.current_hash = n_iter


    def fit_hash(self, label, iteration):
        '''Standard implementation of a perfect hash function for WL using a dictionary and a never decreasing current_hash counter'''
        if label in self.labeldict[iteration]:
            return self.labeldict[iteration][label]
        else:
            self.current_hash += 1
            self.labeldict[iteration][label] = self.current_hash
            return self.current_hash


    def transform_hash(self, label, iteration):
        '''Inductive variant of perfect hash function that returns 'not seen at iteration i' as hash value for unseen neighborhoods at iteration i'''
        if label in self.labeldict[iteration]:
            return self.labeldict[iteration][label]
        else:
            return self.unseendict[iteration]


    def node_label(self, v, label, adj: sp.csr_matrix):
        '''Neighborhood label sorting aggregation for WL. We sort the list of labels of the neighbors and prepend the label of the central node.'''
        neighborlabels = [label[i] for i in adj.indices[adj.indptr[v]:adj.indptr[v+1]]]
        sorted(neighborlabels)
        return tuple([label[v]] + neighborlabels)


    def wl_relabel(self, adj, node_labels=None, hash_function=None):
        '''Iterative color refinement algorithm using the sorting aggregation above and a hash function of choice'''

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
        '''Fit WL to graph given by sparse adjacency matrix adj and return the compressed label vector at the last iteration.
        
        Actual hash values depend on the order of the graph (i.e. the implementation is not permutation in/equivariant)

        While running this method, self.labeldict, self.labels, and self.compressed_labels are populated. 
        You may inspect these to obtain the representation of the current graph.
        '''
        return self.wl_relabel(adj, node_labels=node_labels, hash_function=self.fit_hash)


    def transform(self, adj, node_labels=None):
        '''Compute *known* WL labels of graph nodes given by sparse adjacency matrix adj and return 
        the compressed label vector at the last iteration. That is, if some neighborhood was not seen during an 
        earlier application of self.fit() it will be assigned one of n_iter artificial labels which encode
        'unseen/new label at iteration i'.
        
        Actual hash values depend on the previous calls to self.fit() (i.e. the implementation is not permutation in/equivariant)

        While running this method, self.labels and self.compressed_labels are populated, but self.labeldict *remains unchanged*. 
        You may inspect these to obtain the representation of the current graph.
        '''
        return self.wl_relabel(adj, node_labels=node_labels, hash_function=self.transform_hash)


    def fit_transform(self, adj, node_labels=None):
        '''same as self.fit()'''
        return self.wl_relabel(adj, node_labels=node_labels, hash_function=self.fit_hash)


    def get_cumulative_histogram(self):
        '''Quick and dirty way to obtain a histogram representation of the last graph processed by either .fit() or .transform()'''

        histogram = Counter()

        for i in range(self.n_iter+1):
            histogram.update(compressed_labels[i])
        
        return histogram

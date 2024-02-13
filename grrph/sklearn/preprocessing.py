import numpy as np
from sklearn.base import BaseEstimator,  TransformerMixin
import hashlib
import copy

class VertexLabelHasher(BaseEstimator, TransformerMixin):

    def __init__(self, feature_id_list=[0]):
        self.feature_id_list = feature_id_list

    def hashfct(self, x):        
        # x is a list of int values
        if x.ndim == 0:
            # if we have a single feature, return it as the hash
            return x
        else:
            # we need to hash different values to some uniqueish int values in a deterministic way
            # collisions are not great here, so we try to be smart, but slow
            bts = x.data.tobytes()
            hsh = int.from_bytes(hashlib.sha1(bts).digest(), 'little') & 0xFFFFFFFF
            # print(x, hsh) # if you wanna check if collisions occur and/or hashing works fine
            return hsh

    def fit(self, X, y=None):

        # we do nothing. hash function alrady exists.
        self.feature_id_list = np.array(self.feature_id_list)

        # TODO: we might wanna check if the ids are compatible to the features present in X

        # Return the classifier
        return self

    def transform(self, X):

        Xprime = list()
        for g in X:
            graph = copy.deepcopy(g)
            for v in graph.node_labels:
                a = np.array(graph.node_labels[v])[self.feature_id_list]
                graph.node_labels[v] = self.hashfct(a)
            Xprime.append(graph)

        return Xprime

    
import numpy as np 

class VotingRegressor(object):
    """ Implements a voting regressor for pre-trained regressors
    see https://stackoverflow.com/questions/42920148/using-sklearn-voting-ensemble-with-partial-fit
    """

    def __init__(self, estimators):
        self.estimators = estimators

    def predict(self, X):
        # get values
        Y = np.zeros([X.shape[0], len(self.estimators)], dtype=int)
        for i, reg in enumerate(self.estimators):
            Y[:, i] = reg.predict(X)
        # apply voting 
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y = np.mean(Y, axis=1)
        return y

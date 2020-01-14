import numpy as np
from scipy.stats import mode


class KNearestNeighbor(object):

    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))

        return dists

    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i] = np.sqrt(np.sum((X[i, :] - self.X_train) ** 2, axis=1))
        return dists

    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        dists = np.dot(X, self.X_train.T) * -2
        sq1 = np.sum(np.square(X), axis=1, keepdims=True)
        sq2 = np.sum(np.square(self.X_train), axis=1)

        dists = np.add(dists, sq1)
        dists = np.add(dists, sq2)
        dists = np.sqrt(dists)

        return dists

    def predict_labels(self, dists, k=1):

        y_pred = mode(self.y_train[np.argsort(dists)[:, :k]], axis=1)[0]

        return y_pred.ravel()

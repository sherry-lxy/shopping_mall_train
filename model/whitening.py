import numpy as np


class PCAWhitening:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self.mean = None
        self.eigenvalue = None
        self.eigenvector = None
        self.pca = None

    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        x_ = x - self.mean
        cov = np.dot(x_.T, x_) / x_.shape[0]
        E, D, _ = np.linalg.svd(cov)
        D = np.sqrt(D) + self.epsilon
        self.eigenvalue = D
        self.eigenvector = E
        self.pca = np.dot(np.diag(1.0 / D), E.T)
        return self

    def transform(self, x):
        x_ = x - self.mean
        return np.dot(x_, self.pca.T)

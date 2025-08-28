import numpy as np
import itertools

class LinearSVM:
    def __init__(self, C=1.0, lr=0.001, epochs=1000):
        self.C = C
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y == 1, 1, -1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            for i in range(n_samples):
                condition = y_[i] * (np.dot(X[i], self.w) + self.b)
                if condition >= 1:
                    self.w -= self.lr * (2 * 1e-4 * self.w)
                else:
                    self.w -= self.lr * (2 * 1e-4 * self.w - self.C * y_[i] * X[i])
                    self.b -= self.lr * (-self.C * y_[i])

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)


class MulticlassSVM_OVO:
    def __init__(self, C=1.0, lr=0.001, epochs=1000):
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.classifiers = {}

    def fit(self, X, y):
        self.labels = sorted(np.unique(y))
        for (i, j) in itertools.combinations(self.labels, 2):
            mask = (y == i) | (y == j)
            X_sub = X[mask]
            y_sub = y[mask]
            y_sub = np.where(y_sub == i, 1, -1)

            clf = LinearSVM(C=self.C, lr=self.lr, epochs=self.epochs)
            clf.fit(X_sub, y_sub)
            self.classifiers[(i, j)] = clf

    def predict(self, X):
        votes = np.zeros((X.shape[0], len(self.labels)), dtype=int)
        for (i, j), clf in self.classifiers.items():
            pred = clf.predict(X)
            for idx, val in enumerate(pred):
                if val == 1:
                    votes[idx, i] += 1
                else:
                    votes[idx, j] += 1
        return np.argmax(votes, axis=1)

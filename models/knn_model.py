import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)
    def to_dict(self):
        return {
            "k": self.k,
            "X_train": self.X_train.tolist(),
            "y_train": self.y_train.tolist()
        }

    @classmethod
    def from_dict(cls, model_dict):
        model = cls(k=model_dict["k"])
        model.X_train = np.array(model_dict["X_train"])
        model.y_train = np.array(model_dict["y_train"])
        return model
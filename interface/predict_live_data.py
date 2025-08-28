import json
import numpy as np

class LightweightSVM:
    def __init__(self, w, b):
        self.w = np.array(w)
        self.b = b

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

class OVOClassifier:
    def __init__(self, model_json_path):
        self.classifiers = {}
        self.labels = set()
        self._load_model(model_json_path)

    def _load_model(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        for key, params in data.items():
            i, j = map(int, key.split("_"))
            clf = LightweightSVM(params["w"], params["b"])
            self.classifiers[(i, j)] = clf
            self.labels.update([i, j])
        self.labels = sorted(self.labels)

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

if __name__ == "__main__":
    sample = np.array([[0.452, 0.019, 0.49, 0.40, 0.09, -0.5]])
    clf = OVOClassifier("PJ_PumpGuard/models/svm_model.json")
    y_pred = clf.predict(sample)
    print("pridicted classes list：", y_pred)
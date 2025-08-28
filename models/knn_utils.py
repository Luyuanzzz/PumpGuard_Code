import json
from models.knn_model import KNNClassifier

def export_knn_model_to_json(model, filepath="models/knn_model.json"):
    model_dict = model.to_dict()
    with open(filepath, "w") as f:
        json.dump(model_dict, f)
    print(f" KNN Model saved in {filepath}")

def load_knn_model_from_json(filepath="models/knn_model.json"):
    with open(filepath, "r") as f:
        model_dict = json.load(f)
    return KNNClassifier.from_dict(model_dict)
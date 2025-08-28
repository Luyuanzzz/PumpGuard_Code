from models.knn_model import KNNClassifier
from models.knn_utils import export_knn_model_to_json

def train_model(X, y):
    print("[KNN] Training KNN model...")
    model = KNNClassifier(k=13)
    model.fit(X, y)
    
    return model


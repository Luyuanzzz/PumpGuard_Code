from models.svm_model import MulticlassSVM_OVO
from models.svm_utils import export_model_to_json

def train_model(X, y):
    print("[SVM] Training SVM...")
    model = MulticlassSVM_OVO(C=1.0, lr=0.0005, epochs=100)
    model.fit(X, y)
    
    return model

import numpy as np
import pandas as pd
import argparse
from training import train_svm, train_knn
from models.svm_utils import export_model_to_json
from models.knn_utils import export_knn_model_to_json

MODEL_REGISTRY = {
    "svm": {
        "trainer": train_svm,
        "exporter": export_model_to_json,
        "json_path": "models/trained_models/svm_model.json",
        "class": None  
    },
    "knn": {
        "trainer": train_knn,
        "exporter": export_knn_model_to_json,
        "json_path": "models/trained_models/knn_model.json",
        "class": None  
    }
}

def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values
    labels = sorted(set(y_raw))
    label_to_num = {label: idx for idx, label in enumerate(labels)}
    y = np.array([label_to_num[label] for label in y_raw])
    # Standardization
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X_scaled = (X - X_mean) / X_std
    return X_scaled, y, label_to_num

def shuffle_data(X, y, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]

# k-fold cross validation
def k_fold_cross_validation(model_class, X, y, num_folds=5, **model_params):
    fold_size = len(X) // num_folds
    accuracies = []
    for i in range(num_folds):
        start = i * fold_size
        end = start + fold_size if i < num_folds - 1 else len(X)
        X_val, y_val = X[start:end], y[start:end]
        X_train = np.concatenate([X[:start], X[end:]])
        y_train = np.concatenate([y[:start], y[end:]])

        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = np.sum(y_val == y_pred) / len(y_val)
        accuracies.append(acc)
        print(f" Fold {i+1} Accuracy: {acc:.4f}")

    avg_acc = np.mean(accuracies)
    print(f"\n Average Accuracy over {num_folds} folds: {avg_acc:.4f}")
    return avg_acc


def main(model_name="svm", num_folds=5):
    if model_name not in MODEL_REGISTRY:
        print(f" Unknown model '{model_name}', available: {list(MODEL_REGISTRY.keys())}")
        return

    print(f"\n Training {model_name.upper()} with {num_folds}-fold cross validation...")

    X, y, label_map = load_and_preprocess_data("data/Final_training_data.csv")
    X, y = shuffle_data(X, y)

    trainer = MODEL_REGISTRY[model_name]["trainer"]
    exporter = MODEL_REGISTRY[model_name]["exporter"]
    json_path = MODEL_REGISTRY[model_name]["json_path"]

    # Step 1: cross validation
    if model_name == "svm":
        from models.svm_model import MulticlassSVM_OVO
        MODEL_REGISTRY["svm"]["class"] = MulticlassSVM_OVO
        k_fold_cross_validation(MulticlassSVM_OVO, X, y, num_folds=num_folds,
                                C=1.0, lr=0.0005, epochs=100) # here set epoch number
    elif model_name == "knn":
        from models.knn_model import KNNClassifier
        MODEL_REGISTRY["knn"]["class"] = KNNClassifier
        k_fold_cross_validation(KNNClassifier, X, y, num_folds=num_folds, k=13) # here set neighbor number

    # Step 2: training with full data for the model 
    print("\n Training final model on full dataset and saving to JSON...")
    model = trainer.train_model(X, y)
    exporter(model, filepath=json_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="svm" ) # choose knn or svm
    parser.add_argument("--num_folds", type=int, default=5 ) # set k-folds number 
    args = parser.parse_args()
    main(model_name=args.model, num_folds=args.num_folds)

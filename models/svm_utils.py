import json

def export_model_to_json(model, filepath="models/svm_model.json"):
    export_dict = {}
    for (i, j), clf in model.classifiers.items():
        key = f"{i}_{j}"
        export_dict[key] = {"w": clf.w.tolist(), "b": clf.b}
    with open(filepath, "w") as f:
        json.dump(export_dict, f, indent=2)
    print(f" Model saved as JSON：{filepath}")
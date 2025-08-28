import json
import numpy as np
from interface.predict_live_data import OVOClassifier

clf = OVOClassifier("models/svm_model.json")

def lambda_handler(event, context):
    try:
        features = event.get("features")
        if not features:
            return {"statusCode": 400, "body": "Missing 'features' field"}

        X = np.array([features])
        prediction = clf.predict(X)

        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": int(prediction[0])})
        }
    except Exception as e:
        return {"statusCode": 500, "body": str(e)}
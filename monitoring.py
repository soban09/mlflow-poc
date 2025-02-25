import mlflow
import requests
import json
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri("mysql+pymysql://root:root@host.docker.internal:3306/mlflow")
mlflow.set_experiment("model_monitoring")

# MLflow model serving URL
MLFLOW_MODEL_URL = "http://127.0.0.1:5000/invocations"

@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from request
    input_data = request.get_json()

    # Send request to MLflow model server
    response = requests.post(MLFLOW_MODEL_URL, json=input_data, headers={"Content-Type": "application/json"})
    prediction = response.json()

    # Log prediction in MLflow
    with mlflow.start_run():
        mlflow.log_dict({
            "timestamp": str(datetime.now()),
            "input": input_data,
            "prediction": prediction
        }, "predictions.json")

    return jsonify(prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

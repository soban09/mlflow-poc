import mlflow
from mlflow.models import infer_signature
import pandas as pd
import pymysql 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.autolog()

data = pd.read_csv('./diabetes-dev.csv')
X = data.drop(columns=['Diabetic', 'PatientID'])
y = data['Diabetic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
_f1_score = f1_score(y_test, y_pred)

mlflow.set_tracking_uri("mysql+pymysql://root:root@host.docker.internal:3306/mlflow")

mlflow.set_experiment("MLflow Quickstart")

with mlflow.start_run() as run:

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", _f1_score)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic RandomForrest model for diabetes dataset")

    signature = infer_signature(X_train, model.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forrest_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="random_forrest_model",
    )

    # Get the run ID
    latest_run_id = run.info.run_id
    with open('./logs.txt', 'w') as f:
        f.write(latest_run_id)
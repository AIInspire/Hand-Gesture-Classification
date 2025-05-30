import os
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from mlflow.models.signature import infer_signature

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from datetime import datetime


# ============================
# 1. Load Dataset
# ============================

def load_data(train_path='Data/train_df.csv', test_path='Data/test_df.csv'):
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Train/Test data not found in Data/ directory.")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]

    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    return X_train, X_test, y_train, y_test


# ============================
# 2. Preprocessing
# ============================

def preprocess(y_train, y_test):
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Save label encoder
    label_encoder_path = "artifacts/label_encoder.pkl"
    os.makedirs("artifacts", exist_ok=True)
    with open(label_encoder_path, "wb") as f:
        pickle.dump(le, f)

    return y_train_encoded, y_test_encoded, label_encoder_path


# ============================
# 3. Training and Logging
# ============================

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, label_encoder_path):
    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("accuracy", acc)

        for cls, metrics in report.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{cls}_{metric}", value)

        # Save model as .pkl
        model_path = f"artifacts/{model_name}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Log artifacts
        mlflow.log_artifact(model_path, artifact_path="models")
        mlflow.log_artifact(label_encoder_path, artifact_path="preprocessing")

        # Also log the model with mlflow.sklearn
        signature = infer_signature(X_train, model.predict(X_train[:5]))
        input_example = X_train.head(5)
        mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=input_example)

        print(f"[{model_name}] Accuracy: {acc:.4f}")
        print(f"[{model_name}] Run Link: http://127.0.0.1:5000/#/experiments/{run.info.experiment_id}/runs/{run_id}")


# ============================
# 4. Main
# ============================

def main():
    mlruns_path = Path(".mlruns").resolve()
    mlruns_uri = f"file:///{mlruns_path.as_posix()}"

    mlflow.set_tracking_uri(mlruns_uri)
    mlflow.set_registry_uri(mlruns_uri)
    mlflow.set_experiment("Hand Gesture Classification")

    X_train, X_test, y_train, y_test = load_data()
    y_train_encoded, y_test_encoded, label_encoder_path = preprocess(y_train, y_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, eval_metric="mlogloss")
    }

    for name, model in models.items():
        train_and_log_model(model, name, X_train, X_test, y_train_encoded, y_test_encoded, label_encoder_path)


if __name__ == "__main__":
    main()

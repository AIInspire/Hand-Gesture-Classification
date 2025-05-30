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
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    return y_train_encoded, y_test_encoded


def normalize_landmarks(hand_landmarks):
    """
    Normalize landmarks:
    - Origin: wrist (landmark 0)
    - Scale x by landmark 12 x
    - Scale y by landmark 12 y
    """
    wrist = hand_landmarks.landmark[0]
    mid_tip = hand_landmarks.landmark[12]

    x_scale = mid_tip.x if abs(mid_tip.x) > 1e-6 else 1e-6
    y_scale = mid_tip.y if abs(mid_tip.y) > 1e-6 else 1e-6

    normalized = []
    for lm in hand_landmarks.landmark:
        x_norm = (lm.x - wrist.x) / x_scale
        y_norm = (lm.y - wrist.y) / y_scale
        z_norm = lm.z
        normalized.extend([x_norm, y_norm, z_norm])

    return normalized


# ============================
# 3. Training and Logging
# ============================

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name) as run:
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

        # Infer signature from training data and model prediction
        signature = infer_signature(X_train, model.predict(X_train[:5]))
        input_example = X_train.head(5)

        mlflow.sklearn.log_model(model, model_name,
                                 signature=signature,
                                 input_example=input_example)
        
        # Save the model as a pickle file locally
        pkl_filename = f"{model_name}.pkl"
        with open(pkl_filename, "wb") as f:
            pickle.dump(model, f)

        print(f"[{model_name}] Accuracy: {acc:.4f}")
        print(f"Model saved as: {pkl_filename}")
        print(f"Run URL: {mlflow.get_tracking_uri()}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{run.info.run_id}")


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
    y_train_encoded, y_test_encoded = preprocess(y_train, y_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, eval_metric="mlogloss")
    }

    for name, model in models.items():
        train_and_log_model(model, name, X_train, X_test, y_train_encoded, y_test_encoded)


if __name__ == "__main__":
    main()

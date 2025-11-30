import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# ---- Config ----
TARGET = "Churn Label"
BEST_THRESHOLD = 0.30
EXPERIMENT_NAME = "churn-rf-mlops"

def build_pipeline(X, n_estimators=500, max_depth=None):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample"
    )

    pipe = Pipeline([
        ("prep", preprocess),
        ("model", model)
    ])
    return pipe

def main():
    # === MLflow: pointer vers le serveur ===
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1) Data
    df = pd.read_csv("data/processed.csv")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 2) Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Params (modifie-les pour faire plusieurs runs)
    n_estimators = 500
    max_depth = 12

    pipe = build_pipeline(X, n_estimators=n_estimators, max_depth=max_depth)

    # 4) Entraînement + logging MLflow
    with mlflow.start_run():
        print("Training model...")
        pipe.fit(X_train, y_train)

        val_proba = pipe.predict_proba(X_val)[:, 1]
        val_pred = (val_proba >= BEST_THRESHOLD).astype(int)

        p = precision_score(y_val, val_pred, zero_division=0)
        r = recall_score(y_val, val_pred, zero_division=0)
        f1 = f1_score(y_val, val_pred, zero_division=0)

        print(f"Validation precision={p:.3f} | recall={r:.3f} | f1={f1:.3f} (threshold={BEST_THRESHOLD})")

        # Log params & metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("threshold", BEST_THRESHOLD)

        mlflow.log_metric("val_precision", p)
        mlflow.log_metric("val_recall", r)
        mlflow.log_metric("val_f1", f1)

        # Sauvegarde modèle + log artifact
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipe, "models/model.pkl")
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        print("Model saved to models/model.pkl ✅")
        print("Run logged in MLflow ✅")

if __name__ == "__main__":
    main()

import pandas as pd
import joblib

from config import BEST_THRESHOLD
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

TARGET = "Churn Label"

def build_pipeline(X):
    num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

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
        n_estimators=500,
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
    df = pd.read_csv("data/processed.csv")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline(X)

    print("Training model...")
    pipe.fit(X_train, y_train)

    val_proba = pipe.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= BEST_THRESHOLD).astype(int)


    p = precision_score(y_val, val_pred, zero_division=0)
    r = recall_score(y_val, val_pred, zero_division=0)

    print(f"Validation precision={p:.3f} | recall={r:.3f} {BEST_THRESHOLD}")

    joblib.dump(pipe, "models/model.pkl")
    print("Model saved to models/model.pkl ✅")

if __name__ == "__main__":
    main()

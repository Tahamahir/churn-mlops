import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

TARGET = "Churn Label"

df = pd.read_csv("data/processed.csv")
X = df.drop(columns=[TARGET])
y = df[TARGET]

model = joblib.load("models/model.pkl")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

val_proba = model.predict_proba(X_val)[:, 1]

rows = []
for t in np.arange(0.05, 1.00, 0.05):
    pred = (val_proba >= t).astype(int)
    p = precision_score(y_val, pred, zero_division=0)
    r = recall_score(y_val, pred, zero_division=0)
    f1 = f1_score(y_val, pred, zero_division=0)
    rows.append({"threshold": t, "precision": p, "recall": r, "f1": f1})

table = pd.DataFrame(rows)
print(table.sort_values("threshold"))

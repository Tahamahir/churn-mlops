import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

TARGET = "Churn Label"
PRECISION_MIN = 0.80  # on exige au moins 80% de precision

def main():
    # Charger le dataset
    df = pd.read_csv("data/processed.csv")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Charger le modèle entraîné
    model = joblib.load("models/model.pkl")

    # Refaire le même split que dans train.py
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Probabilités sur la validation
    val_proba = model.predict_proba(X_val)[:, 1]

    best_t, best_r, best_p = 0.5, 0, 0

    # Tester plusieurs seuils
    for t in np.arange(0.05, 1.00, 0.05):
        pred = (val_proba >= t).astype(int)
        p = precision_score(y_val, pred, zero_division=0)
        r = recall_score(y_val, pred, zero_division=0)

        if p >= PRECISION_MIN and r > best_r:
            best_t, best_r, best_p = t, r, p

    print("=== Meilleur seuil trouvé ===")
    print(f"threshold = {best_t:.2f}")
    print(f"precision = {best_p:.3f}")
    print(f"recall    = {best_r:.3f}")

if __name__ == "__main__":
    main()

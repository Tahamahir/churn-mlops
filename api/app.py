# api/app.py
from typing import Dict, Any, List
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

# ---- Config ----
TARGET = "Churn Label"
BEST_THRESHOLD = 0.30
MODEL_PATH = "models/model.pkl"

# ---- Charger modèle ----
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("models/model.pkl introuvable. Entraîne d'abord le modèle.")
model = joblib.load(MODEL_PATH)

# ---- Colonnes attendues (mêmes noms que dans processed.csv) ----
cols = pd.read_csv("data/processed.csv", nrows=0).columns.tolist()
FEATURES = [c for c in cols if c != TARGET]

app = FastAPI(title="Churn API", version="0.1.0")

@app.get("/")
def root():
    return {"status": "ok", "model": "random_forest", "threshold": BEST_THRESHOLD}

# ---------- Helper commun ----------
def predict_one(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforme le JSON en DataFrame 1 ligne, réindexe sur FEATURES
    (les colonnes manquantes deviennent NaN → gérées par l'imputer du pipeline).
    """
    df = pd.DataFrame([payload])
    X = df.reindex(columns=FEATURES)
    proba = float(model.predict_proba(X)[0, 1])
    label = int(proba >= BEST_THRESHOLD)
    return {"probability": proba, "label": label, "threshold_used": BEST_THRESHOLD}

# ---------- Endpoints ----------
@app.post("/predict")
def predict(payload: Dict[str, Any]):
    try:
        return predict_one(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {e}")

@app.post("/predict_many")
def predict_many(batch: Dict[str, List[Dict[str, Any]]]):
    try:
        items = batch.get("items", [])
        if not isinstance(items, list):
            raise ValueError("Body must contain key 'items' as a list.")
        results = [predict_one(item) for item in items]
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur batch: {e}")

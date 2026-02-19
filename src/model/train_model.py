"""
Train XGBoost yield model and persist artefacts to models/.
Replicates the logic from notebooks/modelling.ipynb.

Usage:
    python src/model/train_model.py
"""

import sys
from pathlib import Path
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from constants.path import GOLD_PATH

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_and_save():
    # ── Load data ─────────────────────────────────────────
    df = pd.read_parquet(GOLD_PATH / "gold.parquet")
    df = df.dropna(subset=["yield"])
    print(f"Loaded gold.parquet  →  {df.shape[0]} rows, {df.shape[1]} cols")

    TARGET = "yield"
    LEAKAGE = ["yield", "production", "area", "department"]
    DROP = LEAKAGE + ["year"]

    feature_cols = [c for c in df.columns if c not in DROP]
    X = df[feature_cols].copy()
    y = df[TARGET].copy().values

    # ── One-hot encode departments ────────────────────────
    X = pd.get_dummies(X, columns=["code_dep"], drop_first=False)

    # ── Climate × department interaction features ─────────
    CLIMATE_INTERACT = [
        "heat_days", "hot_dry_days", "total_precip", "gdd",
    ]
    dep_cols = [c for c in X.columns if c.startswith("code_dep_")]
    for v in CLIMATE_INTERACT:
        if v in X.columns:
            for d in dep_cols:
                X[f"{v}_x_{d}"] = X[v] * X[d]
    X = X.drop(columns=dep_cols)

    X = X.astype(np.float32)

    print(f"Feature matrix  →  {X.shape[0]} rows × {X.shape[1]} features")

    # ── Train final model ─────────────────────────────────
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    print("Model trained ✓")

    # ── Persist ───────────────────────────────────────────
    with open(MODELS_DIR / "xgb_yield.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / "feature_cols.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)

    print(f"Saved → {MODELS_DIR / 'xgb_yield.pkl'}")
    print(f"Saved → {MODELS_DIR / 'feature_cols.pkl'}")
    print(f"       ({len(X.columns)} feature columns)")


if __name__ == "__main__":
    train_and_save()

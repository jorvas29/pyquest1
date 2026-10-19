"""Regresión logística multinomial y modelos de árboles (RF, GBM, XGBoost, LightGBM).

Atacan directamente el mercado 1X2: reciben las variables de
``features.FeatureBuilder`` y devuelven P(local), P(empate), P(visita)
sin pasar por el marcador exacto.

La validación es temporal (se entrena con el pasado y se evalúa con el
último 20 % del historial) para medir log-loss y RPS sin mirar el futuro.
Después cada modelo se reentrena con todo el historial.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def rps(probs: np.ndarray, y: np.ndarray) -> float:
    """Ranked Probability Score (menor = mejor). Penaliza más fallar lejos (1 vs 2)."""
    onehot = np.eye(3)[y]
    cum_p = np.cumsum(probs, axis=1)[:, :2]
    cum_o = np.cumsum(onehot, axis=1)[:, :2]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / 2))


def log_loss(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1)
    return float(-np.mean(np.log(p)))


def _model_factories() -> dict:
    models = {
        "Regresión logística": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)
        ),
        "Random Forest": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=400, min_samples_leaf=15, max_features="sqrt", random_state=7, n_jobs=-1),
        ),
        "Gradient Boosting": lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.03, max_iter=150, min_samples_leaf=30, l2_regularization=2.0,
            random_state=7
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = lambda: XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", eval_metric="mlogloss", random_state=7,
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = lambda: LGBMClassifier(
            n_estimators=300, num_leaves=15, learning_rate=0.05, min_child_samples=20, random_state=7, verbose=-1
        )
    except ImportError:
        pass
    return models


def _proba3(model, X) -> np.ndarray:
    """Probabilidades en orden (local, empate, visita) aunque falte una clase."""
    raw = model.predict_proba(X)
    classes = list(model.classes_)
    out = np.zeros((X.shape[0], 3))
    for k, c in enumerate(classes):
        out[:, int(c)] = raw[:, k]
    return out


def train_and_predict(X: np.ndarray, y: np.ndarray, x_target: np.ndarray, holdout: float = 0.2) -> dict:
    """Entrena todos los modelos disponibles y predice el partido objetivo.

    Devuelve ``{modelo: {"1x2": {...}, "logloss": .., "rps": ..}}``.
    """
    results = {}
    n = len(y)
    if n < 60 or len(set(y)) < 3:
        return results
    split = int(n * (1 - holdout))
    base_rate = np.bincount(y[:split], minlength=3) / split
    results["Frecuencia base (referencia)"] = {
        "1x2": dict(zip("1X2", map(float, np.bincount(y, minlength=3) / n))),
        "logloss": log_loss(np.tile(base_rate, (n - split, 1)), y[split:]),
        "rps": rps(np.tile(base_rate, (n - split, 1)), y[split:]),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, factory in _model_factories().items():
            model = factory().fit(X[:split], y[:split])
            val = _proba3(model, X[split:])
            final = factory().fit(X, y)
            p = _proba3(final, x_target)[0]
            results[name] = {
                "1x2": {"1": float(p[0]), "X": float(p[1]), "2": float(p[2])},
                "logloss": log_loss(val, y[split:]),
                "rps": rps(val, y[split:]),
            }
    return results

"""Cuotas: probabilidad implícita y eliminación del margen de la casa."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def overround(odds: list[float]) -> float:
    return float(sum(1 / o for o in odds) - 1)


def devig_proportional(odds: list[float]) -> list[float]:
    raw = np.array([1 / o for o in odds])
    return list(raw / raw.sum())


def devig_shin(odds: list[float]) -> list[float]:
    """Método de Shin: asume apostadores informados; corrige el sesgo favorito-sorpresa."""
    pi = np.array([1 / o for o in odds])
    s = pi.sum()
    if s <= 1:
        return list(pi / s)

    def probs(z):
        return (np.sqrt(z**2 + 4 * (1 - z) * pi**2 / s) - z) / (2 * (1 - z))

    z = brentq(lambda z: probs(z).sum() - 1, 0.0, 0.4)
    p = probs(z)
    return list(p / p.sum())


def devig_power(odds: list[float]) -> list[float]:
    """Método de potencia: p_i = (1/o_i)^k con k tal que sumen 1."""
    pi = np.array([1 / o for o in odds])
    k = brentq(lambda k: (pi**k).sum() - 1, 0.5, 3.0)
    return list(pi**k)


def market_consensus(books: list[dict], keys=("1", "X", "2")) -> dict | None:
    """Promedio entre casas de la probabilidad sin margen (Shin) para un mercado."""
    rows = []
    for book in books:
        if all(book.get(k) for k in keys):
            rows.append(devig_shin([float(book[k]) for k in keys]))
    if not rows:
        return None
    mean = np.mean(rows, axis=0)
    return {k: float(v) for k, v in zip(keys, mean)}


def fair_odds(prob: float) -> float:
    return float("inf") if prob <= 0 else 1 / prob

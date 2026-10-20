"""Cuotas: de probabilidad a cuota y de cuota a probabilidad.

Glosario
--------
cuota justa        1 / p. Es la cuota sin margen que corresponde a la
                   probabilidad del modelo.
p implícita        1 / cuota. Con cuotas reales, la suma de las p implícitas
                   de un mercado supera 1: ese exceso es el margen de la casa.
p sin margen       p implícitas reescaladas para que sumen 1 (método
                   proporcional). Es la mejor estimación simple de lo que
                   "piensa" el mercado.
valor esperado     EV = p_modelo * cuota - 1, por unidad apostada.
                   EV > 0 indica que, SI el modelo acierta en su probabilidad,
                   la apuesta tendría esperanza positiva. No es una garantía.
"""

from __future__ import annotations

import math
from typing import Mapping


def cuota_justa(probabilidad: float) -> float:
    """Cuota sin margen para una probabilidad (inf si p = 0)."""
    if probabilidad <= 0:
        return math.inf
    return 1.0 / probabilidad


def probabilidad_implicita(cuota: float) -> float:
    """1 / cuota (incluye el margen de la casa)."""
    if cuota <= 1.0:
        raise ValueError(f"Una cuota decimal debe ser > 1, se recibió {cuota}")
    return 1.0 / cuota


def margen(cuotas: Mapping[str, float]) -> float:
    """Margen de la casa en un mercado: sum(1 / cuota) - 1."""
    return sum(1.0 / c for c in cuotas.values()) - 1.0


def probabilidades_sin_margen(cuotas: Mapping[str, float]) -> dict[str, float]:
    """Probabilidades implícitas normalizadas para que sumen 1."""
    inversas = {k: probabilidad_implicita(c) for k, c in cuotas.items()}
    total = sum(inversas.values())
    return {k: v / total for k, v in inversas.items()}


def valor_esperado(probabilidad: float, cuota: float) -> float:
    """Beneficio esperado por unidad apostada: p * cuota - 1."""
    return probabilidad * cuota - 1.0

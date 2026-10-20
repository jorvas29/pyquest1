"""Probabilidades de mercado a partir de distribuciones de conteo.

Funciones puras (sin estado) para que sean fáciles de probar:

    Goles (a partir de la matriz conjunta de marcadores):
        probabilidades_1x2   -> {"1", "X", "2"}
        probabilidad_btts    -> P(ambos marcan)
        probabilidad_mas_de  -> P(goles totales > línea)
        marcadores_probables -> los marcadores más probables

    Córners (a partir de las tasas esperadas de cada equipo):
        probabilidades_corners -> más/menos de una línea y quién saca más

Convención de la matriz: matriz[i, j] = P(local marca i, visitante marca j).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def probabilidades_1x2(matriz: np.ndarray) -> dict[str, float]:
    """1 = gana el local (i > j), X = empate (i == j), 2 = gana el visitante (i < j)."""
    m = np.asarray(matriz, dtype=float)
    total = m.sum()
    p_local = np.tril(m, k=-1).sum() / total
    p_empate = np.trace(m) / total
    p_visitante = np.triu(m, k=1).sum() / total
    return {"1": float(p_local), "X": float(p_empate), "2": float(p_visitante)}


def probabilidad_btts(matriz: np.ndarray) -> float:
    """P(ambos marcan) = P(local >= 1 y visitante >= 1).

    Se calcula como 1 - P(local = 0) - P(visitante = 0) + P(0-0), que es la
    suma de la submatriz sin la fila 0 ni la columna 0.
    """
    m = np.asarray(matriz, dtype=float)
    return float(m[1:, 1:].sum() / m.sum())


def probabilidad_mas_de(matriz: np.ndarray, linea: float = 2.5) -> float:
    """P(goles totales > línea)."""
    m = np.asarray(matriz, dtype=float)
    i, j = np.indices(m.shape)
    return float(m[(i + j) > linea].sum() / m.sum())


def marcadores_probables(matriz: np.ndarray, n: int = 5) -> list[tuple[int, int, float]]:
    """Los `n` marcadores más probables como (goles_local, goles_visitante, p)."""
    m = np.asarray(matriz, dtype=float) / np.sum(matriz)
    orden = np.argsort(m.ravel())[::-1][:n]
    return [(int(k // m.shape[1]), int(k % m.shape[1]), float(m.ravel()[k])) for k in orden]


def _pmf_conteo(media: float, dispersion: float, maximo: int) -> np.ndarray:
    """Distribución de un conteo con media dada.

    dispersion = 1  -> Poisson.
    dispersion > 1  -> binomial negativa con varianza = media * dispersion
                       (los córners suelen variar más de lo que dice Poisson).
    """
    k = np.arange(maximo + 1)
    if dispersion <= 1.0 + 1e-9:
        pmf = stats.poisson.pmf(k, media)
    else:
        n = media / (dispersion - 1.0)
        p = 1.0 / dispersion
        pmf = stats.nbinom.pmf(k, n, p)
    return pmf / pmf.sum()


def probabilidades_corners(
    media_local: float,
    media_visitante: float,
    linea: float = 9.5,
    dispersion: float = 1.0,
    maximo: int = 40,
) -> dict[str, float]:
    """Mercados de córners a partir de las medias esperadas de cada equipo.

    Devuelve:
        mas           -> P(total > línea)
        menos         -> P(total < línea)
        nulo          -> P(total == línea); solo > 0 con líneas enteras (apuesta devuelta)
        media_total   -> córners totales esperados
        local_mas     -> P(local saca más córners que el visitante)
        igual         -> P(mismo número de córners)
        visitante_mas -> P(visitante saca más córners)
    """
    pmf_l = _pmf_conteo(media_local, dispersion, maximo)
    pmf_v = _pmf_conteo(media_visitante, dispersion, maximo)

    # Distribución del total = convolución de las dos distribuciones.
    pmf_total = np.convolve(pmf_l, pmf_v)
    totales = np.arange(len(pmf_total))
    mas = float(pmf_total[totales > linea].sum())
    menos = float(pmf_total[totales < linea].sum())
    nulo = float(pmf_total[totales == linea].sum())

    conjunta = np.outer(pmf_l, pmf_v)
    local_mas = float(np.tril(conjunta, k=-1).sum())
    igual = float(np.trace(conjunta))
    visitante_mas = float(np.triu(conjunta, k=1).sum())

    return {
        "mas": mas,
        "menos": menos,
        "nulo": nulo,
        "media_total": float(media_local + media_visitante),
        "local_mas": local_mas,
        "igual": igual,
        "visitante_mas": visitante_mas,
    }

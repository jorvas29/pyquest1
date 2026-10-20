"""Modelo de córners: regresión de Poisson sobre características previas.

Cada partido aporta dos filas, una por equipo:

    córners_del_equipo ~ Poisson(lambda)

    log(lambda) = b0
                + b1 * log(córners a favor promedio del equipo)
                + b2 * log(córners en contra promedio del rival)
                + b3 * es_local
                + b4 * (elo_equipo - elo_rival) / 400

Es un modelo log-lineal (como un GLM de Poisson): los coeficientes se leen
como efectos multiplicativos sobre la media de córners.

Sobredispersión
---------------
Los córners suelen variar más de lo que predice una Poisson. Tras el ajuste
se estima un factor de dispersión phi = sum((y - lambda)^2 / lambda) / (n - p).
Si phi > 1, las probabilidades de los mercados se calculan con una binomial
negativa de media lambda y varianza lambda * phi, en lugar de una Poisson.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from .mercados import probabilidades_corners

NOMBRES_X = ["log_cf_equipo", "log_cc_rival", "es_local", "dif_elo"]


def matrices_corners(caract: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Matrices de diseño (local, visitante) a partir de las características."""
    n = len(caract)
    x_local = np.column_stack(
        [
            np.log(caract["l_cf"].to_numpy(dtype=float)),
            np.log(caract["v_cc"].to_numpy(dtype=float)),
            np.ones(n),
            (caract["l_elo"] - caract["v_elo"]).to_numpy(dtype=float) / 400.0,
        ]
    )
    x_visit = np.column_stack(
        [
            np.log(caract["v_cf"].to_numpy(dtype=float)),
            np.log(caract["l_cc"].to_numpy(dtype=float)),
            np.zeros(n),
            (caract["v_elo"] - caract["l_elo"]).to_numpy(dtype=float) / 400.0,
        ]
    )
    return x_local, x_visit


class ModeloCorners:
    """Regresión de Poisson con regularización L2 suave (sin penalizar el intercepto)."""

    def __init__(self, regularizacion: float = 1.0):
        self.regularizacion = regularizacion
        self.coeficientes: np.ndarray | None = None
        self.media_x: np.ndarray | None = None
        self.desv_x: np.ndarray | None = None
        self.dispersion = 1.0
        self.n_filas = 0

    def _estandarizar(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.media_x) / self.desv_x
        return np.column_stack([np.ones(len(z)), z])

    def ajustar(self, caract: pd.DataFrame, corners_local, corners_visitante) -> "ModeloCorners":
        x_l, x_v = matrices_corners(caract)
        x = np.vstack([x_l, x_v])
        y = np.concatenate(
            [np.asarray(corners_local, dtype=float), np.asarray(corners_visitante, dtype=float)]
        )
        validas = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
        x, y = x[validas], y[validas]
        if len(y) < 30:
            raise ValueError("Se necesitan al menos 15 partidos con córners para ajustar el modelo.")

        self.media_x = x.mean(axis=0)
        self.desv_x = x.std(axis=0)
        self.desv_x[self.desv_x == 0] = 1.0
        z = self._estandarizar(x)
        r = self.regularizacion

        def nll_y_grad(beta):
            eta = np.clip(z @ beta, -10, 10)
            lam = np.exp(eta)
            nll = np.sum(lam - y * eta) + r * np.sum(beta[1:] ** 2)
            grad = z.T @ (lam - y)
            grad[1:] += 2 * r * beta[1:]
            return nll, grad

        beta0 = np.zeros(z.shape[1])
        beta0[0] = np.log(max(y.mean(), 0.1))
        res = optimize.minimize(nll_y_grad, beta0, jac=True, method="L-BFGS-B")
        self.coeficientes = res.x

        lam = np.exp(z @ self.coeficientes)
        grados = max(len(y) - z.shape[1], 1)
        self.dispersion = float(max(1.0, np.sum((y - lam) ** 2 / lam) / grados))
        self.n_filas = int(len(y))
        return self

    def _medias(self, x: np.ndarray) -> np.ndarray:
        if self.coeficientes is None:
            raise RuntimeError("El modelo de córners no está ajustado.")
        return np.exp(self._estandarizar(x) @ self.coeficientes)

    def medias(self, caract: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Córners esperados (local, visitante) para cada fila de características."""
        x_l, x_v = matrices_corners(caract)
        return self._medias(x_l), self._medias(x_v)

    def predecir(self, caract_partido: dict | pd.Series, linea: float = 9.5) -> dict[str, float]:
        """Probabilidades de los mercados de córners para un partido."""
        fila = pd.DataFrame([dict(caract_partido)])
        m_l, m_v = self.medias(fila)
        salida = probabilidades_corners(float(m_l[0]), float(m_v[0]), linea, self.dispersion)
        salida["media_local"] = float(m_l[0])
        salida["media_visitante"] = float(m_v[0])
        salida["linea"] = float(linea)
        return salida

    def coeficientes_originales(self) -> dict[str, float]:
        """Coeficientes en la escala original de las variables (para interpretarlos)."""
        b = self.coeficientes
        pendientes = b[1:] / self.desv_x
        intercepto = b[0] - np.sum(pendientes * self.media_x)
        return {"intercepto": float(intercepto), **{n: float(v) for n, v in zip(NOMBRES_X, pendientes)}}

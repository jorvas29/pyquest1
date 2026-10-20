"""Modelo de goles Dixon-Coles (y Poisson independiente como caso particular).

Idea
----
Los goles de cada equipo siguen una Poisson:

    goles_local     ~ Poisson(lambda)   log(lambda) = c + localia + ataque[local]     + defensa[visitante]
    goles_visitante ~ Poisson(mu)       log(mu)     = c +           ataque[visitante] + defensa[local]

Dixon y Coles (1997) añaden dos mejoras:

1. Un factor tau(x, y) que corrige la probabilidad de los marcadores bajos
   (0-0, 1-0, 0-1, 1-1), donde la Poisson independiente se equivoca más.
   Lo controla un único parámetro rho (rho = 0 -> Poisson independiente).

2. Pesos que decaen con el tiempo: w = exp(-xi * días_de_antigüedad), para
   que los partidos recientes cuenten más que los antiguos.

Además se añade una pequeña regularización L2 sobre ataque y defensa: hace
el modelo identificable (sin fijar un equipo de referencia) y evita valores
extremos para equipos con pocos partidos (por ejemplo, recién ascendidos).

El ajuste es por máxima verosimilitud con L-BFGS-B y gradiente analítico.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special

RHO_MIN, RHO_MAX = -0.25, 0.25


def tau(x, y, lam, mu, rho):
    """Factor de corrección de Dixon-Coles para marcadores bajos (vectorizado)."""
    x = np.asarray(x)
    y = np.asarray(y)
    lam = np.asarray(lam, dtype=float)
    mu = np.asarray(mu, dtype=float)
    t = np.ones(np.broadcast(x, y, lam, mu).shape)
    t = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1.0 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1.0 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1.0 - rho, t)
    return t


def matriz_marcadores(lam: float, mu: float, rho: float = 0.0, max_goles: int = 10) -> np.ndarray:
    """Matriz (max_goles+1) x (max_goles+1) con P(local = i, visitante = j).

    Se normaliza para que sume exactamente 1 (la cola por encima de
    `max_goles` es despreciable para tasas de goles realistas).
    """
    goles = np.arange(max_goles + 1)
    p_local = np.exp(goles * np.log(lam) - lam - special.gammaln(goles + 1))
    p_visit = np.exp(goles * np.log(mu) - mu - special.gammaln(goles + 1))
    matriz = np.outer(p_local, p_visit)
    i, j = np.indices(matriz.shape)
    matriz = matriz * np.clip(tau(i, j, lam, mu, rho), 0.0, None)
    return matriz / matriz.sum()


class ModeloDixonColes:
    """Modelo de goles por equipos con corrección Dixon-Coles opcional.

    Parámetros
    ----------
    xi:
        Velocidad de olvido por día. 0.0019 equivale a una vida media de
        aproximadamente un año (valor habitual en la literatura).
    usar_rho:
        False -> Poisson independiente (rho fijado a 0).
    regularizacion:
        Peso de la penalización L2 sobre ataque y defensa.
    """

    def __init__(self, xi: float = 0.0019, usar_rho: bool = True, regularizacion: float = 2.0):
        self.xi = xi
        self.usar_rho = usar_rho
        self.regularizacion = regularizacion
        self.equipos: list[str] = []
        self.indice: dict[str, int] = {}
        self.intercepto = 0.0
        self.localia = 0.0
        self.ataque = np.zeros(0)
        self.defensa = np.zeros(0)
        self.rho = 0.0
        self.fecha_referencia: pd.Timestamp | None = None
        self.n_partidos = 0

    @property
    def nombre(self) -> str:
        return "Dixon-Coles" if self.usar_rho else "Poisson independiente"

    # ------------------------------------------------------------------ ajuste
    def _desempaquetar(self, theta: np.ndarray, n: int):
        c, g = theta[0], theta[1]
        a = theta[2 : 2 + n]
        d = theta[2 + n : 2 + 2 * n]
        rho = theta[2 + 2 * n] if self.usar_rho else 0.0
        return c, g, a, d, rho

    def _nll_y_gradiente(self, theta, ih, ia, x, y, w, n):
        c, g, a, d, rho = self._desempaquetar(theta, n)
        log_lam = c + g + a[ih] + d[ia]
        log_mu = c + a[ia] + d[ih]
        lam = np.exp(log_lam)
        mu = np.exp(log_mu)

        # Parte Poisson (sin el término constante log(x!)).
        ll_poisson = x * log_lam - lam + y * log_mu - mu
        d_loglam = x - lam
        d_logmu = y - mu
        d_rho = np.zeros_like(lam)

        # Parte tau (solo para marcadores 0-0, 0-1, 1-0, 1-1).
        t = np.clip(tau(x, y, lam, mu, rho), 1e-10, None)
        ll_tau = np.log(t)
        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)
        d_loglam = d_loglam + np.where(m00, -lam * mu * rho / t, 0.0) + np.where(m01, lam * rho / t, 0.0)
        d_logmu = d_logmu + np.where(m00, -lam * mu * rho / t, 0.0) + np.where(m10, mu * rho / t, 0.0)
        d_rho = (
            np.where(m00, -lam * mu / t, 0.0)
            + np.where(m01, lam / t, 0.0)
            + np.where(m10, mu / t, 0.0)
            + np.where(m11, -1.0 / t, 0.0)
        )

        r = self.regularizacion
        nll = -np.sum(w * (ll_poisson + ll_tau)) + r * (np.sum(a**2) + np.sum(d**2))

        gl = w * d_loglam
        gm = w * d_logmu
        grad = np.zeros_like(theta)
        grad[0] = -(gl.sum() + gm.sum())
        grad[1] = -gl.sum()
        g_a = np.bincount(ih, weights=gl, minlength=n) + np.bincount(ia, weights=gm, minlength=n)
        g_d = np.bincount(ia, weights=gl, minlength=n) + np.bincount(ih, weights=gm, minlength=n)
        grad[2 : 2 + n] = -g_a + 2 * r * a
        grad[2 + n : 2 + 2 * n] = -g_d + 2 * r * d
        if self.usar_rho:
            grad[2 + 2 * n] = -np.sum(w * d_rho)
        return nll, grad

    def _datos_ajuste(self, partidos: pd.DataFrame, fecha_referencia=None):
        self.equipos = sorted(set(partidos["local"]) | set(partidos["visitante"]))
        self.indice = {e: k for k, e in enumerate(self.equipos)}
        ih = partidos["local"].map(self.indice).to_numpy()
        ia = partidos["visitante"].map(self.indice).to_numpy()
        x = partidos["goles_local"].to_numpy(dtype=float)
        y = partidos["goles_visitante"].to_numpy(dtype=float)
        ref = pd.Timestamp(fecha_referencia) if fecha_referencia is not None else partidos["fecha"].max()
        dias = (ref - partidos["fecha"]).dt.days.to_numpy(dtype=float)
        w = np.exp(-self.xi * np.clip(dias, 0.0, None))
        return ih, ia, x, y, w, ref

    def ajustar(self, partidos: pd.DataFrame, fecha_referencia=None) -> "ModeloDixonColes":
        """Ajusta el modelo con los partidos dados (todos deben ser pasados)."""
        if len(partidos) < 10:
            raise ValueError("Se necesitan al menos 10 partidos para ajustar el modelo de goles.")
        ih, ia, x, y, w, ref = self._datos_ajuste(partidos, fecha_referencia)
        n = len(self.equipos)

        media = max((x.mean() + y.mean()) / 2.0, 0.1)
        theta0 = np.zeros(2 + 2 * n + (1 if self.usar_rho else 0))
        theta0[0] = np.log(media)
        theta0[1] = 0.2
        limites = [(None, None)] * (2 + 2 * n)
        if self.usar_rho:
            limites.append((RHO_MIN, RHO_MAX))

        res = optimize.minimize(
            self._nll_y_gradiente,
            theta0,
            args=(ih, ia, x, y, w, n),
            jac=True,
            method="L-BFGS-B",
            bounds=limites,
            options={"maxiter": 2000},
        )
        c, g, a, d, rho = self._desempaquetar(res.x, n)
        self.intercepto, self.localia = float(c), float(g)
        self.ataque, self.defensa = a.copy(), d.copy()
        self.rho = float(rho)
        self.fecha_referencia = ref
        self.n_partidos = len(partidos)
        self.convergio = bool(res.success)
        return self

    # -------------------------------------------------------------- predicción
    def _fuerzas(self, equipo: str) -> tuple[float, float]:
        """Ataque y defensa de un equipo; un equipo sin historial cuenta como medio (0, 0)."""
        k = self.indice.get(equipo)
        if k is None:
            return 0.0, 0.0
        return float(self.ataque[k]), float(self.defensa[k])

    def tasas(self, local: str, visitante: str) -> tuple[float, float]:
        """Goles esperados (lambda del local, mu del visitante)."""
        if not self.equipos:
            raise RuntimeError("El modelo no está ajustado.")
        a_l, d_l = self._fuerzas(local)
        a_v, d_v = self._fuerzas(visitante)
        lam = np.exp(self.intercepto + self.localia + a_l + d_v)
        mu = np.exp(self.intercepto + a_v + d_l)
        return float(lam), float(mu)

    def matriz(self, local: str, visitante: str, max_goles: int = 10) -> np.ndarray:
        lam, mu = self.tasas(local, visitante)
        return matriz_marcadores(lam, mu, self.rho, max_goles)

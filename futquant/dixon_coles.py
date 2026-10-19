"""Modelo Dixon-Coles (Poisson modificado, 1997).

Dos usos:

1. ``dc_matrix(lam, mu, rho)``: aplica la corrección de baja puntuación
   (tau) a una matriz de Poisson con goles esperados ya calculados.
   Con ``rho < 0`` sube la probabilidad de 0-0 y 1-1 y baja la de 1-0 / 0-1.

2. ``DixonColes().fit(partidos)``: estima por máxima verosimilitud el
   ataque y la defensa de cada equipo, la ventaja de local y ``rho``,
   con ponderación temporal exponencial (los partidos recientes pesan más).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from .markets import MAX_GOALS, poisson_matrix

DEFAULT_RHO = -0.10  # valor típico en ligas europeas (Dixon & Coles reportan ~ -0.13)


def tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Factor de corrección de Dixon-Coles para marcadores 0-0, 0-1, 1-0 y 1-1."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def rho_bounds(lam: float, mu: float) -> tuple[float, float]:
    """Rango de ``rho`` que mantiene todas las probabilidades positivas."""
    lower = max(-1.0 / lam, -1.0 / mu)
    upper = min(1.0 / (lam * mu), 1.0)
    return lower, upper


def dc_matrix(lam: float, mu: float, rho: float = DEFAULT_RHO, max_goals: int = MAX_GOALS) -> np.ndarray:
    lo, hi = rho_bounds(lam, mu)
    rho = float(np.clip(rho, lo + 1e-6, hi - 1e-6))
    M = poisson_matrix(lam, mu, max_goals)
    for x in (0, 1):
        for y in (0, 1):
            M[x, y] *= tau(x, y, lam, mu, rho)
    return M / M.sum()


def _vector_tau(x, y, lam, mu, rho):
    t = np.ones_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    t[m00] = 1.0 - lam[m00] * mu[m00] * rho
    t[m01] = 1.0 + lam[m01] * rho
    t[m10] = 1.0 + mu[m10] * rho
    t[m11] = 1.0 - rho
    return t


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


@dataclass
class DixonColes:
    """Estimación MLE de Dixon-Coles con decaimiento temporal.

    ``xi`` es la tasa de decaimiento por día (0.0018 ≈ media vida de ~1 año,
    el valor que recomiendan Dixon & Coles).
    """

    xi: float = 0.0018
    teams: list[str] = field(default_factory=list)
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.0
    intercept: float = 0.0
    rho: float = DEFAULT_RHO
    fitted: bool = False

    def fit(self, matches: list[dict], reference_date=None) -> "DixonColes":
        """``matches``: dicts con ``home``, ``away``, ``hg``, ``ag`` y opcional ``date``."""
        self.teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
        index = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        hi = np.array([index[m["home"]] for m in matches])
        ai = np.array([index[m["away"]] for m in matches])
        hg = np.array([int(m["hg"]) for m in matches])
        ag = np.array([int(m["ag"]) for m in matches])

        if all("date" in m and m["date"] for m in matches):
            dates = [_as_date(m["date"]) for m in matches]
            ref = _as_date(reference_date) if reference_date else max(dates)
            age = np.array([(ref - d).days for d in dates], dtype=float)
            weights = np.exp(-self.xi * np.clip(age, 0, None))
        else:
            weights = np.ones(len(matches))

        # Parámetros: intercepto, local, ataque (n-1), defensa (n-1), rho.
        # El último equipo cierra la restricción suma(ataque) = suma(defensa) = 0.
        def unpack(p):
            att = np.append(p[2 : 2 + n - 1], -p[2 : 2 + n - 1].sum())
            dfn = np.append(p[1 + n : 2 * n], -p[1 + n : 2 * n].sum())
            return p[0], p[1], att, dfn, p[-1]

        def neg_log_lik(p):
            c, home, att, dfn, rho = unpack(p)
            lam = np.exp(c + home + att[hi] + dfn[ai])
            mu = np.exp(c + att[ai] + dfn[hi])
            t = _vector_tau(hg, ag, lam, mu, rho)
            if np.any(t <= 0):
                return 1e12
            ll = (
                np.log(t)
                + hg * np.log(lam) - lam - gammaln(hg + 1)
                + ag * np.log(mu) - mu - gammaln(ag + 1)
            )
            return -float(np.sum(weights * ll))

        x0 = np.zeros(2 * n + 1)
        x0[0] = np.log(max(np.mean(np.concatenate([hg, ag])), 0.1))
        x0[1] = 0.25
        x0[-1] = -0.05
        bounds = [(None, None)] * (2 * n) + [(-0.35, 0.35)]
        res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds)
        c, home, att, dfn, rho = unpack(res.x)
        self.intercept, self.home_adv, self.rho = float(c), float(home), float(rho)
        self.attack = {t: float(att[i]) for t, i in index.items()}
        self.defence = {t: float(dfn[i]) for t, i in index.items()}
        self.fitted = True
        return self

    def expected_goals(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        if not self.fitted:
            raise RuntimeError("Primero llama a fit()")
        h = 0.0 if neutral else self.home_adv
        lam = np.exp(self.intercept + h + self.attack[home] + self.defence[away])
        mu = np.exp(self.intercept + self.attack[away] + self.defence[home])
        return float(lam), float(mu)

    def predict_matrix(self, home: str, away: str, neutral: bool = False) -> np.ndarray:
        lam, mu = self.expected_goals(home, away, neutral)
        return dc_matrix(lam, mu, self.rho)

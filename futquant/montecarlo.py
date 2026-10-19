"""Simulación de Monte Carlo con cadena de Markov minuto a minuto.

El estado del partido es ``(minuto, goles_local, goles_visitante, rojas)``.
En cada minuto la probabilidad de transición a un nuevo estado depende de:

* el reparto temporal de goles (se marcan más goles al final; el perfil
  se calibra para que el 45 % caiga en el primer tiempo),
* el marcador (el equipo que pierde arriesga más y el que gana se repliega),
* las expulsiones (el equipo con 10 baja su ritmo, el rival lo sube).

Se simulan ``n_sims`` partidos y se obtiene la matriz empírica de
marcadores, la del descanso y la distribución de los minutos de gol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .markets import MAX_GOALS

MINUTES = 90
FIRST_HALF_SHARE = 0.45


def time_profile(minutes: int = MINUTES, first_half_share: float = FIRST_HALF_SHARE) -> np.ndarray:
    """Pesos por minuto lineales ``a + b t`` con el reparto pedido entre tiempos."""
    t = np.arange(minutes) + 0.5
    half = minutes / 2
    # Resolviendo  ∫0^half (a+bt) / ∫0^min (a+bt) = share  con a = 1.
    num = half - first_half_share * minutes
    den = first_half_share * minutes**2 / 2 - half**2 / 2
    b = num / den if den else 0.0
    w = 1 + b * t
    return w / w.sum()


@dataclass
class MarkovMatchSimulator:
    trailing_boost: float = 0.10   # +10 % de ritmo para quien va perdiendo
    leading_damp: float = 0.05     # -5 % para quien va ganando
    red_card_rate: tuple[float, float] = (0.08, 0.10)  # rojas esperadas por partido (local, visita)
    red_own_factor: float = 0.70
    red_rival_factor: float = 1.25
    seed: int | None = 7

    def _run(self, lam_h: float, lam_a: float, n: int, rng: np.random.Generator):
        prof = time_profile()
        hg = np.zeros(n, dtype=int)
        ag = np.zeros(n, dtype=int)
        red_h = np.zeros(n, dtype=bool)
        red_a = np.zeros(n, dtype=bool)
        ht_h = ht_a = None
        first = np.zeros(n, dtype=int)  # 0 = sin goles, 1 = local, 2 = visita
        minute_hist = np.zeros(MINUTES)
        p_red_h = self.red_card_rate[0] / MINUTES
        p_red_a = self.red_card_rate[1] / MINUTES

        for minute in range(MINUTES):
            rate_h = np.full(n, lam_h * prof[minute])
            rate_a = np.full(n, lam_a * prof[minute])
            diff = hg - ag
            rate_h = np.where(diff < 0, rate_h * (1 + self.trailing_boost), rate_h)
            rate_h = np.where(diff > 0, rate_h * (1 - self.leading_damp), rate_h)
            rate_a = np.where(diff > 0, rate_a * (1 + self.trailing_boost), rate_a)
            rate_a = np.where(diff < 0, rate_a * (1 - self.leading_damp), rate_a)
            rate_h = np.where(red_h, rate_h * self.red_own_factor, rate_h)
            rate_h = np.where(red_a, rate_h * self.red_rival_factor, rate_h)
            rate_a = np.where(red_a, rate_a * self.red_own_factor, rate_a)
            rate_a = np.where(red_h, rate_a * self.red_rival_factor, rate_a)

            goal_h = rng.random(n) < 1 - np.exp(-rate_h)
            goal_a = rng.random(n) < 1 - np.exp(-rate_a)
            first = np.where((first == 0) & goal_h, 1, first)
            first = np.where((first == 0) & goal_a, 2, first)
            hg += goal_h
            ag += goal_a
            minute_hist[minute] += goal_h.sum() + goal_a.sum()

            red_h |= rng.random(n) < p_red_h
            red_a |= rng.random(n) < p_red_a
            if minute == MINUTES // 2 - 1:
                ht_h, ht_a = hg.copy(), ag.copy()
        return hg, ag, ht_h, ht_a, first, minute_hist

    def simulate(self, lam_home: float, lam_away: float, n_sims: int = 10_000) -> dict:
        rng = np.random.default_rng(self.seed)
        # Calibración: los efectos de marcador y rojas desplazan la media,
        # así que se reescalan las tasas para que los goles esperados se respeten.
        cal_h, cal_a = lam_home, lam_away
        for _ in range(2):
            hg, ag, *_ = self._run(cal_h, cal_a, 4000, rng)
            cal_h *= lam_home / max(hg.mean(), 1e-6)
            cal_a *= lam_away / max(ag.mean(), 1e-6)

        hg, ag, ht_h, ht_a, first, minute_hist = self._run(cal_h, cal_a, n_sims, rng)
        size = MAX_GOALS + 1
        M = np.zeros((size, size))
        np.add.at(M, (np.minimum(hg, MAX_GOALS), np.minimum(ag, MAX_GOALS)), 1)
        M_ht = np.zeros((size, size))
        np.add.at(M_ht, (np.minimum(ht_h, MAX_GOALS), np.minimum(ht_a, MAX_GOALS)), 1)

        bins = minute_hist.reshape(6, 15).sum(axis=1)
        labels = ["1-15", "16-30", "31-45", "46-60", "61-75", "76-90"]
        return {
            "matrix": M / n_sims,
            "ht_matrix": M_ht / n_sims,
            "first_goal": {
                "home": float((first == 1).mean()),
                "away": float((first == 2).mean()),
                "none": float((first == 0).mean()),
            },
            "goal_minutes": {lab: float(v / max(bins.sum(), 1)) for lab, v in zip(labels, bins)},
            "n_sims": n_sims,
        }

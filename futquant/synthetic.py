"""Generador de una liga sintética (solo para pruebas y demos).

Los equipos y resultados son FICTICIOS: sirven para comprobar que los
modelos recuperan parámetros conocidos, nunca para analizar un partido real.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np


def synthetic_league(n_teams: int = 12, seasons: int = 2, seed: int = 1, home_adv: float = 0.25,
                     rho: float = -0.1) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    teams = [f"Equipo {chr(65 + i)}" for i in range(n_teams)]
    attack = rng.normal(0, 0.3, n_teams)
    attack -= attack.mean()
    defence = rng.normal(0, 0.25, n_teams)
    defence -= defence.mean()
    base = np.log(1.25)
    matches = []
    day = date(2024, 8, 1)
    from .dixon_coles import dc_matrix

    for _ in range(seasons):
        fixtures = [(i, j) for i in range(n_teams) for j in range(n_teams) if i != j]
        rng.shuffle(fixtures)
        for k, (i, j) in enumerate(fixtures):
            lam = np.exp(base + home_adv + attack[i] + defence[j])
            mu = np.exp(base + attack[j] + defence[i])
            M = dc_matrix(lam, mu, rho)
            idx = rng.choice(M.size, p=M.ravel())
            hg, ag = np.unravel_index(idx, M.shape)
            matches.append({
                "date": (day + timedelta(days=k // (n_teams // 2) * 3)).isoformat(),
                "home": teams[i], "away": teams[j], "hg": int(hg), "ag": int(ag),
                "home_sot": int(rng.poisson(lam * 3)), "away_sot": int(rng.poisson(mu * 3)),
                "home_corners": int(rng.poisson(5 + attack[i] * 3)), "away_corners": int(rng.poisson(4.3 + attack[j] * 3)),
            })
        day += timedelta(days=len(fixtures) // (n_teams // 2) * 3 + 60)
    truth = {"teams": teams, "attack": dict(zip(teams, attack)), "defence": dict(zip(teams, defence)),
             "home_adv": home_adv, "rho": rho}
    return matches, truth

"""Córners y tarjetas (FASE 3.4).

Esperado = (a favor de uno + en contra del otro) / 2 + lo mismo al revés.
En tarjetas se mezcla 50 % la estimación por equipos y 50 % el promedio
del árbitro. Para las líneas de más/menos se usa una binomial negativa
(los córners y tarjetas tienen más varianza que una Poisson); con
``dispersion = 1`` se reduce a Poisson.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import nbinom, poisson


def count_over(mean: float, line: float, dispersion: float = 1.0) -> float:
    k = int(np.floor(line))
    if dispersion <= 1.0:
        return float(1 - poisson.cdf(k, mean))
    p = 1 / dispersion
    r = mean / (dispersion - 1)
    return float(1 - nbinom.cdf(k, r, p))


def expected_corners(home: dict, away: dict) -> dict | None:
    keys = ("corners_favor", "corners_contra")
    if not all(home.get(k) is not None for k in keys) or not all(away.get(k) is not None for k in keys):
        return None
    h = (home["corners_favor"] + away["corners_contra"]) / 2
    a = (away["corners_favor"] + home["corners_contra"]) / 2
    return {"home": h, "away": a, "total": h + a,
            "steps": [f"Local: ({home['corners_favor']} + {away['corners_contra']}) / 2 = {h:.2f}",
                      f"Visita: ({away['corners_favor']} + {home['corners_contra']}) / 2 = {a:.2f}",
                      f"Total esperado = {h + a:.2f}"]}


def expected_cards(home: dict, away: dict, referee: dict | None) -> dict | None:
    keys = ("amarillas_favor", "amarillas_contra")
    if not all(home.get(k) is not None for k in keys) or not all(away.get(k) is not None for k in keys):
        return None
    h = (home["amarillas_favor"] + away["amarillas_contra"]) / 2
    a = (away["amarillas_favor"] + home["amarillas_contra"]) / 2
    teams_total = h + a
    steps = [f"Local: ({home['amarillas_favor']} + {away['amarillas_contra']}) / 2 = {h:.2f}",
             f"Visita: ({away['amarillas_favor']} + {home['amarillas_contra']}) / 2 = {a:.2f}",
             f"Total por equipos = {teams_total:.2f}"]
    ref_avg = (referee or {}).get("amarillas_pp")
    if ref_avg is not None:
        total = 0.5 * teams_total + 0.5 * ref_avg
        steps.append(f"Mezcla 50/50 con árbitro ({ref_avg:.2f}): {total:.2f}")
    else:
        total = teams_total
        steps.append("Árbitro sin promedio verificado: solo equipos")
    share = h / teams_total if teams_total else 0.5
    return {"home": total * share, "away": total * (1 - share), "total": total, "steps": steps}


def lines_table(mean: float, lines, dispersion: float) -> dict[float, dict[str, float]]:
    out = {}
    for line in lines:
        over = count_over(mean, line, dispersion)
        out[line] = {"over": over, "under": 1 - over}
    return out

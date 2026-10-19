"""Derivación de mercados a partir de una matriz de marcadores.

Todas las funciones reciben una matriz ``M`` donde ``M[i, j]`` es la
probabilidad de que el local marque ``i`` goles y el visitante ``j``.
La matriz puede venir de Poisson, Dixon-Coles o de una simulación de
Monte Carlo: los mercados se calculan igual para todos los modelos.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

MAX_GOALS = 10


def poisson_matrix(lam_home: float, lam_away: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Matriz de marcadores con Poisson independiente (normalizada)."""
    goals = np.arange(max_goals + 1)
    home = poisson.pmf(goals, lam_home)
    away = poisson.pmf(goals, lam_away)
    matrix = np.outer(home, away)
    return matrix / matrix.sum()


def result_1x2(M: np.ndarray) -> dict[str, float]:
    home = float(np.tril(M, -1).sum())
    draw = float(np.trace(M))
    away = float(np.triu(M, 1).sum())
    return {"1": home, "X": draw, "2": away}


def double_chance(M: np.ndarray) -> dict[str, float]:
    p = result_1x2(M)
    return {"1X": p["1"] + p["X"], "X2": p["X"] + p["2"], "12": p["1"] + p["2"]}


def draw_no_bet(M: np.ndarray) -> dict[str, float]:
    p = result_1x2(M)
    decisive = p["1"] + p["2"]
    return {"1 (DNB)": p["1"] / decisive, "2 (DNB)": p["2"] / decisive}


def goal_diff_distribution(M: np.ndarray) -> dict[int, float]:
    """P(goles local - goles visitante = d)."""
    n = M.shape[0]
    dist: dict[int, float] = {}
    for i in range(n):
        for j in range(n):
            dist[i - j] = dist.get(i - j, 0.0) + float(M[i, j])
    return dist


def asian_handicap(M: np.ndarray, line: float, side: str = "home") -> dict[str, float]:
    """Hándicap asiático de línea entera o media (``-1``, ``-0.5``, ``+1.5``...).

    ``line`` se aplica al equipo ``side``. Devuelve probabilidad de ganar,
    de devolución (push) y de perder, más la probabilidad "efectiva"
    ``win / (win + lose)`` que es la que se compara con la cuota.
    """
    if (line * 2) % 1 != 0:
        raise ValueError("Solo líneas enteras o medias (las de cuarto se dividen en dos apuestas)")
    dist = goal_diff_distribution(M)
    sign = 1 if side == "home" else -1
    win = push = lose = 0.0
    for diff, prob in dist.items():
        adjusted = sign * diff + line
        if adjusted > 0:
            win += prob
        elif adjusted == 0:
            push += prob
        else:
            lose += prob
    effective = win / (win + lose) if (win + lose) > 0 else 0.0
    return {"win": win, "push": push, "lose": lose, "effective": effective}


def european_handicap(M: np.ndarray, line: int) -> dict[str, float]:
    """Hándicap europeo (3 vías) aplicado al local: 1 / X / 2 tras el ajuste."""
    dist = goal_diff_distribution(M)
    out = {"1": 0.0, "X": 0.0, "2": 0.0}
    for diff, prob in dist.items():
        adjusted = diff + line
        key = "1" if adjusted > 0 else "X" if adjusted == 0 else "2"
        out[key] += prob
    return out


def totals(M: np.ndarray, lines=(0.5, 1.5, 2.5, 3.5, 4.5)) -> dict[float, dict[str, float]]:
    n = M.shape[0]
    total_goals = np.add.outer(np.arange(n), np.arange(n))
    out = {}
    for line in lines:
        over = float(M[total_goals > line].sum())
        out[line] = {"over": over, "under": 1.0 - over}
    return out


def team_totals(M: np.ndarray, lines=(0.5, 1.5, 2.5)) -> dict[str, dict[float, dict[str, float]]]:
    home_marg = M.sum(axis=1)
    away_marg = M.sum(axis=0)
    out: dict[str, dict[float, dict[str, float]]] = {"home": {}, "away": {}}
    goals = np.arange(M.shape[0])
    for key, marg in (("home", home_marg), ("away", away_marg)):
        for line in lines:
            over = float(marg[goals > line].sum())
            out[key][line] = {"over": over, "under": 1.0 - over}
    return out


def btts(M: np.ndarray) -> dict[str, float]:
    yes = float(M[1:, 1:].sum())
    return {"yes": yes, "no": 1.0 - yes}


def clean_sheets(M: np.ndarray) -> dict[str, float]:
    """Valla invicta del local (visitante no marca) y del visitante."""
    return {"home": float(M[:, 0].sum()), "away": float(M[0, :].sum())}


def top_scores(M: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    flat = np.argsort(M, axis=None)[::-1][:k]
    out = []
    for idx in flat:
        i, j = np.unravel_index(idx, M.shape)
        out.append((f"{i}-{j}", float(M[i, j])))
    return out


def expected_goals(M: np.ndarray) -> tuple[float, float]:
    goals = np.arange(M.shape[0])
    return float(M.sum(axis=1) @ goals), float(M.sum(axis=0) @ goals)


def all_markets(M: np.ndarray) -> dict:
    """Paquete completo de mercados de goles para un partido."""
    return {
        "1x2": result_1x2(M),
        "double_chance": double_chance(M),
        "dnb": draw_no_bet(M),
        "ah_home_-1": asian_handicap(M, -1, "home"),
        "ah_away_+1": asian_handicap(M, +1, "away"),
        "ah_home_+1": asian_handicap(M, +1, "home"),
        "ah_away_-1": asian_handicap(M, -1, "away"),
        "ah_home_-1.5": asian_handicap(M, -1.5, "home"),
        "ah_away_+1.5": asian_handicap(M, +1.5, "away"),
        "eh_home_-1": european_handicap(M, -1),
        "totals": totals(M),
        "team_totals": team_totals(M),
        "btts": btts(M),
        "clean_sheets": clean_sheets(M),
        "top_scores": top_scores(M, 3),
        "expected_goals": expected_goals(M),
    }


def ht_ft(M_first_half: np.ndarray, M_full: np.ndarray) -> dict[str, float]:
    """Aproximación medio tiempo / final suponiendo independencia entre tiempos.

    Se construye la matriz del segundo tiempo como Poisson con la diferencia de
    goles esperados, y se combinan ambos tiempos. Útil solo como referencia.
    """
    eh_full, ea_full = expected_goals(M_full)
    eh_1, ea_1 = expected_goals(M_first_half)
    M2 = poisson_matrix(max(eh_full - eh_1, 0.01), max(ea_full - ea_1, 0.01))
    n = M_first_half.shape[0]
    out: dict[str, float] = {}
    for i1 in range(n):
        for j1 in range(n):
            p1 = M_first_half[i1, j1]
            if p1 < 1e-9:
                continue
            ht = "1" if i1 > j1 else "X" if i1 == j1 else "2"
            for i2 in range(n):
                for j2 in range(n):
                    p = p1 * M2[i2, j2]
                    if p < 1e-12:
                        continue
                    h, a = i1 + i2, j1 + j2
                    ft = "1" if h > a else "X" if h == a else "2"
                    key = f"{ht}/{ft}"
                    out[key] = out.get(key, 0.0) + float(p)
    return out

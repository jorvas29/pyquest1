"""Construcción de variables predictivas a partir del historial de partidos.

Recorre el historial en orden cronológico y, ANTES de cada partido, calcula
las variables de cada equipo (sin mirar el resultado de ese partido, para
no filtrar información del futuro). Esas filas entrenan la regresión
logística y los modelos de machine learning.

Variables generadas (diferencia local - visitante salvo que se indique):

* ``elo_diff``, ``glicko_diff``, ``pi_gd``: ratings dinámicos previos.
* ``ppg_diff``, ``ppg5_diff``: puntos por partido últimos 10 y 5.
* ``gf_diff``, ``ga_diff``: goles a favor / en contra últimos 10.
* ``home_gf``, ``home_ga``: del local jugando DE LOCAL.
* ``away_gf``, ``away_ga``: del visitante jugando DE VISITANTE.
* ``rest_diff``: días de descanso (fatiga).
* ``sot_diff``, ``corners_diff``: tiros al arco y córners si el CSV los trae.
* ``xg_diff``: xG si el CSV trae columnas ``home_xg`` / ``away_xg``.
* ``extra_<nombre>``: cualquier columna ``home_<nombre>`` / ``away_<nombre>``
  adicional que el usuario agregue (presupuesto, bajas, altitud...).
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime

import numpy as np

from .ratings import Elo, Glicko2, PiRatings

BASE_COLUMNS = {"date", "home", "away", "hg", "ag", "hthg", "htag"}
STAT_PAIRS = {"sot": ("home_sot", "away_sot"), "corners": ("home_corners", "away_corners"), "xg": ("home_xg", "away_xg")}


def _parse_date(value):
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(str(value))


class TeamHistory:
    def __init__(self, window: int = 10):
        self.window = window
        self.all = defaultdict(lambda: deque(maxlen=window))
        self.home = defaultdict(lambda: deque(maxlen=window))
        self.away = defaultdict(lambda: deque(maxlen=window))
        self.stats = defaultdict(lambda: defaultdict(lambda: deque(maxlen=window)))
        self.last_date: dict[str, datetime] = {}

    @staticmethod
    def _mean(values, idx, default=np.nan):
        return float(np.mean([v[idx] for v in values])) if values else default

    def team_features(self, team: str, venue: str, date) -> dict[str, float]:
        games = self.all[team]
        last5 = list(games)[-5:]
        venue_games = self.home[team] if venue == "home" else self.away[team]
        rest = (date - self.last_date[team]).days if date and team in self.last_date else np.nan
        out = {
            "ppg": self._mean(games, 2),
            "ppg5": self._mean(last5, 2),
            "gf": self._mean(games, 0),
            "ga": self._mean(games, 1),
            "venue_gf": self._mean(venue_games, 0),
            "venue_ga": self._mean(venue_games, 1),
            "rest": min(rest, 14) if not np.isnan(rest) else np.nan,
            "n": len(games),
        }
        for name, values in self.stats[team].items():
            out[name] = float(np.mean(values)) if values else np.nan
        return out

    def record(self, team: str, venue: str, gf: int, ga: int, date, stats: dict[str, float]):
        pts = 3 if gf > ga else 1 if gf == ga else 0
        self.all[team].append((gf, ga, pts))
        (self.home if venue == "home" else self.away)[team].append((gf, ga, pts))
        for name, value in stats.items():
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                self.stats[team][name].append(float(value))
        if date:
            self.last_date[team] = date


def _extra_names(matches: list[dict]) -> list[str]:
    keys = set().union(*(m.keys() for m in matches)) if matches else set()
    names = []
    for k in keys:
        if k.startswith("home_") and f"away_{k[5:]}" in keys:
            name = k[5:]
            if name not in {"sot", "corners", "xg"}:
                names.append(name)
    return sorted(names)


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


class FeatureBuilder:
    def __init__(self, window: int = 10, min_games: int = 3):
        self.window = window
        self.min_games = min_games

    def _row(self, hist, elo, glicko, pi, home, away, date, extras):
        fh = hist.team_features(home, "home", date)
        fa = hist.team_features(away, "away", date)
        row = {
            "elo_diff": elo.rating(home) + elo.home_adv - elo.rating(away),
            "glicko_diff": glicko.player(home).rating - glicko.player(away).rating,
            "pi_gd": pi.expected_goal_diff(home, away),
            "ppg_diff": fh["ppg"] - fa["ppg"],
            "ppg5_diff": fh["ppg5"] - fa["ppg5"],
            "gf_diff": fh["gf"] - fa["gf"],
            "ga_diff": fh["ga"] - fa["ga"],
            "home_gf": fh["venue_gf"],
            "home_ga": fh["venue_ga"],
            "away_gf": fa["venue_gf"],
            "away_ga": fa["venue_ga"],
            "rest_diff": fh["rest"] - fa["rest"],
        }
        for name in STAT_PAIRS:
            if f"{name}_for" in fh or f"{name}_for" in fa:
                row[f"{name}_diff"] = fh.get(f"{name}_for", np.nan) - fa.get(f"{name}_for", np.nan)
        for name, (hv, av) in extras.items():
            row[f"extra_{name}"] = hv - av
        return row, min(fh["n"], fa["n"])

    def build(self, matches: list[dict]):
        """Devuelve (X, y, nombres_columnas, estado) para entrenar.

        ``y``: 0 = gana local, 1 = empate, 2 = gana visitante.
        ``estado`` permite construir luego la fila del partido a predecir.
        """
        matches = sorted(matches, key=lambda m: _parse_date(m.get("date")) or datetime.min)
        extra_names = _extra_names(matches)
        hist = TeamHistory(self.window)
        elo, glicko, pi = Elo(), Glicko2(), PiRatings()
        rows, labels = [], []
        for m in matches:
            date = _parse_date(m.get("date"))
            home, away, hg, ag = m["home"], m["away"], int(m["hg"]), int(m["ag"])
            extras = {n: (_num(m.get(f"home_{n}")), _num(m.get(f"away_{n}"))) for n in extra_names}
            row, n_games = self._row(hist, elo, glicko, pi, home, away, date, extras)
            if n_games >= self.min_games:
                rows.append(row)
                labels.append(0 if hg > ag else 1 if hg == ag else 2)

            stats_h, stats_a = {}, {}
            for name, (hc, ac) in STAT_PAIRS.items():
                hv, av = _num(m.get(hc)), _num(m.get(ac))
                stats_h[f"{name}_for"], stats_a[f"{name}_for"] = hv, av
            hist.record(home, "home", hg, ag, date, stats_h)
            hist.record(away, "away", ag, hg, date, stats_a)
            elo.update(home, away, hg, ag)
            glicko.update(home, away, hg, ag)
            pi.update(home, away, hg, ag)

        columns = sorted({k for r in rows for k in r})
        X = np.array([[r.get(c, np.nan) for c in columns] for r in rows], dtype=float)
        y = np.array(labels, dtype=int)
        state = {"hist": hist, "elo": elo, "glicko": glicko, "pi": pi, "extra_names": extra_names}
        return X, y, columns, state

    def match_row(self, state, columns, home, away, date=None, extras=None) -> np.ndarray:
        extras = extras or {}
        ex = {n: extras.get(n, (np.nan, np.nan)) for n in state["extra_names"]}
        row, _ = self._row(state["hist"], state["elo"], state["glicko"], state["pi"], home, away, _parse_date(date), ex)
        return np.array([[row.get(c, np.nan) for c in columns]], dtype=float)

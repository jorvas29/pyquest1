"""Carga de historiales en CSV.

Acepta dos formatos:

* El de football-data.co.uk (``Date, HomeTeam, AwayTeam, FTHG, FTAG, HTHG,
  HTAG, HST, AST, HC, AC, HY, AY`` ...), que se traduce automáticamente.
* Uno propio con columnas ``date, home, away, hg, ag`` y opcionales
  ``home_xg, away_xg, home_sot, away_sot, home_corners, away_corners`` y
  cualquier par ``home_<x>`` / ``away_<x>`` (se vuelve variable para ML).
"""

from __future__ import annotations

import csv
from pathlib import Path

from .features import _parse_date

FOOTBALL_DATA_MAP = {
    "Date": "date", "HomeTeam": "home", "AwayTeam": "away", "FTHG": "hg", "FTAG": "ag",
    "HTHG": "hthg", "HTAG": "htag", "HST": "home_sot", "AST": "away_sot",
    "HC": "home_corners", "AC": "away_corners", "HY": "home_yellows", "AY": "away_yellows",
    "HF": "home_fouls", "AF": "away_fouls",
}


def load_matches(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    football_data = "HomeTeam" in rows[0]
    out = []
    for r in rows:
        if football_data:
            m = {FOOTBALL_DATA_MAP[k]: v for k, v in r.items() if k in FOOTBALL_DATA_MAP}
        else:
            m = dict(r)
        if not m.get("home") or m.get("hg") in (None, "") or m.get("ag") in (None, ""):
            continue  # partidos sin jugar o filas vacías
        m["hg"], m["ag"] = int(float(m["hg"])), int(float(m["ag"]))
        if m.get("date"):
            m["date"] = _parse_date(m["date"]).date().isoformat()
        out.append(m)
    return out

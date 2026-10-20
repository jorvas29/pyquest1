"""Scraper del xG de la Champions League en FootyStats.

Descarga https://footystats.org/europe/uefa-champions-league/xg y extrae la
tabla de xG por equipo (xG a favor, xG en contra, partidos...). No depende
de clases CSS concretas: recorre todas las tablas de la página y elige la
que tenga encabezados de xG, así sobrevive a cambios de diseño.

Modos de descarga:

* Directo (por defecto): una petición HTTP normal con cabeceras de navegador.
* Bright Data Web Unlocker: si están definidas ``BRIGHTDATA_API_KEY`` y
  ``BRIGHTDATA_UNLOCKER_ZONE`` (útil si FootyStats bloquea la petición).
* Archivo local: ``--html pagina.html`` (página guardada desde el navegador).

Uso:

    python -m futquant.footystats -o ucl_xg.csv
    python -m futquant.footystats --html guardada.html -o ucl_xg.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

UCL_XG_URL = "https://footystats.org/europe/uefa-champions-league/xg"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}


# ------------------------------------------------------------------ descarga

def fetch_direct(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - se reintenta cualquier fallo de red
            if attempt == retries - 1:
                raise RuntimeError(f"No se pudo descargar {url}: {exc}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("inalcanzable")


def fetch_brightdata(url: str, retries: int = 3) -> str:
    api_key = os.environ["BRIGHTDATA_API_KEY"]
    zone = os.environ["BRIGHTDATA_UNLOCKER_ZONE"]
    body = json.dumps({"zone": zone, "url": url, "format": "raw"}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://api.brightdata.com/request", data=body, method="POST",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise RuntimeError(f"Bright Data no pudo descargar {url}: {exc}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("inalcanzable")


def fetch(url: str) -> str:
    if os.environ.get("BRIGHTDATA_API_KEY") and os.environ.get("BRIGHTDATA_UNLOCKER_ZONE"):
        return fetch_brightdata(url)
    return fetch_direct(url)


# ------------------------------------------------------------------ parser

class _TableParser(HTMLParser):
    """Convierte cada <table> del HTML en una lista de filas de texto."""

    def __init__(self):
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._stack: list[list[list[str]]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._stack.append([])
        elif tag == "tr" and self._stack:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._stack:
            if self._row:
                self._stack[-1].append(self._row)
            self._row = None
        elif tag == "table" and self._stack:
            self.tables.append(self._stack.pop())

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _num(text: str) -> float | None:
    m = re.search(r"-?\d+(?:[.,]\d+)?", text or "")
    return float(m.group().replace(",", ".")) if m else None


def _norm(header: str) -> str:
    return re.sub(r"[^a-z0-9%]", "", header.lower())


# Alias de encabezados -> campo de salida (se comparan normalizados).
FIELDS = {
    "team": ("team", "equipo", "club"),
    "mp": ("mp", "pj", "matches", "partidos", "gp"),
    "xg": ("xg", "xgfor", "xgf", "xgfavor"),
    "xga": ("xga", "xgagainst", "xgcontra"),
    "xg_per_match": ("xgmatch", "xgpermatch", "xg90", "xgpartido", "xgpg"),
    "xga_per_match": ("xgamatch", "xgapermatch", "xga90", "xgapartido", "xgapg"),
    "gf": ("gf", "goals", "goalsfor", "golesfavor"),
    "ga": ("ga", "goalsagainst", "golescontra"),
}


def _map_headers(header_row: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, raw in enumerate(header_row):
        h = _norm(raw)
        for field, aliases in FIELDS.items():
            if field not in mapping and h in aliases:
                mapping[field] = idx
    return mapping


def parse_xg_table(html: str) -> list[dict]:
    """Devuelve una fila por equipo con xG/xGA por partido (calculados si hace falta)."""
    parser = _TableParser()
    parser.feed(html)
    best: list[dict] = []
    for table in parser.tables:
        if len(table) < 3:
            continue
        mapping = _map_headers(table[0])
        if "xg" not in mapping and "xg_per_match" not in mapping:
            continue
        if "team" not in mapping:
            # La columna de equipo suele ser la primera con texto no numérico.
            sample = table[1]
            mapping["team"] = next((i for i, c in enumerate(sample) if c and _num(c) is None), 0)
        rows = []
        for raw in table[1:]:
            try:
                row = {"equipo": raw[mapping["team"]]}
                for field, idx in mapping.items():
                    if field != "team" and idx < len(raw):
                        row[field] = _num(raw[idx])
                mp = row.get("mp")
                if row.get("xg_per_match") is None and row.get("xg") is not None and mp:
                    row["xg_per_match"] = round(row["xg"] / mp, 3)
                if row.get("xga_per_match") is None and row.get("xga") is not None and mp:
                    row["xga_per_match"] = round(row["xga"] / mp, 3)
                if row["equipo"] and row.get("xg_per_match") is not None:
                    rows.append(row)
            except (IndexError, KeyError, TypeError):
                continue
        if len(rows) > len(best):
            best = rows
    return best


COLUMNS = ["equipo", "mp", "xg", "xga", "xg_per_match", "xga_per_match", "gf", "ga"]


def save_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="xG de la Champions League desde FootyStats")
    ap.add_argument("--url", default=UCL_XG_URL)
    ap.add_argument("--html", type=Path, help="Usar un HTML guardado en vez de descargar")
    ap.add_argument("-o", "--salida", type=Path, default=Path("ucl_xg.csv"))
    args = ap.parse_args(argv)

    html = args.html.read_text(encoding="utf-8", errors="replace") if args.html else fetch(args.url)
    rows = parse_xg_table(html)
    if not rows:
        print("No se encontró una tabla de xG. Guarda la página desde el navegador y usa --html.", file=sys.stderr)
        return 1
    save_csv(rows, args.salida)
    print(f"{len(rows)} equipos guardados en {args.salida}")
    for r in rows[:5]:
        print(f"  {r['equipo']}: xG {r.get('xg_per_match')} · xGA {r.get('xga_per_match')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

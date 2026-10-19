"""Línea de comandos.

    python -m futquant ficha partido.json [--historial liga.csv] [--sims 10000] [-o ficha.md]
    python -m futquant demo-historial liga_demo.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .analysis import analyze
from .data_io import load_matches
from .ficha import render


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="futquant", description="Ficha cuantitativa de un partido")
    sub = parser.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("ficha", help="Genera la ficha del partido desde un JSON de variables")
    f.add_argument("partido", type=Path)
    f.add_argument("--historial", type=Path, help="CSV de partidos (football-data.co.uk o formato propio)")
    f.add_argument("--sims", type=int, default=10_000, help="Simulaciones de Monte Carlo")
    f.add_argument("-o", "--salida", type=Path, help="Archivo Markdown de salida")

    d = sub.add_parser("demo-historial", help="Escribe un CSV de liga FICTICIA para probar los modelos")
    d.add_argument("salida", type=Path)

    args = parser.parse_args(argv)
    if args.cmd == "demo-historial":
        from .synthetic import synthetic_league

        matches, _ = synthetic_league()
        with open(args.salida, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(matches[0].keys()))
            writer.writeheader()
            writer.writerows(matches)
        print(f"{len(matches)} partidos ficticios en {args.salida}")
        return 0

    data = json.loads(args.partido.read_text(encoding="utf-8"))
    history = load_matches(args.historial) if args.historial else None
    md = render(analyze(data, history, args.sims))
    if args.salida:
        args.salida.write_text(md, encoding="utf-8")
        print(f"Ficha escrita en {args.salida}")
    else:
        sys.stdout.write(md)
    return 0

"""Punto de entrada de línea de comandos.

Ejemplos:

    python -m predictor generar-sintetico
    python -m predictor equipos
    python -m predictor predecir "Sintético 01 FC" "Sintético 02 FC"
    python -m predictor predecir LOCAL VISITANTE --csv datos/E0_2425.csv --cuotas 1=2.10 X=3.40 2=3.60
    python -m predictor evaluar --salida-md informes/evaluacion.md

Si no se indica fuente (--csv o --api-config) se usan datos SINTÉTICOS.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

import pandas as pd

from . import ADVERTENCIA
from .datos import FuenteAPI, FuenteCSV, FuenteDatos, FuenteSintetica, generar_liga_sintetica
from .evaluacion import backtest, calibraciones, resumen_metricas, simular_valor
from .informe import markdown_evaluacion, texto_prediccion
from .prediccion import ModeloConjunto

CLAVES_CUOTAS = {
    "1": "1",
    "x": "X",
    "2": "2",
    "btts_si": "btts_si",
    "btts_no": "btts_no",
    "corners_mas": "corners_mas",
    "corners_menos": "corners_menos",
}


# ------------------------------------------------------------- utilidades
def _anadir_fuente(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("fuente de datos (por defecto: sintética)")
    excl = g.add_mutually_exclusive_group()
    excl.add_argument("--csv", type=Path, help="CSV local (esquema común o football-data.co.uk)")
    excl.add_argument("--api-config", type=Path, help="JSON con url, mapeo y token de una API")
    excl.add_argument("--sintetico", action="store_true", help="usar la liga sintética (por defecto)")
    g.add_argument("--semilla", type=int, default=42, help="semilla de la liga sintética")


def crear_fuente(args: argparse.Namespace) -> FuenteDatos:
    if args.csv:
        return FuenteCSV(args.csv)
    if args.api_config:
        return FuenteAPI.desde_config(args.api_config)
    return FuenteSintetica(semilla=args.semilla)


def parsear_cuotas(textos: list[str] | None) -> dict[str, float]:
    """Convierte ["1=2.10", "X=3.4", "btts_si=1.8"] en {"1": 2.1, "X": 3.4, ...}."""
    cuotas: dict[str, float] = {}
    for texto in textos or []:
        if "=" not in texto:
            raise ValueError(f"Cuota mal escrita: '{texto}'. Usa clave=valor, p. ej. 1=2.10")
        clave, valor = texto.split("=", 1)
        clave_norm = CLAVES_CUOTAS.get(clave.strip().lower())
        if clave_norm is None:
            raise ValueError(f"Clave de cuota desconocida: '{clave}'. Válidas: {', '.join(CLAVES_CUOTAS)}")
        cuota = float(valor.replace(",", "."))
        if cuota <= 1.0:
            raise ValueError(f"La cuota de '{clave}' debe ser mayor que 1 (formato decimal).")
        cuotas[clave_norm] = cuota
    return cuotas


def resolver_equipo(nombre: str, equipos: list[str]) -> str:
    """Busca el equipo sin distinguir mayúsculas; si no existe, sugiere parecidos."""
    por_minusculas = {e.lower(): e for e in equipos}
    if nombre.lower() in por_minusculas:
        return por_minusculas[nombre.lower()]
    parecidos = difflib.get_close_matches(nombre, equipos, n=3, cutoff=0.5)
    pista = f" ¿Quizás: {', '.join(parecidos)}?" if parecidos else ""
    raise ValueError(f"Equipo no encontrado en los datos: '{nombre}'.{pista}")


def _contexto(fuente: FuenteDatos, partidos: pd.DataFrame) -> dict[str, str]:
    return {
        "Fuente de datos": fuente.descripcion,
        "Partidos usados": f"{len(partidos)} (del {partidos['fecha'].min():%Y-%m-%d} "
        f"al {partidos['fecha'].max():%Y-%m-%d})",
    }


# --------------------------------------------------------------- comandos
def cmd_generar(args) -> int:
    df = generar_liga_sintetica(semilla=args.semilla, n_equipos=args.equipos, temporadas=args.temporadas)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.salida, index=False)
    print(f"Liga SINTÉTICA guardada en {args.salida} ({len(df)} partidos, datos inventados).")
    return 0


def cmd_equipos(args) -> int:
    fuente = crear_fuente(args)
    partidos = fuente.cargar()
    equipos = sorted(set(partidos["local"]) | set(partidos["visitante"]))
    print(f"Fuente: {fuente.descripcion}")
    print(f"{len(equipos)} equipos:")
    for e in equipos:
        print(f"  - {e}")
    return 0


def cmd_predecir(args) -> int:
    fuente = crear_fuente(args)
    partidos = fuente.cargar()
    if args.fecha:
        partidos = partidos[partidos["fecha"] < pd.Timestamp(args.fecha)]
        if partidos.empty:
            raise ValueError(f"No hay partidos anteriores a {args.fecha}.")
    equipos = sorted(set(partidos["local"]) | set(partidos["visitante"]))
    local = resolver_equipo(args.local, equipos)
    visitante = resolver_equipo(args.visitante, equipos)
    if local == visitante:
        raise ValueError("El local y el visitante deben ser equipos distintos.")
    cuotas = parsear_cuotas(args.cuotas)

    modelo = ModeloConjunto(xi=args.xi, usar_rho=(args.modelo == "dixon-coles")).ajustar(partidos)
    pred = modelo.predecir(local, visitante, linea_corners=args.linea_corners)
    print(texto_prediccion(pred, _contexto(fuente, partidos), cuotas, args.umbral_valor))
    return 0


def cmd_evaluar(args) -> int:
    fuente = crear_fuente(args)
    partidos = fuente.cargar()
    print(f"Evaluando con validación temporal ({len(partidos)} partidos)...", file=sys.stderr)
    pred = backtest(
        partidos,
        min_entrenamiento=args.min_entrenamiento,
        paso_dias=args.paso_dias,
        linea_corners=args.linea_corners,
        xi=args.xi,
        usar_rho=(args.modelo == "dixon-coles"),
    )
    resumen = resumen_metricas(pred)
    calib = calibraciones(pred)
    valor = simular_valor(pred, args.umbral_valor)
    contexto = _contexto(fuente, partidos)
    contexto.update(
        {
            "Partidos evaluados": f"{len(pred)} (entrenamiento inicial: {args.min_entrenamiento}, "
            f"reajuste cada {args.paso_dias} días)",
            "Modelo de goles": "Dixon-Coles" if args.modelo == "dixon-coles" else "Poisson independiente",
        }
    )
    texto = markdown_evaluacion(resumen, calib, valor, contexto, args.umbral_valor)
    print(texto)
    if args.salida_md:
        args.salida_md.parent.mkdir(parents=True, exist_ok=True)
        args.salida_md.write_text(texto, encoding="utf-8")
        print(f"Informe guardado en {args.salida_md}", file=sys.stderr)
    if args.salida_csv:
        args.salida_csv.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(args.salida_csv, index=False)
        print(f"Predicciones partido a partido guardadas en {args.salida_csv}", file=sys.stderr)
    return 0


# ------------------------------------------------------------------ parser
def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m predictor",
        description="Predicción de BTTS, córners y 1X2 con modelos probabilísticos.",
        epilog=ADVERTENCIA,
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("generar-sintetico", help="guardar una liga sintética en CSV")
    p.add_argument("--salida", type=Path, default=Path("datos/liga_sintetica.csv"))
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--equipos", type=int, default=20)
    p.add_argument("--temporadas", type=int, default=3)
    p.set_defaults(funcion=cmd_generar)

    p = sub.add_parser("equipos", help="listar los equipos de la fuente")
    _anadir_fuente(p)
    p.set_defaults(funcion=cmd_equipos)

    def opciones_modelo(p):
        p.add_argument("--modelo", choices=["dixon-coles", "poisson"], default="dixon-coles")
        p.add_argument("--xi", type=float, default=0.0019, help="olvido temporal por día (0 = sin olvido)")
        p.add_argument("--linea-corners", type=float, default=9.5)
        p.add_argument("--umbral-valor", type=float, default=0.05, help="EV mínimo para marcar valor")

    p = sub.add_parser("predecir", help="probabilidades y cuotas de un partido")
    p.add_argument("local")
    p.add_argument("visitante")
    p.add_argument(
        "--cuotas",
        nargs="*",
        metavar="CLAVE=CUOTA",
        help="cuotas de mercado: 1, X, 2, btts_si, btts_no, corners_mas, corners_menos",
    )
    p.add_argument("--fecha", help="predecir como si fuera esta fecha (usa solo partidos anteriores)")
    _anadir_fuente(p)
    opciones_modelo(p)
    p.set_defaults(funcion=cmd_predecir)

    p = sub.add_parser("evaluar", help="validación temporal, métricas y valor esperado")
    p.add_argument("--min-entrenamiento", type=int, default=150)
    p.add_argument("--paso-dias", type=int, default=7)
    p.add_argument("--salida-md", type=Path, help="guardar el informe en Markdown")
    p.add_argument("--salida-csv", type=Path, help="guardar las predicciones partido a partido")
    _anadir_fuente(p)
    opciones_modelo(p)
    p.set_defaults(funcion=cmd_evaluar)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)
    try:
        return args.funcion(args)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

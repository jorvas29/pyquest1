"""Esquema común de partidos.

Todas las fuentes de datos (CSV, API, sintética) terminan devolviendo un
DataFrame con estas columnas, para que el resto del programa no tenga que
saber de dónde vienen los datos.
"""

from __future__ import annotations

import pandas as pd

# Columnas sin las que no se puede ajustar ningún modelo.
COLUMNAS_OBLIGATORIAS = [
    "fecha",            # fecha del partido (datetime)
    "local",            # nombre del equipo local
    "visitante",        # nombre del equipo visitante
    "goles_local",      # goles del local al final del partido
    "goles_visitante",  # goles del visitante al final del partido
]

# Columnas que se usan si existen.
COLUMNAS_OPCIONALES = [
    "corners_local",
    "corners_visitante",
    "cuota_1",              # cuota de mercado: gana el local
    "cuota_x",              # cuota de mercado: empate
    "cuota_2",              # cuota de mercado: gana el visitante
    "cuota_btts_si",        # cuota de mercado: ambos marcan, sí
    "cuota_btts_no",        # cuota de mercado: ambos marcan, no
    "linea_corners",        # línea de córners totales (p. ej. 9.5)
    "cuota_corners_mas",    # cuota de mercado: más de la línea
    "cuota_corners_menos",  # cuota de mercado: menos de la línea
    "origen",               # etiqueta de procedencia (p. ej. "SINTETICO")
]

COLUMNAS_NUMERICAS = [
    "goles_local",
    "goles_visitante",
    "corners_local",
    "corners_visitante",
    "cuota_1",
    "cuota_x",
    "cuota_2",
    "cuota_btts_si",
    "cuota_btts_no",
    "linea_corners",
    "cuota_corners_mas",
    "cuota_corners_menos",
]


def normalizar(df: pd.DataFrame, dayfirst: bool = False) -> pd.DataFrame:
    """Valida y limpia un DataFrame ya renombrado al esquema común.

    - Comprueba que existen las columnas obligatorias.
    - Convierte la fecha a datetime (sin zona horaria) y los números a float.
    - Descarta partidos sin marcador (por ejemplo, partidos aún no jugados).
    - Ordena por fecha y reinicia el índice.
    """
    faltan = [c for c in COLUMNAS_OBLIGATORIAS if c not in df.columns]
    if faltan:
        raise ValueError(
            f"Faltan columnas obligatorias: {faltan}. "
            f"Columnas recibidas: {list(df.columns)}"
        )

    columnas = [c for c in COLUMNAS_OBLIGATORIAS + COLUMNAS_OPCIONALES if c in df.columns]
    salida = df[columnas].copy()

    fechas = pd.to_datetime(salida["fecha"], dayfirst=dayfirst, utc=True, errors="coerce")
    salida["fecha"] = fechas.dt.tz_localize(None).dt.normalize()

    for col in COLUMNAS_NUMERICAS:
        if col in salida.columns:
            salida[col] = pd.to_numeric(salida[col], errors="coerce")

    salida["local"] = salida["local"].astype(str).str.strip()
    salida["visitante"] = salida["visitante"].astype(str).str.strip()

    salida = salida.dropna(subset=["fecha", "goles_local", "goles_visitante"])
    salida = salida.sort_values("fecha", kind="stable").reset_index(drop=True)

    if salida.empty:
        raise ValueError("No hay partidos jugados (con marcador) en los datos.")
    return salida


def tiene_corners(partidos: pd.DataFrame) -> bool:
    """True si hay datos de córners suficientes para ajustar su modelo."""
    if "corners_local" not in partidos.columns or "corners_visitante" not in partidos.columns:
        return False
    validos = partidos[["corners_local", "corners_visitante"]].notna().all(axis=1)
    return int(validos.sum()) >= 30

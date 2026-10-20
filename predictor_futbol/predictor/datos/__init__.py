"""Obtención de datos: esquema común y fuentes intercambiables."""

from .esquema import COLUMNAS_OBLIGATORIAS, COLUMNAS_OPCIONALES, normalizar, tiene_corners
from .fuentes import (
    MAPEO_FOOTBALL_DATA,
    FuenteAPI,
    FuenteCSV,
    FuenteDatos,
    FuenteSintetica,
)
from .sintetico import generar_liga_sintetica

__all__ = [
    "COLUMNAS_OBLIGATORIAS",
    "COLUMNAS_OPCIONALES",
    "MAPEO_FOOTBALL_DATA",
    "FuenteAPI",
    "FuenteCSV",
    "FuenteDatos",
    "FuenteSintetica",
    "generar_liga_sintetica",
    "normalizar",
    "tiene_corners",
]

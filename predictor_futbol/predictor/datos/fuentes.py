"""Capa de fuentes de datos intercambiable.

Cada fuente implementa `cargar()` y devuelve un DataFrame en el esquema común
(ver `esquema.py`). El resto del programa solo trabaja con ese DataFrame, así
que cambiar de CSV a API (o a datos sintéticos) no toca los modelos.

    FuenteDatos       -> interfaz abstracta
    FuenteCSV         -> fichero CSV local (detecta el formato football-data.co.uk)
    FuenteAPI         -> API HTTP que devuelve JSON, con mapeo de columnas
    FuenteSintetica   -> liga inventada para demostrar el flujo
"""

from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Iterable, Mapping

import pandas as pd

from .esquema import normalizar
from .sintetico import generar_liga_sintetica

# Columnas de los CSV gratuitos de football-data.co.uk -> esquema común.
MAPEO_FOOTBALL_DATA = {
    "Date": "fecha",
    "HomeTeam": "local",
    "AwayTeam": "visitante",
    "FTHG": "goles_local",
    "FTAG": "goles_visitante",
    "HC": "corners_local",
    "AC": "corners_visitante",
    "B365H": "cuota_1",
    "B365D": "cuota_x",
    "B365A": "cuota_2",
}


class FuenteDatos(ABC):
    """Interfaz común: cualquier fuente sabe cargar partidos y describirse."""

    #: Texto corto que se muestra en los informes.
    descripcion: str = "fuente desconocida"

    #: True si los datos son inventados (se avisa en todos los informes).
    es_sintetica: bool = False

    @abstractmethod
    def cargar(self) -> pd.DataFrame:
        """Devuelve los partidos jugados en el esquema común."""


class FuenteCSV(FuenteDatos):
    """Lee un CSV local.

    Si el CSV ya usa el esquema común, se lee tal cual. Si tiene las columnas
    de football-data.co.uk (`HomeTeam`, `FTHG`, ...), se renombran solas. Para
    cualquier otro formato se puede pasar `mapeo` {columna_csv: columna_comun}.
    """

    def __init__(self, ruta: str | Path, mapeo: Mapping[str, str] | None = None):
        self.ruta = Path(ruta)
        self.mapeo = dict(mapeo) if mapeo else None
        self.descripcion = f"CSV local ({self.ruta})"

    def cargar(self) -> pd.DataFrame:
        if not self.ruta.exists():
            raise FileNotFoundError(f"No existe el fichero de datos: {self.ruta}")

        crudo = pd.read_csv(self.ruta)

        mapeo = self.mapeo
        dayfirst = False
        if mapeo is None and "HomeTeam" in crudo.columns:
            # football-data.co.uk escribe las fechas como dd/mm/aa o dd/mm/aaaa.
            mapeo = MAPEO_FOOTBALL_DATA
            dayfirst = True

        if mapeo:
            crudo = crudo.rename(columns=mapeo)

        partidos = normalizar(crudo, dayfirst=dayfirst)
        if "origen" in partidos.columns and (partidos["origen"] == "SINTETICO").all():
            self.es_sintetica = True
            self.descripcion += " [DATOS SINTÉTICOS]"
        return partidos


class FuenteAPI(FuenteDatos):
    """Lee partidos de una API HTTP que devuelve JSON.

    Parámetros
    ----------
    url:
        Dirección completa del recurso.
    mapeo:
        {campo_json: columna_comun}. Los campos anidados se escriben con
        puntos, por ejemplo "score.fullTime.home" (se aplanan con
        `pandas.json_normalize`).
    ruta_registros:
        Claves para llegar a la lista de partidos dentro del JSON, por
        ejemplo ["matches"].
    cabeceras:
        Cabeceras HTTP fijas.
    cabecera_token / variable_token:
        Si se indican, se lee el token de la variable de entorno y se envía
        en esa cabecera (así la clave nunca queda escrita en el código).
    obtener_json:
        Función opcional url -> objeto JSON. Sirve para las pruebas, o para
        usar otra librería HTTP.
    """

    def __init__(
        self,
        url: str,
        mapeo: Mapping[str, str],
        ruta_registros: Iterable[str] = (),
        cabeceras: Mapping[str, str] | None = None,
        cabecera_token: str | None = None,
        variable_token: str | None = None,
        obtener_json: Callable[[str], object] | None = None,
        timeout: float = 30.0,
    ):
        self.url = url
        self.mapeo = dict(mapeo)
        self.ruta_registros = list(ruta_registros)
        self.cabeceras = dict(cabeceras or {})
        self.cabecera_token = cabecera_token
        self.variable_token = variable_token
        self.timeout = timeout
        self._obtener_json = obtener_json or self._obtener_http
        self.descripcion = f"API ({url})"

    @classmethod
    def desde_config(cls, ruta: str | Path) -> "FuenteAPI":
        """Crea la fuente a partir de un fichero JSON de configuración."""
        with open(ruta, encoding="utf-8") as f:
            config = json.load(f)
        return cls(
            url=config["url"],
            mapeo=config["mapeo"],
            ruta_registros=config.get("ruta_registros", []),
            cabeceras=config.get("cabeceras"),
            cabecera_token=config.get("cabecera_token"),
            variable_token=config.get("variable_token"),
        )

    def _obtener_http(self, url: str) -> object:
        cabeceras = dict(self.cabeceras)
        if self.cabecera_token and self.variable_token:
            token = os.environ.get(self.variable_token)
            if not token:
                raise RuntimeError(
                    f"Define la variable de entorno {self.variable_token} con tu clave de la API."
                )
            cabeceras[self.cabecera_token] = token
        peticion = urllib.request.Request(url, headers=cabeceras)
        with urllib.request.urlopen(peticion, timeout=self.timeout) as respuesta:
            return json.loads(respuesta.read().decode("utf-8"))

    def cargar(self) -> pd.DataFrame:
        datos = self._obtener_json(self.url)
        for clave in self.ruta_registros:
            datos = datos[clave]
        crudo = pd.json_normalize(datos)
        crudo = crudo.rename(columns=self.mapeo)
        return normalizar(crudo)


class FuenteSintetica(FuenteDatos):
    """Liga inventada, generada con semilla fija (reproducible).

    Los nombres de equipo ("Sintético 01 FC", ...) y la columna
    `origen = "SINTETICO"` dejan claro que NO son datos reales.
    """

    es_sintetica = True

    def __init__(self, semilla: int = 42, n_equipos: int = 20, temporadas: int = 3):
        self.semilla = semilla
        self.n_equipos = n_equipos
        self.temporadas = temporadas
        self.descripcion = (
            f"SINTÉTICA (datos inventados para demostración, semilla={semilla})"
        )

    def cargar(self) -> pd.DataFrame:
        crudo = generar_liga_sintetica(
            n_equipos=self.n_equipos, temporadas=self.temporadas, semilla=self.semilla
        )
        return normalizar(crudo)

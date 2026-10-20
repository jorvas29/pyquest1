"""Ingeniería de características previas al partido.

Para cada partido se calculan, para el local (prefijo `l_`) y el visitante
(prefijo `v_`), estas variables:

    forma     -> puntos por partido en los últimos `ventana_forma` partidos
    gf, gc    -> goles a favor / en contra por partido (últimos `ventana_promedios`)
    cf, cc    -> córners a favor / en contra por partido (últimos `ventana_promedios`)
    gf_cond   -> goles a favor jugando en la misma condición (casa o fuera)
    gc_cond   -> goles en contra jugando en la misma condición (casa o fuera)
    elo       -> rating Elo: la "ventaja de calidad estimada"
    pj        -> partidos jugados hasta ese momento

y para el partido:

    dif_elo         -> elo local - elo visitante
    prob_local_elo  -> probabilidad de victoria local según Elo (con localía)

Sin fuga de datos del futuro
----------------------------
Los partidos se recorren en orden cronológico agrupados por fecha. Para cada
fecha se calculan PRIMERO las características de todos sus partidos y SOLO
DESPUÉS se actualiza el historial con sus resultados. Así, ningún partido ve
su propio resultado ni el de partidos posteriores (ni el de partidos del
mismo día).

Suavizado
---------
Con pocos partidos jugados, una media simple es muy ruidosa (1 partido con 4
goles daría gf = 4). Por eso cada media se "encoge" hacia la media de la
liga hasta ese momento:  (suma + k * media_liga) / (n + k), con k = `suavizado`.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import pandas as pd

COLUMNAS_EQUIPO = ["forma", "gf", "gc", "cf", "cc", "gf_cond", "gc_cond", "elo", "pj"]


@dataclass
class ConfigCaracteristicas:
    ventana_forma: int = 5
    ventana_promedios: int = 10
    suavizado: float = 2.0
    elo_inicial: float = 1500.0
    elo_k: float = 20.0
    elo_ventaja_local: float = 60.0


def _media_suavizada(valores, prior: float, k: float) -> float:
    n = len(valores)
    return (sum(valores) + k * prior) / (n + k)


def _multiplicador_goles(diferencia: int) -> float:
    """Multiplicador del Elo de fútbol: las goleadas mueven más el rating."""
    d = abs(diferencia)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11.0 + d) / 8.0


class MotorCaracteristicas:
    """Mantiene el historial de cada equipo y calcula características.

    Uso:
        motor = MotorCaracteristicas()
        motor.caracteristicas_partido("A", "B")   # antes del partido
        motor.actualizar("A", "B", 2, 1, 6, 3)     # después del partido
    """

    def __init__(self, config: ConfigCaracteristicas | None = None):
        self.config = config or ConfigCaracteristicas()
        largo = max(self.config.ventana_forma, self.config.ventana_promedios)
        self._resultados: dict[str, deque] = {}
        self._corners: dict[str, deque] = {}
        self._condicion: dict[tuple[str, bool], deque] = {}
        self._elo: dict[str, float] = {}
        self._jugados: dict[str, int] = {}
        self._largo = largo
        # Acumulados de liga (para los valores a priori del suavizado).
        self._goles_casa = [0.0, 0]
        self._goles_fuera = [0.0, 0]
        self._corners_liga = [0.0, 0]
        self._puntos_liga = [0.0, 0]

    # ----------------------------------------------------------- a priori
    @staticmethod
    def _media(acumulado, defecto):
        suma, n = acumulado
        return suma / n if n else defecto

    def _prior_goles(self, en_casa: bool | None = None) -> float:
        casa = self._media(self._goles_casa, 1.5)
        fuera = self._media(self._goles_fuera, 1.2)
        if en_casa is None:
            return (casa + fuera) / 2.0
        return casa if en_casa else fuera

    # ------------------------------------------------------ características
    def caracteristicas_equipo(self, equipo: str, en_casa: bool) -> dict[str, float]:
        c = self.config
        k = c.suavizado
        res = list(self._resultados.get(equipo, []))
        cor = list(self._corners.get(equipo, []))
        cond = list(self._condicion.get((equipo, en_casa), []))

        prior_g = self._prior_goles()
        prior_c = self._media(self._corners_liga, 5.0)
        prior_p = self._media(self._puntos_liga, 1.35)

        ultimos_forma = res[-c.ventana_forma :]
        ultimos = res[-c.ventana_promedios :]
        return {
            "forma": _media_suavizada([r[0] for r in ultimos_forma], prior_p, k),
            "gf": _media_suavizada([r[1] for r in ultimos], prior_g, k),
            "gc": _media_suavizada([r[2] for r in ultimos], prior_g, k),
            "cf": _media_suavizada([r[0] for r in cor], prior_c, k),
            "cc": _media_suavizada([r[1] for r in cor], prior_c, k),
            "gf_cond": _media_suavizada([r[0] for r in cond], self._prior_goles(en_casa), k),
            "gc_cond": _media_suavizada([r[1] for r in cond], self._prior_goles(not en_casa), k),
            "elo": self._elo.get(equipo, c.elo_inicial),
            "pj": float(self._jugados.get(equipo, 0)),
        }

    def caracteristicas_partido(self, local: str, visitante: str) -> dict[str, float]:
        fl = self.caracteristicas_equipo(local, en_casa=True)
        fv = self.caracteristicas_equipo(visitante, en_casa=False)
        fila = {f"l_{k}": v for k, v in fl.items()}
        fila.update({f"v_{k}": v for k, v in fv.items()})
        fila["dif_elo"] = fl["elo"] - fv["elo"]
        fila["prob_local_elo"] = 1.0 / (
            1.0 + 10 ** (-(fila["dif_elo"] + self.config.elo_ventaja_local) / 400.0)
        )
        return fila

    # ---------------------------------------------------------- actualizar
    def actualizar(self, local, visitante, gl, gv, cl=None, cv=None) -> None:
        """Incorpora el resultado de un partido ya jugado al historial."""
        c = self.config
        gl, gv = int(gl), int(gv)
        puntos_l = 3 if gl > gv else (1 if gl == gv else 0)
        puntos_v = 3 if gv > gl else (1 if gl == gv else 0)

        for equipo in (local, visitante):
            self._resultados.setdefault(equipo, deque(maxlen=self._largo))
            self._corners.setdefault(equipo, deque(maxlen=c.ventana_promedios))
            self._jugados[equipo] = self._jugados.get(equipo, 0) + 1
        self._resultados[local].append((puntos_l, gl, gv))
        self._resultados[visitante].append((puntos_v, gv, gl))
        self._condicion.setdefault((local, True), deque(maxlen=c.ventana_promedios)).append((gl, gv))
        self._condicion.setdefault((visitante, False), deque(maxlen=c.ventana_promedios)).append((gv, gl))

        hay_corners = cl is not None and cv is not None and not (
            _es_nan(cl) or _es_nan(cv)
        )
        if hay_corners:
            self._corners[local].append((float(cl), float(cv)))
            self._corners[visitante].append((float(cv), float(cl)))
            self._corners_liga[0] += float(cl) + float(cv)
            self._corners_liga[1] += 2

        self._goles_casa[0] += gl
        self._goles_casa[1] += 1
        self._goles_fuera[0] += gv
        self._goles_fuera[1] += 1
        self._puntos_liga[0] += puntos_l + puntos_v
        self._puntos_liga[1] += 2

        # Elo con ventaja de local y multiplicador por diferencia de goles.
        r_l = self._elo.get(local, c.elo_inicial)
        r_v = self._elo.get(visitante, c.elo_inicial)
        esperado_l = 1.0 / (1.0 + 10 ** (-((r_l + c.elo_ventaja_local) - r_v) / 400.0))
        real_l = 1.0 if gl > gv else (0.5 if gl == gv else 0.0)
        cambio = c.elo_k * _multiplicador_goles(gl - gv) * (real_l - esperado_l)
        self._elo[local] = r_l + cambio
        self._elo[visitante] = r_v - cambio


def _es_nan(valor) -> bool:
    try:
        return math.isnan(float(valor))
    except (TypeError, ValueError):
        return True


def _valor(fila, columna):
    return getattr(fila, columna) if hasattr(fila, columna) else None


def construir_caracteristicas(
    partidos: pd.DataFrame, config: ConfigCaracteristicas | None = None
) -> pd.DataFrame:
    """Características previas de cada partido (mismo índice que `partidos`)."""
    motor = MotorCaracteristicas(config)
    filas: dict = {}
    for _, grupo in partidos.groupby("fecha", sort=True):
        # 1) Primero: calcular con el historial ANTERIOR a esta fecha.
        for p in grupo.itertuples():
            filas[p.Index] = motor.caracteristicas_partido(p.local, p.visitante)
        # 2) Después: incorporar los resultados de esta fecha.
        for p in grupo.itertuples():
            motor.actualizar(
                p.local,
                p.visitante,
                p.goles_local,
                p.goles_visitante,
                _valor(p, "corners_local"),
                _valor(p, "corners_visitante"),
            )
    return pd.DataFrame.from_dict(filas, orient="index").loc[partidos.index]


def perfil_previo(
    partidos: pd.DataFrame,
    local: str,
    visitante: str,
    fecha=None,
    config: ConfigCaracteristicas | None = None,
) -> dict[str, float]:
    """Características de un partido futuro usando solo partidos anteriores a `fecha`."""
    if fecha is not None:
        partidos = partidos[partidos["fecha"] < pd.Timestamp(fecha)]
    motor = MotorCaracteristicas(config)
    for p in partidos.sort_values("fecha", kind="stable").itertuples():
        motor.actualizar(
            p.local,
            p.visitante,
            p.goles_local,
            p.goles_visitante,
            _valor(p, "corners_local"),
            _valor(p, "corners_visitante"),
        )
    return motor.caracteristicas_partido(local, visitante)


def tabla_elo(partidos: pd.DataFrame, config: ConfigCaracteristicas | None = None) -> pd.Series:
    """Rating Elo final de cada equipo, de mayor a menor."""
    motor = MotorCaracteristicas(config)
    for p in partidos.sort_values("fecha", kind="stable").itertuples():
        motor.actualizar(p.local, p.visitante, p.goles_local, p.goles_visitante)
    return pd.Series(motor._elo, name="elo").sort_values(ascending=False)


__all__ = [
    "COLUMNAS_EQUIPO",
    "ConfigCaracteristicas",
    "MotorCaracteristicas",
    "construir_caracteristicas",
    "perfil_previo",
    "tabla_elo",
]

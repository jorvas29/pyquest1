"""Une los modelos de goles y córners para predecir un partido concreto."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .caracteristicas import ConfigCaracteristicas, construir_caracteristicas, perfil_previo
from .datos.esquema import tiene_corners
from .modelos.corners import ModeloCorners
from .modelos.dixon_coles import ModeloDixonColes
from .modelos.mercados import (
    marcadores_probables,
    probabilidad_btts,
    probabilidad_mas_de,
    probabilidades_1x2,
)


@dataclass
class Seleccion:
    """Una apuesta posible dentro de un mercado, con la probabilidad del modelo."""

    mercado: str        # "1X2", "Ambos marcan", "Córners"
    nombre: str         # "1 (local)", "Sí", "Más de 9.5", ...
    clave_cuota: str    # clave usada en --cuotas: "1", "X", "2", "btts_si", ...
    probabilidad: float
    p_nulo: float = 0.0  # probabilidad de apuesta devuelta (líneas enteras)

    @property
    def cuota_justa(self) -> float:
        if self.probabilidad <= 0:
            return float("inf")
        # Con posible devolución, la cuota justa es la que da EV = 0.
        return (1.0 - self.p_nulo) / self.probabilidad


@dataclass
class PrediccionPartido:
    local: str
    visitante: str
    goles_esperados_local: float
    goles_esperados_visitante: float
    p_1x2: dict[str, float]
    p_btts: float
    p_mas_25: float
    marcadores: list[tuple[int, int, float]]
    corners: dict[str, float] | None
    perfil: dict[str, float]
    modelo_goles: str
    rho: float
    selecciones: list[Seleccion] = field(default_factory=list)


class ModeloConjunto:
    """Ajusta todos los modelos con un historial y predice partidos futuros."""

    def __init__(
        self,
        xi: float = 0.0019,
        usar_rho: bool = True,
        config_caracteristicas: ConfigCaracteristicas | None = None,
    ):
        self.goles = ModeloDixonColes(xi=xi, usar_rho=usar_rho)
        self.corners: ModeloCorners | None = None
        self.config_caracteristicas = config_caracteristicas
        self.partidos: pd.DataFrame | None = None

    def ajustar(self, partidos: pd.DataFrame) -> "ModeloConjunto":
        self.partidos = partidos
        self.goles.ajustar(partidos)
        if tiene_corners(partidos):
            caract = construir_caracteristicas(partidos, self.config_caracteristicas)
            self.corners = ModeloCorners().ajustar(
                caract, partidos["corners_local"], partidos["corners_visitante"]
            )
        return self

    def predecir(self, local: str, visitante: str, linea_corners: float = 9.5) -> PrediccionPartido:
        if self.partidos is None:
            raise RuntimeError("Primero hay que llamar a ajustar().")
        matriz = self.goles.matriz(local, visitante)
        lam, mu = self.goles.tasas(local, visitante)
        perfil = perfil_previo(self.partidos, local, visitante, config=self.config_caracteristicas)
        p_1x2 = probabilidades_1x2(matriz)
        p_btts = probabilidad_btts(matriz)

        selecciones = [
            Seleccion("1X2", "1 (local)", "1", p_1x2["1"]),
            Seleccion("1X2", "X (empate)", "X", p_1x2["X"]),
            Seleccion("1X2", "2 (visitante)", "2", p_1x2["2"]),
            Seleccion("Ambos marcan", "Sí", "btts_si", p_btts),
            Seleccion("Ambos marcan", "No", "btts_no", 1.0 - p_btts),
        ]

        corners = None
        if self.corners is not None:
            corners = self.corners.predecir(perfil, linea_corners)
            texto_linea = f"{linea_corners:g}"
            selecciones += [
                Seleccion("Córners", f"Más de {texto_linea}", "corners_mas", corners["mas"], corners["nulo"]),
                Seleccion("Córners", f"Menos de {texto_linea}", "corners_menos", corners["menos"], corners["nulo"]),
            ]

        return PrediccionPartido(
            local=local,
            visitante=visitante,
            goles_esperados_local=lam,
            goles_esperados_visitante=mu,
            p_1x2=p_1x2,
            p_btts=p_btts,
            p_mas_25=probabilidad_mas_de(matriz, 2.5),
            marcadores=marcadores_probables(matriz, 5),
            corners=corners,
            perfil=perfil,
            modelo_goles=self.goles.nombre,
            rho=self.goles.rho,
            selecciones=selecciones,
        )

"""Generador de una liga SINTÉTICA (datos inventados).

Sirve únicamente para demostrar el flujo de extremo a extremo cuando no hay
una fuente de datos real. Nada de lo que sale de aquí describe equipos ni
partidos reales:

    - los equipos se llaman "Sintético 01 FC", "Sintético 02 FC", ...
    - cada fila lleva la columna origen = "SINTETICO"

Cómo se generan los datos
-------------------------
1. Cada equipo recibe una fuerza de ataque y de defensa ocultas, que derivan
   un poco de una temporada a otra.
2. Los goles se muestrean de una distribución Dixon-Coles con esas fuerzas
   (tasas de Poisson + pequeña corrección de marcadores bajos).
3. Los córners siguen una Poisson con sobredispersión (mezcla gamma, es decir,
   binomial negativa), y dependen en parte de la fuerza de ataque.
4. Las "cuotas de mercado" se obtienen de las probabilidades verdaderas con
   ruido y un margen de la casa, para poder probar la detección de valor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..modelos.dixon_coles import matriz_marcadores
from ..modelos.mercados import probabilidad_btts, probabilidades_1x2

LINEA_CORNERS_SINTETICA = 9.5


def _calendario_ida_vuelta(n_equipos: int, rng: np.random.Generator) -> list[list[tuple[int, int]]]:
    """Calendario de liga a doble vuelta con el método del círculo.

    Devuelve una lista de jornadas; cada jornada es una lista de pares
    (local, visitante) con índices de equipo.
    """
    equipos = list(rng.permutation(n_equipos))
    if n_equipos % 2:
        equipos.append(-1)  # equipo fantasma = descansa
    n = len(equipos)
    ida = []
    for jornada in range(n - 1):
        pares = []
        for i in range(n // 2):
            a, b = equipos[i], equipos[n - 1 - i]
            if a == -1 or b == -1:
                continue
            # Alternar localía para que nadie juegue siempre en casa.
            pares.append((a, b) if (jornada + i) % 2 == 0 else (b, a))
        ida.append(pares)
        equipos = [equipos[0]] + [equipos[-1]] + equipos[1:-1]
    vuelta = [[(b, a) for (a, b) in jornada] for jornada in ida]
    return ida + vuelta


def _cuotas_con_margen(probs: np.ndarray, rng: np.random.Generator, ruido: float, margen: float) -> np.ndarray:
    """Convierte probabilidades verdaderas en cuotas de casa de apuestas.

    Se añade ruido en escala logarítmica (el mercado también se equivoca) y
    luego un margen proporcional (la suma de 1/cuota queda en 1 + margen).
    """
    log_p = np.log(np.clip(probs, 1e-6, 1.0)) + rng.normal(0.0, ruido, size=len(probs))
    p_mercado = np.exp(log_p) / np.exp(log_p).sum()
    return np.round(1.0 / (p_mercado * (1.0 + margen)), 2)


def generar_liga_sintetica(
    n_equipos: int = 20,
    temporadas: int = 3,
    semilla: int = 42,
    fecha_inicio: str = "2021-08-14",
    margen: float = 0.05,
) -> pd.DataFrame:
    """Genera una liga sintética completa con resultados, córners y cuotas."""
    rng = np.random.default_rng(semilla)
    nombres = [f"Sintético {i + 1:02d} FC" for i in range(n_equipos)]

    # Fuerzas ocultas (escala logarítmica, centradas en 0).
    ataque = rng.normal(0.0, 0.25, n_equipos)
    defensa = rng.normal(0.0, 0.20, n_equipos)
    ataque -= ataque.mean()
    defensa -= defensa.mean()
    # Estilo de córners: correlacionado con el ataque + componente propia.
    corners_ataque = 0.6 * ataque + rng.normal(0.0, 0.10, n_equipos)
    corners_defensa = -0.3 * defensa + rng.normal(0.0, 0.08, n_equipos)

    base_goles = np.log(1.20)
    ventaja_local = 0.25
    rho = -0.08
    base_corners = np.log(4.8)
    ventaja_local_corners = 0.12
    forma_gamma = 12.0  # sobredispersión de córners: var = mu + mu^2 / 12

    filas = []
    fecha_temporada = pd.Timestamp(fecha_inicio)
    for temporada in range(temporadas):
        if temporada > 0:
            # Deriva entre temporadas (fichajes, entrenadores, ...).
            ataque += rng.normal(0.0, 0.06, n_equipos)
            defensa += rng.normal(0.0, 0.05, n_equipos)
            ataque -= ataque.mean()
            defensa -= defensa.mean()

        for j, jornada in enumerate(_calendario_ida_vuelta(n_equipos, rng)):
            fecha_jornada = fecha_temporada + pd.Timedelta(days=7 * j)
            for (h, a) in jornada:
                fecha = fecha_jornada + pd.Timedelta(days=int(rng.integers(0, 2)))

                # --- Goles (Dixon-Coles verdadero) ---
                lam = np.exp(base_goles + ventaja_local + ataque[h] + defensa[a])
                mu = np.exp(base_goles + ataque[a] + defensa[h])
                matriz = matriz_marcadores(lam, mu, rho, max_goles=10)
                celda = rng.choice(matriz.size, p=matriz.ravel())
                gl, gv = divmod(int(celda), matriz.shape[1])

                # --- Córners (Poisson-gamma) ---
                lam_cl = np.exp(base_corners + ventaja_local_corners + corners_ataque[h] + corners_defensa[a])
                lam_cv = np.exp(base_corners + corners_ataque[a] + corners_defensa[h])
                cl = int(rng.poisson(lam_cl * rng.gamma(forma_gamma, 1.0 / forma_gamma)))
                cv = int(rng.poisson(lam_cv * rng.gamma(forma_gamma, 1.0 / forma_gamma)))

                # --- Cuotas de mercado simuladas ---
                p1x2 = probabilidades_1x2(matriz)
                cuotas_1x2 = _cuotas_con_margen(
                    np.array([p1x2["1"], p1x2["X"], p1x2["2"]]), rng, ruido=0.08, margen=margen
                )
                p_btts = probabilidad_btts(matriz)
                cuotas_btts = _cuotas_con_margen(
                    np.array([p_btts, 1.0 - p_btts]), rng, ruido=0.10, margen=margen
                )
                p_mas = float(stats.poisson.sf(int(LINEA_CORNERS_SINTETICA), lam_cl + lam_cv))
                cuotas_corners = _cuotas_con_margen(
                    np.array([p_mas, 1.0 - p_mas]), rng, ruido=0.10, margen=margen
                )

                filas.append(
                    {
                        "fecha": fecha.strftime("%Y-%m-%d"),
                        "local": nombres[h],
                        "visitante": nombres[a],
                        "goles_local": gl,
                        "goles_visitante": gv,
                        "corners_local": cl,
                        "corners_visitante": cv,
                        "cuota_1": cuotas_1x2[0],
                        "cuota_x": cuotas_1x2[1],
                        "cuota_2": cuotas_1x2[2],
                        "cuota_btts_si": cuotas_btts[0],
                        "cuota_btts_no": cuotas_btts[1],
                        "linea_corners": LINEA_CORNERS_SINTETICA,
                        "cuota_corners_mas": cuotas_corners[0],
                        "cuota_corners_menos": cuotas_corners[1],
                        "origen": "SINTETICO",
                    }
                )
        # Siguiente temporada: tras 2*(n-1) jornadas y un parón de verano.
        fecha_temporada = fecha_jornada + pd.Timedelta(days=84)

    return pd.DataFrame(filas)

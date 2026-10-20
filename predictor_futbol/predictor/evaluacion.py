"""Evaluación con validación temporal (walk-forward) y métricas de calibración.

Validación temporal
-------------------
Nunca se entrena con partidos posteriores a los que se predicen:

    |---- entrenamiento ----|-- prueba (paso_dias) --|
    |------- entrenamiento -------|-- prueba --|
    |----------- entrenamiento -----------|-- prueba --|

En cada ventana se reajustan los modelos SOLO con partidos de fecha
estrictamente anterior al primer día de la ventana de prueba. Las
características de cada partido (forma, promedios, Elo) ya se calculan por
construcción con información previa al partido (ver `caracteristicas.py`).

Métricas (por mercado)
----------------------
log-loss   -mean(log p_asignada_al_resultado_real). Castiga mucho las
           predicciones muy seguras que fallan. Menor es mejor.
Brier      mean(sum_k (p_k - o_k)^2). Error cuadrático de probabilidades.
           Menor es mejor. (En 1X2 se suma sobre las tres clases.)

Se comparan tres fuentes de probabilidad:
    modelo      -> nuestros modelos
    referencia  -> frecuencias históricas del entrenamiento (modelo "tonto")
    mercado     -> probabilidades implícitas sin margen de las cuotas, si existen
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from .caracteristicas import ConfigCaracteristicas, construir_caracteristicas
from .cuotas import probabilidades_sin_margen
from .datos.esquema import tiene_corners
from .modelos.corners import ModeloCorners
from .modelos.dixon_coles import ModeloDixonColes
from .modelos.mercados import probabilidad_btts, probabilidades_1x2, probabilidades_corners

EPS = 1e-15


# ------------------------------------------------------------------ métricas
def log_loss_binaria(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_binario(p, y) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss_multiclase(probs, y_idx) -> float:
    """probs: matriz (n, k) cuyas filas suman 1; y_idx: índice de la clase real."""
    probs = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
    y_idx = np.asarray(y_idx, dtype=int)
    return float(-np.mean(np.log(probs[np.arange(len(y_idx)), y_idx])))


def brier_multiclase(probs, y_idx) -> float:
    probs = np.asarray(probs, dtype=float)
    y_idx = np.asarray(y_idx, dtype=int)
    reales = np.zeros_like(probs)
    reales[np.arange(len(y_idx)), y_idx] = 1.0
    return float(np.mean(np.sum((probs - reales) ** 2, axis=1)))


def tabla_calibracion(p, y, n_bins: int = 10) -> pd.DataFrame:
    """Agrupa las predicciones por tramos de probabilidad.

    Un modelo bien calibrado tiene `frecuencia_real` parecida a `p_media`
    en cada tramo (si dice 60 %, ocurre aproximadamente el 60 % de las veces).
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    tramo = np.clip(np.digitize(p, bordes[1:-1]), 0, n_bins - 1)
    filas = []
    for b in range(n_bins):
        m = tramo == b
        if not m.any():
            continue
        filas.append(
            {
                "tramo": f"{bordes[b]:.1f}-{bordes[b + 1]:.1f}",
                "n": int(m.sum()),
                "p_media": float(p[m].mean()),
                "frecuencia_real": float(y[m].mean()),
            }
        )
    return pd.DataFrame(filas)


# ------------------------------------------------------- validación temporal
def particiones_temporales(
    partidos: pd.DataFrame, min_entrenamiento: int = 150, paso_dias: int = 7
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """Genera pares (índices de entrenamiento, índices de prueba) sin fuga del futuro.

    Todo partido de entrenamiento tiene fecha < cualquier partido de prueba.
    """
    if len(partidos) <= min_entrenamiento:
        raise ValueError(
            f"Hay {len(partidos)} partidos; se necesitan más de {min_entrenamiento} para evaluar."
        )
    fechas = partidos["fecha"]
    inicio = fechas.sort_values().iloc[min_entrenamiento]
    ultima = fechas.max()
    while inicio <= ultima:
        fin = inicio + pd.Timedelta(days=paso_dias)
        entrenamiento = partidos.index[fechas < inicio]
        prueba = partidos.index[(fechas >= inicio) & (fechas < fin)]
        if len(prueba):
            yield entrenamiento, prueba
        inicio = fin


def _sin_margen_o_nan(fila, claves: list[str]) -> list[float]:
    cuotas = [getattr(fila, c, np.nan) for c in claves]
    if any(pd.isna(c) or c <= 1.0 for c in cuotas):
        return [np.nan] * len(claves)
    probs = probabilidades_sin_margen(dict(zip(claves, cuotas)))
    return [probs[c] for c in claves]


def backtest(
    partidos: pd.DataFrame,
    min_entrenamiento: int = 150,
    paso_dias: int = 7,
    linea_corners: float = 9.5,
    xi: float = 0.0019,
    usar_rho: bool = True,
    config_caracteristicas: ConfigCaracteristicas | None = None,
) -> pd.DataFrame:
    """Predice cada partido con modelos entrenados solo con el pasado.

    Devuelve una fila por partido con:
        p_*   probabilidades del modelo
        b_*   probabilidades de referencia (frecuencias del entrenamiento)
        m_*   probabilidades del mercado sin margen (NaN si no hay cuotas)
        y_*   resultado real
        cuota_* cuotas originales (para simular apuestas de valor)
    """
    caract = construir_caracteristicas(partidos, config_caracteristicas)
    con_corners = tiene_corners(partidos)
    registros = []

    for idx_ent, idx_prueba in particiones_temporales(partidos, min_entrenamiento, paso_dias):
        ent = partidos.loc[idx_ent]
        goles = ModeloDixonColes(xi=xi, usar_rho=usar_rho).ajustar(ent)

        modelo_corners = None
        if con_corners:
            validos = ent[["corners_local", "corners_visitante"]].notna().all(axis=1)
            if validos.sum() >= 15:
                modelo_corners = ModeloCorners().ajustar(
                    caract.loc[ent.index[validos]],
                    ent.loc[validos, "corners_local"],
                    ent.loc[validos, "corners_visitante"],
                )

        # Referencia: frecuencias observadas en el entrenamiento.
        dif = ent["goles_local"] - ent["goles_visitante"]
        base_1x2 = [(dif > 0).mean(), (dif == 0).mean(), (dif < 0).mean()]
        base_btts = ((ent["goles_local"] > 0) & (ent["goles_visitante"] > 0)).mean()
        if con_corners:
            total_ent = (ent["corners_local"] + ent["corners_visitante"]).dropna()

        for fila in partidos.loc[idx_prueba].itertuples():
            matriz = goles.matriz(fila.local, fila.visitante)
            p1x2 = probabilidades_1x2(matriz)
            gl, gv = fila.goles_local, fila.goles_visitante
            reg = {
                "fecha": fila.fecha,
                "local": fila.local,
                "visitante": fila.visitante,
                "n_entrenamiento": len(ent),
                "y_1x2": 0 if gl > gv else (1 if gl == gv else 2),
                "p_1": p1x2["1"],
                "p_x": p1x2["X"],
                "p_2": p1x2["2"],
                "b_1": base_1x2[0],
                "b_x": base_1x2[1],
                "b_2": base_1x2[2],
                "y_btts": int(gl > 0 and gv > 0),
                "p_btts": probabilidad_btts(matriz),
                "b_btts": base_btts,
            }
            m1, mx, m2 = _sin_margen_o_nan(fila, ["cuota_1", "cuota_x", "cuota_2"])
            reg.update({"m_1": m1, "m_x": mx, "m_2": m2})
            reg["m_btts"] = _sin_margen_o_nan(fila, ["cuota_btts_si", "cuota_btts_no"])[0]
            for c in ["cuota_1", "cuota_x", "cuota_2", "cuota_btts_si", "cuota_btts_no",
                      "cuota_corners_mas", "cuota_corners_menos"]:
                reg[c] = getattr(fila, c, np.nan)

            if modelo_corners is not None:
                linea = getattr(fila, "linea_corners", np.nan)
                linea = linea_corners if pd.isna(linea) else float(linea)
                m_l, m_v = modelo_corners.medias(caract.loc[[fila.Index]])
                pc = probabilidades_corners(float(m_l[0]), float(m_v[0]), linea, modelo_corners.dispersion)
                total = fila.corners_local + fila.corners_visitante
                reg.update(
                    {
                        "linea_corners": linea,
                        "p_corners_mas": pc["mas"],
                        "p_corners_nulo": pc["nulo"],
                        "b_corners_mas": float((total_ent > linea).mean()),
                        "corners_esperados": pc["media_total"],
                        "corners_reales": total,
                        # NaN si no hay dato o si el total cae justo en la línea (apuesta nula).
                        "y_corners_mas": np.nan if pd.isna(total) or total == linea else float(total > linea),
                        "m_corners_mas": _sin_margen_o_nan(
                            fila, ["cuota_corners_mas", "cuota_corners_menos"]
                        )[0],
                    }
                )
            registros.append(reg)

    return pd.DataFrame(registros)


# ------------------------------------------------------------------ resumen
def resumen_metricas(pred: pd.DataFrame) -> pd.DataFrame:
    """Log-loss y Brier por mercado para modelo, referencia y mercado."""
    filas = []

    y = pred["y_1x2"].to_numpy()
    p_mod = pred[["p_1", "p_x", "p_2"]].to_numpy()
    p_ref = pred[["b_1", "b_x", "b_2"]].to_numpy()
    fila = {
        "mercado": "1X2",
        "n": len(pred),
        "logloss_modelo": log_loss_multiclase(p_mod, y),
        "brier_modelo": brier_multiclase(p_mod, y),
        "logloss_referencia": log_loss_multiclase(p_ref, y),
        "brier_referencia": brier_multiclase(p_ref, y),
    }
    con_mercado = pred[["m_1", "m_x", "m_2"]].notna().all(axis=1).to_numpy()
    fila.update(_metricas_mercado_multi(pred, con_mercado, y, p_mod))
    filas.append(fila)

    y = pred["y_btts"].to_numpy()
    fila = {
        "mercado": "Ambos marcan",
        "n": len(pred),
        "logloss_modelo": log_loss_binaria(pred["p_btts"], y),
        "brier_modelo": brier_binario(pred["p_btts"], y),
        "logloss_referencia": log_loss_binaria(pred["b_btts"], y),
        "brier_referencia": brier_binario(pred["b_btts"], y),
    }
    fila.update(_metricas_mercado_bin(pred, "p_btts", "m_btts", "y_btts"))
    filas.append(fila)

    if "y_corners_mas" in pred.columns:
        sub = pred[pred["y_corners_mas"].notna()]
        y = sub["y_corners_mas"].to_numpy()
        fila = {
            "mercado": "Córners (más de la línea)",
            "n": len(sub),
            "logloss_modelo": log_loss_binaria(sub["p_corners_mas"], y),
            "brier_modelo": brier_binario(sub["p_corners_mas"], y),
            "logloss_referencia": log_loss_binaria(sub["b_corners_mas"], y),
            "brier_referencia": brier_binario(sub["b_corners_mas"], y),
        }
        fila.update(_metricas_mercado_bin(sub, "p_corners_mas", "m_corners_mas", "y_corners_mas"))
        filas.append(fila)

    return pd.DataFrame(filas)


def _metricas_mercado_multi(pred, mascara, y, p_mod) -> dict:
    """Mercado vs modelo en el MISMO subconjunto de partidos (los que tienen cuotas)."""
    if not mascara.any():
        return {"n_mercado": 0}
    p_mer = pred.loc[mascara, ["m_1", "m_x", "m_2"]].to_numpy()
    return {
        "n_mercado": int(mascara.sum()),
        "logloss_mercado": log_loss_multiclase(p_mer, y[mascara]),
        "brier_mercado": brier_multiclase(p_mer, y[mascara]),
        "logloss_modelo_mismos": log_loss_multiclase(p_mod[mascara], y[mascara]),
    }


def _metricas_mercado_bin(pred, col_p, col_m, col_y) -> dict:
    sub = pred[pred[col_m].notna()]
    if sub.empty:
        return {"n_mercado": 0}
    return {
        "n_mercado": len(sub),
        "logloss_mercado": log_loss_binaria(sub[col_m], sub[col_y]),
        "brier_mercado": brier_binario(sub[col_m], sub[col_y]),
        "logloss_modelo_mismos": log_loss_binaria(sub[col_p], sub[col_y]),
    }


def calibraciones(pred: pd.DataFrame, n_bins: int = 10) -> dict[str, pd.DataFrame]:
    """Tablas de calibración de las selecciones principales."""
    salida = {
        "1X2: gana el local": tabla_calibracion(pred["p_1"], pred["y_1x2"] == 0, n_bins),
        "1X2: empate": tabla_calibracion(pred["p_x"], pred["y_1x2"] == 1, n_bins),
        "Ambos marcan: sí": tabla_calibracion(pred["p_btts"], pred["y_btts"], n_bins),
    }
    if "y_corners_mas" in pred.columns:
        sub = pred[pred["y_corners_mas"].notna()]
        salida["Córners: más de la línea"] = tabla_calibracion(sub["p_corners_mas"], sub["y_corners_mas"], n_bins)
    return salida


def simular_valor(pred: pd.DataFrame, umbral: float = 0.05) -> pd.DataFrame:
    """Simula apostar 1 unidad cuando EV del modelo > umbral, con las cuotas históricas.

    Es una comprobación retrospectiva y orientativa: un ROI positivo en el
    pasado NO garantiza beneficios futuros (y con pocas apuestas es ruido).
    """
    definiciones = [
        ("1X2", "1", "p_1", "cuota_1", lambda d: d["y_1x2"] == 0, None),
        ("1X2", "X", "p_x", "cuota_x", lambda d: d["y_1x2"] == 1, None),
        ("1X2", "2", "p_2", "cuota_2", lambda d: d["y_1x2"] == 2, None),
        ("Ambos marcan", "Sí", "p_btts", "cuota_btts_si", lambda d: d["y_btts"] == 1, None),
        ("Ambos marcan", "No", None, "cuota_btts_no", lambda d: d["y_btts"] == 0, None),
    ]
    if "p_corners_mas" in pred.columns:
        definiciones += [
            ("Córners", "Más", "p_corners_mas", "cuota_corners_mas", lambda d: d["y_corners_mas"] == 1, "nulo"),
            ("Córners", "Menos", None, "cuota_corners_menos", lambda d: d["y_corners_mas"] == 0, "nulo"),
        ]

    filas = []
    for mercado, sel, col_p, col_c, gana, con_nulo in definiciones:
        if col_c not in pred.columns:
            continue
        d = pred[pred[col_c].notna()]
        if con_nulo:
            d = d[d["corners_reales"].notna() & d["p_corners_mas"].notna()]
        if d.empty:
            continue
        if col_p is None:  # selección complementaria: "No" o "Menos"
            base = "p_btts" if mercado == "Ambos marcan" else "p_corners_mas"
            nulo = d["p_corners_nulo"] if con_nulo else 0.0
            p = 1.0 - d[base] - nulo
        else:
            p = d[col_p]
            nulo = d["p_corners_nulo"] if con_nulo else 0.0
        ev = p * d[col_c] + nulo - 1.0
        elegidas = d[ev > umbral]
        if elegidas.empty:
            filas.append({"mercado": mercado, "seleccion": sel, "apuestas": 0, "aciertos": 0,
                          "beneficio": 0.0, "roi": np.nan})
            continue
        ganadas = gana(elegidas)
        if con_nulo:
            devueltas = elegidas["corners_reales"] == elegidas["linea_corners"]
        else:
            devueltas = pd.Series(False, index=elegidas.index)
        beneficio = np.where(devueltas, 0.0, np.where(ganadas, elegidas[col_c] - 1.0, -1.0))
        filas.append(
            {
                "mercado": mercado,
                "seleccion": sel,
                "apuestas": len(elegidas),
                "aciertos": int(ganadas.sum()),
                "beneficio": float(beneficio.sum()),
                "roi": float(beneficio.sum() / len(elegidas)),
            }
        )
    return pd.DataFrame(filas)

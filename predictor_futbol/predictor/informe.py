"""Informes de texto: predicción de un partido y evaluación histórica.

Incluye la comparación entre las probabilidades del modelo y las cuotas de
mercado para detectar selecciones con valor esperado positivo.
"""

from __future__ import annotations

import math
from typing import Mapping

import pandas as pd

from . import ADVERTENCIA
from .cuotas import probabilidad_implicita, probabilidades_sin_margen, valor_esperado
from .prediccion import PrediccionPartido

LINEA = "=" * 66
SUBLINEA = "-" * 66


def _pct(p: float) -> str:
    return f"{100 * p:5.1f} %"


def _cuota(c: float) -> str:
    return "  inf" if math.isinf(c) else f"{c:5.2f}"


# ------------------------------------------------------- valor esperado
def tabla_valor(
    pred: PrediccionPartido, cuotas: Mapping[str, float], umbral: float = 0.05
) -> pd.DataFrame:
    """Compara modelo y mercado en cada selección con cuota disponible.

    Columnas:
        p_modelo        probabilidad del modelo
        cuota_justa     cuota sin margen del modelo
        cuota_mercado   cuota ofrecida
        p_mercado       probabilidad implícita del mercado; sin margen si se
                        dieron todas las cuotas del mercado, si no 1/cuota
        ventaja         p_modelo - p_mercado
        ev              valor esperado por unidad: p * cuota (+ p_nulo) - 1
        señal           "VALOR" si ev > umbral
    """
    por_mercado: dict[str, list] = {}
    for s in pred.selecciones:
        por_mercado.setdefault(s.mercado, []).append(s)

    filas = []
    for mercado, selecciones in por_mercado.items():
        disponibles = {s.clave_cuota: cuotas[s.clave_cuota] for s in selecciones if s.clave_cuota in cuotas}
        completo = len(disponibles) == len(selecciones)
        sin_margen = probabilidades_sin_margen(disponibles) if completo else {}
        for s in selecciones:
            if s.clave_cuota not in disponibles:
                continue
            c = disponibles[s.clave_cuota]
            p_mer = sin_margen.get(s.clave_cuota, probabilidad_implicita(c))
            ev = valor_esperado(s.probabilidad, c) + s.p_nulo
            filas.append(
                {
                    "mercado": mercado,
                    "seleccion": s.nombre,
                    "p_modelo": s.probabilidad,
                    "cuota_justa": s.cuota_justa,
                    "cuota_mercado": c,
                    "p_mercado": p_mer,
                    "sin_margen": completo,
                    "ventaja": s.probabilidad - p_mer,
                    "ev": ev,
                    "senal": "VALOR" if ev > umbral else "",
                }
            )
    return pd.DataFrame(filas)


# ---------------------------------------------------------- predicción
def texto_prediccion(
    pred: PrediccionPartido,
    contexto: Mapping[str, str] | None = None,
    cuotas: Mapping[str, float] | None = None,
    umbral: float = 0.05,
) -> str:
    """Informe de un partido en disposición vertical, listo para la consola."""
    p = pred.perfil
    lineas = [
        LINEA,
        f"  PREDICCIÓN: {pred.local} (local) vs {pred.visitante} (visitante)",
        LINEA,
    ]
    for clave, valor in (contexto or {}).items():
        lineas.append(f"{clave:<22}: {valor}")
    lineas.append(f"{'Modelo de goles':<22}: {pred.modelo_goles} (rho = {pred.rho:+.3f})")

    lineas += [
        "",
        "PERFIL PREVIO (solo partidos anteriores)",
        SUBLINEA,
        f"{'':<30}{'Local':>12}{'Visitante':>14}",
        f"{'Forma (puntos/partido)':<30}{p['l_forma']:>12.2f}{p['v_forma']:>14.2f}",
        f"{'Goles a favor / partido':<30}{p['l_gf']:>12.2f}{p['v_gf']:>14.2f}",
        f"{'Goles en contra / partido':<30}{p['l_gc']:>12.2f}{p['v_gc']:>14.2f}",
        f"{'Goles a favor (casa/fuera)':<30}{p['l_gf_cond']:>12.2f}{p['v_gf_cond']:>14.2f}",
        f"{'Córners a favor / partido':<30}{p['l_cf']:>12.2f}{p['v_cf']:>14.2f}",
        f"{'Córners en contra / partido':<30}{p['l_cc']:>12.2f}{p['v_cc']:>14.2f}",
        f"{'Elo (calidad estimada)':<30}{p['l_elo']:>12.0f}{p['v_elo']:>14.0f}",
        f"{'Partidos en el historial':<30}{p['l_pj']:>12.0f}{p['v_pj']:>14.0f}",
        f"{'Diferencia Elo (local - visit.)':<30}{p['dif_elo']:>+12.0f}",
        "",
        "GOLES ESPERADOS",
        SUBLINEA,
        f"  Local     : {pred.goles_esperados_local:.2f}",
        f"  Visitante : {pred.goles_esperados_visitante:.2f}",
        "  Marcadores más probables:",
    ]
    for gl, gv, prob in pred.marcadores:
        lineas.append(f"    {gl}-{gv}   {_pct(prob)}")

    lineas += ["", "PROBABILIDADES Y CUOTAS JUSTAS (sin margen)", SUBLINEA]
    mercado_actual = None
    for s in pred.selecciones:
        if s.mercado != mercado_actual:
            mercado_actual = s.mercado
            lineas.append(f"  {s.mercado}")
        lineas.append(f"    {s.nombre:<18} p = {_pct(s.probabilidad)}   cuota justa = {_cuota(s.cuota_justa)}")
    lineas.append(f"  Extra: más de 2.5 goles  p = {_pct(pred.p_mas_25)}")

    if pred.corners is not None:
        c = pred.corners
        lineas += [
            "",
            "DETALLE DE CÓRNERS",
            SUBLINEA,
            f"  Esperados local      : {c['media_local']:.2f}",
            f"  Esperados visitante  : {c['media_visitante']:.2f}",
            f"  Esperados totales    : {c['media_total']:.2f}",
            f"  Local saca más       : {_pct(c['local_mas'])}",
            f"  Mismo número         : {_pct(c['igual'])}",
            f"  Visitante saca más   : {_pct(c['visitante_mas'])}",
        ]
        if c["nulo"] > 0:
            lineas.append(f"  Total = línea (nula) : {_pct(c['nulo'])}")
    else:
        lineas += ["", "Córners: no hay datos de córners en la fuente; mercado no disponible."]

    if cuotas:
        tabla = tabla_valor(pred, cuotas, umbral)
        lineas += ["", f"MODELO vs MERCADO (señal de valor si EV > {umbral:.0%})", SUBLINEA]
        if tabla.empty:
            lineas.append("  Ninguna de las cuotas indicadas corresponde a un mercado disponible.")
        for f in tabla.itertuples():
            lineas += [
                f"  {f.mercado} · {f.seleccion}",
                f"    p modelo          : {_pct(f.p_modelo)}",
                f"    p mercado{'' if f.sin_margen else ' (con margen)':<9}: {_pct(f.p_mercado)}",
                f"    ventaja           : {100 * f.ventaja:+5.1f} puntos",
                f"    cuota justa       : {_cuota(f.cuota_justa)}",
                f"    cuota mercado     : {_cuota(f.cuota_mercado)}",
                f"    valor esperado    : {100 * f.ev:+5.1f} %   {f.senal}",
            ]

    lineas += ["", LINEA, ADVERTENCIA, LINEA]
    return "\n".join(lineas)


# ----------------------------------------------------------- evaluación
def _tabla_markdown(df: pd.DataFrame, decimales: int = 4) -> str:
    """Tabla Markdown sencilla (sin depender de `tabulate`)."""
    if df.empty:
        return "_(sin datos)_"

    def fmt(v):
        if isinstance(v, float):
            return "" if math.isnan(v) else f"{v:.{decimales}f}"
        return str(v)

    cab = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    cuerpo = ["| " + " | ".join(fmt(v) for v in fila) + " |" for fila in df.itertuples(index=False)]
    return "\n".join([cab, sep, *cuerpo])


def markdown_evaluacion(
    resumen: pd.DataFrame,
    calibraciones: Mapping[str, pd.DataFrame],
    valor: pd.DataFrame,
    contexto: Mapping[str, str],
    umbral: float,
) -> str:
    """Informe Markdown de la validación temporal."""
    partes = ["# Evaluación temporal del predictor", "", f"> {ADVERTENCIA}", ""]
    for clave, v in contexto.items():
        partes.append(f"- **{clave}:** {v}")
    partes += [
        "",
        "## Métricas por mercado",
        "",
        "Menor es mejor. `referencia` = frecuencias históricas del entrenamiento; "
        "`mercado` = probabilidades implícitas sin margen; `logloss_modelo_mismos` = "
        "log-loss del modelo en los mismos partidos que tienen cuotas.",
        "",
        _tabla_markdown(resumen),
        "",
        "## Calibración",
        "",
        "Si el modelo está bien calibrado, `frecuencia_real` se parece a `p_media` en cada tramo.",
    ]
    for nombre, tabla in calibraciones.items():
        partes += ["", f"### {nombre}", "", _tabla_markdown(tabla, 3)]
    partes += [
        "",
        f"## Simulación retrospectiva de apuestas con valor (EV > {umbral:.0%}, 1 unidad)",
        "",
        "Orientativa: un ROI histórico positivo NO garantiza beneficios futuros.",
        "",
        _tabla_markdown(valor, 3),
        "",
    ]
    return "\n".join(partes)

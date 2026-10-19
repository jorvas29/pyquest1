"""Índice de ventaja (FASE 2) y goles esperados ajustados (FASE 3.1).

Cada bloque se puntúa de -2 (favorece claramente al visitante) a +2
(favorece claramente al local). Si el analista da la puntuación en el
JSON se respeta; si no, se sugiere una a partir de los datos numéricos
disponibles. El índice es la suma ponderada (también entre -2 y +2).
"""

from __future__ import annotations

import numpy as np

WEIGHTS = {
    "forma": 0.20,
    "local_visitante": 0.15,
    "ataque_defensa": 0.20,
    "bajas": 0.15,
    "motivacion": 0.10,
    "descanso": 0.10,
    "h2h": 0.05,
    "condiciones": 0.05,
}

LABELS = {
    "forma": "Forma",
    "local_visitante": "Local / visitante",
    "ataque_defensa": "Ataque / defensa (xG)",
    "bajas": "Bajas",
    "motivacion": "Motivación / contexto",
    "descanso": "Descanso / calendario",
    "h2h": "Enfrentamientos directos",
    "condiciones": "Condiciones / árbitro",
}

MAX_ADJUST = 0.15


def _clip_score(x: float) -> int:
    return int(np.clip(round(x), -2, 2))


def _get(d: dict, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d or d[p] is None:
            return default
        d = d[p]
    return d


def suggest_scores(data: dict) -> dict[str, tuple[int, str]]:
    """Puntuaciones automáticas a partir de los números del JSON."""
    h, a = data.get("local", {}), data.get("visitante", {})
    out: dict[str, tuple[int, str]] = {}

    ppg_h, ppg_a = _get(h, "ppg10"), _get(a, "ppg10")
    if ppg_h is not None and ppg_a is not None:
        diff = ppg_h - ppg_a
        out["forma"] = (_clip_score(diff / 0.5), f"PPG últimos 10: {ppg_h:.2f} vs {ppg_a:.2f} (dif {diff:+.2f})")

    ch, ca = _get(h, "condicion", default={}), _get(a, "condicion", default={})
    if ch.get("ppg") is not None and ca.get("ppg") is not None:
        diff = ch["ppg"] - ca["ppg"]
        out["local_visitante"] = (
            _clip_score(diff / 0.5),
            f"PPG local de local {ch['ppg']:.2f} vs visitante de visita {ca['ppg']:.2f}",
        )

    xg_h, xga_h, xg_a, xga_a = ch.get("xg"), ch.get("xga"), ca.get("xg"), ca.get("xga")
    if None not in (xg_h, xga_h, xg_a, xga_a):
        net = (xg_h - xga_h) - (xg_a - xga_a)
        out["ataque_defensa"] = (_clip_score(net / 0.5), f"xG neto en su condición: {xg_h - xga_h:+.2f} vs {xg_a - xga_a:+.2f}")

    w_h = sum(b.get("peso", 0) for b in h.get("bajas", []) if b.get("estado", "CONFIRMADO") == "CONFIRMADO")
    w_a = sum(b.get("peso", 0) for b in a.get("bajas", []) if b.get("estado", "CONFIRMADO") == "CONFIRMADO")
    if h.get("bajas") is not None or a.get("bajas") is not None:
        diff = w_a - w_h
        out["bajas"] = (_clip_score(diff / 1.5), f"Peso de bajas confirmadas: local {w_h} / visitante {w_a}")

    rest_h, rest_a = _get(h, "descanso_dias"), _get(a, "descanso_dias")
    if rest_h is not None and rest_a is not None:
        diff = rest_h - rest_a
        out["descanso"] = (_clip_score(diff / 2), f"Días de descanso: {rest_h} vs {rest_a}")

    h2h = data.get("h2h", {})
    if h2h.get("local_ganados") is not None and h2h.get("visitante_ganados") is not None:
        total = max(h2h.get("partidos", 0), 1)
        diff = (h2h["local_ganados"] - h2h["visitante_ganados"]) / total
        out["h2h"] = (_clip_score(diff * 3), f"H2H: {h2h['local_ganados']}G-{h2h.get('empates', 0)}E-{h2h['visitante_ganados']}P")
    return out


def advantage_index(data: dict) -> dict:
    """Combina puntuaciones manuales (``data['indice']``) con las sugeridas."""
    manual = data.get("indice", {})
    auto = suggest_scores(data)
    blocks = []
    for key, weight in WEIGHTS.items():
        if key in manual and manual[key].get("score") is not None:
            score = _clip_score(manual[key]["score"])
            note = manual[key].get("nota", "")
            origin = "analista"
        elif key in auto:
            score, note = auto[key]
            origin = "auto"
        else:
            score, note, origin = 0, "No verificado (se puntúa 0)", "sin datos"
        blocks.append({"key": key, "label": LABELS[key], "weight": weight, "score": score,
                       "weighted": weight * score, "note": note, "origin": origin})
    value = float(sum(b["weighted"] for b in blocks))
    return {"blocks": blocks, "value": value}


def expected_goals(data: dict, index_value: float) -> dict:
    """λ_local y λ_visita (goles y xG promediados) con ajuste de hasta ±15 %."""
    ch = _get(data, "local", "condicion", default={})
    ca = _get(data, "visitante", "condicion", default={})
    steps = []

    lam_h_goals = (ch["gf"] + ca["gc"]) / 2
    lam_a_goals = (ca["gf"] + ch["gc"]) / 2
    steps.append(f"Goles: λ_local = ({ch['gf']:.2f} + {ca['gc']:.2f}) / 2 = {lam_h_goals:.3f}")
    steps.append(f"Goles: λ_visita = ({ca['gf']:.2f} + {ch['gc']:.2f}) / 2 = {lam_a_goals:.3f}")
    lam_h, lam_a = lam_h_goals, lam_a_goals

    if None not in (ch.get("xg"), ch.get("xga"), ca.get("xg"), ca.get("xga")):
        lam_h_xg = (ch["xg"] + ca["xga"]) / 2
        lam_a_xg = (ca["xg"] + ch["xga"]) / 2
        steps.append(f"xG: λ_local = ({ch['xg']:.2f} + {ca['xga']:.2f}) / 2 = {lam_h_xg:.3f}")
        steps.append(f"xG: λ_visita = ({ca['xg']:.2f} + {ch['xga']:.2f}) / 2 = {lam_a_xg:.3f}")
        lam_h, lam_a = (lam_h_goals + lam_h_xg) / 2, (lam_a_goals + lam_a_xg) / 2
        steps.append(f"Promedio goles/xG: λ_local = {lam_h:.3f} · λ_visita = {lam_a:.3f}")
    else:
        steps.append("xG no disponible: se usan solo goles")

    # Índice ±2  ->  ±7.5 %; ajustes extra por bajas clave (JSON) y tope total ±15 %.
    extra = data.get("ajuste_lambda", {})
    adj_h = 0.075 * index_value + float(extra.get("local", 0.0))
    adj_a = -0.075 * index_value + float(extra.get("visitante", 0.0))
    adj_h, adj_a = (float(np.clip(x, -MAX_ADJUST, MAX_ADJUST)) for x in (adj_h, adj_a))
    base_h, base_a = lam_h, lam_a
    lam_h, lam_a = base_h * (1 + adj_h), base_a * (1 + adj_a)
    steps.append(
        f"Ajuste por índice ({index_value:+.2f}) y bajas: local {adj_h:+.1%} → {lam_h:.3f} · visita {adj_a:+.1%} → {lam_a:.3f}"
    )
    return {"home": lam_h, "away": lam_a, "base_home": base_h, "base_away": base_a, "steps": steps}

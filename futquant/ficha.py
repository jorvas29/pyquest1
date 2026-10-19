"""Renderizado de la FICHA DEL PARTIDO en Markdown (formato fijo de 11 secciones)."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from .picks import build_candidates, select_picks

BOGOTA = ZoneInfo("America/Bogota")
NV = "No verificado"


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _odds(x) -> str:
    return "—" if x is None or x == float("inf") else f"{x:.2f}"


def kickoff_colombia(value: str | None) -> str:
    if not value:
        return NV
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return f"{dt:%Y-%m-%d %H:%M} (zona horaria no indicada)"
    return f"{dt.astimezone(BOGOTA):%Y-%m-%d %H:%M} (hora Colombia, UTC-5)"


def _fav(h, a, higher_is_better=True, tol=0.0):
    if h is None or a is None:
        return NV
    if abs(h - a) <= tol:
        return "Parejo"
    better_home = (h > a) if higher_is_better else (h < a)
    return "Local" if better_home else "Visitante"


def _fmt(v):
    if v is None:
        return NV
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def variables_table(data: dict) -> list[tuple[str, str, str, str]]:
    h, a = data.get("local", {}), data.get("visitante", {})
    ch, ca = h.get("condicion", {}), a.get("condicion", {})
    rows = [
        ("Posición / puntos", _fmt(h.get("tabla")), _fmt(a.get("tabla")), data.get("favorece_tabla", "—")),
        ("Forma últimos 5", _fmt(h.get("forma5")), _fmt(a.get("forma5")), "—"),
        ("PPG últimos 10", _fmt(h.get("ppg10")), _fmt(a.get("ppg10")), _fav(h.get("ppg10"), a.get("ppg10"), tol=0.15)),
        ("PPG en su condición (local / visita)", _fmt(ch.get("ppg")), _fmt(ca.get("ppg")), _fav(ch.get("ppg"), ca.get("ppg"), tol=0.15)),
        ("Goles a favor en su condición", _fmt(ch.get("gf")), _fmt(ca.get("gf")), _fav(ch.get("gf"), ca.get("gf"), tol=0.1)),
        ("Goles en contra en su condición", _fmt(ch.get("gc")), _fmt(ca.get("gc")), _fav(ch.get("gc"), ca.get("gc"), False, 0.1)),
        ("xG en su condición", _fmt(ch.get("xg")), _fmt(ca.get("xg")), _fav(ch.get("xg"), ca.get("xg"), tol=0.1)),
        ("xGA en su condición", _fmt(ch.get("xga")), _fmt(ca.get("xga")), _fav(ch.get("xga"), ca.get("xga"), False, 0.1)),
        ("Córners a favor / en contra", f"{_fmt(h.get('corners_favor'))} / {_fmt(h.get('corners_contra'))}",
         f"{_fmt(a.get('corners_favor'))} / {_fmt(a.get('corners_contra'))}", "—"),
        ("Amarillas propias / del rival", f"{_fmt(h.get('amarillas_favor'))} / {_fmt(h.get('amarillas_contra'))}",
         f"{_fmt(a.get('amarillas_favor'))} / {_fmt(a.get('amarillas_contra'))}", "—"),
        ("Días de descanso", _fmt(h.get("descanso_dias")), _fmt(a.get("descanso_dias")),
         _fav(h.get("descanso_dias"), a.get("descanso_dias"), tol=0)),
        ("Bajas confirmadas (peso)",
         _bajas(h), _bajas(a), "—"),
    ]
    for r in data.get("variables", []):
        rows.append((r.get("variable", ""), _fmt(r.get("local")), _fmt(r.get("visitante")), r.get("favorece", "—")))
    return rows


def _bajas(team: dict) -> str:
    bajas = team.get("bajas")
    if bajas is None:
        return NV
    if not bajas:
        return "Sin bajas reportadas"
    return "; ".join(f"{b['jugador']} ({b.get('rol', '?')}, {b.get('estado', 'CONFIRMADO')}, peso {b.get('peso', 0)})" for b in bajas)


def render(res: dict) -> str:
    d = res["data"]
    p = d["partido"]
    H, A = p["local"], p["visitante"]
    cands = build_candidates(res)
    picks = select_picks(cands)
    mk = res["markets"]
    L = []
    w = L.append

    w(f"# FICHA DEL PARTIDO — {H} vs {A}\n")

    # 1. Ficha
    w("## 1. Ficha\n")
    w("| Campo | Dato |\n|---|---|")
    w(f"| Partido | {H} vs {A} |")
    w(f"| Competición | {p.get('competicion', NV)} |")
    w(f"| Fecha y hora | {kickoff_colombia(p.get('fecha_hora'))} |")
    w(f"| Estadio | {p.get('estadio', NV)} |")
    ref = d.get("arbitro_stats", {})
    ref_txt = p.get("arbitro", NV)
    if ref:
        ref_txt += f" (amarillas {_fmt(ref.get('amarillas_pp'))}, rojas {_fmt(ref.get('rojas_pp'))}, penales {_fmt(ref.get('penales_pp'))} por partido)"
    w(f"| Árbitro | {ref_txt} |")
    w(f"| Clima | {p.get('clima', NV)} |\n")

    # 2. Resumen
    w("## 2. Resumen\n")
    for line in (p.get("resumen") or [NV])[:3]:
        w(f"- {line}")
    w("")

    # 3. Variables
    w("## 3. Tabla de variables\n")
    w(f"| Variable | {H} | {A} | Favorece a |\n|---|---|---|---|")
    for row in variables_table(d):
        w("| " + " | ".join(row) + " |")
    w("")

    # 4. Índice
    w("## 4. Índice de ventaja\n")
    w("| Bloque | Peso | Puntuación | Ponderado | Explicación |\n|---|---|---|---|---|")
    for b in res["index"]["blocks"]:
        w(f"| {b['label']} | {b['weight']:.0%} | {b['score']:+d} | {b['weighted']:+.2f} | {b['note']} ({b['origin']}) |")
    v = res["index"]["value"]
    lean = H if v > 0.15 else A if v < -0.15 else "ninguno (parejo)"
    w(f"\n**Índice = {v:+.2f}** (escala -2 a +2) → inclina hacia: **{lean}**\n")

    # 5. Probabilidades
    w("## 5. Probabilidades calculadas\n")
    w("### Goles esperados\n")
    for s in res["lambda"]["steps"]:
        w(f"- {s}")
    w(f"- ρ Dixon-Coles = {res['rho']:+.3f}" + (" (estimado por MLE del historial)" if "dc_fit" in res else " (valor por defecto de la literatura)"))
    if "dc_fit" in res and "lambda" in res["dc_fit"]:
        lh, la = res["dc_fit"]["lambda"]
        w(f"- Dixon-Coles MLE por equipo (historial de {res['dc_fit']['n']} partidos): λ_local {lh:.2f} · λ_visita {la:.2f}")
    w("")

    w("### 1X2 por modelo\n")
    w("| Modelo | 1 | X | 2 | Peso en ensamble |\n|---|---|---|---|---|")
    for name, pr in res["models_1x2"].items():
        w(f"| {name} | {_pct(pr['1'])} | {_pct(pr['X'])} | {_pct(pr['2'])} | {res['weights'][name]:.1%} |")
    f = res["final_1x2"]
    w(f"| **ENSAMBLE** | **{_pct(f['1'])}** | **{_pct(f['X'])}** | **{_pct(f['2'])}** | 100% |\n")
    if res.get("ml_metrics"):
        w("Validación temporal (último 20 % del historial; menor = mejor):\n")
        w("| Modelo | Log-loss | RPS | ¿Entra al ensamble? |\n|---|---|---|---|")
        for name, m in res["ml_metrics"].items():
            used = "referencia" if "in_ensemble" not in m else ("Sí" if m["in_ensemble"] else "No (peor que la base)")
            w(f"| {name} | {m['logloss']:.3f} | {m['rps']:.4f} | {used} |")
        w("")
    if res.get("ratings"):
        w("Ratings: " + " · ".join(
            f"{k}: {H} {a:.2f}" + (f" / {A} {b:.2f}" if b is not None else "") for k, (a, b) in res["ratings"].items()) + "\n")

    dc = mk["double_chance"]
    w("### Resultado y hándicap\n")
    w("| Mercado | Probabilidad | Cuota justa |\n|---|---|---|")
    rows = [
        (f"1X ({H} o empate)", dc["1X"]), (f"X2 ({A} o empate)", dc["X2"]), ("12 (no empate)", dc["12"]),
        (f"{H} empate no válido", mk["dnb"]["1 (DNB)"]), (f"{A} empate no válido", mk["dnb"]["2 (DNB)"]),
        (f"{H} -1 asiático (sin devolución)", mk["ah_home_-1"]["effective"]),
        (f"{A} +1 asiático (sin devolución)", mk["ah_away_+1"]["effective"]),
        (f"{H} -1.5", mk["ah_home_-1.5"]["win"]), (f"{A} +1.5", mk["ah_away_+1.5"]["win"]),
        (f"Hándicap europeo {H} -1: 1 / X / 2",
         None),
    ]
    for label, prob in rows[:-1]:
        w(f"| {label} | {_pct(prob)} | {_odds(1 / prob if prob else None)} |")
    eh = mk["eh_home_-1"]
    w(f"| {rows[-1][0]} | {_pct(eh['1'])} / {_pct(eh['X'])} / {_pct(eh['2'])} | — |\n")

    w("### Goles\n")
    w("| Línea | Más | Menos |\n|---|---|---|")
    for line, t in mk["totals"].items():
        w(f"| {line} | {_pct(t['over'])} | {_pct(t['under'])} |")
    w(f"\nAmbos marcan: **Sí {_pct(mk['btts']['yes'])}** · No {_pct(mk['btts']['no'])}  ")
    w(f"Valla invicta: {H} {_pct(mk['clean_sheets']['home'])} · {A} {_pct(mk['clean_sheets']['away'])}\n")
    w(f"| Goles por equipo | Más de 0.5 | Más de 1.5 | Más de 2.5 |\n|---|---|---|---|")
    for side, name in (("home", H), ("away", A)):
        tt = mk["team_totals"][side]
        w(f"| {name} | {_pct(tt[0.5]['over'])} | {_pct(tt[1.5]['over'])} | {_pct(tt[2.5]['over'])} |")
    w("")

    w("### Marcadores más probables\n")
    for score, pr in mk["top_scores"]:
        w(f"- {score}: {_pct(pr)}")
    sim = res["simulation"]
    w(f"\nMonte Carlo ({sim['n_sims']:,} simulaciones): primer gol {H} {_pct(sim['first_goal']['home'])} · "
      f"{A} {_pct(sim['first_goal']['away'])} · sin goles {_pct(sim['first_goal']['none'])}  ")
    w("Reparto de goles por tramo: " + " · ".join(f"{k}' {v:.0%}" for k, v in sim["goal_minutes"].items()) + "\n")

    w("### Primer tiempo\n")
    ht = res["ht"]
    w(f"- λ 1T = {ht['lambda'][0]:.2f} + {ht['lambda'][1]:.2f} (λ × 0.45)")
    w(f"- Más de 0.5: {_pct(ht['totals'][0.5]['over'])} · Más de 1.5: {_pct(ht['totals'][1.5]['over'])}")
    w(f"- Resultado al descanso: 1 {_pct(ht['1x2']['1'])} · X {_pct(ht['1x2']['X'])} · 2 {_pct(ht['1x2']['2'])}")
    top_htft = sorted(ht["ht_ft"].items(), key=lambda kv: -kv[1])[:3]
    w("- Descanso/final más probables: " + " · ".join(f"{k} {_pct(v)}" for k, v in top_htft) + "\n")

    for kind, title in (("corners", "Córners"), ("cards", "Tarjetas")):
        w(f"### {title}\n")
        block = res.get(kind)
        if not block:
            w(f"{NV}: faltan promedios a favor / en contra.\n")
            continue
        for s in block["steps"]:
            w(f"- {s}")
        w("\n| Línea | Más | Menos |\n|---|---|---|")
        for line, t in block["lines"].items():
            w(f"| {line} | {_pct(t['over'])} | {_pct(t['under'])} |")
        w("")

    w("### Contraste con el mercado\n")
    if not res["contrast"]:
        w("Sin cuotas cargadas: no hay contraste.\n")
    else:
        w("| Mercado | Modelo | Mercado (sin margen) | Diferencia |\n|---|---|---|---|")
        for c in res["contrast"]:
            flag = " ⚠️ >10 pts" if c["flag"] else ""
            w(f"| {c['market']} | {_pct(c['model'])} | {_pct(c['market_p'])} | {c['diff'] * 100:+.1f} pts{flag} |")
        explanations = d.get("explicacion_diferencias")
        if any(c["flag"] for c in res["contrast"]):
            w("\nDiferencia mayor a 10 puntos: revisar datos. " + (explanations or
              "Posibles causas: bajas no reflejadas, rotación esperada o información que el mercado ya descuenta."))
        w("")

    # 6. Pick principal
    w("## 6. PICK PRINCIPAL\n")
    m = picks["main"]
    if m is None:
        w("No hay selección entre 60 % y 88 %: **no se recomienda pick principal**.\n")
    else:
        if picks["main_warning"]:
            w(f"> {picks['main_warning']}\n")
        w(f"**{m['label']}** · Probabilidad **{_pct(m['prob'])}** · Confianza **{m['confidence']}** · Cuota justa **{_odds(m['fair'])}**")
        if m.get("user_odds"):
            w(f"\nTu cuota {m['user_odds']:.2f}: {m['user_vs_fair']}.")
        if m.get("note"):
            w(f"\n_{m['note']}_")
        w("\nRazones:")
        for _, text in [s for s in m["signals"] if s[0]][:3]:
            w(f"1. {text}")
        against = [t for ok, t in m["signals"] if not ok]
        risk = d.get("mayor_riesgo") or (against[0] if against else "Varianza propia del fútbol: un gol temprano cambia el guion.")
        w(f"\n**Mayor riesgo:** {risk}\n")

    # 7. Picks por mercado
    w("## 7. PICKS POR MERCADO\n")
    w("| Mercado | Pick | Probabilidad | Confianza | Cuota justa | Variables a favor / en contra |\n|---|---|---|---|---|---|")
    for g, x in picks["by_group"].items():
        extra = f" (tu cuota {x['user_odds']:.2f}: {x['user_vs_fair'].split(' (')[0]})" if x.get("user_odds") else ""
        w(f"| {g} | {x['label']}{extra} | {_pct(x['prob'])} | {x['confidence']} | {_odds(x['fair'])} | {x['support']} / {x['against']} |")
    w("")

    # 8. Picks a evitar
    w("## 8. Picks a evitar\n")
    if not picks["avoid"]:
        w("- Ninguno destacado.")
    for x, why in picks["avoid"][:8]:
        w(f"- {x['label']} ({_pct(x['prob'])}): {why}.")
    w("- Combinadas: la probabilidad conjunta se multiplica (dos picks de 70 % = 49 %).\n")

    # 9. Qué revisar
    w("## 9. Qué revisar antes de apostar\n")
    w("- Alineaciones oficiales 1 h antes del inicio (sobre todo las bajas marcadas como RUMOR o duda).")
    w("- Bajas de último minuto y calentamiento.")
    w("- Cambios de cuota fuertes (si el mercado se mueve contra el pick, revisar la noticia que lo causa).")
    for item in d.get("revisar", []):
        w(f"- {item}")
    w("")

    # 10. Fuentes
    w("## 10. Fuentes\n")
    sources = d.get("fuentes", [])
    if not sources:
        w(f"- {NV}: no se cargaron fuentes.")
    for s in sources:
        w(f"- [{s.get('titulo', s.get('url'))}]({s.get('url')}) — consultado {s.get('fecha', NV)}")
    for n in res["notes"]:
        w(f"- Nota del modelo: {n}")
    w("")

    # 11. Cierre
    w("## 11. Cierre\n")
    w("**Pronóstico estadístico, no garantía. Apuesta solo lo que puedas perder.**")
    return "\n".join(L) + "\n"

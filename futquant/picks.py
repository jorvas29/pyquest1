"""Selección de picks (FASE 4).

Cada candidato lleva:

* ``prob``: probabilidad final del ensamble.
* ``signals``: lista de variables a favor (True) o en contra (False).
* ``grave``: variable grave en contra (p. ej. bajas -2 hacia el otro lado).

Reglas:

* Se descartan picks triviales (> 88 %).
* PICK PRINCIPAL: mayor probabilidad entre 60 % y 88 % con ≥ 4 variables a
  favor y sin variable grave en contra.
* Confianza ALTA (≥ 70 % y variables coinciden), MEDIA (60-70 % o alguna en
  contra), BAJA (< 60 % o datos incompletos).
"""

from __future__ import annotations

from .odds import fair_odds

TRIVIAL = 0.88
MIN_MAIN = 0.60
MIN_SUPPORT = 4


def _pct(team: dict, key: str):
    return team.get("tendencias", {}).get(key)


def _blocks(res):
    return {b["key"]: b for b in res["index"]["blocks"]}


def _side_signals(res, side: str) -> tuple[list[tuple[bool, str]], bool]:
    """side: 'home' o 'away'. Señales = bloques del índice + modelos + mercado."""
    sign = 1 if side == "home" else -1
    signals = []
    for b in res["index"]["blocks"]:
        if b["origin"] == "sin datos" or b["score"] == 0:
            continue
        signals.append((sign * b["score"] > 0, f"{b['label']} ({b['score']:+d})"))
    key = "1" if side == "home" else "2"
    other = "2" if side == "home" else "1"
    agree = sum(1 for p in res["models_1x2"].values() if p[key] > p[other])
    signals.append((agree >= len(res["models_1x2"]) * 0.75, f"Modelos que lo favorecen: {agree}/{len(res['models_1x2'])}"))
    blocks = _blocks(res)
    grave = (sign * blocks["bajas"]["score"] <= -2) or (sign * res["index"]["value"] < -0.5)
    return signals, grave


def _total_signals(res, line: float, over: bool) -> list[tuple[bool, str]]:
    d = res["data"]
    h, a = d.get("local", {}), d.get("visitante", {})
    lam_t = res["lambda"]["home"] + res["lambda"]["away"]
    key = f"over{str(line).replace('.', '')}"
    s = []
    s.append(((lam_t > line + 0.25) if over else (lam_t < line - 0.25), f"Goles esperados {lam_t:.2f} vs línea {line}"))
    for name, team in ((d["partido"]["local"], h), (d["partido"]["visitante"], a)):
        pct = _pct(team, key)
        if pct is not None:
            s.append(((pct >= 55) if over else (pct <= 45), f"{name}: {pct}% de partidos con más de {line}"))
    h2h = d.get("h2h", {}).get(key + "_pct")
    if h2h is not None:
        s.append(((h2h >= 55) if over else (h2h <= 45), f"H2H: {h2h}% con más de {line}"))
    ch, ca = h.get("condicion", {}), a.get("condicion", {})
    if ch.get("xg") is not None and ca.get("xg") is not None:
        xg_t = ch["xg"] + ca["xg"]
        s.append(((xg_t > line) if over else (xg_t < line), f"xG sumado {xg_t:.2f}"))
    mc = res["matrices"]["Monte Carlo / Markov"]
    from .markets import totals
    p_mc = totals(mc, (line,))[line]["over"]
    s.append(((p_mc > 0.5) if over else (p_mc < 0.5), f"Monte Carlo: {p_mc:.0%} más de {line}"))
    return s


def _btts_signals(res, yes: bool) -> list[tuple[bool, str]]:
    d = res["data"]
    h, a = d.get("local", {}), d.get("visitante", {})
    s = []
    for name, team, lam in ((d["partido"]["local"], h, res["lambda"]["home"]),
                            (d["partido"]["visitante"], a, res["lambda"]["away"])):
        s.append(((lam >= 1.0) if yes else (lam < 0.9), f"λ {name} = {lam:.2f}"))
        pct = _pct(team, "btts")
        if pct is not None:
            s.append(((pct >= 55) if yes else (pct <= 45), f"{name}: ambos marcan en {pct}%"))
        cs = _pct(team, "vallas_invictas")
        if cs is not None:
            s.append(((cs <= 30) if yes else (cs >= 40), f"{name}: valla invicta en {cs}%"))
    h2h = d.get("h2h", {}).get("btts_pct")
    if h2h is not None:
        s.append(((h2h >= 55) if yes else (h2h <= 45), f"H2H: ambos marcan en {h2h}%"))
    return s


def _team_total_signals(res, side: str, line: float, over: bool) -> list[tuple[bool, str]]:
    d = res["data"]
    team = d.get("local" if side == "home" else "visitante", {})
    rival = d.get("visitante" if side == "home" else "local", {})
    lam = res["lambda"][side]
    s = [((lam > line + 0.4) if over else (lam < line - 0.2), f"λ = {lam:.2f}")]
    marca = _pct(team, "marca")
    if marca is not None:
        s.append(((marca >= 70) if over else (marca <= 50), f"Marca en {marca}% de sus partidos"))
    cs = _pct(rival, "vallas_invictas")
    if cs is not None:
        s.append(((cs <= 30) if over else (cs >= 40), f"Rival con valla invicta en {cs}%"))
    cond = team.get("condicion", {})
    if cond.get("xg") is not None:
        s.append(((cond["xg"] > line + 0.3) if over else (cond["xg"] < line), f"xG en su condición {cond['xg']:.2f}"))
    return s


def _prop_signals(res, kind: str, line: float, over: bool) -> list[tuple[bool, str]]:
    d = res["data"]
    exp = res[kind]["total"]
    s = [((exp > line + 0.5) if over else (exp < line - 0.5), f"Esperado {exp:.2f} vs línea {line}")]
    key = f"{'corners' if kind == 'corners' else 'tarjetas'}_over{str(line).replace('.', '')}"
    for name, team in ((d["partido"]["local"], d.get("local", {})), (d["partido"]["visitante"], d.get("visitante", {}))):
        pct = _pct(team, key)
        if pct is not None:
            s.append(((pct >= 55) if over else (pct <= 45), f"{name}: {pct}% sobre {line}"))
    if kind == "cards":
        ref = d.get("arbitro_stats", {}).get("amarillas_pp")
        if ref is not None:
            s.append(((ref > line) if over else (ref < line), f"Árbitro: {ref} amarillas por partido"))
    return s


def _ht_signals(res, line: float, over: bool) -> list[tuple[bool, str]]:
    d = res["data"]
    lam_ht = sum(res["ht"]["lambda"])
    s = [((lam_ht > line + 0.2) if over else (lam_ht < line - 0.2), f"λ 1er tiempo = {lam_ht:.2f}")]
    for name, team in ((d["partido"]["local"], d.get("local", {})), (d["partido"]["visitante"], d.get("visitante", {}))):
        pct = _pct(team, "gol_1t")
        if pct is not None:
            s.append(((pct >= 60) if over else (pct <= 40), f"{name}: hay gol en el 1T en {pct}%"))
    share = res["simulation"]["goal_minutes"]
    first_half = share["1-15"] + share["16-30"] + share["31-45"]
    s.append(((first_half >= 0.45) if over else (first_half < 0.45), f"Simulación: {first_half:.0%} de goles en el 1T"))
    return s


def build_candidates(res: dict) -> list[dict]:
    d = res["data"]
    H, A = d["partido"]["local"], d["partido"]["visitante"]
    mk = res["markets"]
    c: list[dict] = []

    def add(group, pid, label, prob, signals, grave=False, note=""):
        c.append({"group": group, "id": pid, "label": label, "prob": prob, "signals": signals,
                  "grave": grave, "note": note})

    home_sig, home_grave = _side_signals(res, "home")
    away_sig, away_grave = _side_signals(res, "away")
    p = res["final_1x2"]
    dc = mk["double_chance"]
    draw_sig = [(abs(res["index"]["value"]) < 0.3, f"Índice equilibrado ({res['index']['value']:+.2f})"),
                (p["X"] >= 0.28, f"Empate {p['X']:.0%}")]
    grp = "Resultado / doble oportunidad"
    add(grp, "1X2 1", f"Gana {H}", p["1"], home_sig, home_grave)
    add(grp, "1X2 X", "Empate", p["X"], draw_sig)
    add(grp, "1X2 2", f"Gana {A}", p["2"], away_sig, away_grave)
    add(grp, "DO 1X", f"{H} o empate (1X)", dc["1X"], home_sig, home_grave)
    add(grp, "DO X2", f"{A} o empate (X2)", dc["X2"], away_sig, away_grave)
    add(grp, "DO 12", "No hay empate (12)", dc["12"], [(not s, t) for s, t in draw_sig])
    add(grp, "DNB 1", f"{H} empate no válido", mk["dnb"]["1 (DNB)"], home_sig, home_grave, "Devuelve si empatan")
    add(grp, "DNB 2", f"{A} empate no válido", mk["dnb"]["2 (DNB)"], away_sig, away_grave, "Devuelve si empatan")

    grp = "Hándicap"
    ah = mk["ah_home_-1"]
    add(grp, "AH 1 -1", f"{H} -1 (asiático)", ah["effective"], home_sig, home_grave,
        f"Gana {ah['win']:.0%}, devuelve {ah['push']:.0%}")
    ah = mk["ah_away_+1"]
    add(grp, "AH 2 +1", f"{A} +1 (asiático)", ah["effective"], away_sig, away_grave,
        f"Gana {ah['win']:.0%}, devuelve {ah['push']:.0%}")
    add(grp, "AH 1 -1.5", f"{H} -1.5", mk["ah_home_-1.5"]["win"], home_sig, home_grave)
    add(grp, "AH 2 +1.5", f"{A} +1.5", mk["ah_away_+1.5"]["win"], away_sig, away_grave)
    ah = mk["ah_home_+1"]
    add(grp, "AH 1 +1", f"{H} +1 (asiático)", ah["effective"], home_sig, home_grave,
        f"Gana {ah['win']:.0%}, devuelve {ah['push']:.0%}")
    ah = mk["ah_away_-1"]
    add(grp, "AH 2 -1", f"{A} -1 (asiático)", ah["effective"], away_sig, away_grave,
        f"Gana {ah['win']:.0%}, devuelve {ah['push']:.0%}")

    grp = "Goles"
    for line in (1.5, 2.5, 3.5):
        t = mk["totals"][line]
        add(grp, f"Más de {line}", f"Más de {line} goles", t["over"], _total_signals(res, line, True))
        add(grp, f"Menos de {line}", f"Menos de {line} goles", t["under"], _total_signals(res, line, False))

    grp = "Ambos marcan"
    add(grp, "BTTS Sí", "Ambos marcan: Sí", mk["btts"]["yes"], _btts_signals(res, True))
    add(grp, "BTTS No", "Ambos marcan: No", mk["btts"]["no"], _btts_signals(res, False))

    grp = "Goles por equipo"
    for side, name in (("home", H), ("away", A)):
        for line in (0.5, 1.5):
            t = mk["team_totals"][side][line]
            add(grp, f"{side} más de {line}", f"{name} más de {line}", t["over"], _team_total_signals(res, side, line, True))
            add(grp, f"{side} menos de {line}", f"{name} menos de {line}", t["under"], _team_total_signals(res, side, line, False))

    if res.get("corners"):
        grp = "Córners"
        for line, t in res["corners"]["lines"].items():
            add(grp, f"Córners más de {line}", f"Más de {line} córners", t["over"], _prop_signals(res, "corners", line, True))
            add(grp, f"Córners menos de {line}", f"Menos de {line} córners", t["under"], _prop_signals(res, "corners", line, False))

    if res.get("cards"):
        grp = "Tarjetas"
        for line, t in res["cards"]["lines"].items():
            add(grp, f"Tarjetas más de {line}", f"Más de {line} tarjetas", t["over"], _prop_signals(res, "cards", line, True))
            add(grp, f"Tarjetas menos de {line}", f"Menos de {line} tarjetas", t["under"], _prop_signals(res, "cards", line, False))

    grp = "1er tiempo"
    for line in (0.5, 1.5):
        t = res["ht"]["totals"][line]
        add(grp, f"1T más de {line}", f"1er tiempo: más de {line}", t["over"], _ht_signals(res, line, True))
        add(grp, f"1T menos de {line}", f"1er tiempo: menos de {line}", t["under"], _ht_signals(res, line, False))
    ht = res["ht"]["1x2"]
    add(grp, "1T X", "1er tiempo: empate", ht["X"], [(abs(res["index"]["value"]) < 0.5, "Índice equilibrado")])

    user_odds = d.get("cuotas_usuario", {})
    incomplete = sum(1 for b in res["index"]["blocks"] if b["origin"] == "sin datos") >= 3
    for cand in c:
        cand["support"] = sum(1 for ok, _ in cand["signals"] if ok)
        cand["against"] = sum(1 for ok, _ in cand["signals"] if not ok)
        cand["fair"] = fair_odds(cand["prob"])
        cand["confidence"] = confidence(cand, incomplete)
        if cand["id"] in user_odds:
            o = float(user_odds[cand["id"]])
            cand["user_odds"] = o
            cand["user_vs_fair"] = "por encima (paga más que la justa)" if o > cand["fair"] else "por debajo (paga menos que la justa)"
    return c


def confidence(cand: dict, incomplete: bool) -> str:
    p = cand["prob"]
    if p < MIN_MAIN or incomplete or cand["grave"] or cand["against"] > cand["support"]:
        return "BAJA"
    if p >= 0.70 and cand["support"] >= MIN_SUPPORT and cand["against"] <= 1 and not cand["grave"]:
        return "ALTA"
    return "MEDIA"


def select_picks(candidates: list[dict]) -> dict:
    valid = [x for x in candidates if x["prob"] is not None and x["prob"] <= TRIVIAL]
    trivial = [x for x in candidates if x["prob"] is not None and x["prob"] > TRIVIAL]

    main_pool = [x for x in valid if x["prob"] >= MIN_MAIN and x["support"] >= MIN_SUPPORT and not x["grave"]]
    main = max(main_pool, key=lambda x: (x["prob"], x["support"]), default=None)
    main_warning = None
    if main is None:
        fallback = [x for x in valid if x["prob"] >= MIN_MAIN]
        main = max(fallback, key=lambda x: (x["support"], x["prob"]), default=None)
        main_warning = "Ningún pick cumple ≥4 variables a favor sin variable grave en contra: se muestra el mejor disponible."

    # Mejor pick de cada mercado: sin variable grave y con más variables a favor
    # que en contra; si ninguno cumple, se relaja la condición paso a paso.
    by_group: dict[str, dict] = {}
    for g in dict.fromkeys(x["group"] for x in valid):
        pool = [x for x in valid if x["group"] == g]
        for rule in (lambda x: not x["grave"] and x["support"] > x["against"],
                     lambda x: not x["grave"],
                     lambda x: True):
            eligible = [x for x in pool if rule(x)]
            if eligible:
                by_group[g] = max(eligible, key=lambda x: (x["prob"], x["support"]))
                break

    avoid = []
    for x in valid:
        if x["grave"] and x["prob"] >= 0.5:
            avoid.append((x, "variable grave en contra (bajas o índice hacia el rival)"))
        elif x["prob"] >= 0.55 and x["against"] > x["support"]:
            avoid.append((x, f"la probabilidad es alta pero {x['against']} variables van en contra"))
    for x in trivial:
        avoid.append((x, f"trivial ({x['prob']:.1%}): no aporta, cuota sin recorrido"))
    return {"main": main, "main_warning": main_warning, "by_group": by_group, "avoid": avoid}

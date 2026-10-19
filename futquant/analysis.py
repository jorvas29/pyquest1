"""Orquestador: corre todos los modelos y los combina en un ensamble.

Flujo
-----
1. Índice de ventaja (FASE 2) y λ ajustados (FASE 3.1).
2. Matrices de marcador: Poisson, Dixon-Coles y Monte Carlo/Markov.
   Con historial además Dixon-Coles MLE por equipo.
3. Modelos 1X2 directos: Elo, pi-ratings, Glicko-2, regresión logística,
   Random Forest, Gradient Boosting (+ XGBoost / LightGBM si están
   instalados) y consenso del mercado sin margen.
4. Ensamble 1X2 ponderado. La matriz combinada se reescala para que su
   1X2 coincida con el ensamble y así todos los mercados de goles sean
   coherentes con el resultado final.
5. Mercados, córners, tarjetas, primer tiempo y contraste con el mercado.
"""

from __future__ import annotations

import numpy as np

from . import markets as mk
from .dixon_coles import DEFAULT_RHO, DixonColes, dc_matrix
from .features import FeatureBuilder
from .index import advantage_index, expected_goals
from .ml import train_and_predict
from .montecarlo import MarkovMatchSimulator
from .odds import market_consensus
from .props import expected_cards, expected_corners, lines_table
from .ratings import Elo, Glicko2, PiRatings, davidson_1x2

MATRIX_WEIGHTS = {"Poisson": 0.2, "Dixon-Coles": 0.4, "Monte Carlo / Markov": 0.4}

# Peso de cada familia en el ensamble 1X2 (se renormaliza con las disponibles).
FAMILY_WEIGHTS = {"marcador": 0.45, "ratings": 0.15, "ml": 0.20, "mercado": 0.20}
FAMILY_OF = {
    "Poisson": "marcador", "Dixon-Coles": "marcador", "Monte Carlo / Markov": "marcador",
    "Dixon-Coles MLE (historial)": "marcador",
    "Elo": "ratings", "pi-ratings": "ratings", "Glicko-2": "ratings",
    "Regresión logística": "ml", "Random Forest": "ml", "Gradient Boosting": "ml",
    "XGBoost": "ml", "LightGBM": "ml",
    "Mercado (sin margen, Shin)": "mercado",
}


def rescale_to_1x2(M: np.ndarray, target: dict[str, float]) -> np.ndarray:
    current = mk.result_1x2(M)
    out = M.copy()
    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            key = "1" if i > j else "X" if i == j else "2"
            out[i, j] *= target[key] / current[key] if current[key] > 0 else 0
    return out / out.sum()


def ensemble_1x2(models: dict[str, dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    families: dict[str, list[str]] = {}
    for name in models:
        families.setdefault(FAMILY_OF.get(name, "ml"), []).append(name)
    total_w = sum(FAMILY_WEIGHTS[f] for f in families)
    weights = {}
    for fam, names in families.items():
        for name in names:
            weights[name] = FAMILY_WEIGHTS[fam] / total_w / len(names)
    p = {k: sum(weights[n] * models[n][k] for n in models) for k in "1X2"}
    s = sum(p.values())
    return {k: v / s for k, v in p.items()}, weights


def analyze(data: dict, history: list[dict] | None = None, n_sims: int = 10_000) -> dict:
    home_name = data["partido"]["local"]
    away_name = data["partido"]["visitante"]
    res: dict = {"data": data, "notes": []}

    # ---------------------------------------------------------------- FASE 2 y 3.1
    idx = advantage_index(data)
    lam = expected_goals(data, idx["value"])
    res["index"], res["lambda"] = idx, lam

    # ------------------------------------------------------ modelos con historial
    models_1x2: dict[str, dict[str, float]] = {}
    ml_metrics: dict[str, dict] = {}
    rho = data.get("dc_rho")
    dc_hist_matrix = None
    if history:
        teams = {m["home"] for m in history} | {m["away"] for m in history}
        known = home_name in teams and away_name in teams
        if not known:
            res["notes"].append(f"El historial no contiene a {home_name} o {away_name}: se omiten modelos por equipo.")
        dc = DixonColes().fit(history)
        rho = dc.rho if rho is None else rho
        res["dc_fit"] = {"rho": dc.rho, "home_adv": float(np.exp(dc.home_adv)), "n": len(history)}
        if known:
            dc_hist_matrix = dc.predict_matrix(home_name, away_name)
            res["dc_fit"]["lambda"] = dc.expected_goals(home_name, away_name)
            models_1x2["Dixon-Coles MLE (historial)"] = mk.result_1x2(dc_hist_matrix)
            elo = Elo().fit(history)
            pi = PiRatings().fit(history)
            gl = Glicko2().fit(history)
            models_1x2["Elo"] = elo.predict_1x2(home_name, away_name)
            models_1x2["pi-ratings"] = pi.predict_1x2(home_name, away_name)
            models_1x2["Glicko-2"] = gl.predict_1x2(home_name, away_name)
            res["ratings"] = {
                "Elo": (elo.rating(home_name), elo.rating(away_name)),
                "pi-ratings (dif. goles esperada)": (pi.expected_goal_diff(home_name, away_name), None),
                "Glicko-2": (gl.player(home_name).rating, gl.player(away_name).rating),
            }
            fb = FeatureBuilder()
            X, y, cols, state = fb.build(history)
            x_t = fb.match_row(state, cols, home_name, away_name, data["partido"].get("fecha_hora", "")[:10] or None)
            ml_res = train_and_predict(X, y, x_t)
            base_ll = ml_res.get("Frecuencia base (referencia)", {}).get("logloss", float("inf"))
            for name, r in ml_res.items():
                ml_metrics[name] = r
                if name.startswith("Frecuencia"):
                    continue
                # Un modelo que en validación no supera a la frecuencia base no entra al ensamble.
                r["in_ensemble"] = r["logloss"] <= base_ll
                if r["in_ensemble"]:
                    models_1x2[name] = r["1x2"]
                else:
                    res["notes"].append(f"{name} excluido del ensamble: log-loss de validación "
                                        f"{r['logloss']:.3f} peor que la frecuencia base {base_ll:.3f}.")
            res["ml_features"] = cols
            if not ml_res:
                res["notes"].append("Historial insuficiente (<60 partidos con forma previa) para entrenar ML.")
    else:
        res["notes"].append("Sin historial CSV: no se entrenan Dixon-Coles MLE, pi-ratings, Glicko-2 ni ML.")

    # Elo dado a mano (p. ej. clubelo.com / eloratings.net) si no hay historial.
    elo_in = data.get("ratings", {}).get("elo")
    if "Elo" not in models_1x2 and elo_in:
        diff = elo_in["local"] + elo_in.get("ventaja_local", 65) - elo_in["visitante"]
        models_1x2["Elo"] = davidson_1x2(diff)
        res["ratings"] = {"Elo": (elo_in["local"], elo_in["visitante"])}

    rho = DEFAULT_RHO if rho is None else rho
    res["rho"] = rho

    # ------------------------------------------------------- matrices de marcador
    sim_cfg = data.get("simulacion", {})
    rr = data.get("arbitro_stats", {}).get("rojas_pp")
    red_rate = (rr / 2, rr / 2) if rr is not None else (0.08, 0.10)
    sim = MarkovMatchSimulator(red_card_rate=red_rate, seed=sim_cfg.get("semilla", 7)).simulate(
        lam["home"], lam["away"], sim_cfg.get("n", n_sims))
    matrices = {
        "Poisson": mk.poisson_matrix(lam["home"], lam["away"]),
        "Dixon-Coles": dc_matrix(lam["home"], lam["away"], rho),
        "Monte Carlo / Markov": sim["matrix"],
    }
    for name, M in matrices.items():
        models_1x2[name] = mk.result_1x2(M)
    blended = sum(MATRIX_WEIGHTS[n] * M for n, M in matrices.items())
    if dc_hist_matrix is not None:
        blended = 0.75 * blended + 0.25 * dc_hist_matrix
    blended /= blended.sum()

    # ------------------------------------------------------------------ mercado
    books = data.get("cuotas", {}).get("casas", [])
    consensus = market_consensus(books)
    if consensus:
        models_1x2["Mercado (sin margen, Shin)"] = consensus
    market_ou = market_consensus(books, ("over25", "under25"))
    market_btts = market_consensus(books, ("btts_si", "btts_no"))

    final_1x2, weights = ensemble_1x2(models_1x2)
    final_M = rescale_to_1x2(blended, final_1x2)
    res["models_1x2"], res["weights"], res["final_1x2"] = models_1x2, weights, final_1x2
    res["ml_metrics"] = ml_metrics
    res["matrices"] = matrices
    res["matrix"] = final_M
    res["markets"] = mk.all_markets(final_M)
    res["simulation"] = {k: v for k, v in sim.items() if "matrix" not in k}

    # --------------------------------------------------------------- 1er tiempo
    ht_M = 0.5 * mk.poisson_matrix(lam["home"] * 0.45, lam["away"] * 0.45) + 0.5 * sim["ht_matrix"]
    ht_M /= ht_M.sum()
    res["ht"] = {
        "lambda": (lam["home"] * 0.45, lam["away"] * 0.45),
        "1x2": mk.result_1x2(ht_M),
        "totals": mk.totals(ht_M, (0.5, 1.5)),
        "ht_ft": mk.ht_ft(ht_M, final_M),
    }

    # ------------------------------------------------------- córners y tarjetas
    h, a = data.get("local", {}), data.get("visitante", {})
    disp = data.get("dispersion", {})
    corners = expected_corners(h, a)
    if corners:
        corners["lines"] = lines_table(corners["total"], (7.5, 8.5, 9.5, 10.5, 11.5), disp.get("corners", 1.15))
    cards = expected_cards(h, a, data.get("arbitro_stats"))
    if cards:
        cards["lines"] = lines_table(cards["total"], (2.5, 3.5, 4.5, 5.5, 6.5), disp.get("tarjetas", 1.2))
    res["corners"], res["cards"] = corners, cards

    # ------------------------------------------------ contraste con el mercado
    contrast = []
    if consensus:
        for k, label in (("1", "Local"), ("X", "Empate"), ("2", "Visitante")):
            contrast.append((f"1X2 {label}", final_1x2[k], consensus[k]))
    if market_ou:
        contrast.append(("Más de 2.5", res["markets"]["totals"][2.5]["over"], market_ou["over25"]))
    if market_btts:
        contrast.append(("Ambos marcan Sí", res["markets"]["btts"]["yes"], market_btts["btts_si"]))
    res["contrast"] = [
        {"market": m, "model": p, "market_p": q, "diff": p - q, "flag": abs(p - q) > 0.10} for m, p, q in contrast
    ]
    return res

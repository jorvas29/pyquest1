import math

import numpy as np
import pytest

from futquant import markets as mk
from futquant.analysis import analyze, rescale_to_1x2
from futquant.dixon_coles import DixonColes, dc_matrix
from futquant.ficha import render
from futquant.index import advantage_index, expected_goals
from futquant.montecarlo import MarkovMatchSimulator, time_profile
from futquant.odds import devig_power, devig_proportional, devig_shin
from futquant.picks import build_candidates, select_picks
from futquant.props import count_over
from futquant.ratings import Elo, GlickoPlayer, PiRatings, davidson_1x2, glicko2_update
from futquant.synthetic import synthetic_league


def test_poisson_markets_consistent():
    M = mk.poisson_matrix(1.5, 1.1)
    p = mk.result_1x2(M)
    assert math.isclose(sum(p.values()), 1, abs_tol=1e-9)
    t = mk.totals(M)
    assert t[2.5]["over"] + t[2.5]["under"] == pytest.approx(1)
    eh, ea = mk.expected_goals(M)
    assert eh == pytest.approx(1.5, abs=1e-3) and ea == pytest.approx(1.1, abs=1e-3)
    ah = mk.asian_handicap(M, -1, "home")
    assert ah["win"] + ah["push"] + ah["lose"] == pytest.approx(1)
    # -1.5 del local = ganar por 2 o más = hándicap europeo -1 "1"
    assert mk.asian_handicap(M, -1.5)["win"] == pytest.approx(mk.european_handicap(M, -1)["1"])


def test_dixon_coles_inflates_low_draws():
    P = mk.poisson_matrix(1.4, 1.1)
    D = dc_matrix(1.4, 1.1, rho=-0.12)
    assert D[0, 0] > P[0, 0] and D[1, 1] > P[1, 1]
    assert D[1, 0] < P[1, 0] and D[0, 1] < P[0, 1]
    assert D.sum() == pytest.approx(1)


def test_dixon_coles_fit_recovers_parameters():
    matches, truth = synthetic_league(n_teams=10, seasons=6, seed=3)
    dc = DixonColes(xi=0).fit(matches)
    est = np.array([dc.attack[t] for t in truth["teams"]])
    real = np.array([truth["attack"][t] for t in truth["teams"]])
    assert np.corrcoef(est, real)[0, 1] > 0.8
    assert dc.home_adv == pytest.approx(truth["home_adv"], abs=0.1)
    assert -0.35 <= dc.rho <= 0.35


def test_elo_margin_and_symmetry():
    a, b = Elo(), Elo()
    a.update("X", "Y", 1, 0)
    b.update("X", "Y", 3, 0)
    assert b.rating("X") - 1500 > a.rating("X") - 1500 > 0
    assert a.rating("X") + a.rating("Y") == pytest.approx(3000)
    p = davidson_1x2(0)
    assert p["1"] == pytest.approx(p["2"]) and p["X"] == pytest.approx(0.27, abs=0.01)


def test_glicko2_reference_example():
    # Ejemplo del paper de Glickman (Glicko-2): 1500/200 -> 1464.06 / 151.52
    p = GlickoPlayer(1500, 200, 0.06)
    games = [(GlickoPlayer(1400, 30), 1), (GlickoPlayer(1550, 100), 0), (GlickoPlayer(1700, 300), 0)]
    new = glicko2_update(p, games, tau=0.5)
    assert new.rating == pytest.approx(1464.06, abs=0.05)
    assert new.rd == pytest.approx(151.52, abs=0.05)
    assert new.vol == pytest.approx(0.05999, abs=1e-4)


def test_pi_ratings_move_in_right_direction():
    pi = PiRatings()
    for _ in range(10):
        pi.update("Fuerte", "Débil", 3, 0)
    assert pi.expected_goal_diff("Fuerte", "Débil") > 0.5
    p = pi.predict_1x2("Fuerte", "Débil")
    assert p["1"] > p["2"]


def test_devig_methods_sum_to_one():
    odds = [1.85, 3.7, 4.2]
    for fn in (devig_proportional, devig_shin, devig_power):
        assert sum(fn(odds)) == pytest.approx(1)
    # Shin rebaja al favorito menos que el proporcional quita a la sorpresa
    assert devig_shin(odds)[0] > devig_proportional(odds)[0]


def test_montecarlo_respects_lambdas_and_half_share():
    w = time_profile()
    assert w[:45].sum() == pytest.approx(0.45, abs=1e-6)
    r = MarkovMatchSimulator(seed=1).simulate(1.6, 1.0, 20000)
    eh, ea = mk.expected_goals(r["matrix"])
    assert eh == pytest.approx(1.6, abs=0.06) and ea == pytest.approx(1.0, abs=0.06)


def test_rescale_matches_target():
    M = mk.poisson_matrix(1.3, 1.2)
    target = {"1": 0.5, "X": 0.25, "2": 0.25}
    R = rescale_to_1x2(M, target)
    for k, v in mk.result_1x2(R).items():
        assert v == pytest.approx(target[k])


def test_count_over_poisson_and_nb():
    assert count_over(10, 9.5, 1.0) == pytest.approx(1 - 0.4579297, abs=1e-4)
    assert count_over(10, 14.5, 1.3) > count_over(10, 14.5, 1.0)  # más varianza, más cola


def _data():
    return {
        "partido": {"local": "Equipo A", "visitante": "Equipo B", "fecha_hora": "2026-10-25T20:00:00+02:00"},
        "local": {"ppg10": 2.0, "condicion": {"ppg": 2.2, "gf": 1.9, "gc": 0.9, "xg": 1.8, "xga": 1.0},
                  "corners_favor": 6, "corners_contra": 4, "amarillas_favor": 1.8, "amarillas_contra": 2.2,
                  "descanso_dias": 6, "bajas": []},
        "visitante": {"ppg10": 1.2, "condicion": {"ppg": 1.0, "gf": 1.0, "gc": 1.6, "xg": 1.1, "xga": 1.6},
                      "corners_favor": 4, "corners_contra": 6, "amarillas_favor": 2.3, "amarillas_contra": 1.9,
                      "descanso_dias": 3, "bajas": [{"jugador": "Z", "estado": "CONFIRMADO", "peso": 2}]},
        "arbitro_stats": {"amarillas_pp": 4.5},
        "cuotas": {"casas": [{"1": 1.6, "X": 4.0, "2": 5.5}]},
    }


def test_index_and_lambda_cap():
    d = _data()
    idx = advantage_index(d)
    assert 0 < idx["value"] <= 2
    lam = expected_goals(d, 2.0)
    assert lam["home"] <= lam["base_home"] * 1.15 + 1e-9
    assert lam["away"] >= lam["base_away"] * 0.85 - 1e-9


def test_full_pipeline_with_history_and_rules():
    history, _ = synthetic_league(seasons=2)
    res = analyze(_data(), history, n_sims=3000)
    assert sum(res["final_1x2"].values()) == pytest.approx(1)
    picks = select_picks(build_candidates(res))
    main = picks["main"]
    assert main is None or 0.60 <= main["prob"] <= 0.88
    assert all(x["prob"] <= 0.88 for x in picks["by_group"].values())
    md = render(res)
    for n in range(1, 12):
        assert f"## {n}." in md
    assert "Pronóstico estadístico, no garantía" in md
    assert "hora Colombia" in md


def test_pipeline_without_history():
    res = analyze(_data(), None, n_sims=2000)
    assert "Mercado (sin margen, Shin)" in res["models_1x2"]
    assert res["final_1x2"]["1"] > res["final_1x2"]["2"]

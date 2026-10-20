"""Lógica de probabilidades: 1X2, BTTS, más/menos y córners."""

import numpy as np
import pytest
from scipy import stats

from predictor.modelos.dixon_coles import matriz_marcadores
from predictor.modelos.mercados import (
    marcadores_probables,
    probabilidad_btts,
    probabilidad_mas_de,
    probabilidades_1x2,
    probabilidades_corners,
)

CASOS = [
    (1.5, 1.1, 0.0),
    (2.3, 0.7, -0.1),
    (0.6, 0.6, 0.05),
    (3.0, 2.5, -0.2),
]


@pytest.mark.parametrize("lam, mu, rho", CASOS)
def test_matriz_es_distribucion_valida(lam, mu, rho):
    m = matriz_marcadores(lam, mu, rho)
    assert np.all(m >= 0)
    assert m.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("lam, mu, rho", CASOS)
def test_1x2_suma_uno(lam, mu, rho):
    p = probabilidades_1x2(matriz_marcadores(lam, mu, rho))
    assert set(p) == {"1", "X", "2"}
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(0.0 < v < 1.0 for v in p.values())


@pytest.mark.parametrize("lam, mu, rho", CASOS)
def test_btts_coherente_con_la_distribucion(lam, mu, rho):
    m = matriz_marcadores(lam, mu, rho)
    p_local_cero = m[0, :].sum()
    p_visit_cero = m[:, 0].sum()
    esperado = 1.0 - p_local_cero - p_visit_cero + m[0, 0]
    assert probabilidad_btts(m) == pytest.approx(esperado, abs=1e-12)


@pytest.mark.parametrize("lam, mu", [(1.5, 1.1), (2.3, 0.7), (0.4, 3.1)])
def test_btts_poisson_independiente_forma_cerrada(lam, mu):
    # Con rho = 0: P(BTTS) = (1 - e^-lam) * (1 - e^-mu)
    m = matriz_marcadores(lam, mu, 0.0, max_goles=30)
    assert probabilidad_btts(m) == pytest.approx((1 - np.exp(-lam)) * (1 - np.exp(-mu)), abs=1e-9)


def test_rho_cero_es_poisson_independiente():
    lam, mu = 1.7, 0.9
    m = matriz_marcadores(lam, mu, 0.0, max_goles=30)
    k = np.arange(31)
    esperado = np.outer(stats.poisson.pmf(k, lam), stats.poisson.pmf(k, mu))
    np.testing.assert_allclose(m, esperado / esperado.sum(), atol=1e-12)


def test_rho_negativo_sube_empates_bajos():
    poisson = matriz_marcadores(1.3, 1.1, 0.0)
    dc = matriz_marcadores(1.3, 1.1, -0.15)
    assert dc[0, 0] > poisson[0, 0]
    assert dc[1, 1] > poisson[1, 1]
    assert dc[1, 0] < poisson[1, 0]
    assert probabilidades_1x2(dc)["X"] > probabilidades_1x2(poisson)["X"]


def test_equipos_iguales_dan_1x2_simetrico():
    p = probabilidades_1x2(matriz_marcadores(1.4, 1.4, -0.1))
    assert p["1"] == pytest.approx(p["2"], abs=1e-12)


def test_mas_goles_esperados_aumenta_victoria_local():
    anteriores = [probabilidades_1x2(matriz_marcadores(lam, 1.2))["1"] for lam in (0.8, 1.2, 1.6, 2.0)]
    assert anteriores == sorted(anteriores)


def test_mas_menos_goles_complementarios():
    m = matriz_marcadores(1.6, 1.2, -0.05)
    i, j = np.indices(m.shape)
    assert probabilidad_mas_de(m, 2.5) + m[(i + j) <= 2].sum() == pytest.approx(1.0)


def test_marcadores_probables_ordenados():
    top = marcadores_probables(matriz_marcadores(1.5, 1.0), 5)
    probs = [p for _, _, p in top]
    assert probs == sorted(probs, reverse=True)
    assert len(top) == 5


@pytest.mark.parametrize("dispersion", [1.0, 1.6])
def test_corners_suman_uno(dispersion):
    c = probabilidades_corners(5.2, 4.4, 9.5, dispersion)
    assert c["mas"] + c["menos"] + c["nulo"] == pytest.approx(1.0, abs=1e-9)
    assert c["nulo"] == 0.0
    assert c["local_mas"] + c["igual"] + c["visitante_mas"] == pytest.approx(1.0, abs=1e-9)
    assert c["media_total"] == pytest.approx(9.6)


def test_corners_linea_entera_tiene_nula():
    c = probabilidades_corners(5.0, 5.0, 10.0)
    assert c["nulo"] == pytest.approx(stats.poisson.pmf(10, 10.0), abs=1e-9)
    assert c["mas"] + c["menos"] + c["nulo"] == pytest.approx(1.0, abs=1e-9)


def test_corners_poisson_coincide_con_scipy():
    # Suma de dos Poisson independientes = Poisson con la suma de medias.
    c = probabilidades_corners(5.5, 4.0, 9.5, dispersion=1.0)
    assert c["mas"] == pytest.approx(stats.poisson.sf(9, 9.5), abs=1e-9)


def test_sobredispersion_ensancha_la_distribucion():
    # Con más varianza, las colas pesan más: sube P(total > 13.5).
    poisson = probabilidades_corners(5.0, 5.0, 13.5, dispersion=1.0)
    nb = probabilidades_corners(5.0, 5.0, 13.5, dispersion=1.8)
    assert nb["mas"] > poisson["mas"]

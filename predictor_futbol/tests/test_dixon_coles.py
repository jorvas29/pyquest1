"""Ajuste del modelo de goles."""

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from predictor.modelos.dixon_coles import ModeloDixonColes


def test_gradiente_analitico_correcto(liga_pequena):
    modelo = ModeloDixonColes()
    ih, ia, x, y, w, _ = modelo._datos_ajuste(liga_pequena.head(120))
    n = len(modelo.equipos)
    rng = np.random.default_rng(0)
    theta = rng.normal(0, 0.1, 2 + 2 * n + 1)
    theta[-1] = -0.05
    f = lambda t: modelo._nll_y_gradiente(t, ih, ia, x, y, w, n)[0]
    g = lambda t: modelo._nll_y_gradiente(t, ih, ia, x, y, w, n)[1]
    error = optimize.check_grad(f, g, theta)
    assert error < 1e-3 * np.linalg.norm(g(theta))


def test_recupera_parametros_de_la_liga_sintetica(liga_completa):
    # La liga sintética usa ventaja local 0.25 y rho -0.08.
    modelo = ModeloDixonColes().ajustar(liga_completa)
    assert modelo.convergio
    assert modelo.localia == pytest.approx(0.25, abs=0.08)
    assert -0.2 < modelo.rho < 0.0


def test_equipo_fuerte_favorito():
    filas = []
    fecha = pd.Timestamp("2024-01-01")
    for k in range(30):
        filas.append({"fecha": fecha + pd.Timedelta(days=7 * k), "local": "Fuerte", "visitante": "Débil",
                      "goles_local": 3, "goles_visitante": 0})
        filas.append({"fecha": fecha + pd.Timedelta(days=7 * k + 3), "local": "Débil", "visitante": "Fuerte",
                      "goles_local": 0, "goles_visitante": 2})
    modelo = ModeloDixonColes(usar_rho=False).ajustar(pd.DataFrame(filas))
    lam, mu = modelo.tasas("Fuerte", "Débil")
    assert lam > mu
    assert modelo.rho == 0.0


def test_equipo_desconocido_cuenta_como_medio(liga_pequena):
    modelo = ModeloDixonColes().ajustar(liga_pequena)
    lam, mu = modelo.tasas("Equipo Nuevo", "Otro Nuevo")
    assert lam == pytest.approx(np.exp(modelo.intercepto + modelo.localia))
    assert mu == pytest.approx(np.exp(modelo.intercepto))


def test_pocos_partidos_da_error(liga_pequena):
    with pytest.raises(ValueError):
        ModeloDixonColes().ajustar(liga_pequena.head(5))

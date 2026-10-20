"""Validación temporal y métricas."""

import math

import numpy as np
import pytest

from predictor.evaluacion import (
    backtest,
    brier_binario,
    brier_multiclase,
    log_loss_binaria,
    log_loss_multiclase,
    particiones_temporales,
    resumen_metricas,
    simular_valor,
    tabla_calibracion,
)


def test_particiones_sin_fuga(liga_pequena):
    vistos = []
    for ent, prueba in particiones_temporales(liga_pequena, min_entrenamiento=60, paso_dias=7):
        assert liga_pequena.loc[ent, "fecha"].max() < liga_pequena.loc[prueba, "fecha"].min()
        vistos.extend(prueba)
    # Cada partido posterior al entrenamiento inicial se evalúa exactamente una vez.
    assert len(vistos) == len(set(vistos))
    inicio = liga_pequena["fecha"].sort_values().iloc[60]
    assert set(vistos) == set(liga_pequena.index[liga_pequena["fecha"] >= inicio])


def test_metricas_valores_conocidos():
    assert log_loss_binaria([0.5, 0.5], [1, 0]) == pytest.approx(math.log(2))
    assert brier_binario([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert brier_binario([0.5], [1]) == pytest.approx(0.25)
    probs = np.array([[1 / 3, 1 / 3, 1 / 3]] * 3)
    assert log_loss_multiclase(probs, [0, 1, 2]) == pytest.approx(math.log(3))
    assert brier_multiclase(np.array([[1.0, 0.0, 0.0]]), [0]) == pytest.approx(0.0)
    assert brier_multiclase(np.array([[0.0, 1.0, 0.0]]), [0]) == pytest.approx(2.0)


def test_calibracion_cuenta_todo():
    p = np.linspace(0.05, 0.95, 50)
    y = (p > 0.5).astype(float)
    t = tabla_calibracion(p, y, 10)
    assert t["n"].sum() == 50


def test_backtest_produce_probabilidades_validas(liga_pequena):
    pred = backtest(liga_pequena, min_entrenamiento=90, paso_dias=14)
    assert len(pred) > 0
    np.testing.assert_allclose(pred[["p_1", "p_x", "p_2"]].sum(axis=1), 1.0, atol=1e-9)
    assert pred["p_btts"].between(0, 1).all()
    assert pred["p_corners_mas"].between(0, 1).all()
    # El tamaño del entrenamiento crece con el tiempo (ventana expansiva).
    assert pred["n_entrenamiento"].is_monotonic_increasing

    resumen = resumen_metricas(pred)
    assert set(resumen["mercado"]) == {"1X2", "Ambos marcan", "Córners (más de la línea)"}
    assert (resumen["logloss_modelo"] > 0).all()

    valor = simular_valor(pred, umbral=0.05)
    assert (valor["aciertos"] <= valor["apuestas"]).all()

"""Conversión entre probabilidades y cuotas."""

import math

import pytest

from predictor.cuotas import (
    cuota_justa,
    margen,
    probabilidad_implicita,
    probabilidades_sin_margen,
    valor_esperado,
)


def test_cuota_justa_es_inversa():
    assert cuota_justa(0.5) == pytest.approx(2.0)
    assert cuota_justa(0.25) == pytest.approx(4.0)
    assert math.isinf(cuota_justa(0.0))


def test_sin_margen_suma_uno_y_conserva_orden():
    cuotas = {"1": 2.10, "X": 3.40, "2": 3.60}
    p = probabilidades_sin_margen(cuotas)
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["1"] > p["X"] > p["2"]


def test_margen():
    assert margen({"si": 1.90, "no": 1.90}) == pytest.approx(2 / 1.9 - 1)
    assert margen({"si": 2.0, "no": 2.0}) == pytest.approx(0.0)


def test_valor_esperado():
    assert valor_esperado(0.5, 2.0) == pytest.approx(0.0)
    assert valor_esperado(0.6, 2.0) == pytest.approx(0.2)
    assert valor_esperado(0.4, 2.0) == pytest.approx(-0.2)


def test_cuota_invalida():
    with pytest.raises(ValueError):
        probabilidad_implicita(1.0)

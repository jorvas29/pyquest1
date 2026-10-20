"""Punto de entrada de línea de comandos."""

import pytest

from predictor import ADVERTENCIA
from predictor.cli import main, parsear_cuotas, resolver_equipo
from predictor.prediccion import ModeloConjunto


def test_parsear_cuotas():
    assert parsear_cuotas(["1=2.10", "x=3,40", "BTTS_SI=1.8"]) == {"1": 2.10, "X": 3.40, "btts_si": 1.8}
    with pytest.raises(ValueError):
        parsear_cuotas(["ganador=2"])
    with pytest.raises(ValueError):
        parsear_cuotas(["1=0.9"])


def test_resolver_equipo():
    equipos = ["Sintético 01 FC", "Sintético 02 FC"]
    assert resolver_equipo("sintético 01 fc", equipos) == "Sintético 01 FC"
    with pytest.raises(ValueError, match="Quizás"):
        resolver_equipo("Sintetico 01", equipos)


def test_prediccion_completa(liga_pequena):
    pred = ModeloConjunto().ajustar(liga_pequena).predecir("Sintético 01 FC", "Sintético 02 FC")
    assert sum(pred.p_1x2.values()) == pytest.approx(1.0)
    por_mercado = {}
    for s in pred.selecciones:
        por_mercado.setdefault(s.mercado, 0.0)
        por_mercado[s.mercado] += s.probabilidad + s.p_nulo
    # Cada mercado reparte la probabilidad completa entre sus selecciones.
    for total in por_mercado.values():
        assert total == pytest.approx(1.0, abs=1e-9)


def test_cli_predecir(capsys):
    codigo = main(["predecir", "Sintético 01 FC", "Sintético 02 FC", "--cuotas", "1=2.0", "X=3.5", "2=3.8"])
    salida = capsys.readouterr().out
    assert codigo == 0
    assert "MODELO vs MERCADO" in salida
    assert "Ambos marcan" in salida and "Córners" in salida
    assert ADVERTENCIA in salida


def test_cli_equipo_inexistente(capsys):
    assert main(["predecir", "Real Inventado", "Sintético 02 FC"]) == 2
    assert "Equipo no encontrado" in capsys.readouterr().err

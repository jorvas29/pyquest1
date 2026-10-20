"""Datos compartidos por las pruebas (siempre SINTÉTICOS)."""

import pytest

from predictor.datos import FuenteSintetica


@pytest.fixture(scope="session")
def liga_pequena():
    """Liga sintética pequeña (10 equipos, 2 temporadas) para pruebas rápidas."""
    return FuenteSintetica(semilla=7, n_equipos=10, temporadas=2).cargar()


@pytest.fixture(scope="session")
def liga_completa():
    """Liga sintética por defecto (20 equipos, 3 temporadas)."""
    return FuenteSintetica().cargar()

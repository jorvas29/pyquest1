"""Capa de fuentes de datos."""

import pandas as pd
import pytest

from predictor.datos import FuenteAPI, FuenteCSV, FuenteSintetica, normalizar


def test_csv_formato_football_data(tmp_path):
    ruta = tmp_path / "liga.csv"
    pd.DataFrame(
        {
            "Div": ["X1", "X1"],
            "Date": ["05/01/2025", "12/01/2025"],
            "HomeTeam": ["Equipo A", "Equipo B"],
            "AwayTeam": ["Equipo B", "Equipo A"],
            "FTHG": [2, 0],
            "FTAG": [1, 0],
            "HC": [6, 3],
            "AC": [4, 5],
            "B365H": [1.9, 2.5],
            "B365D": [3.4, 3.1],
            "B365A": [4.0, 2.9],
        }
    ).to_csv(ruta, index=False)
    df = FuenteCSV(ruta).cargar()
    assert list(df["local"]) == ["Equipo A", "Equipo B"]
    # dd/mm/aaaa: 05/01 es el 5 de enero, no el 1 de mayo.
    assert df["fecha"].iloc[0] == pd.Timestamp("2025-01-05")
    assert df["corners_local"].tolist() == [6, 3]
    assert df["cuota_1"].iloc[0] == 1.9


def test_csv_descarta_partidos_sin_jugar(tmp_path):
    ruta = tmp_path / "comun.csv"
    pd.DataFrame(
        {
            "fecha": ["2025-01-01", "2025-01-08"],
            "local": ["A", "B"],
            "visitante": ["B", "A"],
            "goles_local": [1, None],
            "goles_visitante": [1, None],
        }
    ).to_csv(ruta, index=False)
    assert len(FuenteCSV(ruta).cargar()) == 1


def test_faltan_columnas():
    with pytest.raises(ValueError, match="Faltan columnas"):
        normalizar(pd.DataFrame({"fecha": ["2025-01-01"], "local": ["A"]}))


def test_api_con_json_anidado():
    respuesta = {
        "matches": [
            {"utcDate": "2025-02-01T15:00:00Z", "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"},
             "score": {"fullTime": {"home": 2, "away": 2}}},
            {"utcDate": "2025-02-08T15:00:00Z", "homeTeam": {"name": "B"}, "awayTeam": {"name": "A"},
             "score": {"fullTime": {"home": None, "away": None}}},
        ]
    }
    fuente = FuenteAPI(
        url="https://ejemplo.invalid/partidos",
        mapeo={
            "utcDate": "fecha",
            "homeTeam.name": "local",
            "awayTeam.name": "visitante",
            "score.fullTime.home": "goles_local",
            "score.fullTime.away": "goles_visitante",
        },
        ruta_registros=["matches"],
        obtener_json=lambda url: respuesta,
    )
    df = fuente.cargar()
    assert len(df) == 1
    assert df.iloc[0]["goles_local"] == 2
    assert df.iloc[0]["fecha"] == pd.Timestamp("2025-02-01")


def test_sintetica_esta_etiquetada():
    fuente = FuenteSintetica(semilla=1, n_equipos=6, temporadas=1)
    df = fuente.cargar()
    assert fuente.es_sintetica
    assert (df["origen"] == "SINTETICO").all()
    assert df["local"].str.startswith("Sintético").all()
    # 6 equipos a doble vuelta = 30 partidos.
    assert len(df) == 30


def test_sintetica_reproducible():
    a = FuenteSintetica(semilla=3, n_equipos=6, temporadas=1).cargar()
    b = FuenteSintetica(semilla=3, n_equipos=6, temporadas=1).cargar()
    pd.testing.assert_frame_equal(a, b)

"""Características previas: sin fuga de datos del futuro."""

import pandas as pd
import pytest

from predictor.caracteristicas import construir_caracteristicas, perfil_previo


def test_sin_fuga_al_cambiar_resultados_futuros(liga_pequena):
    corte = liga_pequena["fecha"].iloc[len(liga_pequena) // 2]
    alterada = liga_pequena.copy()
    futuro = alterada["fecha"] >= corte
    alterada.loc[futuro, "goles_local"] = 9
    alterada.loc[futuro, "corners_local"] = 25

    original = construir_caracteristicas(liga_pequena)
    cambiada = construir_caracteristicas(alterada)

    # Todos los partidos hasta la fecha de corte (incluida) no cambian:
    # sus características solo dependen de partidos de fechas anteriores.
    hasta = liga_pequena["fecha"] <= corte
    pd.testing.assert_frame_equal(original[hasta], cambiada[hasta])
    # Y los posteriores sí cambian (el historial se usa de verdad).
    assert not original[~hasta].equals(cambiada[~hasta])


def test_primer_partido_usa_valores_a_priori(liga_pequena):
    c = construir_caracteristicas(liga_pequena).iloc[0]
    assert c["l_pj"] == 0 and c["v_pj"] == 0
    assert c["l_elo"] == 1500 and c["v_elo"] == 1500


def test_partidos_del_mismo_dia_no_se_ven_entre_si():
    df = pd.DataFrame(
        {
            "fecha": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-08"]),
            "local": ["A", "C", "A"],
            "visitante": ["B", "D", "C"],
            "goles_local": [5, 0, 1],
            "goles_visitante": [0, 4, 1],
        }
    )
    c = construir_caracteristicas(df)
    assert c.loc[1, "l_pj"] == 0 and c.loc[1, "l_elo"] == 1500
    # El tercer partido ya conoce los dos del día 1.
    assert c.loc[2, "l_pj"] == 1 and c.loc[2, "v_pj"] == 1
    assert c.loc[2, "l_elo"] > 1500 > c.loc[2, "v_elo"]


def test_perfil_previo_respeta_la_fecha(liga_pequena):
    fecha = liga_pequena["fecha"].iloc[100]
    p = perfil_previo(liga_pequena, "Sintético 01 FC", "Sintético 02 FC", fecha=fecha)
    anteriores = liga_pequena[liga_pequena["fecha"] < fecha]
    jugados = ((anteriores["local"] == "Sintético 01 FC") | (anteriores["visitante"] == "Sintético 01 FC")).sum()
    assert p["l_pj"] == jugados


def test_elo_suma_cero(liga_pequena):
    from predictor.caracteristicas import tabla_elo

    elo = tabla_elo(liga_pequena)
    assert elo.mean() == pytest.approx(1500.0)

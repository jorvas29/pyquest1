"""Comprueba el modelo con el ejemplo Coritiba vs Palmeiras."""

import unittest

from futbol.prediccion import (
    estimar_xg,
    margen,
    matriz_marcadores,
    predecir,
    probabilidades_sin_margen,
)

XG_LOCAL = 0.625
XG_VISITANTE = 1.423

CUOTAS = {
    "mas_0.5": 1.05,
    "mas_1.5": 1.3,
    "mas_2.5": 1.95,
    "menos_0.5": 9,
    "menos_2.5": 1.833,
}


class TestPrediccion(unittest.TestCase):

    def setUp(self):
        self.p = predecir(XG_LOCAL, XG_VISITANTE, cuotas=CUOTAS)

    def test_1x2(self):
        x12 = self.p["1x2"]
        self.assertAlmostEqual(x12["local"]["probabilidad"], 0.163, places=3)
        self.assertAlmostEqual(x12["empate"]["probabilidad"], 0.272, places=3)
        self.assertAlmostEqual(x12["visitante"]["probabilidad"], 0.565, places=3)

    def test_1x2_suma_uno(self):
        total = sum(m["probabilidad"] for m in self.p["1x2"].values())
        self.assertAlmostEqual(total, 1, places=6)

    def test_ambos_marcan(self):
        self.assertAlmostEqual(
            self.p["ambos_marcan"]["si"]["probabilidad"], 0.353, places=3
        )

    def test_mas_menos(self):
        self.assertAlmostEqual(self.p["mas"][0.5]["probabilidad"], 0.871, places=3)
        self.assertAlmostEqual(self.p["mas"][2.5]["probabilidad"], 0.336, places=3)
        self.assertAlmostEqual(self.p["menos"][4.5]["probabilidad"], 0.943, places=3)

    def test_impacto_igual_que_la_tabla(self):
        # Valores de la tabla original
        self.assertAlmostEqual(self.p["mas"][0.5]["impacto"], -0.0855, places=3)
        self.assertAlmostEqual(self.p["mas"][1.5]["impacto"], -0.2111, places=3)
        self.assertAlmostEqual(self.p["menos"][0.5]["impacto"], 0.1610, places=3)
        self.assertAlmostEqual(self.p["menos"][2.5]["impacto"], 0.2166, places=3)

    def test_sin_cuota_no_hay_impacto(self):
        self.assertIsNone(self.p["1x2"]["local"]["impacto"])


    def test_doble_oportunidad(self):
        dc = self.p["doble_oportunidad"]
        self.assertAlmostEqual(dc["1x"]["probabilidad"], 0.163 + 0.272, places=2)
        self.assertAlmostEqual(dc["x2"]["probabilidad"], 0.272 + 0.565, places=2)


class TestExcelDenHaagEmmen(unittest.TestCase):
    """Datos del Excel MODELO POISSON 25-26 (Den Haag vs Emmen)."""

    def setUp(self):
        # Goles por partido: Den Haag en casa y Emmen fuera.
        # Medias de la liga: 1,8 goles el local y 1,5048 el visitante.
        self.xg = estimar_xg(
            gf_casa_local=2.9,
            gc_casa_local=0.9,
            gf_fuera_visitante=13 / 11,
            gc_fuera_visitante=23 / 11,
            media_goles_local=1.8,
            media_goles_visitante=158 / 105,
        )

    def test_mismo_lambda_que_el_excel(self):
        self.assertAlmostEqual(self.xg[0], 3.3687, places=3)
        self.assertAlmostEqual(self.xg[1], 0.7068, places=3)

    def test_la_matriz_suma_uno(self):
        m = matriz_marcadores(*self.xg)
        self.assertAlmostEqual(sum(map(sum, m)), 1, places=9)

    def test_menos_de_05_correcto(self):
        # El Excel daba 7,3 % por cortar la matriz en 6 goles.
        p = predecir(*self.xg)
        self.assertAlmostEqual(p["menos"][0.5]["probabilidad"], 0.0170, places=3)
        self.assertAlmostEqual(p["1x2"]["local"]["probabilidad"], 0.866, places=2)


class TestDixonColes(unittest.TestCase):

    def test_rho_cero_es_poisson(self):
        a = predecir(0.625, 1.423)
        b = predecir(0.625, 1.423, rho=0.0)
        self.assertEqual(a["1x2"]["empate"], b["1x2"]["empate"])

    def test_rho_negativo_sube_empate(self):
        base = predecir(0.625, 1.423)["1x2"]["empate"]["probabilidad"]
        dc = predecir(0.625, 1.423, rho=-0.1)["1x2"]["empate"]["probabilidad"]
        self.assertGreater(dc, base)


class TestMargen(unittest.TestCase):

    def test_margen_1x2(self):
        self.assertAlmostEqual(margen([1.37, 2.75, 3.7]), 0.364, places=3)

    def test_sin_margen_suma_uno(self):
        self.assertAlmostEqual(sum(probabilidades_sin_margen([1.9, 1.9])), 1)


if __name__ == "__main__":
    unittest.main()

"""Comprueba el modelo con el ejemplo Coritiba vs Palmeiras."""

import unittest

from futbol.prediccion import predecir

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


if __name__ == "__main__":
    unittest.main()

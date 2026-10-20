# pyquest1

## Predicción de partidos (modelo de Poisson)

```
python -m futbol "Coritiba" 0.625 "Palmeiras" 1.423 \
    --cuota mas_2.5=1.95 --cuota menos_2.5=1.833 \
    --oportunidades 3 --rho -0.1
```

- `futbol/prediccion.py`: 1X2, doble oportunidad, ambos marcan,
  más/menos goles, grandes ocasiones y el impacto
  (`probabilidad · cuota − 1`).
- `estimar_xg(...)`: λ desde la tabla de liga (GF casa × GC fuera del
  rival ÷ media de la liga), como el Excel MODELO POISSON 25-26.
- `rho`: corrección Dixon-Coles opcional para 0-0, 1-0, 0-1 y 1-1.
- `margen(...)` y `probabilidades_sin_margen(...)`: lo que piensa la casa.
- Tests: `python -m unittest tests.test_prediccion`

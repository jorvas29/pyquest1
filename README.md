# pyquest1

## Predicción de partidos (modelo de Poisson)

```
python -m futbol "Coritiba" 0.625 "Palmeiras" 1.423 \
    --cuota mas_2.5=1.95 --cuota menos_2.5=1.833 \
    --oportunidades 3
```

- `futbol/prediccion.py`: 1X2, ambos marcan, más/menos goles,
  grandes ocasiones y el impacto (`probabilidad · cuota − 1`).
- Tests: `python -m unittest tests.test_prediccion`

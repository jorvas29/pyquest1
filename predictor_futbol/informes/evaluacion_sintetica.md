# Evaluación temporal del predictor

> AVISO: ningún modelo garantiza resultados en las apuestas. Las probabilidades son estimaciones con incertidumbre basadas en datos históricos y en supuestos simplificados. Úsalas solo como apoyo a la decisión y apuesta con responsabilidad.

- **Fuente de datos:** SINTÉTICA (datos inventados para demostración, semilla=42)
- **Partidos usados:** 1140 (del 2021-08-14 al 2024-03-17)
- **Partidos evaluados:** 990 (entrenamiento inicial: 150, reajuste cada 7 días)
- **Modelo de goles:** Dixon-Coles

## Métricas por mercado

Menor es mejor. `referencia` = frecuencias históricas del entrenamiento; `mercado` = probabilidades implícitas sin margen; `logloss_modelo_mismos` = log-loss del modelo en los mismos partidos que tienen cuotas.

| mercado | n | logloss_modelo | brier_modelo | logloss_referencia | brier_referencia | n_mercado | logloss_mercado | brier_mercado | logloss_modelo_mismos |
|---|---|---|---|---|---|---|---|---|---|
| 1X2 | 990 | 1.0525 | 0.6350 | 1.0706 | 0.6470 | 990 | 1.0480 | 0.6317 | 1.0525 |
| Ambos marcan | 990 | 0.6665 | 0.2368 | 0.6770 | 0.2419 | 990 | 0.6558 | 0.2319 | 0.6665 |
| Córners (más de la línea) | 990 | 0.6778 | 0.2423 | 0.6795 | 0.2432 | 990 | 0.6655 | 0.2364 | 0.6778 |

## Calibración

Si el modelo está bien calibrado, `frecuencia_real` se parece a `p_media` en cada tramo.

### 1X2: gana el local

| tramo | n | p_media | frecuencia_real |
|---|---|---|---|
| 0.0-0.1 | 1 | 0.091 | 0.000 |
| 0.1-0.2 | 11 | 0.170 | 0.364 |
| 0.2-0.3 | 113 | 0.265 | 0.310 |
| 0.3-0.4 | 218 | 0.353 | 0.381 |
| 0.4-0.5 | 277 | 0.452 | 0.426 |
| 0.5-0.6 | 219 | 0.547 | 0.530 |
| 0.6-0.7 | 113 | 0.643 | 0.611 |
| 0.7-0.8 | 36 | 0.731 | 0.611 |
| 0.8-0.9 | 2 | 0.815 | 0.000 |

### 1X2: empate

| tramo | n | p_media | frecuencia_real |
|---|---|---|---|
| 0.1-0.2 | 95 | 0.176 | 0.305 |
| 0.2-0.3 | 736 | 0.252 | 0.265 |
| 0.3-0.4 | 158 | 0.322 | 0.241 |
| 0.4-0.5 | 1 | 0.427 | 0.000 |

### Ambos marcan: sí

| tramo | n | p_media | frecuencia_real |
|---|---|---|---|
| 0.2-0.3 | 3 | 0.267 | 0.333 |
| 0.3-0.4 | 59 | 0.366 | 0.508 |
| 0.4-0.5 | 210 | 0.455 | 0.467 |
| 0.5-0.6 | 338 | 0.551 | 0.615 |
| 0.6-0.7 | 289 | 0.645 | 0.654 |
| 0.7-0.8 | 84 | 0.734 | 0.679 |
| 0.8-0.9 | 7 | 0.827 | 0.714 |

### Córners: más de la línea

| tramo | n | p_media | frecuencia_real |
|---|---|---|---|
| 0.3-0.4 | 2 | 0.382 | 0.500 |
| 0.4-0.5 | 112 | 0.468 | 0.473 |
| 0.5-0.6 | 541 | 0.555 | 0.595 |
| 0.6-0.7 | 331 | 0.632 | 0.613 |
| 0.7-0.8 | 4 | 0.704 | 0.500 |

## Simulación retrospectiva de apuestas con valor (EV > 5%, 1 unidad)

Orientativa: un ROI histórico positivo NO garantiza beneficios futuros.

| mercado | seleccion | apuestas | aciertos | beneficio | roi |
|---|---|---|---|---|---|
| 1X2 | 1 | 356 | 157 | 11.860 | 0.033 |
| 1X2 | X | 136 | 29 | -10.150 | -0.075 |
| 1X2 | 2 | 287 | 81 | 1.660 | 0.006 |
| Ambos marcan | Sí | 276 | 143 | 8.010 | 0.029 |
| Ambos marcan | No | 230 | 77 | -36.800 | -0.160 |
| Córners | Más | 255 | 127 | 8.430 | 0.033 |
| Córners | Menos | 389 | 131 | 0.570 | 0.001 |

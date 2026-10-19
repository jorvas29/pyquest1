# pyquest1

## futquant — motor cuantitativo de pronósticos de fútbol

Calcula la **FICHA DEL PARTIDO** (11 secciones fijas) combinando todos los
modelos en un ensamble. Las cuotas son una variable más, no el objetivo.

### Modelos

| Familia | Modelo | Archivo | Qué aporta |
|---|---|---|---|
| Marcador | Poisson | `markets.py` | Matriz de marcadores base |
| Marcador | Dixon-Coles (ρ) | `dixon_coles.py` | Corrige 0-0 / 1-1; MLE de ataque, defensa, local y ρ con decaimiento temporal |
| Marcador | Monte Carlo + cadena de Markov | `montecarlo.py` | 90 minutos, estado por marcador y rojas, 10 000 simulaciones |
| Ratings | Elo (con margen de victoria) | `ratings.py` | Fuerza relativa, 3-0 mueve más que 1-0 |
| Ratings | pi-ratings (Constantinou & Fenton) | `ratings.py` | Rating separado de local y visitante |
| Ratings | Glicko-2 | `ratings.py` | Fuerza + incertidumbre (RD) |
| 1X2 directo | Regresión logística multinomial | `ml.py` | Variables → P(1, X, 2) sin pasar por goles |
| 1X2 directo | Random Forest, Gradient Boosting, XGBoost, LightGBM | `ml.py` | Relaciones no lineales entre variables |
| Mercado | Cuotas sin margen (Shin, potencia, proporcional) | `odds.py` | Qué espera el mercado |

### Flujo (FASES 1-4)

1. **Variables** → JSON del partido (`examples/partido_ejemplo.json`).
2. **Índice de ventaja** (`index.py`) → 8 bloques de -2 a +2 con sus pesos.
3. **λ** = promedio goles / xG, ajuste máximo ±15 % por índice y bajas.
4. **Matrices** Poisson + Dixon-Coles + Monte Carlo (+ Dixon-Coles MLE si hay historial).
5. **Ensamble 1X2** por familias (marcador 45 %, ratings 15 %, ML 20 %, mercado 20 %).
   - Los modelos ML que en validación temporal no superan la frecuencia base quedan fuera.
   - La matriz final se reescala para que su 1X2 coincida con el ensamble.
6. **Mercados**: 1X2, doble oportunidad, empate no válido, hándicaps, totales,
   ambos marcan, goles por equipo, marcadores, 1er tiempo, córners y tarjetas.
7. **Picks** (`picks.py`): descarta > 88 %, pick principal 60-88 % con ≥ 4
   variables a favor y sin variable grave en contra, confianza y cuota justa.

### Uso

```bash
pip install -r requirements.txt

# Ficha solo con el JSON del partido
python -m futquant ficha examples/partido_ejemplo.json

# Con historial (CSV de football-data.co.uk o formato propio) para
# Dixon-Coles MLE, Elo, pi-ratings, Glicko-2 y machine learning
python -m futquant ficha partido.json --historial E0.csv -o ficha.md

# Liga FICTICIA para probar el motor
python -m futquant demo-historial liga_demo.csv

python -m pytest -q
```

El ejemplo y la liga de demo son **ficticios**: en un partido real cada dato
del JSON debe venir de una fuente verificada (y listarse en `fuentes`).

> Pronóstico estadístico, no garantía. Apuesta solo lo que puedas perder.

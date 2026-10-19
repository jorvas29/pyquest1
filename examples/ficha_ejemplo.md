# FICHA DEL PARTIDO — Equipo A vs Equipo B

## 1. Ficha

| Campo | Dato |
|---|---|
| Partido | Equipo A vs Equipo B |
| Competición | Liga de ejemplo (ficticia) |
| Fecha y hora | 2026-10-25 13:00 (hora Colombia, UTC-5) |
| Estadio | Estadio Ejemplo (altitud 300 m) |
| Árbitro | Árbitro Ejemplo (amarillas 4.60, rojas 0.18, penales 0.30 por partido) |
| Clima | No verificado |

## 2. Resumen

- Equipo A llega 3.º y necesita ganar para seguir en zona de Champions (dato ficticio).
- Equipo B suma 4 partidos sin perder fuera de casa, pero pierde a su central titular.
- Ambos producen más de 1.4 xG por partido en su condición.

## 3. Tabla de variables

| Variable | Equipo A | Equipo B | Favorece a |
|---|---|---|---|
| Posición / puntos | 3.º · 18 pts · DG +7 | 8.º · 13 pts · DG +1 | — |
| Forma últimos 5 | GGEPG | EGEGP | — |
| PPG últimos 10 | 1.90 | 1.50 | Local |
| PPG en su condición (local / visita) | 2.20 | 1.60 | Local |
| Goles a favor en su condición | 1.90 | 1.30 | Local |
| Goles en contra en su condición | 0.90 | 1.40 | Local |
| xG en su condición | 1.75 | 1.35 | Local |
| xGA en su condición | 1.05 | 1.50 | Local |
| Córners a favor / en contra | 6.10 / 4.00 | 4.40 / 5.60 | — |
| Amarillas propias / del rival | 1.80 / 2.20 | 2.30 / 1.90 | — |
| Días de descanso | 6 | 3 | Local |
| Bajas confirmadas (peso) | Mediocampista X (creador, RUMOR, peso 1) | Central Y (central titular, CONFIRMADO, peso 2) | — |

## 4. Índice de ventaja

| Bloque | Peso | Puntuación | Ponderado | Explicación |
|---|---|---|---|---|
| Forma | 20% | +1 | +0.20 | PPG últimos 10: 1.90 vs 1.50 (dif +0.40) (auto) |
| Local / visitante | 15% | +1 | +0.15 | PPG local de local 2.20 vs visitante de visita 1.60 (auto) |
| Ataque / defensa (xG) | 20% | +2 | +0.40 | xG neto en su condición: +0.70 vs -0.15 (auto) |
| Bajas | 15% | +1 | +0.15 | Peso de bajas confirmadas: local 0 / visitante 2 (auto) |
| Motivación / contexto | 10% | +1 | +0.10 | Local pelea puestos de Champions; visitante en mitad de tabla (analista) |
| Descanso / calendario | 10% | +2 | +0.20 | Días de descanso: 6 vs 3 (auto) |
| Enfrentamientos directos | 5% | +1 | +0.05 | H2H: 3G-2E-1P (auto) |
| Condiciones / árbitro | 5% | +0 | +0.00 | Sin altitud ni clima extremo; árbitro neutro (analista) |

**Índice = +1.25** (escala -2 a +2) → inclina hacia: **Equipo A**

## 5. Probabilidades calculadas

### Goles esperados

- Goles: λ_local = (1.90 + 1.40) / 2 = 1.650
- Goles: λ_visita = (1.30 + 0.90) / 2 = 1.100
- xG: λ_local = (1.75 + 1.50) / 2 = 1.625
- xG: λ_visita = (1.35 + 1.05) / 2 = 1.200
- Promedio goles/xG: λ_local = 1.637 · λ_visita = 1.150
- Ajuste por índice (+1.25) y bajas: local +9.4% → 1.791 · visita -12.4% → 1.008
- ρ Dixon-Coles = +0.044 (estimado por MLE del historial)
- Dixon-Coles MLE por equipo (historial de 264 partidos): λ_local 1.41 · λ_visita 1.47

### 1X2 por modelo

| Modelo | 1 | X | 2 | Peso en ensamble |
|---|---|---|---|---|
| Dixon-Coles MLE (historial) | 36.6% | 23.9% | 39.5% | 14.1% |
| Elo | 36.3% | 27.0% | 36.7% | 6.2% |
| pi-ratings | 32.8% | 24.3% | 42.9% | 6.2% |
| Glicko-2 | 36.2% | 27.0% | 36.8% | 6.2% |
| Poisson | 55.8% | 23.2% | 21.1% | 14.1% |
| Dixon-Coles | 56.2% | 22.2% | 21.6% | 14.1% |
| Monte Carlo / Markov | 56.2% | 24.2% | 19.6% | 14.1% |
| Mercado (sin margen, Shin) | 51.7% | 25.8% | 22.5% | 25.0% |
| **ENSAMBLE** | **48.3%** | **24.5%** | **27.2%** | 100% |

Validación temporal (último 20 % del historial; menor = mejor):

| Modelo | Log-loss | RPS | ¿Entra al ensamble? |
|---|---|---|---|
| Frecuencia base (referencia) | 1.086 | 0.2101 | referencia |
| Regresión logística | 1.211 | 0.2185 | No (peor que la base) |
| Random Forest | 1.107 | 0.2058 | No (peor que la base) |
| Gradient Boosting | 1.440 | 0.2591 | No (peor que la base) |

Ratings: Elo: Equipo A 1480.00 / Equipo B 1546.95 · pi-ratings (dif. goles esperada): Equipo A -0.21 · Glicko-2: Equipo A 1465.02 / Equipo B 1533.32

### Resultado y hándicap

| Mercado | Probabilidad | Cuota justa |
|---|---|---|
| 1X (Equipo A o empate) | 72.8% | 1.37 |
| X2 (Equipo B o empate) | 51.7% | 1.93 |
| 12 (no empate) | 75.5% | 1.32 |
| Equipo A empate no válido | 64.0% | 1.56 |
| Equipo B empate no válido | 36.0% | 2.77 |
| Equipo A -1 asiático (sin devolución) | 33.5% | 2.99 |
| Equipo B +1 asiático (sin devolución) | 66.5% | 1.50 |
| Equipo A -1.5 | 26.0% | 3.85 |
| Equipo B +1.5 | 74.0% | 1.35 |
| Hándicap europeo Equipo A -1: 1 / X / 2 | 26.0% / 22.3% / 51.7% | — |

### Goles

| Línea | Más | Menos |
|---|---|---|
| 0.5 | 94.0% | 6.0% |
| 1.5 | 76.9% | 23.1% |
| 2.5 | 53.2% | 46.8% |
| 3.5 | 31.2% | 68.8% |
| 4.5 | 15.3% | 84.7% |

Ambos marcan: **Sí 55.1%** · No 44.9%  
Valla invicta: Equipo A 31.6% · Equipo B 19.4%

| Goles por equipo | Más de 0.5 | Más de 1.5 | Más de 2.5 |
|---|---|---|---|
| Equipo A | 80.6% | 48.8% | 23.0% |
| Equipo B | 68.4% | 32.1% | 11.3% |

### Marcadores más probables

- 1-1: 11.6%
- 1-0: 9.9%
- 2-1: 9.1%

Monte Carlo (10,000 simulaciones): primer gol Equipo A 61.0% · Equipo B 33.0% · sin goles 6.1%  
Reparto de goles por tramo: 1-15' 14% · 16-30' 15% · 31-45' 16% · 46-60' 17% · 61-75' 19% · 76-90' 19%

### Primer tiempo

- λ 1T = 0.81 + 0.45 (λ × 0.45)
- Más de 0.5: 71.6% · Más de 1.5: 36.0%
- Resultado al descanso: 1 41.3% · X 40.3% · 2 18.4%
- Descanso/final más probables: 1/1 32.5% · X/X 14.5% · X/1 14.4%

### Córners

- Local: (6.1 + 5.6) / 2 = 5.85
- Visita: (4.4 + 4.0) / 2 = 4.20
- Total esperado = 10.05

| Línea | Más | Menos |
|---|---|---|
| 7.5 | 76.5% | 23.5% |
| 8.5 | 65.7% | 34.3% |
| 9.5 | 53.9% | 46.1% |
| 10.5 | 42.3% | 57.7% |
| 11.5 | 31.6% | 68.4% |

### Tarjetas

- Local: (1.8 + 1.9) / 2 = 1.85
- Visita: (2.3 + 2.2) / 2 = 2.25
- Total por equipos = 4.10
- Mezcla 50/50 con árbitro (4.60): 4.35

| Línea | Más | Menos |
|---|---|---|
| 2.5 | 78.2% | 21.8% |
| 3.5 | 61.0% | 39.0% |
| 4.5 | 43.3% | 56.7% |
| 5.5 | 28.1% | 71.9% |
| 6.5 | 16.8% | 83.2% |

### Contraste con el mercado

| Mercado | Modelo | Mercado (sin margen) | Diferencia |
|---|---|---|---|
| 1X2 Local | 48.3% | 51.7% | -3.4 pts |
| 1X2 Empate | 24.5% | 25.8% | -1.3 pts |
| 1X2 Visitante | 27.2% | 22.5% | +4.7 pts |
| Más de 2.5 | 53.2% | 52.4% | +0.9 pts |
| Ambos marcan Sí | 55.1% | 54.2% | +0.9 pts |

## 6. PICK PRINCIPAL

**Equipo A +1 (asiático)** · Probabilidad **87.3%** · Confianza **ALTA** · Cuota justa **1.15**

_Gana 73%, devuelve 17%_

Razones:
1. Forma (+1)
1. Local / visitante (+1)
1. Ataque / defensa (xG) (+2)

**Mayor riesgo:** Modelos que lo favorecen: 4/8

## 7. PICKS POR MERCADO

| Mercado | Pick | Probabilidad | Confianza | Cuota justa | Variables a favor / en contra |
|---|---|---|---|---|---|
| Resultado / doble oportunidad | No hay empate (12) | 75.5% | MEDIA | 1.32 | 2 / 0 |
| Hándicap | Equipo A +1 (asiático) | 87.3% | ALTA | 1.15 | 7 / 1 |
| Goles | Más de 1.5 goles (tu cuota 1.28: por debajo) | 76.9% | ALTA | 1.30 | 5 / 0 |
| Ambos marcan | Ambos marcan: Sí | 55.1% | BAJA | 1.82 | 6 / 1 |
| Goles por equipo | Equipo A más de 0.5 | 80.6% | ALTA | 1.24 | 4 / 0 |
| Córners | Más de 7.5 córners | 76.5% | MEDIA | 1.31 | 1 / 0 |
| Tarjetas | Menos de 6.5 tarjetas | 83.2% | MEDIA | 1.20 | 2 / 0 |
| 1er tiempo | 1er tiempo: más de 0.5 | 71.6% | MEDIA | 1.40 | 3 / 1 |

## 8. Picks a evitar

- Equipo B o empate (X2) (51.7%): variable grave en contra (bajas o índice hacia el rival).
- Equipo B +1 (asiático) (66.5%): variable grave en contra (bajas o índice hacia el rival).
- Equipo B +1.5 (74.0%): variable grave en contra (bajas o índice hacia el rival).
- Menos de 10.5 córners (57.7%): la probabilidad es alta pero 1 variables van en contra.
- Menos de 4.5 tarjetas (56.7%): la probabilidad es alta pero 2 variables van en contra.
- Combinadas: la probabilidad conjunta se multiplica (dos picks de 70 % = 49 %).

## 9. Qué revisar antes de apostar

- Alineaciones oficiales 1 h antes del inicio (sobre todo las bajas marcadas como RUMOR o duda).
- Bajas de último minuto y calentamiento.
- Cambios de cuota fuertes (si el mercado se mueve contra el pick, revisar la noticia que lo causa).

## 10. Fuentes

- [Fuente de ejemplo (reemplazar por FBref/Sofascore/etc.)](https://example.com) — consultado 2026-10-19
- Nota del modelo: Regresión logística excluido del ensamble: log-loss de validación 1.211 peor que la frecuencia base 1.086.
- Nota del modelo: Random Forest excluido del ensamble: log-loss de validación 1.107 peor que la frecuencia base 1.086.
- Nota del modelo: Gradient Boosting excluido del ensamble: log-loss de validación 1.440 peor que la frecuencia base 1.086.

## 11. Cierre

**Pronóstico estadístico, no garantía. Apuesta solo lo que puedas perder.**

# Predictor de fútbol: BTTS, córners y 1X2

Herramienta de línea de comandos en Python que estima probabilidades para tres
mercados de un partido de fútbol y las compara con las cuotas del mercado:

- **1X2** (gana el local, empate, gana el visitante)
- **Ambos marcan** (BTTS sí / no)
- **Córners** (más / menos de una línea, y qué equipo saca más)

> **AVISO:** ningún modelo garantiza resultados en las apuestas. Las
> probabilidades son estimaciones con incertidumbre basadas en datos históricos
> y en supuestos simplificados. Úsalas solo como apoyo a la decisión y apuesta
> con responsabilidad.

> **Datos:** el repositorio NO contiene datos reales. `datos/liga_sintetica.csv`
> es una liga **inventada** (equipos "Sintético 01 FC", ..., columna
> `origen = SINTETICO`) que sirve para demostrar el flujo completo. Para usar
> datos reales, pasa tu propio CSV (`--csv`) o configura una API (`--api-config`).

---

## Estructura de carpetas

```
predictor_futbol/
│
├── README.md
├── requirements.txt          ← numpy, pandas, scipy, pytest
├── pyproject.toml            ← configuración de pytest y del paquete
│
├── predictor/                ← paquete principal
│   ├── __init__.py           ← ADVERTENCIA y versión
│   ├── __main__.py           ← permite `python -m predictor`
│   ├── cli.py                ← comandos: generar-sintetico, equipos, predecir, evaluar
│   │
│   ├── datos/                ← OBTENCIÓN DE DATOS
│   │   ├── esquema.py        ← columnas comunes + limpieza
│   │   ├── fuentes.py        ← FuenteCSV, FuenteAPI, FuenteSintetica (intercambiables)
│   │   └── sintetico.py      ← generador de la liga inventada
│   │
│   ├── caracteristicas.py    ← INGENIERÍA DE CARACTERÍSTICAS (forma, goles, córners, localía, Elo)
│   │
│   ├── modelos/              ← MODELOS
│   │   ├── dixon_coles.py    ← goles: Dixon-Coles / Poisson independiente
│   │   ├── corners.py        ← córners: regresión de Poisson (+ binomial negativa)
│   │   └── mercados.py       ← 1X2, BTTS, más/menos a partir de las distribuciones
│   │
│   ├── prediccion.py         ← une los modelos para un partido concreto
│   ├── cuotas.py             ← cuota justa, margen, probabilidades sin margen, EV
│   ├── evaluacion.py         ← EVALUACIÓN: walk-forward, log-loss, Brier, calibración
│   └── informe.py            ← informes de consola y Markdown (modelo vs mercado)
│
├── config/
│   └── api_ejemplo.json      ← plantilla para una API (football-data.org v4)
│
├── datos/
│   └── liga_sintetica.csv    ← liga SINTÉTICA de demostración (3 temporadas, 20 equipos)
│
├── informes/
│   └── evaluacion_sintetica.md  ← ejemplo de informe de evaluación
│
└── tests/                    ← 56 pruebas con pytest
    ├── conftest.py
    ├── test_mercados.py      ← 1X2 suma 1, BTTS coherente, córners, ...
    ├── test_cuotas.py
    ├── test_dixon_coles.py   ← gradiente, recuperación de parámetros
    ├── test_caracteristicas.py  ← sin fuga de datos del futuro
    ├── test_evaluacion.py
    ├── test_fuentes.py
    └── test_cli.py
```

---

## Instalación

Requiere Python 3.10 o superior.

```bash
cd predictor_futbol

# 1) Entorno virtual (recomendado)
python3 -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate

# 2) Dependencias
pip install -r requirements.txt
```

---

## Ejecución

Todos los comandos se lanzan desde la carpeta `predictor_futbol/`.

### 1. Ver los equipos disponibles

```bash
python -m predictor equipos
```

Sin `--csv` ni `--api-config`, se usa la liga **sintética**.

### 2. Predecir un partido

```bash
python -m predictor predecir "Sintético 01 FC" "Sintético 07 FC"
```

Con cuotas de mercado, para buscar valor esperado:

```bash
python -m predictor predecir "Sintético 01 FC" "Sintético 07 FC" \
    --cuotas 1=2.30 X=3.30 2=3.20 \
             btts_si=1.75 btts_no=2.05 \
             corners_mas=1.90 corners_menos=1.90 \
    --linea-corners 9.5
```

Claves de `--cuotas`:

| Clave           | Selección                    |
|-----------------|------------------------------|
| `1`             | gana el local                |
| `X`             | empate                       |
| `2`             | gana el visitante            |
| `btts_si`       | ambos marcan: sí             |
| `btts_no`       | ambos marcan: no             |
| `corners_mas`   | más córners que la línea     |
| `corners_menos` | menos córners que la línea   |

Otras opciones útiles:

| Opción               | Qué hace                                                        |
|----------------------|-----------------------------------------------------------------|
| `--modelo poisson`   | goles con Poisson independiente en lugar de Dixon-Coles         |
| `--xi 0.0019`        | olvido temporal por día (0 = todos los partidos pesan igual)    |
| `--fecha 2023-12-01` | predice "como si fuera" esa fecha (solo usa partidos anteriores) |
| `--umbral-valor 0.05`| EV mínimo para marcar una selección como VALOR                  |

La salida (en vertical) incluye:

1. Perfil previo de cada equipo (forma, goles, córners, Elo).
2. Goles esperados y marcadores más probables.
3. Probabilidad y **cuota justa** (= 1 / p) de cada selección.
4. Detalle de córners.
5. Si hay cuotas: **modelo vs mercado** con ventaja, cuota justa, cuota de
   mercado y **valor esperado** (EV = p · cuota − 1).

### 3. Evaluar el modelo (validación temporal)

```bash
python -m predictor evaluar --salida-md informes/evaluacion.md
```

Opciones: `--min-entrenamiento 150` (partidos antes de la primera predicción),
`--paso-dias 7` (cada cuántos días se reajusta), `--salida-csv` (predicciones
partido a partido).

### 4. Usar tus propios datos

**CSV** (esquema común, o los CSV de football-data.co.uk directamente):

```bash
python -m predictor predecir "Arsenal" "Chelsea" --csv ruta/E0.csv
```

Columnas del esquema común:

| Columna               | Obligatoria | Descripción                         |
|-----------------------|-------------|-------------------------------------|
| `fecha`               | sí          | fecha del partido                   |
| `local`, `visitante`  | sí          | nombres de los equipos              |
| `goles_local`, `goles_visitante` | sí | marcador final                 |
| `corners_local`, `corners_visitante` | no | activa el mercado de córners  |
| `cuota_1`, `cuota_x`, `cuota_2` | no  | cuotas 1X2 (para evaluar vs mercado) |
| `cuota_btts_si`, `cuota_btts_no` | no | cuotas de ambos marcan          |
| `linea_corners`, `cuota_corners_mas`, `cuota_corners_menos` | no | cuotas de córners |

Si el CSV trae `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `HC`, `AC`, `B365H`,
`B365D`, `B365A` (formato football-data.co.uk), se traduce automáticamente y las
fechas se leen como dd/mm/aaaa.

**API**: copia `config/api_ejemplo.json`, ajusta `url` y `mapeo`
(`campo.anidado` → columna común), define la variable de entorno con tu clave
y ejecuta:

```bash
export FOOTBALL_DATA_TOKEN="tu_clave"
python -m predictor predecir "Real Madrid CF" "FC Barcelona" --api-config config/api_ejemplo.json
```

La plantilla sigue la estructura de la API v4 de football-data.org, pero no se
ha probado contra el servicio real. Esa API no ofrece córners, así que con ella
el mercado de córners queda desactivado.

**Desde Python**, cualquier clase que herede de `FuenteDatos` e implemente
`cargar()` sirve:

```python
from predictor.datos import FuenteCSV
from predictor.prediccion import ModeloConjunto

partidos = FuenteCSV("mis_partidos.csv").cargar()
modelo = ModeloConjunto().ajustar(partidos)
pred = modelo.predecir("Equipo A", "Equipo B", linea_corners=9.5)
print(pred.p_1x2, pred.p_btts, pred.corners)
```

### 5. Pruebas

```bash
python -m pytest
```

---

## Decisiones de modelado

### Goles, 1X2 y ambos marcan: Dixon-Coles

1. Cada equipo tiene un parámetro de **ataque** y otro de **defensa**, más una
   **ventaja de local** común:

   ```
   goles_local     ~ Poisson(λ),   log λ = c + localía + ataque[local] + defensa[visitante]
   goles_visitante ~ Poisson(μ),   log μ = c + ataque[visitante] + defensa[local]
   ```

2. La corrección **tau** de Dixon-Coles ajusta los marcadores 0-0, 1-0, 0-1 y
   1-1 con un parámetro ρ. Con ρ = 0 es Poisson independiente.

3. **Pesos temporales** `exp(−ξ · días)`: lo reciente pesa más
   (ξ = 0.0019 ≈ vida media de un año).

4. **Regularización L2** suave en ataque y defensa: hace el modelo identificable
   y evita valores extremos en equipos con pocos partidos.

5. Ajuste por máxima verosimilitud (L-BFGS-B, gradiente analítico verificado en
   las pruebas).

6. De ahí sale la **matriz de marcadores** P(i, j), y todos los mercados de
   goles se derivan de ella, de modo que son coherentes entre sí:

   - 1X2: suma bajo la diagonal / diagonal / sobre la diagonal.
   - BTTS: suma de la matriz sin la fila 0 ni la columna 0.
   - Más de 2.5: suma de las celdas con i + j > 2.5.

### Córners: regresión de Poisson con sobredispersión

1. Una fila por equipo y partido. Variables (calculadas antes del partido):

   - log(córners a favor promedio del equipo)
   - log(córners en contra promedio del rival)
   - es_local
   - diferencia de Elo / 400

2. Modelo log-lineal ajustado por máxima verosimilitud con L2 suave.

3. Los córners varían más de lo que predice una Poisson. Se estima la
   dispersión φ y, si φ > 1, se usa una **binomial negativa** con varianza
   λ · φ. El total del partido es la convolución de ambos equipos.

4. Con líneas enteras (p. ej. 10) se calcula también la probabilidad de
   **apuesta nula**, y la cuota justa y el EV la tienen en cuenta.

### Características

- **Forma**: puntos por partido en los últimos 5.
- **Goles y córners** a favor y en contra: últimos 10 partidos.
- **Localía**: goles a favor y en contra solo en casa (local) o solo fuera
  (visitante).
- **Ventaja de calidad estimada**: rating **Elo** con ventaja de local
  (+60 puntos) y multiplicador por diferencia de goles.
- Las medias se **suavizan** hacia la media de la liga para que pocos
  partidos no den valores extremos.
- **Sin fuga del futuro**: se recorren las fechas en orden. En cada fecha se
  calculan primero las características y solo después se añaden los
  resultados. Una prueba lo comprueba modificando resultados futuros.

### Evaluación

- **Walk-forward**: se reentrena cada `paso_dias` solo con partidos de fecha
  estrictamente anterior a la ventana de prueba.
- **Log-loss** y **Brier** por mercado, comparando tres fuentes:
  - el modelo,
  - una referencia ingenua (frecuencias históricas),
  - el mercado (probabilidades implícitas sin margen, si hay cuotas).
- **Tablas de calibración** por tramos de probabilidad.
- **Simulación de valor**: apostar 1 unidad cuando EV > umbral con las cuotas
  históricas. Es orientativa: un ROI pasado no asegura nada.

### Resultados con la liga sintética (990 partidos evaluados)

| Mercado      | Log-loss modelo | Log-loss referencia | Log-loss mercado |
|--------------|-----------------|---------------------|------------------|
| 1X2          | 1.0525          | 1.0706              | 1.0480           |
| Ambos marcan | 0.6665          | 0.6770              | 0.6558           |
| Córners 9.5  | 0.6778          | 0.6795              | 0.6655           |

Lectura honesta:

- El modelo supera a la referencia ingenua en los tres mercados.
- El "mercado" sintético es algo mejor que el modelo, porque se generó a
  partir de las probabilidades verdaderas más ruido. Es lo esperable también
  con datos reales: las casas de apuestas son difíciles de batir.
- Dixon-Coles mejora ligeramente a Poisson independiente (1X2: 1.0525 frente a
  1.0538).

### Limitaciones

- No considera lesiones, alineaciones, calendario, motivación ni clima.
- Los parámetros de los equipos asumen fuerzas estables (con olvido temporal).
- Los equipos sin historial se tratan como equipos medios.
- La comparación con cuotas usa el método proporcional para quitar el margen,
  que es una aproximación.

"""
Predicción de un partido con el modelo de Poisson.

Idea del modelo
---------------
Los goles de cada equipo se tratan como sucesos raros e
independientes, así que siguen una distribución de Poisson:

    P(k goles) = e^(-λ) · λ^k / k!

donde λ son los goles esperados (xG) del equipo en ese partido.

Con el λ del local y el λ del visitante se arma la matriz de
marcadores (0-0, 1-0, 0-1, ...) y de ella salen todos los mercados:
1X2, doble oportunidad, ambos marcan y más/menos goles.

La matriz llega hasta 10 goles por equipo y se normaliza para que
sume 1. Así no se pierde probabilidad cuando un λ es alto (el Excel
MODELO POISSON 25-26 cortaba en 6 goles y perdía hasta un 5,6 %).

Opcionalmente se aplica la corrección de Dixon-Coles (ρ), que
ajusta los marcadores bajos 0-0, 1-0, 0-1 y 1-1.

"Impacto" es el valor esperado de apostar 1 unidad a la cuota dada:

    impacto = probabilidad · cuota − 1

Positivo  → la cuota paga más de lo que el modelo cree justo.
Negativo  → la cuota paga menos.
"""

from math import exp, factorial

# Marcadores por encima de este número tienen una probabilidad
# despreciable (con λ ≈ 3 la cola desde 11 goles es < 0,03 %).
MAX_GOLES = 10

LINEAS_GOLES = (0.5, 1.5, 2.5, 3.5, 4.5)


# ─────────────────────────────────────────────
# 1. Piezas básicas
# ─────────────────────────────────────────────

def poisson(k, lam):
    """Probabilidad de marcar exactamente k goles con media lam."""
    return exp(-lam) * lam ** k / factorial(k)


def impacto(probabilidad, cuota):
    """Valor esperado (en tanto por uno) de apostar a esa cuota."""
    if cuota is None:
        return None
    return probabilidad * cuota - 1


def cuota_justa(probabilidad):
    """Cuota a la que la apuesta no gana ni pierde a la larga."""
    if probabilidad <= 0:
        return float("inf")
    return 1 / probabilidad


def estimar_xg(
    gf_casa_local,
    gc_casa_local,
    gf_fuera_visitante,
    gc_fuera_visitante,
    media_goles_local,
    media_goles_visitante,
):
    """
    λ de cada equipo a partir de la tabla de liga (ataque × defensa).

    Todos los valores son goles por partido:
        - el local, solo en sus partidos en casa
        - el visitante, solo en sus partidos fuera
        - media_goles_local:     goles que marca de media el equipo
                                 de casa en la liga
        - media_goles_visitante: goles que marca de media el equipo
                                 de fuera en la liga

        xG local     = GF casa local     × GC fuera visitante
                       ÷ media_goles_local

        xG visitante = GF fuera visitante × GC casa local
                       ÷ media_goles_visitante

    Es el mismo cálculo que hace el Excel MODELO POISSON 25-26.
    """
    xg_local = gf_casa_local * gc_fuera_visitante / media_goles_local
    xg_visitante = gf_fuera_visitante * gc_casa_local / media_goles_visitante
    return xg_local, xg_visitante


def mezclar(xg_tabla, xg_manual, peso_tabla):
    """
    Mezcla el λ de la tabla con el xG que tengas de otra fuente.

        peso_tabla = 1   → solo la tabla
        peso_tabla = 0   → solo el xG manual
    """
    return peso_tabla * xg_tabla + (1 - peso_tabla) * xg_manual


def margen(cuotas):
    """
    Margen de la casa en un mercado completo (ej. las 3 cuotas del 1X2):

        margen = suma(1 / cuota) − 1
    """
    return sum(1 / c for c in cuotas) - 1


def probabilidades_sin_margen(cuotas):
    """Lo que piensa la casa: (1 / cuota) repartido para sumar 1."""
    inversas = [1 / c for c in cuotas]
    total = sum(inversas)
    return [x / total for x in inversas]


# ─────────────────────────────────────────────
# 2. Matriz de marcadores
# ─────────────────────────────────────────────

def dixon_coles(i, j, xg_local, xg_visitante, rho):
    """
    Factor de corrección de Dixon-Coles para marcadores bajos.

    Con ρ < 0 suben el 0-0 y el 1-1 y bajan el 1-0 y el 0-1.
    Con ρ = 0 el factor es 1 (Poisson puro).
    """
    if i == 0 and j == 0:
        return 1 - xg_local * xg_visitante * rho
    if i == 0 and j == 1:
        return 1 + xg_local * rho
    if i == 1 and j == 0:
        return 1 + xg_visitante * rho
    if i == 1 and j == 1:
        return 1 - rho
    return 1


def matriz_marcadores(xg_local, xg_visitante, max_goles=MAX_GOLES, rho=0.0):
    """
    matriz[i][j] = probabilidad de que el partido acabe i - j.

    Como los dos equipos son independientes:
        P(i, j) = P_local(i) · P_visitante(j) · factor Dixon-Coles

    Al final se divide todo por la suma para que la matriz sume 1.
    """
    local = [poisson(i, xg_local) for i in range(max_goles + 1)]
    visitante = [poisson(j, xg_visitante) for j in range(max_goles + 1)]

    matriz = [
        [
            max(0.0, p_i * p_j * dixon_coles(i, j, xg_local, xg_visitante, rho))
            for j, p_j in enumerate(visitante)
        ]
        for i, p_i in enumerate(local)
    ]

    total = sum(sum(fila) for fila in matriz)
    return [[p / total for p in fila] for fila in matriz]


# ─────────────────────────────────────────────
# 3. Mercados
# ─────────────────────────────────────────────

def resultado_1x2(matriz):
    """Suma la matriz por debajo, sobre y encima de la diagonal."""
    gana_local = empate = gana_visitante = 0.0

    for i, fila in enumerate(matriz):
        for j, p in enumerate(fila):
            if i > j:
                gana_local += p
            elif i == j:
                empate += p
            else:
                gana_visitante += p

    return {
        "local": gana_local,
        "empate": empate,
        "visitante": gana_visitante,
    }


def doble_oportunidad(x12):
    """1X, 12 y X2 son sumas de dos resultados del 1X2."""
    return {
        "1x": x12["local"] + x12["empate"],
        "12": x12["local"] + x12["visitante"],
        "x2": x12["empate"] + x12["visitante"],
    }


def ambos_marcan(matriz):
    """
    Sí = el local marca al menos 1 Y el visitante marca al menos 1,
    es decir, toda la matriz menos la fila 0 y la columna 0.

    Sin Dixon-Coles equivale a:
        P(sí) = (1 − e^(−λ_local)) · (1 − e^(−λ_visitante))
    """
    si = sum(
        p
        for i, fila in enumerate(matriz)
        for j, p in enumerate(fila)
        if i > 0 and j > 0
    )
    return {"si": si, "no": 1 - si}


def mas_menos(matriz, lineas=LINEAS_GOLES):
    """
    Suma las casillas de la matriz según los goles totales i + j.

    Para la línea 2,5:
        menos = casillas con 0, 1 o 2 goles en total
        más   = 1 − menos

    Sin Dixon-Coles equivale a una Poisson con media
    xG local + xG visitante.
    """
    resultado = {}

    for linea in lineas:
        menos = sum(
            p
            for i, fila in enumerate(matriz)
            for j, p in enumerate(fila)
            if i + j < linea
        )
        resultado[linea] = {"mas": 1 - menos, "menos": menos}

    return resultado


def grandes_oportunidades(esperadas, umbrales=(1, 2, 3)):
    """
    Probabilidad de que haya al menos k grandes ocasiones en el
    partido, tratando su número también como una Poisson.

        P(≥ k) = 1 − [P(0) + ... + P(k − 1)]
    """
    return {
        k: 1 - sum(poisson(n, esperadas) for n in range(k))
        for k in umbrales
    }


# ─────────────────────────────────────────────
# 4. Todo junto
# ─────────────────────────────────────────────

def predecir(xg_local, xg_visitante, cuotas=None, oportunidades=None, rho=0.0):
    """
    Devuelve un diccionario con todas las probabilidades.

    cuotas (opcional) usa estas claves, todas opcionales:
        "local", "empate", "visitante",
        "dc_1x", "dc_12", "dc_x2",
        "ambos_si", "ambos_no",
        "mas_2.5", "menos_2.5", ... (una por línea)

    oportunidades (opcional): grandes ocasiones esperadas en el
    partido. Si falta no se calcula ese mercado.

    rho (opcional): corrección de Dixon-Coles. 0 = Poisson puro;
    los valores habituales van de −0,15 a −0,05.
    """
    cuotas = cuotas or {}
    xg_total = xg_local + xg_visitante
    matriz = matriz_marcadores(xg_local, xg_visitante, rho=rho)

    def mercado(nombre, probabilidad):
        cuota = cuotas.get(nombre)
        return {
            "probabilidad": probabilidad,
            "cuota": cuota,
            "cuota_justa": cuota_justa(probabilidad),
            "impacto": impacto(probabilidad, cuota),
        }

    x12 = resultado_1x2(matriz)
    dc = doble_oportunidad(x12)
    btts = ambos_marcan(matriz)
    goles = mas_menos(matriz)

    salida = {
        "xg_local": xg_local,
        "xg_visitante": xg_visitante,
        "xg_total": xg_total,
        "1x2": {k: mercado(k, p) for k, p in x12.items()},
        "doble_oportunidad": {
            k: mercado(f"dc_{k}", p) for k, p in dc.items()
        },
        "ambos_marcan": {
            "si": mercado("ambos_si", btts["si"]),
            "no": mercado("ambos_no", btts["no"]),
        },
        "mas": {
            linea: mercado(f"mas_{linea}", p["mas"])
            for linea, p in goles.items()
        },
        "menos": {
            linea: mercado(f"menos_{linea}", p["menos"])
            for linea, p in goles.items()
        },
    }

    if oportunidades is not None:
        salida["grandes_oportunidades"] = {
            k: mercado(f"oportunidades_{k}", p)
            for k, p in grandes_oportunidades(oportunidades).items()
        }

    return salida


def tabla(local, visitante, prediccion):
    """Texto listo para imprimir en consola, como la tabla original."""

    def fila(nombre, m):
        prob = f"{m['probabilidad']:6.1%}"
        justa = f"{m['cuota_justa']:6.2f}"
        cuota = f"{m['cuota']:6.2f}" if m["cuota"] else "     —"
        imp = f"{m['impacto']:+8.2%}" if m["impacto"] is not None else "       —"
        return f"  {nombre:<22}{prob}   {justa}   {cuota}   {imp}"

    p = prediccion
    lineas = [
        f"{local} (xG {p['xg_local']:.3f})  vs  "
        f"{visitante} (xG {p['xg_visitante']:.3f})",
        f"  {'MERCADO':<22}{'PROB':>6}   {'JUSTA':>6}   "
        f"{'CUOTA':>6}   {'IMPACTO':>8}",
        fila(f"Gana {local}", p["1x2"]["local"]),
        fila("Empate", p["1x2"]["empate"]),
        fila(f"Gana {visitante}", p["1x2"]["visitante"]),
        fila("1X", p["doble_oportunidad"]["1x"]),
        fila("12", p["doble_oportunidad"]["12"]),
        fila("X2", p["doble_oportunidad"]["x2"]),
        fila("Ambos marcan: sí", p["ambos_marcan"]["si"]),
        fila("Ambos marcan: no", p["ambos_marcan"]["no"]),
    ]

    for k, m in p.get("grandes_oportunidades", {}).items():
        lineas.append(fila(f"{k}+ grandes ocasiones", m))

    for linea, m in p["mas"].items():
        lineas.append(fila(f"Más de {linea}", m))

    for linea, m in p["menos"].items():
        lineas.append(fila(f"Menos de {linea}", m))

    return "\n".join(lineas)

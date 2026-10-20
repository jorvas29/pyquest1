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
1X2, ambos marcan y más/menos goles.

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


def estimar_xg(gf_local, gc_local, gf_visitante, gc_visitante):
    """
    Estimación sencilla de xG cuando no se tiene el dato.

    Ataque de un equipo y defensa del rival pesan lo mismo:

        xG local     = (GF local     + GC visitante) / 2
        xG visitante = (GF visitante + GC local)     / 2
    """
    xg_local = (gf_local + gc_visitante) / 2
    xg_visitante = (gf_visitante + gc_local) / 2
    return xg_local, xg_visitante


# ─────────────────────────────────────────────
# 2. Matriz de marcadores
# ─────────────────────────────────────────────

def matriz_marcadores(xg_local, xg_visitante, max_goles=MAX_GOLES):
    """
    matriz[i][j] = probabilidad de que el partido acabe i - j.

    Como los dos equipos son independientes:
        P(i, j) = P_local(i) · P_visitante(j)
    """
    local = [poisson(i, xg_local) for i in range(max_goles + 1)]
    visitante = [poisson(j, xg_visitante) for j in range(max_goles + 1)]

    return [
        [p_i * p_j for p_j in visitante]
        for p_i in local
    ]


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


def ambos_marcan(xg_local, xg_visitante):
    """
    Sí = el local marca al menos 1 Y el visitante marca al menos 1.

        P(sí) = (1 − e^(−λ_local)) · (1 − e^(−λ_visitante))
    """
    si = (1 - exp(-xg_local)) * (1 - exp(-xg_visitante))
    return {"si": si, "no": 1 - si}


def mas_menos(xg_total, lineas=LINEAS_GOLES):
    """
    La suma de dos Poisson independientes es otra Poisson cuya
    media es xG local + xG visitante.

    Para la línea 2,5:
        menos = P(0) + P(1) + P(2)
        más   = 1 − menos
    """
    resultado = {}

    for linea in lineas:
        tope = int(linea)  # 2,5 → hasta 2 goles es "menos"
        menos = sum(poisson(k, xg_total) for k in range(tope + 1))
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

def predecir(xg_local, xg_visitante, cuotas=None, oportunidades=None):
    """
    Devuelve un diccionario con todas las probabilidades.

    cuotas (opcional) usa estas claves, todas opcionales:
        "local", "empate", "visitante",
        "ambos_si", "ambos_no",
        "mas_2.5", "menos_2.5", ... (una por línea)

    oportunidades (opcional): grandes ocasiones esperadas en el
    partido. Si falta no se calcula ese mercado.
    """
    cuotas = cuotas or {}
    xg_total = xg_local + xg_visitante
    matriz = matriz_marcadores(xg_local, xg_visitante)

    def mercado(nombre, probabilidad):
        cuota = cuotas.get(nombre)
        return {
            "probabilidad": probabilidad,
            "cuota": cuota,
            "cuota_justa": cuota_justa(probabilidad),
            "impacto": impacto(probabilidad, cuota),
        }

    x12 = resultado_1x2(matriz)
    btts = ambos_marcan(xg_local, xg_visitante)
    goles = mas_menos(xg_total)

    salida = {
        "xg_local": xg_local,
        "xg_visitante": xg_visitante,
        "xg_total": xg_total,
        "1x2": {k: mercado(k, p) for k, p in x12.items()},
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

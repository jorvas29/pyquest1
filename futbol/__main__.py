"""
Uso desde consola:

    python -m futbol "Coritiba" 0.625 "Palmeiras" 1.423 \
        --cuota mas_2.5=1.95 --cuota menos_2.5=1.833
"""

import argparse

from .prediccion import predecir, tabla


def main():
    parser = argparse.ArgumentParser(description="Predicción Poisson de un partido")
    parser.add_argument("local")
    parser.add_argument("xg_local", type=float)
    parser.add_argument("visitante")
    parser.add_argument("xg_visitante", type=float)
    parser.add_argument(
        "--cuota",
        action="append",
        default=[],
        metavar="MERCADO=CUOTA",
        help="ej. local=2.10, dc_1x=1.25, ambos_si=1.80, mas_2.5=1.95",
    )
    parser.add_argument(
        "--oportunidades",
        type=float,
        help="grandes ocasiones esperadas en el partido",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.0,
        help="corrección Dixon-Coles (0 = Poisson puro, habitual −0.15 a −0.05)",
    )
    args = parser.parse_args()

    cuotas = {}
    for texto in args.cuota:
        nombre, valor = texto.split("=")
        cuotas[nombre] = float(valor.replace(",", "."))

    resultado = predecir(
        args.xg_local,
        args.xg_visitante,
        cuotas=cuotas,
        oportunidades=args.oportunidades,
        rho=args.rho,
    )
    print(tabla(args.local, args.visitante, resultado))


if __name__ == "__main__":
    main()

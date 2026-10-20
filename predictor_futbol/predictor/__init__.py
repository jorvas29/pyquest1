"""Predictor de fútbol: probabilidades para BTTS, córners y 1X2.

Paquete organizado en capas:

    datos/            -> fuentes intercambiables (CSV, API, sintética)
    caracteristicas   -> forma, goles, córners, localía, Elo
    modelos/          -> Dixon-Coles (goles, 1X2, BTTS) y Poisson (córners)
    cuotas            -> cuotas justas, margen y valor esperado
    evaluacion        -> validación temporal, log-loss, Brier, calibración
    informe           -> textos y tablas para la consola o Markdown
    cli               -> punto de entrada de línea de comandos

ADVERTENCIA: ningún modelo garantiza resultados en las apuestas. Las
probabilidades que calcula este paquete son estimaciones con incertidumbre.
"""

ADVERTENCIA = (
    "AVISO: ningún modelo garantiza resultados en las apuestas. Las "
    "probabilidades son estimaciones con incertidumbre basadas en datos "
    "históricos y en supuestos simplificados. Úsalas solo como apoyo a la "
    "decisión y apuesta con responsabilidad."
)

__version__ = "0.1.0"

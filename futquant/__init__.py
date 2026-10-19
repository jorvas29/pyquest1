"""futquant: motor cuantitativo para pronósticos de fútbol.

Modelos incluidos: Poisson, Dixon-Coles, Monte Carlo / cadena de Markov,
Elo, pi-ratings, Glicko-2, regresión logística multinomial, Random Forest,
Gradient Boosting (y XGBoost / LightGBM si están instalados), más el
consenso del mercado sin margen. Ver ``analysis.analyze`` y ``ficha.render``.
"""

from .analysis import analyze
from .ficha import render

__all__ = ["analyze", "render"]

"""Sistemas de rating dinámico: Elo, pi-ratings y Glicko-2.

Los tres miden la "fuerza" relativa y se actualizan partido a partido.
Para convertir una diferencia de rating en probabilidades 1X2 se usa el
modelo de Davidson (Bradley-Terry con empates), con un parámetro ``nu``
que fija la probabilidad de empate entre equipos iguales:

    P(empate | iguales) = nu / (2 + nu)      -> nu = 0.74 da ~27 %
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from scipy.stats import norm

DEFAULT_NU = 0.74


def davidson_1x2(diff: float, scale: float = 400.0, nu: float = DEFAULT_NU) -> dict[str, float]:
    """1X2 a partir de una diferencia de rating (local - visitante, con ventaja de local)."""
    pi_home = 10 ** (diff / (2 * scale))
    pi_away = 10 ** (-diff / (2 * scale))
    tie = nu * math.sqrt(pi_home * pi_away)
    total = pi_home + pi_away + tie
    return {"1": pi_home / total, "X": tie / total, "2": pi_away / total}


def _result(hg: int, ag: int) -> float:
    return 1.0 if hg > ag else 0.5 if hg == ag else 0.0


# --------------------------------------------------------------------------- Elo

@dataclass
class Elo:
    """Elo estilo World Football Elo Ratings.

    El margen de victoria multiplica K: ganar 3-0 mueve más que ganar 1-0,
    y ganarle a un rival fuerte (expectativa baja) mueve más que a uno débil.
    """

    k: float = 20.0
    home_adv: float = 65.0
    initial: float = 1500.0
    nu: float = DEFAULT_NU
    ratings: dict[str, float] = field(default_factory=dict)

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial)

    @staticmethod
    def margin_multiplier(goal_diff: int) -> float:
        n = abs(goal_diff)
        if n <= 1:
            return 1.0
        if n == 2:
            return 1.5
        return (11 + n) / 8

    def expected(self, home: str, away: str, neutral: bool = False) -> float:
        diff = self.rating(home) + (0 if neutral else self.home_adv) - self.rating(away)
        return 1.0 / (1.0 + 10 ** (-diff / 400))

    def update(self, home: str, away: str, hg: int, ag: int, neutral: bool = False) -> None:
        e = self.expected(home, away, neutral)
        delta = self.k * self.margin_multiplier(hg - ag) * (_result(hg, ag) - e)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta

    def fit(self, matches: list[dict]) -> "Elo":
        for m in matches:
            self.update(m["home"], m["away"], int(m["hg"]), int(m["ag"]))
        return self

    def predict_1x2(self, home: str, away: str, neutral: bool = False) -> dict[str, float]:
        diff = self.rating(home) + (0 if neutral else self.home_adv) - self.rating(away)
        return davidson_1x2(diff, 400.0, self.nu)


# ----------------------------------------------------------------------- pi-ratings

@dataclass
class PiRatings:
    """pi-ratings de Constantinou & Fenton (2013).

    Cada equipo tiene un rating de local y otro de visitante. El rating se
    expresa en "goles": la diferencia esperada de goles es
    ``g(R_local_casa) - g(R_visita_fuera)`` con ``g(r) = sign(r)(b^{|r|/c} - 1)``.
    """

    lam: float = 0.035   # tasa de aprendizaje
    gamma: float = 0.7   # cuánto se traslada lo aprendido de local a visitante y viceversa
    b: float = 10.0
    c: float = 3.0
    sigma: float = 1.6   # desviación de la diferencia de goles para pasar a 1X2
    home: dict[str, float] = field(default_factory=dict)
    away: dict[str, float] = field(default_factory=dict)

    def _g(self, r: float) -> float:
        return math.copysign(self.b ** (abs(r) / self.c) - 1, r)

    def expected_goal_diff(self, home: str, away: str) -> float:
        return self._g(self.home.get(home, 0.0)) - self._g(self.away.get(away, 0.0))

    def update(self, home: str, away: str, hg: int, ag: int) -> None:
        expected = self.expected_goal_diff(home, away)
        observed = hg - ag
        error = abs(observed - expected)
        psi = self.c * math.log10(1 + error)
        psi_home = psi if observed > expected else -psi
        psi_away = -psi_home

        rh_home = self.home.get(home, 0.0)
        ra_home = self.away.get(home, 0.0)
        ra_away = self.away.get(away, 0.0)
        rh_away = self.home.get(away, 0.0)

        new_rh_home = rh_home + psi_home * self.lam
        self.home[home] = new_rh_home
        self.away[home] = ra_home + (new_rh_home - rh_home) * self.gamma

        new_ra_away = ra_away + psi_away * self.lam
        self.away[away] = new_ra_away
        self.home[away] = rh_away + (new_ra_away - ra_away) * self.gamma

    def fit(self, matches: list[dict]) -> "PiRatings":
        for m in matches:
            self.update(m["home"], m["away"], int(m["hg"]), int(m["ag"]))
        return self

    def predict_1x2(self, home: str, away: str) -> dict[str, float]:
        """Diferencia de goles ~ Normal(esperada, sigma); empate = |dif| < 0.5."""
        mu = self.expected_goal_diff(home, away)
        p_away = norm.cdf(-0.5, mu, self.sigma)
        p_home = 1 - norm.cdf(0.5, mu, self.sigma)
        return {"1": float(p_home), "X": float(1 - p_home - p_away), "2": float(p_away)}


# ------------------------------------------------------------------------ Glicko-2

GLICKO_SCALE = 173.7178


@dataclass
class GlickoPlayer:
    rating: float = 1500.0
    rd: float = 350.0
    vol: float = 0.06

    @property
    def mu(self) -> float:
        return (self.rating - 1500) / GLICKO_SCALE

    @property
    def phi(self) -> float:
        return self.rd / GLICKO_SCALE


def _g(phi: float) -> float:
    return 1 / math.sqrt(1 + 3 * phi**2 / math.pi**2)


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1 / (1 + math.exp(-_g(phi_j) * (mu - mu_j)))


def glicko2_update(player: GlickoPlayer, games: list[tuple[GlickoPlayer, float]], tau: float = 0.5) -> GlickoPlayer:
    """Un periodo de Glicko-2 (algoritmo de Glickman, pasos 2-8)."""
    mu, phi, sigma = player.mu, player.phi, player.vol
    if not games:
        phi_star = math.sqrt(phi**2 + sigma**2)
        return GlickoPlayer(player.rating, phi_star * GLICKO_SCALE, sigma)

    v_inv = 0.0
    delta_sum = 0.0
    for opp, score in games:
        g = _g(opp.phi)
        e = _E(mu, opp.mu, opp.phi)
        v_inv += g**2 * e * (1 - e)
        delta_sum += g * (score - e)
    v = 1 / v_inv
    delta = v * delta_sum

    a = math.log(sigma**2)

    def f(x):
        ex = math.exp(x)
        return ex * (delta**2 - phi**2 - v - ex) / (2 * (phi**2 + v + ex) ** 2) - (x - a) / tau**2

    A = a
    if delta**2 > phi**2 + v:
        B = math.log(delta**2 - phi**2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        B = a - k * tau
    fA, fB = f(A), f(B)
    while abs(B - A) > 1e-6:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA /= 2
        B, fB = C, fC
    new_sigma = math.exp(A / 2)

    phi_star = math.sqrt(phi**2 + new_sigma**2)
    new_phi = 1 / math.sqrt(1 / phi_star**2 + 1 / v)
    new_mu = mu + new_phi**2 * delta_sum
    return GlickoPlayer(new_mu * GLICKO_SCALE + 1500, new_phi * GLICKO_SCALE, new_sigma)


@dataclass
class Glicko2:
    """Glicko-2 actualizado partido a partido (cada partido = un periodo).

    Además de la fuerza, guarda la incertidumbre (RD): un equipo con RD alto
    (pocos partidos o cambios recientes) produce probabilidades más
    conservadoras.
    """

    home_adv: float = 65.0
    tau: float = 0.5
    nu: float = DEFAULT_NU
    players: dict[str, GlickoPlayer] = field(default_factory=dict)

    def player(self, team: str) -> GlickoPlayer:
        return self.players.setdefault(team, GlickoPlayer())

    def update(self, home: str, away: str, hg: int, ag: int) -> None:
        ph, pa = self.player(home), self.player(away)
        s = _result(hg, ag)
        ph_adj = GlickoPlayer(ph.rating + self.home_adv, ph.rd, ph.vol)
        pa_adj = GlickoPlayer(pa.rating - self.home_adv, pa.rd, pa.vol)
        new_h = glicko2_update(ph, [(pa_adj, s)], self.tau)
        new_a = glicko2_update(pa, [(ph_adj, 1 - s)], self.tau)
        self.players[home], self.players[away] = new_h, new_a

    def fit(self, matches: list[dict]) -> "Glicko2":
        for m in matches:
            self.update(m["home"], m["away"], int(m["hg"]), int(m["ag"]))
        return self

    def predict_1x2(self, home: str, away: str, neutral: bool = False) -> dict[str, float]:
        ph, pa = self.player(home), self.player(away)
        diff = ph.rating + (0 if neutral else self.home_adv) - pa.rating
        # La incertidumbre combinada encoge la diferencia (factor g de Glicko).
        shrink = _g(math.sqrt(ph.phi**2 + pa.phi**2))
        return davidson_1x2(diff * shrink, 400.0, self.nu)

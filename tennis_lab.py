"""

A minimal laboratory for watching standard textbook statistic misbehave once you
put *humans* inside the sum. We now that in finance fat-tail statistics, nontrivial 
correlations and black swans are almost ubiquitous. But it is instructive to see 
how non-trivial statistics may pop-up in (seemingly) less complicated scenarios.


Here, the setting is a tennis match. At the bottom there is a Bernoulli trial: 
player A wins a point with probability ``p``. Under the textbook assumptions (i.i.d.
points) the number of points A wins in ``n`` points is exactly Binomial(n, p),
the CLT applies with the usual sqrt(n) scaling, and everything is comfortable. 
However, here we break this comfort with **Absorbing barriers / nonlinear aggregation**
and **Serial correlation ("momentum" / the hot hand)**.

Author: Marco Galoppo
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "TennisSimulator",
    "MatchResult",
    "logit",
    "sigmoid",
    "game_win_probability",
    "set_win_probability",
    "match_win_probability_iid",
    "race_win_probability",
]


# --------------------------------------------------------------------------- #
# small function helpers
# --------------------------------------------------------------------------- #
def logit(p: float | np.ndarray) -> float | np.ndarray:
    """Log-odds. Defined on (0, 1)."""
    return np.log(p / (1.0 - p))


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Inverse of :func:`logit`, numerically stable for large |x|."""
    x = np.asarray(x)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out if out.ndim else float(out)


def _sigmoid_scalar(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)

# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass
class MatchResult:
    """Everything worth recording about one simulated match."""

    winner: int                 # 1 if player A won the match, 0 otherwise
    points_a: int
    points_b: int
    games_a: int
    games_b: int
    sets_a: int
    sets_b: int
    longest_run_a: int          # longest streak of consecutive points won by A
    set_scores: list = field(default_factory=list)   # [(gamesA, gamesB), ...]

    @property
    def total_points(self) -> int:
        return self.points_a + self.points_b

    @property
    def point_share(self) -> float:
        return self.points_a / self.total_points


# --------------------------------------------------------------------------- #
# the simulator
# --------------------------------------------------------------------------- #
class TennisSimulator:
    """
    Simulate tennis at the point level, with optional order-1 serial correlation.

    Parameters
    ----------
    p : float
        Target probability that player A wins any given point. If
        ``calibrate=True`` (default) this is the *stationary* point-win
        probability, so it means the same thing whether or not correlation is on.
    k : float
        Momentum strength, in log-odds. ``k = 0`` gives i.i.d. points. Positive
        ``k`` means winning the previous point makes you more likely to win the
        next one (hot hand); negative ``k`` means mean reversion ("they always
        drops serve right after breaking"). 
    memory : {"none", "markov"}
        ``"none"`` forces ``k`` to be ignored. Convenience switch so the same
        object can be flipped between regimes.
    calibrate : bool
        Solve for the offset ``c`` so the stationary point-win probability equals
        ``p`` exactly. Without it, correlation shifts the mean too and you can no 
        longer attribute what you see to variance alone.
    best_of : int
        3 or 5.
    games_to_win_set, tiebreak, tiebreak_points, points_to_win_game :
        Scoring knobs. Defaults are real tennis.
    stationary_start : bool
        Draw the "previous point" state from the stationary distribution at the
        start of each match, so the chain is stationary from t=0.
    seed : int or None
        Seed for the internal ``numpy`` Generator.

    Notes
    -----
    The momentum state persists *across* games and sets within a match - it is a
    property of the player, not of the game. It resets between matches. Not p though,
    we are not looking at players which evolve outside of the game.
    """

    def __init__(
        self,
        p: float = 0.55,
        k: float = 0.0,
        memory: str = "markov",
        calibrate: bool = True,
        best_of: int = 3,
        games_to_win_set: int = 6,
        tiebreak: bool = True,
        tiebreak_points: int = 7,
        points_to_win_game: int = 4,
        stationary_start: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 < p < 1.0:
            raise ValueError("p must lie strictly inside (0, 1)")
        if memory not in ("none", "markov"):
            raise ValueError("memory must be 'none' or 'markov'")
        if best_of not in (3, 5):
            raise ValueError("best_of must be 3 or 5")

        self.p = float(p)
        self.k = 0.0 if memory == "none" else float(k)
        self.memory = memory
        self.calibrate = bool(calibrate)
        self.best_of = int(best_of)
        self.sets_needed = (best_of + 1) // 2
        self.games_to_win_set = int(games_to_win_set)
        self.tiebreak = bool(tiebreak)
        self.tiebreak_points = int(tiebreak_points)
        self.points_to_win_game = int(points_to_win_game)
        self.stationary_start = bool(stationary_start)

        self.rng = np.random.default_rng(seed)

        self._L = float(logit(self.p))
        self._c = self._solve_offset() if (self.calibrate and self.k != 0.0) else 0.0

        # transition probabilities of the order-1 chain, they are conditional!
        self._a = _sigmoid_scalar(self._L + self._c + self.k)   # P(win | won last)
        self._b = _sigmoid_scalar(self._L + self._c - self.k)   # P(win | lost last)

        # buffered uniforms - drawing them in blocks is far faster than one at a time
        self._buf: np.ndarray = np.empty(0)
        self._buf_i: int = 0
        self._block: int = 65536

    # ------------------------------------------------------------------ #
    # calibration and analytic properties
    # ------------------------------------------------------------------ #
    def _stationary_given_offset(self, c: float) -> float:
        a = _sigmoid_scalar(self._L + c + self.k)
        b = _sigmoid_scalar(self._L + c - self.k)
        return b / (1.0 - a + b)

    def _solve_offset(self, tol: float = 1e-13, max_iter: int = 200) -> float:
        """Bisection for c such that the stationary win probability equals p."""
        lo, hi = -40.0, 40.0
        if self._stationary_given_offset(lo) > self.p:
            return lo
        if self._stationary_given_offset(hi) < self.p:
            return hi
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if self._stationary_given_offset(mid) < self.p:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return 0.5 * (lo + hi)

    @property
    def transition_probs(self) -> Tuple[float, float]:
        """``(a, b)`` = (P(win | won last point), P(win | lost last point))."""
        return self._a, self._b

    @property
    def stationary_p(self) -> float:
        """Long-run point-win probability actually implied by ``(p, k)``."""
        return self._b / (1.0 - self._a + self._b)

    @property
    def rho(self) -> float:
        """Lag-1 autocorrelation of the point-outcome sequence, ``a - b``."""
        return self._a - self._b

    @property
    def variance_inflation(self) -> float:
        """
        Asymptotic ratio Var(S_n) / [n p (1-p)] for the correlated sequence.

        Equals ``(1 + rho) / (1 - rho)``. This is the number that tells you how
        badly a naive binomial standard error understates reality.
        """
        r = self.rho
        return (1.0 + r) / (1.0 - r)

    def exact_variance(self, n: int) -> float:
        """
        Exact Var(S_n) for the stationary chain over ``n`` points.

        Since Cov(X_t, X_{t+h}) = p(1-p) rho^h,

            Var(S_n) = p(1-p) * [ n + 2 * sum_{h=1}^{n-1} (n - h) rho^h ]

        This converges to ``n p(1-p) * (1+rho)/(1-rho)`` but from below, which is
        why a finite simulation lands slightly under the asymptotic factor.
        """
        n = int(n)
        h = np.arange(1, n)
        s = float(np.sum((n - h) * self.rho ** h))
        return self.p * (1.0 - self.p) * (n + 2.0 * s)

    @property
    def effective_sample_size_ratio(self) -> float:
        """
        ``n_eff / n``. The reciprocal of the variance inflation factor: how many
        genuinely independent points a stretch of ``n`` correlated points is
        worth. At rho = 0.5, 1000 points carry the information of ~333.
        """
        return 1.0 / self.variance_inflation

    def summary(self) -> Dict[str, float]:
        """Analytic characterisation of the current parameter setting."""
        return {
            "p_target": self.p,
            "k": self.k,
            "offset_c": self._c,
            "P(win|won last)": self._a,
            "P(win|lost last)": self._b,
            "stationary_p": self.stationary_p,
            "rho": self.rho,
            "variance_inflation": self.variance_inflation,
            "n_eff_over_n": self.effective_sample_size_ratio,
            "mean_run_length_A": 1.0 / (1.0 - self._a) if self._a < 1 else np.inf,
        }

    # ------------------------------------------------------------------ #
    # random number plumbing
    # ------------------------------------------------------------------ #
    def _u(self) -> float:
        if self._buf_i >= self._buf.size:
            self._buf = self.rng.random(self._block)
            self._buf_i = 0
        u = self._buf[self._buf_i]
        self._buf_i += 1
        return u

    def _draw_initial_state(self) -> int:
        if not self.stationary_start:
            return 1 if self._u() < self.p else -1
        return 1 if self._u() < self.stationary_p else -1

    def _p_next(self, s: int) -> float:
        "The simple order-one chain will use only the sign of the streak,"
        "the long memory will use directly the magnitude."
        if self.k == 0.0:
            return self.p
        return self._a if s > 0 else self._b

    # ------------------------------------------------------------------ #
    # point sequences 
    # ------------------------------------------------------------------ #
    def simulate_point_sequences(
        self, n_sequences: int, n_points: int, return_paths: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Simulate ``n_sequences`` independent stretches of ``n_points`` points and
        return the number of points won by A in each.

        The simple "sum of Bernoullis" experiment. With ``k = 0`` the returned 
        counts are exactly Binomial(n_points, p). With ``k > 0`` they are not 
        - same mean, wider distribution, heavier tails, longer runs. With long 
        memory enabled things are even funkier than with the order-one chain.

        Vectorised across sequences.

        Returns
        -------
        counts : (n_sequences,) int array
        paths  : (n_sequences, n_points) uint8 array, only if ``return_paths``
        """
        n_sequences, n_points = int(n_sequences), int(n_points)
        state = (self.rng.random(n_sequences) < self.stationary_p, 1, -1)

        counts = np.zeros(n_sequences, dtype=np.int64)
        paths = (
            np.zeros((n_sequences, n_points), dtype=np.uint8) if return_paths else None
        )

        a, b, k = self._a, self._b, self.k
        for t in range(n_points):
            pt = self.p if k == 0.0 else np.where(state > 0, a, b)
            win = (self.rng.random(n_sequences) < pt).astype(np.int8)
            counts += win
            if return_paths:
                paths[:, t] = win
            state = np.where(win == 1, np.max(state,0) + 1, np.min(state,0) -1)

        return (counts, paths) if return_paths else counts

    # ------------------------------------------------------------------ #
    # the scoring hierarchy
    # ------------------------------------------------------------------ #
    def _play_game(self, state: int, run:int, best_run:int, tb: bool = False) -> Tuple[int, int, int, int, int]:
        """
        Play one game (or tiebreak). Returns
        ``(winner, points_a, points_b, new_state, longest_run_a)``.
        """
        target = self.tiebreak_points if tb else self.points_to_win_game
        pa = pb = 0
        while True:
            if self._u() < self._p_next(state):
                pa += 1
                state = state + 1 if state > 0 else 1
                run += 1
                if run > best_run:
                    best_run = run
            else:
                pb += 1
                state = state - 1 if state < 0 else -1
                run = 0
            if pa >= target and pa - pb >= 2:
                return 1, pa, pb, state, run, best_run
            if pb >= target and pb - pa >= 2:
                return 0, pa, pb, state, run, best_run

    def _play_set(self, state: int, run: int, best_run: int ) -> Tuple[int, int, int, int, int, int, int]:
        """Returns ``(winner, games_a, games_b, points_a, points_b, state, best_run)``."""
        ga = gb = pa = pb = 0
        G = self.games_to_win_set
        while True:
            is_tb = self.tiebreak and ga == G and gb == G
            w, x, y, state, run, best_run = self._play_game(state, run, best_run, tb=is_tb)
            pa += x
            pb += y
            if w == 1:
                ga += 1
            else:
                gb += 1
            if is_tb:
                return (1 if w == 1 else 0), ga, gb, pa, pb, state, run, best_run
            if ga >= G and ga - gb >= 2:
                return 1, ga, gb, pa, pb, state, run, best_run
            if gb >= G and gb - ga >= 2:
                return 0, ga, gb, pa, pb, state, run, best_run

    def simulate_match(self) -> MatchResult:
        """Play one complete match. Momentum persists across games and sets."""
        state = self._draw_initial_state()
        sa = sb = ga_tot = gb_tot = pa_tot = pb_tot = 0
        run = best_run = 0
        scores = []
        while sa < self.sets_needed and sb < self.sets_needed:
            w, ga, gb, pa, pb, state, run, best_run = self._play_set(state, run, best_run)
            sa += w
            sb += 1 - w
            ga_tot += ga
            gb_tot += gb
            pa_tot += pa
            pb_tot += pb
            scores.append((ga, gb))
        return MatchResult(
            winner=1 if sa > sb else 0,
            points_a=pa_tot,
            points_b=pb_tot,
            games_a=ga_tot,
            games_b=gb_tot,
            sets_a=sa,
            sets_b=sb,
            longest_run_a=best_run,
            set_scores=scores,
        )

    def simulate_matches(self, n_matches: int, as_dict: bool = True):
        """
        Play ``n_matches`` independent matches.

        Returns a dict of numpy arrays (default) or a list of
        :class:`MatchResult`. The dict form is what the notebook plots.
        """
        n_matches = int(n_matches)
        results = [self.simulate_match() for _ in range(n_matches)]
        if not as_dict:
            return results
        return {
            "winner": np.fromiter((r.winner for r in results), int, n_matches),
            "points_a": np.fromiter((r.points_a for r in results), int, n_matches),
            "points_b": np.fromiter((r.points_b for r in results), int, n_matches),
            "total_points": np.fromiter(
                (r.total_points for r in results), int, n_matches
            ),
            "point_share": np.fromiter(
                (r.point_share for r in results), float, n_matches
            ),
            "games_a": np.fromiter((r.games_a for r in results), int, n_matches),
            "games_b": np.fromiter((r.games_b for r in results), int, n_matches),
            "sets_a": np.fromiter((r.sets_a for r in results), int, n_matches),
            "sets_b": np.fromiter((r.sets_b for r in results), int, n_matches),
            "longest_run_a": np.fromiter(
                (r.longest_run_a for r in results), int, n_matches
            ),
        }

    def match_win_probability(self, n_matches: int = 20000) -> float:
        """Monte-Carlo estimate of P(A wins the match) under current settings."""
        return float(self.simulate_matches(n_matches)["winner"].mean())

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #
    def with_params(self, **kwargs) -> "TennisSimulator":
        """Return a fresh simulator with some parameters overridden."""
        base = dict(
            p=self.p,
            k=self.k,
            memory=self.memory,
            calibrate=self.calibrate,
            best_of=self.best_of,
            games_to_win_set=self.games_to_win_set,
            tiebreak=self.tiebreak,
            tiebreak_points=self.tiebreak_points,
            points_to_win_game=self.points_to_win_game,
            stationary_start=self.stationary_start,
        )
        base.update(kwargs)
        return TennisSimulator(**base)

    def __repr__(self) -> str:
        return (
            f"TennisSimulator(p={self.p:.4f}, k={self.k:.3f}, "
            f"rho={self.rho:.4f}, var_inflation={self.variance_inflation:.3f}, "
            f"best_of={self.best_of})"
        )


# --------------------------------------------------------------------------- #
# closed-form reference: probability of winning a first-to-N-win-by-2 race
# --------------------------------------------------------------------------- #
def game_win_probability(p: float, target: int = 4) -> float:
    """
    Exact probability of winning a first-to-``target``, win-by-2 race with i.i.d.
    Bernoulli(p) points. Used in the notebook as an analytic check on the
    simulator and to draw the nonlinear 'amplification' curve without noise.
    """
    from math import comb

    total = 0.0
    # win outright: opponent reaches at most target-2
    for j in range(target - 1):
        total += comb(target - 1 + j, j) * p ** target * (1 - p) ** j
    # reach deuce at (target-1, target-1), then win the win-by-2 duel
    p_deuce = comb(2 * (target - 1), target - 1) * (p * (1 - p)) ** (target - 1)
    q = p * p / (p * p + (1 - p) ** 2)   # P(win from deuce)
    return total + p_deuce * q


def set_win_probability(p: float, tiebreak: bool = True, tiebreak_points: int = 7) -> float:
    """
    Exact probability of winning a set with i.i.d. Bernoulli(p) points.

    Games are won with ``pg = game_win_probability(p)`` and are independent under
    the i.i.d.-point assumption, so the set is a first-to-6-win-by-2 race capped
    at 7-6, where 6-6 is resolved by a first-to-7-win-by-2 tiebreak *in points*.
    """
    from math import comb

    pg = game_win_probability(p, 4)
    qg = 1.0 - pg

    # win 6-0 through 6-4
    total = sum(comb(5 + g, g) * pg ** 6 * qg ** g for g in range(5))

    # reach 5-5
    p55 = comb(10, 5) * (pg * qg) ** 5
    if tiebreak:
        p_tb = game_win_probability(p, tiebreak_points)
        total += p55 * (pg ** 2 + 2 * pg * qg * p_tb)
    else:
        # 5-5 then an unbounded win-by-2 duel in games
        total += p55 * (pg * pg / (pg * pg + qg * qg))
    return total


def match_win_probability_iid(
    p: float, best_of: int = 3, tiebreak: bool = True
) -> float:
    """
    Exact P(player A wins the match) with i.i.d. Bernoulli(p) points.

    This is the function ``f(p)`` used throughout the fragility analysis. It is
    an extremely steep S-curve: the whole point of the exercise is that a
    quantity you can only estimate to within a few points of a percent gets
    pushed through ``f`` before it means anything.
    """
    ps = set_win_probability(p, tiebreak=tiebreak)
    if best_of == 3:
        return ps ** 2 * (3.0 - 2.0 * ps)
    if best_of == 5:
        qs = 1.0 - ps
        return ps ** 3 * (1.0 + 3.0 * qs + 6.0 * qs ** 2)
    raise ValueError("best_of must be 3 or 5")


def race_win_probability(p_unit: float, target: int, margin: int = 2) -> float:
    """
    Probability of winning a first-to-``target`` race (win by ``margin``) where
    each unit is won independently with probability ``p_unit``. Reduces to
    :func:`game_win_probability` for ``target=4, margin=2``.
    """
    if margin != 2:
        raise NotImplementedError("only margin=2 is implemented")
    return game_win_probability(p_unit, target=target)


if __name__ == "__main__":  # pragma: no cover
    for k in (0.0, 0.5, 1.0):
        sim = TennisSimulator(p=0.55, k=k, seed=0)
        print(sim)
        for key, val in sim.summary().items():
            print(f"    {key:>20s} : {val: .5f}")

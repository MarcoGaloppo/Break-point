# Statistics in human affairs — a tennis laboratory

How much do textbook statistics comfortable machinery survives contact with a
process that has **absorbing barriers**, **serial correlation**, and **parameter uncertainty**? 
Tennis is a clean (and fun!) place to ask, because the bottom of the process really is a
Bernoulli trial. Additionally, everything above it is bookkeeping that we control exactly. 
Although finance is messy sometimes is good to start with a ground truth and build up 
some basic intuition (without being fooled too much by it possibly!)

## The set-up

Player A wins any point with probability `p`. Under textbook assumptions the number of points
A wins in `n` points is exactly `Binomial(n, p)` and `p̂` has standard error `√(p(1-p)/n)`.
Here, we break this simple scenario in various ways:

**1. Nonlinear aggregation.** Tennis counts games (first to 4, win by 2), sets (first to 6
games, win by 2, tiebreak at 6–6), and matches (best of 3 or 5). Every layer is an effective
absorbing-barrier stopping problem, and the composition `f(p) = P(A wins the match)` is a
steep S-curve. It is trivial to say, but scoring a point and winning a game are not the 
same thing. And they do not carry the same probability given a set chance of winning a point.
**Note, this is present even with perfectly i.i.d. points.**

**2. Serial correlation.** The point-win probability depends on the previous point via

(1) p_t = σ( logit(p) + c + k · s_{t-1} ),      s_{t-1} ∈ {−1, +1}.

The offset `c` is solved numerically so the **stationary** win probability stays exactly `p`.
This is done because turning correlation on then changes the variance and path structure
**without moving the mean**. Also, note that to try and avoid exiting completely from
CLT territory we are consider short-time correlations within each game (at this stage).

**3. Long memory, specified through the run-length hazard.**  What actually
controls the memory is the tail of the run-length distribution, so we specify that directly to 
try to introduce fat-tails and long-memory:

(2) h(ℓ) = P(run ends next | run has reached ℓ) = β / (ℓ + γ).

The survival function behaves as `P(L > ℓ) ~ C ℓ^(−β)`, so β is literally a Pareto tail index. 
For `1 < β < 2` the run variance is infinite, `Σ_h ρ_h` diverges, `Var(S_n) ~ n^(3−β)`, and the failure is 
in the scaling *exponent* of the variance. This would be in theory. When fat-tails meet absorbing barriers, 
however, things can change drastically. 

## Headline results

1. An order-1 chain (short-term memory), effectively generates excess kurtosis which still decays 
to zero. Correlation of this kind does not buy you fat tails — it buys you **overconfidence**. And 
this is a problem. Because, then, every normality test on correctly-standardised residuals will pass 
while your error bars stay too small with respect to the actual scenaro you would like to describe. 
There is more volatility in the system, even in this case. Quite fun!

2. Momentum is a variance pump, and in a barrier-crossing game variance is worth something to whoever is behind. 
We have an asymettric situation in which the (relatively) close-to-win-but-still-losing player benefits more from luck (i.e., more things can go better than worse for them) than the close-to-losing-but-still-winning player (which has a more capped ceiling on good luck than bad luck). At p = 0.55 held fixed, the favourite's match-win probability falls from 91% to 77% as k goes from 0 to 1. **A model fitted on point-level data and aggregated under independence systematically overprices favourites.** If a player is a (slight) underdog, volatility will benefit them more!

3. **The absorbing barriers kill the long-memory in tennis.** We consider the underlying 
chosen models within the constraint of *real matches*. As such, a match starts at the first point 
and ends when someone has won two sets. The longest run physically possible in a best-of-3 is **53** points. 
We found that by keeping the mean run length fixed at 4.0 and sweeping β from 3.5 down to 1.3 — i.e., from finite
to infinite variance — the longest run ever observed is **52 or 53 every single time**. In other words: the
extreme is pinned to the format, not to the underlying process. In fact, the clipped runs would be those that carry
*all* of the variance for β < 2.0. As such we discovered the simple fact that **a truncated power law is not a power law**. 

## Files

1.  `tennis_lab.py`: The simulator. `TennisSimulator` (point → game → set → match, with deuce and tiebreak, optional order-1 momentum) plus exact closed forms `game_win_probability`, `set_win_probability`, `match_win_probability_iid`. Depends only on `numpy`. 
2. `narrative_tennis_lab.ipynb` | The analysis: eight sections, with narrative. Outputs are committed, so it reads on GitHub without running. |
3. `baseline.py`: a regression fingerprint. Any change meant to preserve behaviour must leave its output byte-identical — which is what made the signed-streak refactor safe to do.

Key knobs to turn: `p` (target stationary point-win probability), `k` (momentum strength in
log-odds; 0 = i.i.d., 1.0 = strong hot hand), `best_of`, `calibrate`, `seed`. For the
long-memory regime: `memory="long"`, `beta` (tail index) and `gamma` (scale, most usefully set
via `gamma_for_mean_run`).

Quantities that are theorems about the order-1 chain (`rho`, `variance_inflation`,
`exact_variance`, `effective_sample_size_ratio`) raise `NotImplementedError` under
`memory="long"` rather than returning a plausible-looking wrong number, and point at
`hurst_exponent` / `run_tail_index` instead.

## Quick Validations

The module is checked against closed:

- `k = 0` reproduces `Binomial(n, p)` to Monte-Carlo error.
- Simulated variance matches `exact_variance(n)` — the untruncated covariance sum
  `p(1−p)[n + 2Σ(n−h)ρ^h]` — across `k ∈ [0, 1.6]`.
- Simulated match-win frequency matches `match_win_probability_iid` (0.9089 vs 0.9100 on
  30,000 best-of-3 matches; 0.9526 vs 0.9530 on best-of-5).
- `stationary_p` is pinned at the target `p` to six decimals for every `k`.
- Empirical run-length survival matches the exact Gamma-ratio formula `run_survival`
  (0.03565 vs 0.03556 for `P(L > 10)` at β = 2.5).
- `gamma_for_mean_run` reproduces `E[L_A]` to machine precision, and the closed-form
  calibration `p = γ_A/(γ_A+γ_B)` holds for every β.
- `simulate_match_points` returns exactly `total_points` outcomes per match, so recording
  provably does not perturb the random stream.
- The 53-point ceiling was confirmed by exhaustive search over 150,000 deliberately streaky
  matches (77 for best-of-5), never once exceeded.

## Requirements

`numpy` for the module; `pandas`, `matplotlib`, `jupyter` additionally for the notebook.
No `scipy` — the binomial pmf and the normal quantile function are implemented inline.

```
pip install -r requirements.txt
```

## Where this goes next

- **Inference, not simulation** — given one observed match, what is the posterior over
  `(p, k)`? Or, sharper still after Part 7: fit a Hill estimator to run lengths from simulated
  matches and watch it report a tail index on the wrong side of β = 2, confidently and with
  tight standard errors, for a process built with infinite variance.


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

## Headline results

1. An order-1 chain (short-term memory), effectively generates excess kurtosis which still decays 
to zero. Correlation of this kind does not buy you fat tails — it buys you **overconfidence**. And 
this is a problem. Because, then, every normality test on correctly-standardised residuals will pass 
while your error bars stay too small with respect to the actual scenaro you would like to describe. 
There is more volatility in the system, even in this case. Quite fun!

2. Momentum is a variance pump, and in a barrier-crossing game variance is worth something to whoever is behind. 
We have an asymettric situation in which the (relatively) close-to-win-but-still-losing player benefits more from luck (i.e., more things can go better than worse for them) than the close-to-losing-but-still-winning player (which has a more capped ceiling on good luck than bad luck). At p = 0.55 held fixed, the favourite's match-win probability falls from 91% to 77% as k goes from 0 to 1. **A model fitted on point-level data and aggregated under independence systematically overprices favourites.** If a player is a (slight) underdog, volatility will benefit them more!

## Files

1.  `tennis_clt.py`: The simulator. `TennisSimulator` (point → game → set → match, with deuce and tiebreak, optional order-1 momentum) plus exact closed forms `game_win_probability`, `set_win_probability`, `match_win_probability_iid`. Depends only on `numpy`. 
2. `clt_breakdown_tennis.ipynb` | The analysis: eight sections, nine figures, with narrative. Outputs are committed, so it reads on GitHub without running. |

Key knobs to turn: `p` (target stationary point-win probability), `k` (momentum strength in
log-odds; 0 = i.i.d., 1.0 = strong hot hand), `best_of`, `calibrate`, `seed`.

## Quick Validations

The module is checked against closed:

- `k = 0` reproduces `Binomial(n, p)` to Monte-Carlo error.
- Simulated variance matches `exact_variance(n)` — the untruncated covariance sum
  `p(1−p)[n + 2Σ(n−h)ρ^h]` — across `k ∈ [0, 1.6]`.
- Simulated match-win frequency matches `match_win_probability_iid` (0.9089 vs 0.9100 on
  30,000 best-of-3 matches; 0.9526 vs 0.9530 on best-of-5).
- `stationary_p` is pinned at the target `p` to six decimals for every `k`.

## Requirements

`numpy` for the module; `pandas`, `matplotlib`, `jupyter` additionally for the notebook.
No `scipy` — the binomial pmf and the normal quantile function are implemented inline.

```
pip install -r requirements.txt
```

## Where this goes next

- **Random `p` as a first-class feature** — a slow random walk or two-state regime for `p_t`
  within a match (fatigue is a downward drift). This is where genuinely non-vanishing tails
  appear, as opposed to the merely mis-scaled ones.
- **Long memory**: replace the order-1 chain with a streak-length-dependent boost, where
  $\rho$ decays slowly enough that $\sum_h \rho_h$ diverges. That is the regime where the
  CLT genuinely can fail.
- **Inference, not simulation** — given one observed match, what is the posterior over
  `(p, k)`?

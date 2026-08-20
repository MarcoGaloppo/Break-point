"""Inject Part 8 (inference) into narrative_tennis_lab.ipynb. Idempotent."""
import json
import pathlib

NB = pathlib.Path(__file__).with_name("narrative_tennis_lab.ipynb")
MARK = "part8"

nb = json.loads(NB.read_text(encoding="utf-8"))
cells = [c for c in nb["cells"] if c.get("metadata", {}).get("tag") != MARK]

new = []
def md(s):
    new.append({"id": f"p8m{len(new):02d}", "cell_type": "markdown",
                "metadata": {"tag": MARK}, "source": s.strip("\n").splitlines(keepends=True)})
def code(s):
    new.append({"id": f"p8c{len(new):02d}", "cell_type": "code",
                "metadata": {"tag": MARK}, "execution_count": None, "outputs": [],
                "source": s.strip("\n").splitlines(keepends=True)})

# --------------------------------------------------------------------------- #
md(r"""
---
## Part 8 — Inference: what one match actually tells you

Everything so far has been the *forward* problem: given $(p, k, \beta)$, what happens? This
part runs it backwards, which is the only problem anyone holding data actually has. Given an
observed match, what can be said about the parameters — and in particular, can we tell a hot
hand from long memory?

### The likelihoods, and one thing that must be got right

Both models have **exactly computable likelihoods**, so no MCMC is needed; a grid suffices.

* **Order-1.** The sequence likelihood is a product of transition probabilities, so the
  sufficient statistics are the four counts $(n_{11}, n_{10}, n_{01}, n_{00})$.
* **Long memory.** A product of $h(\ell)$ or $1-h(\ell)$ over run ages, so the sufficient
  statistic is the multiset of run lengths. The final run of every match is
  **right-censored** — the match stopped, the run did not end — and contributes
  $\mathbb{P}(L \ge \ell)$ rather than $\mathbb{P}(L = \ell)$. Ignoring that biases every
  run-length estimate downward.

A subtlety worth a sentence: **the stopping rule contributes nothing to the likelihood.**
"The match ends when someone wins two sets" is a deterministic function of the sequence, so
no term appears for it. It nonetheless changes the *information*, because a one-sided match
ends sooner and yields fewer points.

### The two models are not rivals. One nests the other.

Under the $(p, m, \beta)$ parameterisation the A-side hazard is
$h(\ell) = \beta/(\ell + m(\beta-1))$. Let $\beta \to \infty$:

$$h(\ell) \;\longrightarrow\; \frac{\beta}{\ell + m\beta} \;\longrightarrow\; \frac{1}{m}$$

a **constant** hazard — geometric run lengths — which is exactly the order-1 chain. So the
long-memory model contains the order-1 chain as the boundary case $1/\beta = 0$. Verified
numerically: at $\beta = 60$ the log-likelihood of order-1 data is $-2047.39$ against the
order-1 model's own $-2047.21$.

That changes the question. It is not *"which model?"* but *"**is $1/\beta$ greater than
zero?**"* — a nested hypothesis test with one extra parameter, which is a far better-posed
thing to ask.

*(A trap we walked into first: our two log-likelihoods were not likelihoods of the same data.
One included an unconditional first-point term per match and the other conditioned on it. The
offset grows linearly in the number of matches and silently became a preference for the
larger model — model-selection accuracy went **down** with more data. Both are now conditional
on the first point of each match. If a comparison gets worse as data accumulates, suspect the
bookkeeping before the statistics.)*
""")

# --------------------------------------------------------------------------- #
md(r"""
### 8.1 — A parameter that hides from its own estimator
""")

code(r"""
rows = []
for p_ in (0.50, 0.52, 0.55, 0.58, 0.60, 0.65):
    L = np.array([len(x) for x in TennisSimulator(p=p_, k=0.0, seed=SEED).simulate_match_points(2_500)])
    rows.append({"p": p_, "E[points per match]": L.mean(),
                 "sd(p-hat) from one match": np.sqrt(p_ * (1 - p_) / L.mean())})
info = pd.DataFrame(rows).set_index("p")
display(info.style.format({"E[points per match]": "{:.1f}",
                           "sd(p-hat) from one match": "{:.4f}"})
        .set_caption("A one-sided match ends sooner, so it carries less information"))
""")

md(r"""
The stronger the player, the shorter his matches, and so the *less* each one tells you about
him: 163 points at $p=0.50$ against 90 at $p=0.65$, and the standard error on $\hat p$ rises
from $0.039$ to $0.050$. **The parameter degrades its own estimability.** Nothing pathological
is happening — it is the absorbing barrier again, now acting on the sample size rather than on
the outcome.
""")

# --------------------------------------------------------------------------- #
md(r"""
### 8.2 — A posterior that looks fine until you use it
""")

code(r"""
P_GRID = np.linspace(0.30, 0.80, 1_001)
F_GRID = np.array([match_win_probability_iid(x, 3) for x in P_GRID])

def posterior_p(paths, k=0.0):
    '''Flat-prior posterior over p on P_GRID, given a list of match paths.'''
    st = markov_stats(paths)
    ll = np.array([loglik_markov(st, float(x), k) for x in P_GRID])
    w = np.exp(ll - ll.max())
    return w / w.sum()

def credible(w, values, lo=0.025, hi=0.975):
    c = np.cumsum(w)
    return values[np.searchsorted(c, lo)], values[np.searchsorted(c, hi)]

sim = TennisSimulator(p=0.55, k=0.0, seed=SEED)
example = sim.simulate_match_points(1)
w = posterior_p(example)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
ax1.plot(P_GRID, w / np.gradient(P_GRID), color=C0, lw=2)
lo_p, hi_p = credible(w, P_GRID)
ax1.axvspan(lo_p, hi_p, color=C0, alpha=.12)
ax1.axvline(0.55, color="k", ls="--", lw=1.2, label="truth")
ax1.set_xlim(0.35, 0.75); ax1.set_xlabel("p"); ax1.set_ylabel("posterior density")
ax1.set_title(f"Posterior over p from ONE match\n95% interval [{lo_p:.3f}, {hi_p:.3f}]")
ax1.legend(fontsize=8.5)

order = np.argsort(F_GRID)
ax2.plot(F_GRID[order], (w / np.maximum(np.gradient(F_GRID), 1e-12))[order], color=C1, lw=2)
lo_f, hi_f = credible(w, F_GRID)
ax2.axvspan(lo_f, hi_f, color=C1, alpha=.12)
ax2.axvline(match_win_probability_iid(0.55, 3), color="k", ls="--", lw=1.2, label="truth")
ax2.set_xlim(0, 1); ax2.set_yscale("log")
ax2.set_xlabel("f(p) = P(A wins the match)"); ax2.set_ylabel("posterior density (log)")
ax2.set_title(f"The same posterior, pushed through f\n95% interval [{lo_f:.3f}, {hi_f:.3f}]")
ax2.legend(fontsize=8.5)
plt.tight_layout(); plt.show()

widths_p, widths_f = [], []
sim = TennisSimulator(p=0.55, k=0.0, seed=SEED + 1)
for _ in range(300):
    w_ = posterior_p(sim.simulate_match_points(1))
    a, b = credible(w_, P_GRID); widths_p.append(b - a)
    a, b = credible(w_, F_GRID); widths_f.append(b - a)
print(f"over 300 matches, median width of the 95% credible interval")
print(f"    on p     : {np.median(widths_p):.3f}")
print(f"    on f(p)  : {np.median(widths_f):.3f}   <- three quarters of the probability scale")
""")

md(r"""
This is Part 1 attacking inference instead of prediction. The posterior on $p$ is perfectly
respectable — a 95% interval about $0.165$ wide, or $\pm 0.08$. Push it through $f$, where
$f'(\tfrac12) = 10.73$, and the median 95% interval on **the probability he wins the next
match** is $0.76$ wide.

Three quarters of the entire probability scale, after watching a complete match.

A posterior can be well-behaved on the natural parameter and useless on the derived one, and
which of the two you report is a presentation choice with no statistical content. The
parameter is not the quantity of interest; nobody bets on $p$.
""")

# --------------------------------------------------------------------------- #
md(r"""
### 8.3 — Can you see a hot hand?
""")

code(r"""
PS_FIT = np.linspace(0.42, 0.68, 53)
KS_FIT = np.linspace(-0.6, 2.2, 71)
_A = np.zeros((len(PS_FIT), len(KS_FIT))); _B = np.zeros_like(_A)
for _i, _p in enumerate(PS_FIT):
    for _j, _k in enumerate(KS_FIT):
        _A[_i, _j], _B[_i, _j] = calibrated_transitions(float(_p), float(_k))
_LA, _L1A, _LB, _L1B = np.log(_A), np.log1p(-_A), np.log(_B), np.log1p(-_B)

def fit_pk(paths):
    '''ML estimate of (p, k) - grid precomputed, so each fit is a few flops.'''
    n11, n10, n01, n00 = markov_stats(paths)
    ll = n11 * _LA + n10 * _L1A + n01 * _LB + n00 * _L1B
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    return PS_FIT[i], KS_FIT[j]

ns, sds = [1, 2, 5, 10, 25, 50, 100], []
sim = TennisSimulator(p=0.55, k=0.0, seed=SEED + 2)
for n in ns:
    R = max(40, min(250, 2_500 // n))
    sds.append(np.std([fit_pk(sim.simulate_match_points(n))[1] for _ in range(R)], ddof=1))
sds = np.array(sds)
detect_k = 1.96 * sds
detect_rho = np.array([calibrated_transitions(0.55, float(kk))[0]
                       - calibrated_transitions(0.55, float(kk))[1] for kk in detect_k])

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.loglog(ns, detect_rho, "o-", color=C1, ms=5, lw=1.8, label=r"minimum detectable $\rho$ (2 s.e.)")
ax.axhspan(0.02, 0.05, color=C3, alpha=.15)
ax.text(1.15, 0.031, "effect sizes reported in the\nhot-hand literature", fontsize=8, color=C3)
ax.set_xlabel("matches observed"); ax.set_ylabel(r"smallest detectable lag-1 correlation $\rho$")
ax.set_title("A strong hot hand is visible in one match. A realistic one is not.")
ax.legend(fontsize=8.5)
plt.tight_layout(); plt.show()

display(pd.DataFrame({"matches": ns, "sd(k-hat)": sds,
                      "min detectable k": detect_k, "min detectable rho": detect_rho})
        .style.format({"sd(k-hat)": "{:.4f}", "min detectable k": "{:.3f}",
                       "min detectable rho": "{:.3f}"}).hide(axis="index"))
""")

md(r"""
The honest answer is **not** "you cannot tell". From a single match the smallest detectable
$\rho$ is about $0.18$, corresponding to $k \approx 0.36$ — so the strong hot hand of Part 2
($k=1$, $\rho = 0.46$) would be caught comfortably, at better than five standard errors.

The problem is the effect sizes anyone actually claims. The hot-hand literature argues over
correlations of $\rho \sim 0.02$–$0.05$, and reaching those needs **25 to 100 matches** of
the same player at a *constant* $p$ — which is itself an assumption no real season satisfies.
The detectable region and the interesting region barely overlap.
""")

# --------------------------------------------------------------------------- #
md(r"""
### 8.4 — Hot hand, or long memory?
""")

code(r"""
P0, M0 = 0.55, 4.0
_lo, _hi = 0.0, 3.0                       # order-1 k matched to a mean A-run of 4.0
for _ in range(80):
    _mid = (_lo + _hi) / 2
    _lo, _hi = (_mid, _hi) if calibrated_transitions(P0, _mid)[0] < 0.75 else (_lo, _mid)
K_MATCHED = (_lo + _hi) / 2

TAUS = np.concatenate(([1e-3], np.linspace(0.02, 0.88, 44)))   # tau = 1/beta; 0 = order-1
BETAS_FIT = 1.0 / TAUS
GP = np.linspace(0.44, 0.66, 12)
GM = np.linspace(2.0, 7.0, 11)

def profile_tau(paths):
    '''Profile log-likelihood in 1/beta, maximised over p and the mean run.'''
    return np.nanmax(loglik_long_grid(run_stats(paths), GP, GM, BETAS_FIT), axis=(0, 1))

R = 80
ns2 = [1, 2, 5, 10, 25]
false_pos, power, ci_lo, ci_hi = [], [], [], []
for n in ns2:
    s_short = TennisSimulator(p=P0, k=K_MATCHED, seed=SEED + 3)
    s_long = TennisSimulator(p=P0, memory="long", beta=1.5,
                             gamma=gamma_for_mean_run(M0, 1.5, P0), seed=SEED + 4)
    fp = tp = 0; los, his = [], []
    for _ in range(R):
        pr = profile_tau(s_short.simulate_match_points(n)); fp += 2 * (pr.max() - pr[0]) > 3.84
        pr = profile_tau(s_long.simulate_match_points(n));  tp += 2 * (pr.max() - pr[0]) > 3.84
        ok = TAUS[2 * (pr.max() - pr) < 3.84]
        los.append(ok.min()); his.append(ok.max())
    false_pos.append(fp / R); power.append(tp / R)
    ci_lo.append(1 / np.median(his)); ci_hi.append(1 / np.median(los))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
ax1.semilogx(ns2, power, "o-", color=C1, ms=6, lw=1.9, label=r"detect $1/\beta>0$ when true $\beta=1.5$")
ax1.semilogx(ns2, false_pos, "s-", color=C0, ms=5, lw=1.6, label="false positive (truth = order-1)")
ax1.axhline(0.05, color="0.5", ls=":", lw=1.2)
ax1.set_ylim(-0.03, 1.05); ax1.set_xlabel("matches observed"); ax1.set_ylabel("rejection rate")
ax1.set_title("Detecting long memory is easy"); ax1.legend(fontsize=8.5, loc="center right")

ax2.fill_between(ns2, ci_lo, ci_hi, color=C1, alpha=.20, label=r"median 95% interval on $\beta$")
ax2.axhline(1.5, color="k", ls="--", lw=1.3, label=r"truth $\beta=1.5$")
ax2.axhline(2.0, color=C2, ls="-", lw=1.6, label=r"$\beta=2$: variance of $L$ diverges")
ax2.set_xscale("log"); ax2.set_ylim(1, 7)
ax2.set_xlabel("matches observed"); ax2.set_ylabel(r"$\beta$")
ax2.set_title("Measuring it is not"); ax2.legend(fontsize=8.5)
plt.tight_layout(); plt.show()

display(pd.DataFrame({"matches": ns2, "power (true beta=1.5)": power,
                      "false positive (truth order-1)": false_pos,
                      "95% interval on beta": [f"[{a:.2f}, {b:.1f}]" for a, b in zip(ci_lo, ci_hi)]})
        .style.format({"power (true beta=1.5)": "{:.0%}",
                       "false positive (truth order-1)": "{:.0%}"}).hide(axis="index"))
""")

# --------------------------------------------------------------------------- #
md(r"""
Both models are calibrated to the same $p$ and the same mean run length, so they differ
**only** in the shape of the run-length distribution. The nested test asks whether $1/\beta$
is distinguishable from zero.

**Detection is easy.** 81% power from a *single* match, 98% from two, 100% from five, with a
false-positive rate of 0–2%.

*(That rate sits below the nominal 5% because $1/\beta = 0$ is on the **boundary** of the
parameter space, where the likelihood-ratio statistic is not $\chi^2_1$ but a half-and-half
mixture of $\chi^2_0$ and $\chi^2_1$. Using the $\chi^2_1$ threshold is therefore
conservative. Left as is: erring toward not crying wolf is the right direction here.)*

**Measurement is hopeless.** The 95% interval on $\beta$ from one match is $[1.2, 6.6]$. It
does not exclude $\beta = 2$ — the point at which the run variance becomes infinite and the
process changes regime — until about **ten matches**. So a single match will tell you the
memory is heavy-tailed, and will not tell you whether it is the kind that breaks the CLT.

### Why this does not contradict Part 7

Part 7 showed the barrier annihilates the tail: runs capped at 53, the maximum blind to
$\beta$, the Hurst exponent unmeasurable. It would be natural to expect $\beta$ to be
undetectable too. It isn't — because the discrimination is carried by the **body** of the
run-length distribution, not its tail. Runs of 5 to 20 points are far more common under a
power law than under a geometric, and those fit comfortably inside a match.

Which gives the sharpest formulation of the whole notebook:

> **The presence of long memory is easy to detect. Its consequences are impossible to
> measure.** Two matches establish that runs are heavier-tailed than geometric. No number of
> matches recovers the scaling exponent, because the runs that would determine it cannot occur.

That gap — between detecting a property and quantifying what it does to you — is where
Part 1's amplification, Part 7's censoring and Part 8's inference all meet. A model can be
correctly identified, fitted with tight standard errors on the parameter that was identified,
and still say nothing about the quantity anyone cares about.
""")

idx = next(i for i, c in enumerate(cells) if "Natural extensions" in "".join(c["source"]))
cells[idx:idx] = new
nb["cells"] = cells
NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"inserted {len(new)} cells at {idx}; total {len(cells)}")

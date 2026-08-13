"""Insert the 1000-points-in-matches subsection into Part 7. Idempotent."""
import json
import pathlib

NB = pathlib.Path(__file__).with_name("narrative_tennis_lab.ipynb")
MARK = "part7b"

nb = json.loads(NB.read_text(encoding="utf-8"))
cells = [c for c in nb["cells"] if c.get("metadata", {}).get("tag") != MARK]

new = []
def md(s):
    new.append({"id": f"p7bm{len(new):02d}", "cell_type": "markdown",
                "metadata": {"tag": MARK}, "source": s.strip("\n").splitlines(keepends=True)})
def code(s):
    new.append({"id": f"p7bc{len(new):02d}", "cell_type": "code",
                "metadata": {"tag": MARK}, "execution_count": None, "outputs": [],
                "source": s.strip("\n").splitlines(keepends=True)})

# --------------------------------------------------------------------------- #
md(r"""
### A thousand points, played as tennis

Part 2 looked at the distribution of points won out of 1000 in the *bare* process. Repeat it
here, but with the 1000 points delivered the way tennis delivers them: as roughly **eight
consecutive matches**, each starting from a fresh scoreboard and a freshly drawn streak state.

That construction is the whole experiment. The barrier does not merely cap runs — it **chops
the stream into independent blocks of about 120 points**. Whatever memory the process has, it
is reset every match and cannot accumulate past the match boundary. So we should expect the
variance inflation to *saturate* at roughly its value at $n \approx 48$–$120$ and go no
further, no matter how heavy the tail we specified.
""")

# --------------------------------------------------------------------------- #
code(r"""
BLOCK, N_BLOCKS = 1_000, 8_000
SD_NAIVE = np.sqrt(BLOCK * PQ)

def points_in_blocks(kw, n_blocks=N_BLOCKS, block=BLOCK, seed=SEED):
    '''Points won by A in consecutive `block`-point stretches of real match play.

    Matches are played back to back and their point streams concatenated, so a
    block spans several matches and every match boundary inside it resets both
    the scoreboard and the momentum state.
    '''
    sim = TennisSimulator(seed=seed, **kw)
    need, parts, total = n_blocks * block, [], 0
    while total < need:
        batch = sim.simulate_match_points(max(1_000, (need - total) // 100))
        parts.extend(batch)
        total += sum(len(x) for x in batch)
    return np.concatenate(parts)[:need].reshape(n_blocks, block).sum(axis=1)


BLOCK_CFG = [
    ("i.i.d.",                   dict(p=P, k=0.0), C0),
    ("order-1  k = 1",           dict(p=P, k=1.0), C2),
    ("long memory  beta = 2.5",  dict(p=P, memory="long", beta=2.5,
                                      gamma=gamma_for_mean_run(MEAN_RUN, 2.5, P)), C3),
    ("long memory  beta = 1.3",  dict(p=P, memory="long", beta=1.3,
                                      gamma=gamma_for_mean_run(MEAN_RUN, 1.3, P)), "#d35400"),
]
counts = {label: points_in_blocks(kw).astype(float) for label, kw, _ in BLOCK_CFG}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
lo = int(min(c.min() for c in counts.values())) - 10
hi = int(max(c.max() for c in counts.values())) + 10
bins = np.arange(lo - 0.5, hi + 1, 6)          # integer-aligned: counts are discrete
zbins = (bins - BLOCK * P) / SD_NAIVE

grid = np.arange(400, 720)
ax1.plot(grid, binom_pmf(BLOCK, P, grid), "k--", lw=1.7,
         label=f"Binomial(1000, {P})  [what you would assume]")
for label, _, col in BLOCK_CFG:
    ax1.hist(counts[label], bins=bins, density=True, histtype="step", lw=1.8,
             color=col, label=label)
    ax2.hist((counts[label] - BLOCK * P) / SD_NAIVE, bins=zbins, density=True,
             histtype="step", lw=1.8, color=col, label=label)

ax1.set_xlim(400, 720); ax1.set_ylim(0, 0.031)
ax1.set_xlabel("points won by A out of 1000 (played in matches)")
ax1.set_ylabel("density")
ax1.set_title("1000 points delivered as ~8 consecutive matches")
ax1.legend(fontsize=7.5, loc="upper left")

zz = np.linspace(-8, 8, 600)
ax2.plot(zz, np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1.5, label="N(0,1)")
ax2.set_yscale("log"); ax2.set_ylim(1e-5, 1); ax2.set_xlim(-8, 8)
ax2.set_xlabel("z, standardised by the NAIVE binomial s.d.")
ax2.set_title("Log scale: where the 3-sigma went")
ax2.legend(fontsize=7.5, loc="lower center")

plt.tight_layout(); plt.show()

summary = []
for label, _, _ in BLOCK_CFG:
    c = counts[label]
    z_naive = (c - BLOCK * P) / SD_NAIVE
    z_true = (c - c.mean()) / c.std(ddof=1)
    summary.append({"config": label,
                    "realised share": c.mean() / BLOCK,
                    "s.d. of count": c.std(ddof=1),
                    "Var/npq at n=1000": c.var(ddof=1) / (BLOCK * PQ),
                    "excess kurtosis": np.mean(z_true ** 4) - 3,
                    "P(|z_naive|>3)": np.mean(np.abs(z_naive) > 3)})
s = pd.DataFrame(summary).set_index("config")
s["vs the budgeted 0.0027"] = s["P(|z_naive|>3)"] / 0.0027
display(s.style.format({"realised share": "{:.4f}", "s.d. of count": "{:.2f}",
                        "Var/npq at n=1000": "{:.2f}", "excess kurtosis": "{:+.3f}",
                        "P(|z_naive|>3)": "{:.4f}", "vs the budgeted 0.0027": "{:.0f}x"}))
""")

# --------------------------------------------------------------------------- #
md(r"""
Three readings, and the third was not something we set out to find.

**1. The variance inflation has saturated.** Compare against the $n=48$ column of the first
table:

| | $\mathrm{Var}/npq$ at $n=48$ | at $n=1000$ |
|---|---|---|
| i.i.d. | 1.00 | 0.99 |
| order-1 $k=1$ | 2.62 | 2.54 |
| long $\beta=2.5$ | 5.10 | 4.93 |
| long $\beta=1.3$ | 6.90 | 6.61 |

Twenty times as many points and the inflation factor does not move. In the bare process
$\beta=1.3$ would have $\mathrm{Var}(S_n)/n$ growing like $n^{0.7}$ and by $n=1000$ it would
be enormous. Here it is stuck at $6.6$, because **the match boundary resets everything**. The
memory is real, it is large, and it is confined to a window of about 120 points. Long memory
that cannot outlive the horizon is, operationally, not long memory.

**2. There are still no fat tails.** Excess kurtosis is $-0.024$, $-0.038$, $-0.034$,
$-0.009$ — indistinguishable from Gaussian in all four regimes, including the one built with
infinite run variance. The right-hand panel looks alarming only because it is standardised by
the *naive binomial* s.d.: a nominal "3-sigma" event happens $28\%$ of the time at
$\beta=1.3$, a factor of **105** over the $0.27\%$ budgeted. But that is a Gaussian of the
wrong width, not a heavy-tailed law. Divide by the correct s.d. and the curves land on the
dashed line. This is the Part 4 conclusion surviving intact through the strongest memory we
can build inside a tennis match.

**3. The barrier taxes the favourite's mean.** Look at the realised share: $0.5503$, $0.5484$,
$0.5428$, $0.5332$. We calibrated $p=0.55$ exactly — but that calibration,
$p=\gamma_A/(\gamma_A+\gamma_B)$, is exact for the *unbounded* process. A's runs are longer
than B's by construction ($\mathbb{E}[L_A]=4.00$ against $\mathbb{E}[L_B]=3.27$), so A has
more mass beyond 53, and truncation costs A proportionally more. The barrier is an
**asymmetric tax on whoever has the longer streaks** — which is the favourite.

That is a third distinct mechanism working against the favourite, on top of the variance pump
of Part 6 and the Jensen gap of parameter uncertainty. It is also the most invisible of the
three: nothing in the specification hints at it, the calibration code is provably correct,
and the drift only appears once the process is run through the format it will actually be
played in.
""")

idx = next(i for i, c in enumerate(cells)
           if c.get("metadata", {}).get("tag") == "part7"
           and "BETAS = [3.5" in "".join(c["source"]))
cells[idx:idx] = new
nb["cells"] = cells
NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"inserted {len(new)} cells at index {idx}; total {len(cells)}")

import numpy as np
from tennis_lab import run_hazard, run_survival, hurst_from_beta

rng = np.random.default_rng(0)


def sample_run_length(beta, gamma):
    """Draw one run length by walking the hazard."""
    ell = 1
    while rng.random() >= run_hazard(ell, beta, gamma):
        ell += 1
    return ell


for beta in (1.5, 2.5):
    gamma = beta
    lengths = np.array([sample_run_length(beta, gamma) for _ in range(200_000)])
    print(f"\nbeta={beta}  gamma={gamma}  H={hurst_from_beta(beta):.3f}")
    print(f"  mean {lengths.mean():.3f}   max {lengths.max()}")
    print("    l   empirical P(L>l)   exact")
    for L in (1, 2, 5, 10, 50, 200):
        emp = (lengths > L).mean()
        print(f"  {L:>4}   {emp:>15.5f}   {float(run_survival(L, beta, gamma)):.5f}")
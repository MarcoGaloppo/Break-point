import numpy as np
from tennis_lab import TennisSimulator, run_hazard, run_survival

# 1. the inlined arithmetic must match the reference function
sim = TennisSimulator(p=0.55, memory="long", beta=1.5, gamma=2.0, seed=1)
for m in (1, 2, 7, 40):
    assert np.isclose(sim._p_next(m), 1.0 - run_hazard(m, sim.beta, sim._gamma_a))
    assert np.isclose(sim._p_next(-m), run_hazard(m, sim.beta, sim._gamma_b))
print("inlined hazard matches run_hazard")

# 2. the calibration must hit p, for every beta
for beta in (1.3, 1.5, 2.0, 2.5, 3.5):
    s = TennisSimulator(p=0.55, memory="long", beta=beta, gamma=4.0, seed=2)
    share = s.simulate_point_sequences(4_000, 4_000).mean() / 4_000
    print(f"  beta={beta}: mean share {share:.4f}  (target 0.5500)")

# 3. the payoff: Var(S_n)/n should DIVERGE, not converge
print("\nVar(S_n)/n against n  -- flat means short memory, rising means long")
for beta in (2.5, 1.5):
    s = TennisSimulator(p=0.5, memory="long", beta=beta, gamma=4.0, seed=3)
    row = []
    for n in (100, 400, 1600, 6400):
        c = s.simulate_point_sequences(20_000, n)
        row.append(f"n={n}: {c.var(ddof=1)/n:7.2f}")
    print(f"  beta={beta} (H={(3-beta)/2 if beta<2 else 0.5:.2f}):  " + "   ".join(row))
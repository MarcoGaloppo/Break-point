
# To make sure no commit/change done (although branch protected etc. etc.) touches what has been done before
import hashlib
import json

from tennis_lab import TennisSimulator

CASES = [
    dict(p=0.55, k=0.0, seed=1),
    dict(p=0.55, k=1.0, seed=2),
    dict(p=0.47, k=0.5, seed=3, best_of=5),
]

TRACKED = ("winner", "points_a", "total_points", "longest_run_a")


def fingerprint(n_matches=2000):
    out = {}
    for case in CASES:
        sim = TennisSimulator(**case)
        d = sim.simulate_matches(n_matches)
        key = ",".join(f"{k}={v}" for k, v in sorted(case.items()))
        out[key] = {
            name: [int(d[name].sum()), int(d[name].max())] for name in TRACKED
        }
    return out


if __name__ == "__main__":
    fp = fingerprint()
    print(json.dumps(fp, indent=2, sort_keys=True))
    digest = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()
    print("\nSHA256:", digest[:16])
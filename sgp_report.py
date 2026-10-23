"""Summarize sgp_validate.py output: calibration of joint probabilities and of correlation lift."""
import sys

import numpy as np
import pandas as pd

from features import PROC


def table(d, by, labels=None):
    g = d.groupby(by, observed=True).agg(n=("hit", "size"), pred_joint=("p_joint", "mean"),
                                         real_joint=("hit", "mean"), pred_indep=("p_indep", "mean"),
                                         pred_lift=("lift", "mean"))
    g["real_lift_vs_indep"] = g.real_joint / g.pred_indep
    g["joint_err"] = g.real_joint - g.pred_joint
    return g.round(4)


if __name__ == "__main__":
    f = sys.argv[1] if len(sys.argv) > 1 else "sgp_validation.csv.gz"
    d = pd.read_csv(PROC / f)
    d = d[~d.mechanical]
    print(f"{len(d):,} non-mechanical combos from {d.gameId.nunique()} games\n")
    d["pbin"] = pd.cut(d.p_joint, [0.04, 0.06, 0.09, 0.14, 0.22, 0.35, 1.0])
    print("== Calibration of predicted joint probability ==")
    print(table(d, "pbin").to_string())
    d["lbin"] = pd.cut(d.lift, [0, 0.9, 1.1, 1.3, 1.5, 1.7, 5])
    print("\n== Does predicted correlation lift show up? (real_lift_vs_indep should track pred_lift) ==")
    print(table(d, "lbin").to_string())
    big = d[d.p_joint >= 0.06]
    print("\n== By combo type (p_joint >= 6%) ==")
    t = table(big, "sig")
    print(t[t.n >= 2000].sort_values("pred_lift", ascending=False).to_string())
    top = big[big.lift >= 1.4]
    se = np.sqrt(top.hit.mean() * (1 - top.hit.mean()) / len(top))
    print(f"\nHigh-lift (>=1.4) plays: n={len(top):,} pred {top.p_joint.mean():.4f} real {top.hit.mean():.4f} +/- {se:.4f} (note: games overlap, so true error is larger)")

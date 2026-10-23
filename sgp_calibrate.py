"""Derive sgp_calibration.json (probability scale factors) and docs/data/sgp_validation.json
from the full backtest. Factors are fit on the last full season and checked on the current season
(held out), then refit on both for production."""
import json

import numpy as np
import pandas as pd

import sgp_model as M
from features import PROC
from seasons import CUR

EDGES = [0.04, 0.06, 0.09, 0.14, 0.22, 0.35, 1.0]


def fit_scale(d):
    out = {}
    for has_sog, g in d.groupby(d.sig.str.contains("sog")):
        rows = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            b = g[(g.p_joint > lo) & (g.p_joint <= hi)]
            k = b.hit.mean() / b.p_joint.mean() if len(b) > 500 else 1.0
            rows.append([lo, hi, float(np.clip(k, 0.75, 1.25))])
        out["sog" if has_sog else "nosog"] = rows
    return out


def apply(d, scale):
    k = np.ones(len(d))
    has = d.sig.str.contains("sog").values
    for key, mask in (("sog", has), ("nosog", ~has)):
        for lo, hi, f in scale[key]:
            k[mask & (d.p_joint.values > lo) & (d.p_joint.values <= hi)] = f
    return d.p_joint.values * k


def brier(p, y):
    return float(np.mean((p - y) ** 2))


if __name__ == "__main__":
    d = pd.read_csv(PROC / "sgp_validation.csv.gz")
    d = d[~d.mechanical].copy()
    season = d.gameId // 1_000_000
    tr, te = d[season == CUR - 1], d[season == CUR]
    sc = fit_scale(tr)
    for name, x in (("train (last season)", tr), ("held-out (current season)", te)):
        raw, cal = x.p_joint.values, apply(x, sc)
        print(f"{name}: n={len(x):,} games={x.gameId.nunique()}  Brier raw {brier(raw, x.hit):.5f} -> calibrated {brier(cal, x.hit):.5f}; "
              f"mean pred raw {raw.mean():.4f} cal {cal.mean():.4f} real {x.hit.mean():.4f}")
    final = fit_scale(d)
    sog_r = M.SOG_R
    out_path = PROC.parent.parent / "sgp_calibration.json"
    out_path.write_text(json.dumps(dict(sog_r=sog_r, p_scale=final["nosog"], p_scale_sog=final["sog"],
                                        fit_games=int(d.gameId.nunique())), indent=1))
    # public validation summary for the dashboard
    d["cal"] = apply(d, final)
    d["lbin"] = pd.cut(d.lift, [0, 0.9, 1.1, 1.3, 1.5, 1.7, 5], labels=["<0.9", "0.9-1.1", "1.1-1.3", "1.3-1.5", "1.5-1.7", "1.7+"])
    big = d[d.p_joint >= 0.06]
    lt = big.groupby("lbin", observed=True).agg(n=("hit", "size"), pred_joint=("p_joint", "mean"), real_joint=("hit", "mean"),
                                                 pred_indep=("p_indep", "mean"), pred_lift=("lift", "mean"))
    lt["real_lift"] = lt.real_joint / lt.pred_indep
    sig = big.groupby("sig").agg(n=("hit", "size"), pred_joint=("p_joint", "mean"), real_joint=("hit", "mean"),
                                 pred_indep=("p_indep", "mean"), pred_lift=("lift", "mean"))
    sig["real_lift"] = sig.real_joint / sig.pred_indep
    sig = sig[sig.n >= 5000].sort_values("pred_lift", ascending=False)
    (PROC.parent.parent / "docs" / "data" / "sgp_validation.json").write_text(json.dumps(dict(
        games=int(d.gameId.nunique()), combos=int(len(d)),
        by_lift=lt.round(4).reset_index().astype({"lbin": str}).to_dict("records"),
        by_type=sig.round(4).reset_index().to_dict("records")), indent=1))
    print(lt.round(3).to_string()); print(sig.round(3).head(8).to_string())

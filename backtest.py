"""Weighted-composite win model with out-of-sample backtests.

Composite = w1*z(xGF% diff) + w2*z(goalie GSAx diff) + w3*z(schedule diff);
win prob = sigmoid(a + c*composite). Scales (z), the schedule sub-model and (a, c)
are fitted on a training window that ends before the test window starts.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from features import PROC
from seasons import CUR, label

W = (0.75, 0.15, 0.10)


def sched_model(tr):
    X, y = tr[["d_b2b", "d_rest"]].values, tr.home_win.values
    return LogisticRegression(C=1e6).fit(X, y)


def fit(tr, w=W):
    p = {"sched": sched_model(tr)}
    raw = components_raw(tr, p)
    p["mu"], p["sd"] = raw.mean(0), raw.std(0)
    z = (raw - p["mu"]) / p["sd"]
    comp = z @ np.array(w)
    p["cal"] = LogisticRegression(C=1e6).fit(comp[:, None], tr.home_win.values)
    return p


def components_raw(df, p):
    return np.column_stack([
        df.d_xg.values, df.d_gsax.values,
        p["sched"].decision_function(df[["d_b2b", "d_rest"]].values) - p["sched"].intercept_[0]])


def predict(df, p, w=W):
    z = (components_raw(df, p) - p["mu"]) / p["sd"]
    return p["cal"].predict_proba((z @ np.array(w))[:, None])[:, 1]


def metrics(y, pr, name):
    pr = np.clip(pr, 1e-6, 1 - 1e-6)
    return dict(model=name, n=len(y), acc=((pr > .5) == y).mean(),
                logloss=-(y * np.log(pr) + (1 - y) * np.log(1 - pr)).mean(),
                brier=((pr - y) ** 2).mean())


def run(df, train_mask, test_mask, label):
    tr, te = df[train_mask], df[test_mask]
    y = te.home_win.values
    rows = []
    base = tr.home_win.mean()
    rows.append(metrics(y, np.full(len(te), base), "home-ice baseline"))
    p = fit(tr)
    pr = predict(te, p)
    rows.append(metrics(y, pr, "composite 75/15/10"))
    # free-weight logistic on the same z-components (what the data would choose)
    z_tr = (components_raw(tr, p) - p["mu"]) / p["sd"]
    free = LogisticRegression(C=1e6).fit(z_tr, tr.home_win.values)
    z_te = (components_raw(te, p) - p["mu"]) / p["sd"]
    rows.append(metrics(y, free.predict_proba(z_te)[:, 1], "free-weight logistic"))
    for name, w in [("xG only", (1, 0, 0)), ("goalie only", (0, 1, 0)), ("sched only", (0, 0, 1))]:
        pw = fit(tr, w)
        rows.append(metrics(y, predict(te, pw, w), name))
    res = pd.DataFrame(rows).set_index("model")
    print(f"\n=== {label}: train n={len(tr)}, test n={len(te)} ===")
    print(res.round(4).to_string())
    coef = free.coef_[0]
    print("free-weight shares (|coef|):", dict(zip(["xG", "GSAx", "sched"], (abs(coef) / abs(coef).sum()).round(2))))
    print("sched coefs [b2b, rest]:", p["sched"].coef_[0].round(3), "home-ice intercept:", round(p["sched"].intercept_[0], 3))
    # weight sensitivity
    sens = []
    for w1 in (0.4, 0.5, 0.6, 0.75, 0.9):
        rest = 1 - w1
        w = (w1, rest * 0.6, rest * 0.4)
        pw = fit(tr, w)
        m = metrics(y, predict(te, pw, w), f"w1={w1:.2f}")
        sens.append(m)
    print("\nxG-weight sensitivity (rest split 60/40 goalie/sched):")
    print(pd.DataFrame(sens).set_index("model").round(4).to_string())
    te = te.copy()
    te["p_home"] = pr
    return te


def calibration(te):
    te = te.copy()
    te["conf"] = np.maximum(te.p_home, 1 - te.p_home)
    te["pick_home"] = te.p_home > .5
    te["correct"] = te.pick_home == (te.home_win == 1)
    te["bucket"] = pd.cut(te.conf, [0.5, 0.55, 0.58, 0.62, 1.0])
    print("\nAccuracy by pick confidence:")
    print(te.groupby("bucket", observed=True).agg(n=("correct", "size"), acc=("correct", "mean"),
                                                   avg_conf=("conf", "mean")).round(3).to_string())


if __name__ == "__main__":
    df = pd.read_csv(PROC / "game_features.csv", parse_dates=["date"])
    # CUR-3 is feature burn-in only; train on CUR-2, test on the last full season (CUR-1),
    # then test the current season to date with CUR-2 and CUR-1 as training.
    tr1, last, cur = CUR - 2, CUR - 1, CUR
    te_last = run(df, df.season == tr1, df.season == last,
                  f"{label(last)} full season (train {label(tr1)})")
    calibration(te_last)
    te_cur = run(df, df.season.isin([tr1, last]), df.season == cur,
                 f"{label(cur)} season to date (train {label(tr1)}-{label(last)})")
    pd.concat([te_last, te_cur]).to_csv(PROC / "backtest_predictions.csv", index=False)

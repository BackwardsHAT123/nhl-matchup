"""Write dashboard data to docs/data/ (served by GitHub Pages).

predictions.json   upcoming slate (today + next 2 days) with the model's inputs
prediction_log.csv every pre-game prediction ever made (frozen once the day starts) -> live record
summary.json       live record, backtest metrics, calibration, cumulative accuracy, team table
Starter overrides: overrides.json  {"2026-10-20": {"TOR": "Woll"}}
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
import features as F
import predict
from seasons import CUR, label

ROOT = Path(__file__).parent
OUT = ROOT / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "prediction_log.csv"
LOG_COLS = ["gameId", "date", "home", "away", "p_home", "h_goalie", "a_goalie",
            "h_assumed", "a_assumed", "logged_at"]


def et_today():
    # NHL "game day" in US/Eastern; UTC-4/-5 approximated by UTC-5 shift is safe for a morning/noon run
    return (datetime.now(timezone.utc) - timedelta(hours=5)).date()


def slate():
    today = et_today()
    overrides = json.loads((ROOT / "overrides.json").read_text()) if (ROOT / "overrides.json").exists() else {}
    games = []
    for i in range(3):
        d = (today + timedelta(days=i)).isoformat()
        games += predict.compute_slate(d, overrides.get(d, {}))
    return today, games


def update_log(games, today):
    log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=LOG_COLS)
    now = datetime.now(timezone.utc).isoformat(timespec="minutes")
    new = pd.DataFrame([{**{k: g[k] for k in LOG_COLS[:9]}, "logged_at": now} for g in games])
    new["p_home"] = new.p_home.round(4)
    if len(new):
        # rows for days before today are frozen; today/future rows refresh as starters firm up
        frozen = log[log.date < today.isoformat()]
        keep = log[(log.date >= today.isoformat()) & ~log.gameId.isin(new.gameId)]
        log = pd.concat([frozen, keep, new]).drop_duplicates("gameId", keep="last")
    log.sort_values(["date", "gameId"]).to_csv(LOG, index=False)
    return log


def grade(log):
    res = pd.read_csv(F.PROC / "nhl_schedule_results.csv")
    res = res[res.state.isin(["OFF", "FINAL"])].copy()
    res["home_win"] = (res.home_score > res.away_score).astype(int)
    g = log.merge(res[["gameId", "home_win"]], on="gameId")
    p = g.p_home.clip(1e-6, 1 - 1e-6)
    g["correct"] = ((g.p_home > .5) == (g.home_win == 1)).astype(int)
    g["ll"] = -(g.home_win * np.log(p) + (1 - g.home_win) * np.log(1 - p))
    return g


def team_table():
    tg = pd.read_csv(F.PROC / "team_games.csv", parse_dates=["date"])
    rows = []
    for t, g in tg.groupby("team"):
        g = g.sort_values("date").tail(F.XG_WINDOW)
        f, a = g.xgf.sum(), g.xga.sum()
        rows.append(dict(team=t, xgf_pct=(f + F.XG_PRIOR) / (f + a + 2 * F.XG_PRIOR),
                         last_game=g.date.max().date().isoformat(), gp_window=len(g)))
    return sorted(rows, key=lambda r: -r["xgf_pct"])


def backtest_summary():
    bt = pd.read_csv(F.PROC / "backtest_predictions.csv", parse_dates=["date"])
    bt = bt.sort_values("date")
    y, p = bt.home_win.values, bt.p_home.clip(1e-6, 1 - 1e-6).values
    bt["correct"] = ((bt.p_home > .5) == (bt.home_win == 1)).astype(int)
    bt["conf"] = np.maximum(bt.p_home, 1 - bt.p_home)
    cal = bt.groupby(pd.cut(bt.conf, [.5, .55, .6, .65, 1.0]), observed=True).agg(
        n=("correct", "size"), acc=("correct", "mean"), conf=("conf", "mean")).reset_index(drop=True)
    cum = bt.assign(cum=bt.correct.expanding().mean(), n=np.arange(1, len(bt) + 1))
    cum = cum.iloc[::20][["date", "n", "cum"]]
    cum["date"] = cum.date.dt.date.astype(str)
    return dict(
        label=f"{label(CUR - 1)} + {label(CUR)}", n=len(bt), acc=float(bt.correct.mean()),
        logloss=float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        home_rate=float(bt.home_win.mean()),
        calibration=cal.round(4).to_dict("records"), cumulative=cum.round(4).to_dict("records"))


if __name__ == "__main__":
    today, games = slate()
    now = datetime.now(timezone.utc).isoformat(timespec="minutes")
    (OUT / "predictions.json").write_text(json.dumps(
        dict(generated_at=now, today=today.isoformat(), games=games), default=float, indent=1))
    log = update_log(games, today)
    g = grade(log)
    live = dict(n=len(g), acc=float(g.correct.mean()) if len(g) else None,
                logloss=float(g.ll.mean()) if len(g) else None,
                recent=g.sort_values("date").tail(30)[["date", "home", "away", "p_home", "home_win", "correct"]]
                .round(3).to_dict("records"))
    (OUT / "summary.json").write_text(json.dumps(
        dict(generated_at=now, live=live, backtest=backtest_summary(), teams=team_table()), indent=1))
    print(f"{len(games)} upcoming games, live record n={live['n']}, wrote {OUT}")

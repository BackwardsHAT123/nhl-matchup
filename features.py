"""Pre-game features for every game, using only information available before it.

Components (home minus away):
  xg    1. rolling 5v5 xGF% (last XG_WINDOW team games, carried across seasons, shrunk to 50%)
  gsax  2. starting goalie GSAx per start over last GS_WINDOW starts (shrunk to 0)
  sched 3. rest days / back-to-back (home ice is handled by the model intercept)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
PROC = ROOT / "data" / "processed"

XG_WINDOW = 40       # team games in the rolling xGF% window
XG_PRIOR = 20.0      # pseudo-xG added to each side (~8 games) for shrinkage to 50%
GS_WINDOW = 25       # goalie starts in the GSAx window (user spec: 20-30)
GS_PRIOR = 10.0      # pseudo-starts at 0 GSAx
REST_CAP = 4


def team_features(tg: pd.DataFrame) -> pd.DataFrame:
    tg = tg.sort_values(["team", "date", "gameId"]).copy()
    g = tg.groupby("team")
    roll = lambda c: g[c].transform(lambda s: s.rolling(XG_WINDOW, min_periods=1).sum().shift(1))
    f, a = roll("xgf"), roll("xga")
    tg["xgf_pct"] = (f.fillna(0) + XG_PRIOR) / (f.fillna(0) + a.fillna(0) + 2 * XG_PRIOR)
    prev = g["date"].shift(1)
    gap = (tg["date"] - prev).dt.days
    tg["rest_days"] = (gap - 1).clip(upper=REST_CAP).fillna(REST_CAP)
    tg["b2b"] = (tg.rest_days == 0).astype(int)
    return tg


def goalie_features(gg: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    gg = gg.merge(dates, on=["gameId", "team"]).sort_values(["goalie_id", "date", "gameId"])
    g = gg.groupby("goalie_id")["gsax"]
    s = g.transform(lambda x: x.rolling(GS_WINDOW, min_periods=1).sum().shift(1)).fillna(0)
    n = g.transform(lambda x: x.rolling(GS_WINDOW, min_periods=1).count().shift(1)).fillna(0)
    gg["gsax_per_start"] = s / (n + GS_PRIOR)
    gg["prior_starts"] = n
    return gg


def build() -> pd.DataFrame:
    tg = pd.read_csv(PROC / "team_games.csv", parse_dates=["date"])
    gg = pd.read_csv(PROC / "goalie_games.csv")
    tf = team_features(tg)
    gf = goalie_features(gg, tf[["gameId", "team", "date"]])
    tf = tf.merge(gf[["gameId", "team", "goalie_id", "goalie", "gsax_per_start", "prior_starts"]],
                  on=["gameId", "team"], how="left")
    h = tf[tf.home == 1].add_prefix("h_")
    a = tf[tf.home == 0].add_prefix("a_")
    m = h.merge(a, left_on="h_gameId", right_on="a_gameId")
    out = pd.DataFrame({
        "gameId": m.h_gameId, "season": m.h_season, "date": m.h_date,
        "home": m.h_team, "away": m.a_team,
        "home_win": m.h_win,
        "h_xgf_pct": m.h_xgf_pct, "a_xgf_pct": m.a_xgf_pct,
        "h_goalie": m.h_goalie, "a_goalie": m.a_goalie,
        "h_gsax": m.h_gsax_per_start, "a_gsax": m.a_gsax_per_start,
        "h_rest": m.h_rest_days, "a_rest": m.a_rest_days,
        "h_b2b": m.h_b2b, "a_b2b": m.a_b2b,
    }).dropna(subset=["h_gsax", "a_gsax"]).sort_values("date").reset_index(drop=True)
    # MoneyPuck goal totals exclude the shootout goal, so take winners from the NHL API
    res = pd.read_csv(PROC / "nhl_schedule_results.csv")
    res = res[res.state.isin(["OFF", "FINAL"])]
    res["home_win"] = (res.home_score > res.away_score).astype(int)
    out = out.drop(columns="home_win").merge(res[["gameId", "home_win"]], on="gameId")
    out["d_xg"] = (out.h_xgf_pct - out.a_xgf_pct) * 100          # percentage points
    out["d_gsax"] = out.h_gsax - out.a_gsax                       # goals per start
    out["d_b2b"] = out.h_b2b - out.a_b2b
    out["d_rest"] = out.h_rest - out.a_rest
    return out


if __name__ == "__main__":
    d = build()
    d.to_csv(PROC / "game_features.csv", index=False)
    print(d.groupby("season").size())
    print(d[["d_xg", "d_gsax", "d_b2b", "d_rest"]].describe().T)

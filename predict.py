"""Predict upcoming NHL games.

  python3 update_data.py                       # refresh MoneyPuck + NHL API data first
  python3 predict.py                           # today's slate
  python3 predict.py --date 2026-10-20 --goalie TOR="Joseph Woll" --goalie BOS=Swayman

Starters are NOT published by the NHL API ahead of game day, so unless you pass --goalie the tool
assumes each team's most-used recent starter (the other goalie if the primary played last night).
Confirm starters (morning skate / beat reports) and override; the model is only as good as that input.
"""
import argparse
import json
import urllib.request
from datetime import date, datetime

import numpy as np
import pandas as pd

import backtest
import features as F

PROC = F.PROC


def api(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def team_state(tg, team, asof):
    t = tg[(tg.team == team) & (tg.date < asof)].sort_values("date").tail(F.XG_WINDOW)
    f, a = t.xgf.sum(), t.xga.sum()
    pct = (f + F.XG_PRIOR) / (f + a + 2 * F.XG_PRIOR)
    last = t.date.max()
    rest = F.REST_CAP if pd.isna(last) else min((asof - last).days - 1, F.REST_CAP)
    return pct, rest, len(t)


def goalie_stats(gg, gid):
    h = gg[gg.goalie_id == gid].sort_values("date").tail(F.GS_WINDOW)
    return h.gsax.sum() / (len(h) + F.GS_PRIOR), len(h)


def pick_goalie(gg, team, asof, override):
    pool = gg[(gg.team == team) & (gg.date < asof)].sort_values("date")
    recent = pool.tail(10)
    if override:
        names = pool.drop_duplicates("goalie_id", keep="last")
        hit = names[names.goalie.str.lower().str.contains(override.lower())]
        if hit.empty:  # goalie may have changed teams / not started for this team yet
            hit = gg[gg.goalie.str.lower().str.contains(override.lower())].drop_duplicates("goalie_id", keep="last")
        if hit.empty:
            raise SystemExit(f"No goalie matching '{override}' in data")
        r = hit.iloc[-1]
        return int(r.goalie_id), r.goalie, False
    order = recent.goalie_id.value_counts().index.tolist()
    yday = pool.tail(1)
    played_yday = (not yday.empty) and (asof - yday.date.iloc[0]).days == 1
    gid = order[0]
    if played_yday and yday.goalie_id.iloc[0] == gid and len(order) > 1:
        gid = order[1]
    name = pool[pool.goalie_id == gid].goalie.iloc[-1]
    return int(gid), name, True


def compute_slate(day, over=None, w1=0.75):
    """Predictions for every regular-season game on `day` (YYYY-MM-DD). Returns list of dicts."""
    over = over or {}
    asof = pd.Timestamp(day)
    w2 = (1 - w1) * 0.6 if w1 != 0.75 else 0.15
    w = (w1, w2, 1 - w1 - w2)
    tg = pd.read_csv(PROC / "team_games.csv", parse_dates=["date"])
    gg = pd.read_csv(PROC / "goalie_games.csv").merge(tg[["gameId", "team", "date"]], on=["gameId", "team"])
    df = pd.read_csv(PROC / "game_features.csv", parse_dates=["date"])
    params = backtest.fit(df[df.season >= 2024], w)
    sched = api(f"https://api-web.nhle.com/v1/schedule/{day}")
    games = [g for d in sched["gameWeek"] if d["date"] == day for g in d["games"] if g["gameType"] == 2]
    out = []
    for g in games:
        h, a = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
        hx, hr, hn = team_state(tg, h, asof)
        ax, ar, an = team_state(tg, a, asof)
        hg = pick_goalie(gg, h, asof, over.get(h))
        ag = pick_goalie(gg, a, asof, over.get(a))
        hgs, _ = goalie_stats(gg, hg[0])
        ags, _ = goalie_stats(gg, ag[0])
        x = pd.DataFrame([{"d_xg": (hx - ax) * 100, "d_gsax": hgs - ags,
                           "d_b2b": int(hr == 0) - int(ar == 0), "d_rest": hr - ar}])
        out.append(dict(gameId=g["id"], date=day, start_utc=g.get("startTimeUTC"), home=h, away=a,
                        p_home=float(backtest.predict(x, params, w)[0]),
                        h_xg=hx, a_xg=ax, h_goalie=hg[1], a_goalie=ag[1],
                        h_assumed=hg[2], a_assumed=ag[2], h_gsax=hgs, a_gsax=ags,
                        h_rest=int(hr), a_rest=int(ar), weights=list(w)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--goalie", action="append", default=[], help='TEAM="name substring"')
    ap.add_argument("--w1", type=float, default=0.75, help="xGF%% weight (default 0.75)")
    args = ap.parse_args()
    games = compute_slate(args.date, dict(x.split("=", 1) for x in args.goalie), args.w1)
    if not games:
        raise SystemExit(f"No regular-season games on {args.date}")
    rows = []
    for r in games:
        p = r["p_home"]
        rows.append({
            "game": f"{r['away']} @ {r['home']}", "P(home)": round(p, 3),
            "pick": r["home"] if p >= .5 else r["away"], "conf": round(max(p, 1 - p), 3),
            "xGF% a/h": f"{r['a_xg']:.1%} / {r['h_xg']:.1%}",
            "starter a": f"{r['a_goalie']}{'*' if r['a_assumed'] else ''} ({r['a_gsax']:+.2f})",
            "starter h": f"{r['h_goalie']}{'*' if r['h_assumed'] else ''} ({r['h_gsax']:+.2f})",
            "rest a/h": f"{r['a_rest']}{' B2B' if r['a_rest'] == 0 else ''} / {r['h_rest']}{' B2B' if r['h_rest'] == 0 else ''}"})
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(f"\nNHL matchups {args.date}  weights xG/GSAx/sched = {tuple(round(x, 2) for x in games[0]['weights'])}")
    print(pd.DataFrame(rows).sort_values("conf", ascending=False).to_string(index=False))
    print("\n* = assumed starter (override with --goalie TEAM=name). GSAx per start in parens (last 25 starts, shrunk).")


if __name__ == "__main__":
    main()

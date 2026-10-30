"""Team + player stats for the dashboard's matchup drill-down -> docs/data/team_stats.json

Player stats: NHL API club-stats (current season, regular season). Team stats: record/GF/GA/last 5 from
official results, shots from MoneyPuck shot data, rolling xGF% from the main model's tables.
Each player carries league ranks (1 = best) for shots, goals, assists, points.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import odds
import predict
from features import PROC
import features as F
from seasons import CUR

ROOT = Path(__file__).parent
OUT = ROOT / "docs" / "data" / "team_stats.json"
STATS = ("shots", "goals", "assists", "points")
TOP_SKATERS = 12


def fetch(team):
    try:
        return team, predict.api(f"https://api-web.nhle.com/v1/club-stats/{team}/now")
    except Exception as e:
        print("club-stats failed", team, repr(e)[:80])
        return team, None


def team_summary():
    res = pd.read_csv(PROC / "nhl_schedule_results.csv", parse_dates=["date"])
    res = res[(res.season == CUR) & res.state.isin(["OFF", "FINAL"])]
    ts = pd.read_csv(PROC / "team_shots.csv")
    ts = ts[ts.gameId // 1_000_000 == CUR].set_index(["gameId", "team"])
    tg = pd.read_csv(PROC / "team_games.csv", parse_dates=["date"])
    out = {}
    for team in odds.TEAM_KEY:
        h = res[res.home == team].assign(gf=lambda d: d.home_score, ga=lambda d: d.away_score, opp=lambda d: d.away, venue="H")
        a = res[res.away == team].assign(gf=lambda d: d.away_score, ga=lambda d: d.home_score, opp=lambda d: d.home, venue="A")
        g = pd.concat([h, a]).sort_values("date")
        if g.empty:
            out[team] = dict(gp=0)
            continue
        win = g.gf > g.ga
        otl = (~win) & g.last_period.isin(["OT", "SO"])
        sf = [ts.loc[(i, team)].sog if (i, team) in ts.index else None for i in g.gameId]
        sa = []
        for i, o in zip(g.gameId, g.opp):
            sa.append(ts.loc[(i, o)].sog if (i, o) in ts.index else None)
        sf, sa = pd.Series(sf).dropna(), pd.Series(sa).dropna()
        t = tg[(tg.team == team) & (tg.season == tg.season.max())].sort_values("date").tail(F.XG_WINDOW)
        f, ag = t.xgf.sum(), t.xga.sum()
        out[team] = dict(
            gp=int(len(g)), w=int(win.sum()), l=int((~win & ~otl).sum()), otl=int(otl.sum()),
            gf_pg=round(g.gf.mean(), 2), ga_pg=round(g.ga.mean(), 2),
            sf_pg=round(sf.mean(), 1) if len(sf) else None, sa_pg=round(sa.mean(), 1) if len(sa) else None,
            xgf_pct=round((f + F.XG_PRIOR) / (f + ag + 2 * F.XG_PRIOR), 4),
            last5=[dict(r="W" if w else "L", s=f"{int(x)}-{int(y)}", opp=o, v=v, ot=bool(p in ("OT", "SO")))
                   for w, x, y, o, v, p in zip(win.tail(5)[::-1], g.gf.tail(5)[::-1], g.ga.tail(5)[::-1],
                                                g.opp.tail(5)[::-1], g.venue.tail(5)[::-1], g.last_period.tail(5)[::-1])])
    return out


def main():
    teams = {}
    for attempt in range(6):   # re-try only the teams that failed
        todo = [t for t in odds.TEAM_KEY if t not in teams]
        if not todo:
            break
        time.sleep(attempt)
        with ThreadPoolExecutor(3) as ex:
            for team, d in ex.map(fetch, todo):
                if d:
                    teams[team] = d
    rows = []
    for team, d in teams.items():
        for s in d["skaters"]:
            rows.append(dict(team=team, id=s["playerId"], name=f'{s["firstName"]["default"]} {s["lastName"]["default"]}',
                             pos=s["positionCode"], gp=s["gamesPlayed"], goals=s["goals"], assists=s["assists"],
                             points=s["points"], shots=s["shots"], toi=round(s.get("avgTimeOnIcePerGame", 0) / 60, 1)))
    df = pd.DataFrame(rows)
    for st in STATS:   # league rank, ties share the best rank
        df["rk_" + st] = df[st].rank(method="min", ascending=False).astype(int)
    summ = team_summary()
    out = {}
    for team, d in teams.items():
        p = df[df.team == team].sort_values(["points", "goals", "shots"], ascending=False).head(TOP_SKATERS)
        leaders = {}
        for st in STATS:
            top = df[df.team == team].sort_values([st, "points"], ascending=False).iloc[0]
            leaders[st] = dict(name=top["name"], value=int(top[st]), gp=int(top.gp), rank=int(top["rk_" + st]), id=int(top.id))
        goalies = [dict(id=g["playerId"], name=f'{g["firstName"]["default"]} {g["lastName"]["default"]}', gp=g["gamesPlayed"],
                        w=g["wins"], l=g["losses"], otl=g["overtimeLosses"],
                        sv=round(g["savePercentage"], 3), gaa=round(g["goalsAgainstAverage"], 2))
                   for g in sorted(d["goalies"], key=lambda g: -g["gamesPlayed"])[:2] if g["gamesPlayed"]]
        out[team] = dict(summary=summ.get(team, {}), leaders=leaders, goalies=goalies,
                         skaters=p.drop(columns=["team"]).to_dict("records"))
    toi = {str(r.id): r.toi for r in df.itertuples()}   # usual minutes/game for every skater (used by the results review)
    OUT.write_text(json.dumps(dict(generated_at=datetime.now(timezone.utc).isoformat(timespec="minutes"),
                                   season=CUR, teams=out, toi=toi), separators=(",", ":")))
    print("wrote", OUT, len(out), "teams,", OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()

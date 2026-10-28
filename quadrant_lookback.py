"""Lookback: how do matchups between xplot quadrants play out (winner, totals)?

Each game is labelled with point-in-time quadrants for both teams (only games before it), in two views like the xplot:
  full = season-to-date xGF% and the actual starter's season-to-date GSAx per start
  l5   = team's last 5 games xGF% and the starter's last 5 starts GSAx
Quadrant: xGF% vs 50%, GSAx vs 0. "MID" = inside the xplot diamond (|xGF%-50%|/5% + |GSAx|/0.5 <= 1).
The starter is the goalie who actually played (known after the fact), so this measures the signal, not a live pick.
No historical lines are on file, so totals are graded against fixed 5.5 and 6.5 (final score incl. OT/SO).

    python3 quadrant_lookback.py          # prints tables, writes data/processed/quadrant_lookback.csv
"""
import numpy as np
import pandas as pd

import features as F

MIN_TEAM_GP, MIN_STARTS = 5, 3
NAMES = {"TR": "Strong/Hot", "BR": "Strong/Cold", "BL": "Weak/Cold", "TL": "Weak/Hot", "MID": "Middle"}


def quad(xgf, gsax):
    if abs(xgf - 0.5) / 0.05 + abs(gsax) / 0.5 <= 1:
        return "MID"
    return ("T" if gsax >= 0 else "B") + ("R" if xgf >= 0.5 else "L")


def build():
    tg = pd.read_csv(F.PROC / "team_games.csv", parse_dates=["date"]).sort_values(["date", "gameId"])
    gg = pd.read_csv(F.PROC / "goalie_games.csv")
    tg = tg.merge(gg[["gameId", "team", "goalie_id", "gsax"]], on=["gameId", "team"], how="left")
    rows = []
    for team, t in tg.groupby("team"):
        t = t.reset_index(drop=True)
        for i, r in t.iterrows():
            prior = t.iloc[:i]
            cur = prior[prior.season == r.season]
            if len(cur) < MIN_TEAM_GP:
                continue
            st = prior[prior.goalie_id == r.goalie_id]
            st_cur = st[st.season == r.season]
            xg = lambda d: d.xgf.sum() / (d.xgf.sum() + d.xga.sum())
            rec = dict(gameId=r.gameId, team=team, season=r.season)
            if len(st_cur) >= MIN_STARTS:
                rec["full"] = quad(xg(cur), st_cur.gsax.mean())
            if len(st) >= MIN_STARTS:
                rec["l5"] = quad(xg(cur.tail(5)), st.tail(5).gsax.mean())
            rows.append(rec)
    q = pd.DataFrame(rows)
    sc = pd.read_csv(F.PROC / "nhl_schedule_results.csv")
    sc = sc[sc.state == "OFF"].copy()
    sc["total"] = sc.home_score + sc.away_score
    sc["home_win"] = (sc.home_score > sc.away_score).astype(int)
    out = sc[["gameId", "season", "date", "home", "away", "total", "home_win"]]
    for side, col in (("home", "h"), ("away", "a")):
        out = out.merge(q.rename(columns={"team": side, "full": f"{col}_full", "l5": f"{col}_l5"})[["gameId", side, f"{col}_full", f"{col}_l5"]],
                        on=["gameId", side], how="left")
    return out


def tally(g, view):
    d = g.dropna(subset=[f"h_{view}", f"a_{view}"]).copy()
    base_home, base5, base6 = d.home_win.mean(), (d.total > 5.5).mean(), (d.total > 6.5).mean()
    order = list(NAMES)
    out = []
    for i, a in enumerate(order):
        for b in order[i:]:
            # A = first-listed quadrant; orient so A's team is "A" regardless of venue
            m_ah = d[(d[f"h_{view}"] == a) & (d[f"a_{view}"] == b)]
            m_ha = d[(d[f"h_{view}"] == b) & (d[f"a_{view}"] == a)] if a != b else m_ah.iloc[0:0]
            n = len(m_ah) + len(m_ha)
            if not n:
                continue
            a_wins = m_ah.home_win.sum() + (len(m_ha) - m_ha.home_win.sum())
            tot = pd.concat([m_ah.total, m_ha.total])
            home_w = pd.concat([m_ah.home_win, m_ha.home_win]).mean()
            o5, o6 = (tot > 5.5).mean(), (tot > 6.5).mean()
            z = lambda p, p0: (p - p0) / np.sqrt(p0 * (1 - p0) / n)
            out.append(dict(view=view, A=a, B=b, n=n,
                            A_win=a_wins / n if a != b else np.nan, A_win_z=z(a_wins / n, .5) if a != b else np.nan,
                            home_win=home_w, home_z=z(home_w, base_home),
                            avg_total=tot.mean(), over55=o5, over55_z=z(o5, base5), over65=o6, over65_z=z(o6, base6)))
    return pd.DataFrame(out), (len(d), base_home, d.total.mean(), base5, base6)


def summary():
    """JSON-ready dict for the dashboard: per view, baseline rates and one row per quadrant pairing."""
    g, out = build(), {}
    for view in ("full", "l5"):
        t, (n, bh, bt, b5, b6) = tally(g, view)
        cells = [{k: (None if isinstance(v, float) and np.isnan(v) else (round(v, 3) if isinstance(v, float) else v))
                  for k, v in r.items() if k != "view"} for r in t.to_dict("records")]
        out[view] = dict(n=int(n), home_win=round(bh, 3), avg_total=round(bt, 2), over55=round(b5, 3), over65=round(b6, 3), cells=cells)
    return out


def main():
    g = build()
    allr = []
    for view, label in (("full", "SEASON TO DATE"), ("l5", "LAST 5")):
        t, (n, bh, bt, b5, b6) = tally(g, view)
        allr.append(t)
        print(f"\n=== {label}: {n} games | home win {bh:.1%} | avg total {bt:.2f} | over 5.5 {b5:.1%} | over 6.5 {b6:.1%}")
        t = t.assign(A=t.A.map(NAMES), B=t.B.map(NAMES))
        pd.set_option("display.width", 200)
        print(t.drop(columns="view").round(3).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    pd.concat(allr).to_csv(F.PROC / "quadrant_lookback.csv", index=False)


if __name__ == "__main__":
    main()

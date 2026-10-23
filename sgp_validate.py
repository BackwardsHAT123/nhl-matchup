"""Backtest the SGP simulator: do predicted joint probabilities and lifts match reality?

For every game in the last full season + current season (pre-game inputs only):
  simulate -> enumerate 2-3 leg combos -> compare predicted P(joint) with what happened.
Caveat: the lineup proxy is 'shot in the team's previous game'; scratched players count as 0 shots.
"""
import sys
import time

import numpy as np
import pandas as pd

import sgp_model as M
from features import PROC
from seasons import CUR

N_SIM = 6000


def estimate_sog_dispersion(st, seasons):
    """Residual variance of team SOG around the pre-game predicted mean -> NB size parameter."""
    res = []
    for r in st.t[st.t.gameId // 1_000_000.0 >= 0].itertuples():
        pass
    rows = []
    for team, d in st.by_team.items():
        d = d[d.gameId // 1_000_000 >= min(seasons)]
        for r in d.itertuples():
            tr, orr = st.team_rates(team, r.date), st.team_rates(r.opposingTeam, r.date)
            if tr["n"] < 10:
                continue
            mu = tr["sog_for"] + orr["sog_against"] - st.league_sog
            rows.append((mu, r.sog))
    a = np.array(rows)
    resid = a[:, 1] - a[:, 0]
    mu = a[:, 0].mean()
    var = resid.var()
    return mu * mu / max(var - mu, 1.0), var, mu


def actuals(st):
    res = pd.read_csv(PROC / "nhl_schedule_results.csv").set_index("gameId")
    ts = pd.read_csv(PROC / "team_shots.csv").set_index(["gameId", "team"])
    return res, ts


def realized_game(meta_legs, g, res, ts, pg_game):
    r = res.loc[g.gameId]
    hs, as_ = r.home_score, r.away_score
    so = (r.last_period == "SO")
    tot = hs + as_ - (1 if so else 0)
    home_win = hs > as_
    th, ta = ts.loc[(g.gameId, g.home)], ts.loc[(g.gameId, g.away)]
    saves = {"h": ta.sog - ta.goals, "a": th.sog - th.goals}
    out = {}
    for name, (_, m) in meta_legs.items():
        k = m["kind"]
        if k == "ml":
            v = home_win if m["side"] == "h" else not home_win
        elif k == "total":
            v = tot > m["line"] if m["dir"] == "over" else tot < m["line"]
        elif k in ("atg", "sog"):
            row = pg_game[(pg_game.player_id == m["player"])]
            sog, goals = (row.sog.iloc[0], row.goals.iloc[0]) if len(row) else (0, 0)
            v = goals >= 1 if k == "atg" else sog > m["line"]
        else:  # saves
            s = saves[m["side"]]
            v = s > m["line"] if m["dir"] == "over" else s < m["line"]
        out[name] = bool(v)
    return out


if __name__ == "__main__":
    st = M.Store()
    r_est, var, mu = estimate_sog_dispersion(st, [CUR - 2])
    print(f"SOG: mean {mu:.1f}, residual var {var:.1f} -> NB size R = {r_est:.1f} (using {M.SOG_R})")
    M.SOG_R = float(r_est)
    res, ts = actuals(st)
    lineups = M.load_lineups()
    bt = pd.read_csv(PROC / "backtest_predictions.csv", parse_dates=["date"])
    bt = bt[bt.season >= CUR - 1].sort_values("date")
    if len(sys.argv) > 1:
        bt = bt.iloc[::int(sys.argv[1])]
    sigs, rows, leg_rows, t0 = {}, [], [], time.time()
    for i, g in enumerate(bt.itertuples()):
        try:
            inp = M.build_inputs(st, g.home, g.away, g.date, g.p_home,
                                 st.prev_lineup(g.home, g.date, lineups), st.prev_lineup(g.away, g.date, lineups))
            if inp.players_h.empty or inp.players_a.empty:
                continue
            sim = M.simulate(inp, N_SIM, seed=i)
            legs = M.make_legs(inp, sim)
            df = M.rank_combos(legs)
            if df.empty:
                continue
            R = realized_game(legs, g, res, ts, st.pg_by_game[g.gameId])
            kinds = {n: m["kind"] for n, (_, m) in legs.items()}
            leg_rows += [(g.gameId, kinds[n], m.get("line"), float(a.mean()), R[n]) for n, (a, m) in legs.items()]
            df["hit"] = [all(R[n] for n in idx) for idx in df.idx]
            df["sig"] = ["+".join(sorted(kinds[n] for n in idx)) for idx in df.idx]
            df["gameId"] = g.gameId
            rows.append(df[["gameId", "sig", "p_joint", "p_indep", "lift", "hit", "mechanical"]])
        except Exception as e:
            print("skip", g.gameId, repr(e)[:80])
        if i % 100 == 0:
            print(i, len(bt), f"{time.time() - t0:.0f}s", flush=True)
    out = pd.concat(rows)
    out.to_parquet(PROC / "sgp_validation.parquet") if False else out.to_csv(PROC / "sgp_validation.csv.gz", index=False)
    pd.DataFrame(leg_rows, columns=["gameId", "kind", "line", "p", "hit"]).to_csv(PROC / "sgp_validation_legs.csv.gz", index=False)
    print("combos evaluated:", len(out))

"""Tune SHARE_SHRINK / check leg calibration on a game subsample (fast; no combo enumeration)."""
import sys
import numpy as np
import pandas as pd
import sgp_model as M
import sgp_validate as V
from features import PROC
from seasons import CUR


def run(kappa, step=5, n=2500, r=75.5):
    M.SHARE_SHRINK, M.SOG_R = kappa, r
    st = M.Store()
    res, ts = V.actuals(st)
    lineups = M.load_lineups()
    bt = pd.read_csv(PROC / "backtest_predictions.csv", parse_dates=["date"])
    bt = bt[bt.season >= CUR - 1].sort_values("date").iloc[::step]
    rows = []
    for i, g in enumerate(bt.itertuples()):
        inp = M.build_inputs(st, g.home, g.away, g.date, g.p_home,
                             st.prev_lineup(g.home, g.date, lineups), st.prev_lineup(g.away, g.date, lineups))
        if inp.players_h.empty or inp.players_a.empty:
            continue
        sim = M.simulate(inp, n, seed=i)
        legs = M.make_legs(inp, sim)
        R = V.realized_game(legs, g, res, ts, st.pg_by_game[g.gameId])
        for name, (a, m) in legs.items():
            rows.append((m["kind"], m.get("line"), m.get("dir"), float(a.mean()), R[name]))
    return pd.DataFrame(rows, columns=["kind", "line", "dir", "p", "hit"])


def ll(d):
    p = d.p.clip(.005, .995)
    return -(d.hit * np.log(p) + (1 - d.hit) * np.log(1 - p)).mean()


if __name__ == "__main__":
    for k in [float(x) for x in sys.argv[1:]] or [0.0, 0.2, 0.4]:
        d = run(k)
        a, s, t = d[d.kind == "atg"], d[d.kind == "sog"], d[d.kind == "total"]
        print(f"kappa={k}: ATG pred {a.p.mean():.3f} real {a.hit.mean():.3f} ll {ll(a):.4f} | "
              f"SOG pred {s.p.mean():.3f} real {s.hit.mean():.3f} ll {ll(s):.4f} | "
              f"total over5.5 pred {t[(t.dir=='over')&(t.line==5.5)].p.mean():.3f} real {t[(t.dir=='over')&(t.line==5.5)].hit.mean():.3f} "
              f"over6.5 pred {t[(t.dir=='over')&(t.line==6.5)].p.mean():.3f} real {t[(t.dir=='over')&(t.line==6.5)].hit.mean():.3f}", flush=True)

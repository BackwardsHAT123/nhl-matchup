"""Generate same-game-parlay candidates for upcoming games -> docs/data/sgp.json

Reads docs/data/predictions.json (main model's win probabilities), takes tonight's lineups from each
team's most recent boxscore, simulates each game, ranks 2-3 leg combos by correlation lift.
Calibration (from sgp_validate.py) lives in sgp_calibration.json and is applied to joint probabilities.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import predict
import sgp_model as M
from features import PROC
from seasons import CUR

ROOT = Path(__file__).parent
OUT = ROOT / "docs" / "data" / "sgp.json"
N_SIM = 20000
MIN_JOINT = 0.06      # skip long shots: tiny probabilities are model noise
TOP_PER_GAME = 8        # by lift; plus up to 6 likelier plays with lift >= MIN_LIFT_LIKELY
MIN_LIFT_LIKELY = 1.25


def load_calibration():
    f = ROOT / "sgp_calibration.json"
    return json.loads(f.read_text()) if f.exists() else {"sog_r": M.SOG_R, "p_scale": []}


def apply_scale(p, cal, has_sog):
    """Multiplicative correction of joint probability by predicted-probability bin."""
    for lo, hi, k in cal.get("p_scale_sog" if has_sog else "p_scale", []):
        if lo < p <= hi:
            return p * k
    return p


def last_lineup(res, team, asof, cache):
    r = res[((res.home == team) | (res.away == team)) & (res.date < asof)].sort_values("date")
    if r.empty:
        return None
    gid, side = int(r.iloc[-1].gameId), ("homeTeam" if r.iloc[-1].home == team else "awayTeam")
    if gid not in cache:
        cache[gid] = predict.api(f"https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore")["playerByGameStats"]
    t = cache[gid][side]
    return [p["playerId"] for p in t["forwards"] + t["defense"]]


def main():
    cal = load_calibration()
    M.SOG_R = cal.get("sog_r", M.SOG_R)
    pred = json.loads((ROOT / "docs" / "data" / "predictions.json").read_text())
    st = M.Store()
    res = pd.read_csv(PROC / "nhl_schedule_results.csv", parse_dates=["date"])
    res = res[(res.season == CUR) & res.state.isin(["OFF", "FINAL"])]
    cache, games = {}, []
    for i, g in enumerate(pred["games"]):
        asof = pd.Timestamp(g["date"])
        try:
            lh, la = last_lineup(res, g["home"], asof, cache), last_lineup(res, g["away"], asof, cache)
            inp = M.build_inputs(st, g["home"], g["away"], asof, g["p_home"], lh, la)
            if inp.players_h.empty or inp.players_a.empty:
                continue
            sim = M.simulate(inp, N_SIM, seed=i)
            legs = M.make_legs(inp, sim)
            df = M.rank_combos(legs, min_joint=MIN_JOINT)
        except Exception as e:  # one bad game must not kill the slate
            print("skip", g["home"], g["away"], repr(e)[:100])
            continue
        df = df[~df.mechanical]
        if df.empty:
            continue
        df["sig"] = ["+".join(sorted(legs[n][1]["kind"] for n in idx)) for idx in df.idx]
        df["p_cal"] = [apply_scale(p, cal, "sog" in sg) for p, sg in zip(df.p_joint, df.sig)]
        df["lift_cal"] = df.p_cal / df.p_indep
        by_lift = df.sort_values("lift_cal", ascending=False).head(TOP_PER_GAME)
        likely = df[df.lift_cal >= MIN_LIFT_LIKELY].sort_values("p_cal", ascending=False).head(6)
        top = pd.concat([by_lift, likely]).drop_duplicates(subset=["p_joint", "p_indep"])
        plays = [dict(legs=r.legs, leg_probs=[round(x, 3) for x in r.leg_probs], p_joint=round(r.p_cal, 4),
                      p_raw=round(r.p_joint, 4), p_indep=round(r.p_indep, 4), lift=round(r.lift_cal, 3),
                      fair_odds=M.american(r.p_cal), sig=r.sig) for r in top.itertuples()]
        games.append(dict(gameId=g["gameId"], date=g["date"], home=g["home"], away=g["away"],
                          start_utc=g["start_utc"], p_home=round(g["p_home"], 4),
                          exp_total=round(float(sim["total"].mean()), 2), plays=plays))
        print(f"{g['away']} @ {g['home']}: {len(df)} combos, best lift {top.lift_cal.max():.2f}", flush=True)
    OUT.write_text(json.dumps(dict(generated_at=datetime.now(timezone.utc).isoformat(timespec="minutes"),
                                   min_joint=MIN_JOINT, n_sim=N_SIM, games=games), indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()

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

import odds
import predict
import sgp_model as M
import sgp_pricing as P
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


def play_specs(legs, idx):
    """Machine-readable leg definitions (for grading after the game)."""
    out = []
    for n in idx:
        m = legs[n][1]
        d = {}
        for k in ("kind", "team", "side", "player", "name", "line", "dir"):
            v = m.get(k)
            if v is not None:
                d[k] = int(v) if k == "player" else float(v) if k == "line" else v
        out.append(d)
    return out


def build_game(g, i, st, res, cache, cal, books=None, gmap=None):
    """One game's parlay candidates. Returns (entry | None, n_priced). `g` is a predict.compute_slate row."""
    books = books or {}
    asof = pd.Timestamp(g["date"])
    starters = {"h": odds.norm(g["h_goalie"]), "a": odds.norm(g["a_goalie"])}
    try:
        lh, la = last_lineup(res, g["home"], asof, cache), last_lineup(res, g["away"], asof, cache)
        inp = M.build_inputs(st, g["home"], g["away"], asof, g["p_home"], lh, la)
        if inp.players_h.empty or inp.players_a.empty:
            return None, 0
        sim = M.simulate(inp, N_SIM, seed=i)
        extra = None
        if books:
            names = [odds.norm(n) for pl in (inp.players_h, inp.players_a) for n in pl.name]
            extra = P.extra_lines(books, g["home"], g["away"], names, gmap or {}, starters)
        legs = M.make_legs(inp, sim, extra=extra)
        df = M.rank_combos(legs, min_joint=MIN_JOINT)
    except Exception as e:  # one bad game must not kill the slate
        print("skip", g["home"], g["away"], repr(e)[:100])
        return None, 0
    df = df[~df.mechanical]
    if df.empty:
        return None, 0
    df["sig"] = ["+".join(sorted(legs[n][1]["kind"] for n in idx)) for idx in df.idx]
    df["p_cal"] = [apply_scale(p, cal, "sog" in sg) for p, sg in zip(df.p_joint, df.sig)]
    df["lift_cal"] = df.p_cal / df.p_indep
    by_lift = df.sort_values("lift_cal", ascending=False).head(TOP_PER_GAME)
    likely = df[df.lift_cal >= MIN_LIFT_LIKELY].sort_values("p_cal", ascending=False).head(6)
    parts = [by_lift, likely]
    mk = {}
    if books:   # price every correlated candidate, then also surface the best market EVs
        cand = df[(df.lift_cal >= 1.15) & (df.p_cal >= MIN_JOINT)]
        for r in cand.itertuples():
            pr = P.price_play(books, [legs[n][1] for n in r.idx], r.p_cal, gmap, starters, r.lift_cal)
            if pr:
                mk[r.Index] = pr
        if mk:
            parts.append(df.loc[sorted(mk, key=lambda k: -mk[k]["f_star"])[:6]])
    top = pd.concat(parts)
    top = top[~top.index.duplicated()]
    plays, n_priced = [], 0
    for r in top.itertuples():
        play = dict(legs=r.legs, leg_probs=[round(x, 3) for x in r.leg_probs], p_joint=round(r.p_cal, 4),
                    p_raw=round(r.p_joint, 4), p_indep=round(r.p_indep, 4), lift=round(r.lift_cal, 3),
                    fair_odds=M.american(r.p_cal), sig=r.sig, specs=play_specs(legs, r.idx))
        pr = mk.get(r.Index) or (P.price_play(books, [legs[n][1] for n in r.idx], r.p_cal, gmap, starters, r.lift_cal) if books else None)
        if pr:
            play["market"] = dict(book=pr["book"], naive_odds=M.american(1 / pr["dec"]) if pr["dec"] > 1 else None,
                                  naive_dec=round(pr["dec"], 2), ev=round(pr["ev"], 4), ev_half=round(pr["ev_half"], 4),
                                  ev_full=round(pr["ev_full"], 4), f_star=round(pr["f_star"], 3),
                                  n_books=pr["n_books"], leg_mkt_p=[None if x is None else round(x, 3) for x in pr["mkt_p"]])
            n_priced += 1
        plays.append(play)
    lineup = lambda pl: [[int(r.player_id), r.name, round(float(r.share), 4)] for r in pl.itertuples()]
    entry = dict(gameId=g["gameId"], date=g["date"], home=g["home"], away=g["away"],
                 start_utc=g["start_utc"], p_home=round(g["p_home"], 4), priced=bool(books),
                 exp_total=round(float(sim["total"].mean()), 2), plays=plays,
                 starters=dict(h=g["h_goalie"], a=g["a_goalie"], h_assumed=bool(g["h_assumed"]), a_assumed=bool(g["a_assumed"])),
                 exp_sog=dict(h=round(inp.mu_sog_h, 1), a=round(inp.mu_sog_a, 1)),
                 lineup_h=lineup(inp.players_h), lineup_a=lineup(inp.players_a))
    print(f"{g['away']} @ {g['home']}: {len(df)} combos, best lift {top.lift_cal.max():.2f}"
          f"{', priced ' + str(len(mk)) if books else ''}", flush=True)
    return entry, n_priced


def public(entry):
    """Dashboard version: drop grading-only fields."""
    e = {k: v for k, v in entry.items() if k not in ("lineup_h", "lineup_a", "starters", "exp_sog")}
    e["plays"] = [{k: v for k, v in p.items() if k != "specs"} for p in entry["plays"]]
    return e


def main():
    cal = load_calibration()
    M.SOG_R = cal.get("sog_r", M.SOG_R)
    pred = json.loads((ROOT / "docs" / "data" / "predictions.json").read_text())
    st = M.Store()
    res = pd.read_csv(PROC / "nhl_schedule_results.csv", parse_dates=["date"])
    res = res[(res.season == CUR) & res.state.isin(["OFF", "FINAL"])]
    market = odds.fetch_slate()
    gmap = P.goalie_team_map() if market else {}
    cache, games, n_priced = {}, [], 0
    for i, g in enumerate(pred["games"]):
        entry, n = build_game(g, i, st, res, cache, cal, market.get((g["away"], g["home"])), gmap)
        if entry:
            games.append(entry)
            n_priced += n
    meta = dict(generated_at=datetime.now(timezone.utc).isoformat(timespec="minutes"),
                min_joint=MIN_JOINT, n_sim=N_SIM, odds_games=len(market), priced_plays=n_priced)
    (PROC / "sgp_full.json").write_text(json.dumps(dict(meta, games=games)))   # for review.py (logging/grading)
    OUT.write_text(json.dumps(dict(meta, games=[public(e) for e in games]), indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()

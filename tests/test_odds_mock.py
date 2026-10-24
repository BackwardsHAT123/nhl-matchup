"""End-to-end check of the odds -> pricing -> slate path using a mock response in The Odds API format."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

import odds
import sgp_model as M
import sgp_pricing as P
import sgp_slate



def mock_event(inp, sim, home_full, away_full):
    """Prices = model prob with a 7% margin, in the documented response shape."""
    legs = M.make_legs(inp, sim)
    dec = lambda p: round(1 / (p * 1.07), 3)
    def p_of(label): return legs[label][0].mean()
    out = []
    for bk, jitter in (("bookA", 1.0), ("bookB", 0.97)):
        mk = []
        hp, ap = p_of(f"{inp.home} ML"), p_of(f"{inp.away} ML")
        mk.append(dict(key="h2h", outcomes=[dict(name=home_full, price=dec(hp) * jitter), dict(name=away_full, price=dec(ap) * jitter)]))
        mk.append(dict(key="totals", outcomes=[
            dict(name="Over", point=5.5, price=dec(p_of("Over 5.5")) * jitter), dict(name="Under", point=5.5, price=dec(p_of("Under 5.5")) * jitter),
            dict(name="Over", point=6.0, price=1.95), dict(name="Under", point=6.0, price=1.95)]))
        atg, sog = [], []
        for pl in (inp.players_h, inp.players_a):
            for r in pl.assign(eg=pl.share * pl.q).sort_values("eg", ascending=False).head(5).itertuples():
                atg.append(dict(name="Yes", description=r.name, price=dec(p_of(f"{r.name} anytime goal")) * jitter))
                for line in (1.5, 2.5):
                    lab = f"{r.name} over {line} SOG"
                    if lab in legs:
                        p = p_of(lab)
                        sog.append(dict(name="Over", description=r.name, point=line, price=dec(p) * jitter))
                        sog.append(dict(name="Under", description=r.name, point=line, price=dec(1 - p)))
        mk += [dict(key="player_goal_scorer_anytime", outcomes=atg), dict(key="player_shots_on_goal", outcomes=sog)]
        out.append(dict(key=bk, title=bk.upper(), markets=mk))
    return dict(bookmakers=out)


def test_pipeline():
    pred = json.loads((sgp_slate.ROOT / "docs" / "data" / "predictions.json").read_text())
    g = next(x for x in pred["games"] if x["home"] == "CHI" or True)
    st = M.Store()
    asof = pd.Timestamp(g["date"])
    inp = M.build_inputs(st, g["home"], g["away"], asof, g["p_home"])
    sim = M.simulate(inp, 6000, seed=1)
    ev = mock_event(inp, sim, *[x for x in (odds_full(g["home"]), odds_full(g["away"]))])
    books = odds.parse_event(ev)
    assert set(books) == {"bookA", "bookB"}
    assert books["bookA"]["ml"][g["home"]] > 1 and 5.5 in books["bookA"]["totals"]
    legs = M.make_legs(inp, sim, extra=P.extra_lines(books, g["home"], g["away"],
                       [odds.norm(n) for pl in (inp.players_h, inp.players_a) for n in pl.name], {}, {}))
    df = M.rank_combos(legs)
    df = df[~df.mechanical]
    n, last = 0, None
    for r in df.sort_values("lift", ascending=False).head(300).itertuples():
        pr = P.price_play(books, [legs[x][1] for x in r.idx], r.p_joint, {}, {}, r.lift)
        if pr:
            n, last = n + 1, pr
            assert pr["n_books"] in (1, 2) and pr["dec"] > 1 and -1 < pr["ev"] < 50 and pr["ev_full"] <= pr["ev_half"] <= pr["ev"] and -1 <= pr["f_star"] <= 2
    print("priced", n, "of 300 combos; example EV", round(last["ev"], 3), "f*", round(last["f_star"], 2), last["book"])
    assert n > 0


def odds_full(abbr):
    return {"CHI": "Chicago Blackhawks", "MTL": "Montréal Canadiens"}.get(abbr) or next(
        (f"X {k.title()}" for a, k in odds.TEAM_KEY.items() if a == abbr))


if __name__ == "__main__":
    test_pipeline()
    print("ok")

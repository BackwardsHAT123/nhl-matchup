"""Attach market prices (from odds.py) to simulated parlay legs.

The Odds API only gives single-leg prices. For each bookmaker that quotes every leg we multiply the
leg prices ("naive" parlay price). A real SGP pays *less* than this because books discount correlated
legs, so EV vs the naive price is an UPPER bound; `cushion` is how big a payout haircut the book can
apply before the play stops being +EV by our model.
"""
import numpy as np
import pandas as pd

from features import PROC
from odds import fair_prob, norm


def goalie_team_map():
    """normalised goalie name -> team of his most recent start."""
    gg = pd.read_csv(PROC / "goalie_games.csv")
    tg = pd.read_csv(PROC / "team_games.csv", parse_dates=["date"])[["gameId", "team", "date"]]
    gg = gg.merge(tg, on=["gameId", "team"]).sort_values("date").drop_duplicates("goalie_id", keep="last")
    return {norm(r.goalie): r.team for r in gg.itertuples()}


def book_goalies(book, team, gmap, prefer=None):
    """Goalie keys in this book's saves market that belong to `team` (assumed starter first)."""
    c = [g for g in book["saves"] if gmap.get(g) == team]
    return sorted(c, key=lambda g: g != prefer)


def extra_lines(books, home, away, names, gmap, starters):
    """Lines the books post that the default leg grid lacks -> make_legs(extra=...)."""
    totals, sog = set(), {}
    saves = {"h": set(), "a": set()}
    for b in books.values():
        totals |= {pt for pt in b["totals"] if pt % 1 == 0.5}
        for n in names:
            sog.setdefault(n, set()).update(b["sog"].get(n, {}).keys())
        for side, team in (("h", home), ("a", away)):
            gs = book_goalies(b, team, gmap, starters.get(side))
            if gs:
                saves[side] |= set(b["saves"][gs[0]].keys())
    # saves lines far from the market main line add noise: keep the 3 most central
    for side in saves:
        s = sorted(saves[side])
        saves[side] = s[len(s) // 2 - 1: len(s) // 2 + 2] if len(s) > 3 else s
    return {"totals": sorted(totals), "sog": {k: sorted(v) for k, v in sog.items()}, "saves": {k: sorted(v) for k, v in saves.items()}}


def leg_price(book, meta, gmap, starters):
    """(decimal price, vig-free-or-raw market probability) for one leg at one book, or None."""
    k = meta["kind"]
    if k == "ml":
        p = book["ml"].get(meta["team"])
        return (p, 1 / p) if p else None
    if k == "total":
        t = book["totals"].get(meta["line"])
        if not t or meta["dir"] not in t:
            return None
        other = t.get("under" if meta["dir"] == "over" else "over")
        pr = t[meta["dir"]]
        return pr, fair_prob(pr, other)
    if k == "atg":
        p = book["atg"].get(norm(meta["name"]))
        return (p, 1 / p) if p else None
    if k in ("sog", "saves"):
        if k == "sog":
            table = book["sog"].get(norm(meta["name"]), {})
            d = "over"
        else:
            gs = book_goalies(book, meta["team"], gmap, starters.get(meta["side"]))
            table = book["saves"].get(gs[0], {}) if gs else {}
            d = meta["dir"]
        t = table.get(meta["line"])
        if not t or d not in t:
            return None
        other = t.get("under" if d == "over" else "over")
        pr = t[d]
        p_over = fair_prob(t.get("over"), t.get("under")) if t.get("over") else None
        return pr, (p_over if d == "over" else (1 - p_over if p_over is not None else None))
    return None


def price_play(books, metas, p_model, gmap, starters, lift=1.0):
    """Best bookmaker (highest multiplied price among books quoting all legs) or None."""
    best, n_books = None, 0
    for key, b in books.items():
        legs = [leg_price(b, m, gmap, starters) for m in metas]
        if any(x is None for x in legs):
            continue
        n_books += 1
        dec = float(np.prod([x[0] for x in legs]))
        if best is None or dec > best["dec"]:
            best = dict(book=b["title"], dec=dec, mkt_p=[x[1] for x in legs])
    if best is None:
        return None
    # Scenario EVs: the book's SGP price = legs multiplied / lift^f, where f is the share of the correlation
    # it prices in. f=0 is the naive upper bound; f=1 means it prices our full lift (EV then = leg edges only).
    best["ev"] = p_model * best["dec"] - 1
    L = max(lift, 1.0001)
    best["ev_half"] = p_model * best["dec"] / L ** 0.5 - 1
    best["ev_full"] = p_model * best["dec"] / L - 1
    # largest share of the correlation the book can price before EV hits zero (capped at 2 = "robust")
    best["f_star"] = float(np.clip(np.log(max(1 + best["ev"], 1e-6)) / np.log(L), -1, 2))
    best["n_books"] = n_books
    return best

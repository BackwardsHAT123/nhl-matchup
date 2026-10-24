"""Predict each team's starting goalie.

Candidates = goalies who started for the team in its last WINDOW games. Features capture usage, tandem
alternation, rest and back-to-backs; a classifier scores each candidate and the best-scored one is the
guess (scores are normalised to probabilities within the team's candidates).

  python3 starters.py            # backtest vs. the old rule (rows: season, accuracy)
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from features import PROC
from seasons import CUR

WINDOW = 30
FEATS = ["n5", "n10", "n20", "gap", "days_gap", "started_prev", "started_prev2", "consec", "rest", "plays_tomorrow",
         "prev_x_b2b", "prev_x_tomorrow", "share40", "is_top10", "is_top20",
         "gp_season", "s_this", "s_this_share", "dense", "last_gsax", "last_pulled"]
MODEL_FILE = PROC / "starter_model.pkl"


def team_history():
    """{team: DataFrame(date, gameId, starter_id, starter)} sorted by date."""
    tg = pd.read_csv(PROC / "team_games.csv", parse_dates=["date"])[["gameId", "team", "date"]]
    gg = pd.read_csv(PROC / "goalie_games.csv")[["gameId", "team", "goalie_id", "goalie", "gsax", "sa"]]
    h = tg.merge(gg, on=["gameId", "team"], how="left").sort_values(["team", "date"])
    return {t: d.reset_index(drop=True) for t, d in h.groupby("team")}


def cand_rows(dates, starters, i, rest, plays_tomorrow, asof=None, season=None, gsax=None, sa=None):
    """Candidate feature rows for predicting game i from history before it. Returns (ids, X)."""
    asof = dates[i] if asof is None else asof
    win = starters[max(0, i - WINDOW):i]
    if len(win) == 0:
        return [], np.zeros((0, len(FEATS)))
    ids = [g for g in dict.fromkeys(win[::-1]) if g == g]            # most recent first, drop NaN
    last5, last10, last20, last40 = win[-5:], win[-10:], win[-20:], starters[max(0, i - 40):i]
    cnt = lambda arr, g: sum(1 for x in arr if x == g)
    gp_season = sum(1 for j in range(i) if season is not None and season[j] == season[i]) if season is not None else 0
    dense = sum(1 for j in range(max(0, i - 4), i) if (asof - dates[j]).days <= 6)   # games in the last 6 days
    top10 = max(set(last10), key=lambda g: cnt(last10, g)) if len(last10) else None
    top20 = max(set(last20), key=lambda g: cnt(last20, g)) if len(last20) else None
    rows = []
    for g in ids:
        idx = [j for j in range(max(0, i - WINDOW), i) if starters[j] == g]
        gap = i - idx[-1]
        consec = 0
        for j in range(i - 1, max(-1, i - 12), -1):
            if starters[j] == g:
                consec += 1
            else:
                break
        sp = float(gap == 1)
        s_this = sum(1 for j in idx if season is not None and season[j] == season[i])
        lj = idx[-1]
        last_gsax = float(gsax[lj]) if gsax is not None and gsax[lj] == gsax[lj] else 0.0
        last_pulled = float(sa is not None and sa[lj] == sa[lj] and sa[lj] < 15)    # few shots faced -> pulled early
        rows.append([cnt(last5, g), cnt(last10, g), cnt(last20, g), min(gap, 10),
                     min((asof - dates[idx[-1]]).days, 21), sp, float(i >= 2 and starters[i - 2] == g), consec,
                     rest, plays_tomorrow, sp * float(rest == 0), sp * plays_tomorrow,
                     cnt(last40, g) / max(len(last40), 1), float(g == top10), float(g == top20),
                     min(gp_season, 30), s_this, s_this / max(gp_season, 1), dense, last_gsax, last_pulled])
    return ids, np.array(rows, dtype=float)


def build_training(hist):
    X, y, meta = [], [], []
    for team, h in hist.items():
        dates = list(h.date)
        starters = list(h.goalie_id)
        season, gsax, sa = list(h.gameId // 1_000_000), list(h.gsax), list(h.sa)
        for i in range(1, len(h)):
            if starters[i] != starters[i]:      # unknown actual starter
                continue
            rest = min((dates[i] - dates[i - 1]).days - 1, 4)
            tmrw = float(i + 1 < len(h) and (dates[i + 1] - dates[i]).days == 1)
            ids, F = cand_rows(dates, starters, i, rest, tmrw, season=season, gsax=gsax, sa=sa)
            if not len(ids):
                continue
            for g, row in zip(ids, F):
                X.append(row)
                y.append(int(g == starters[i]))
                meta.append((team, i, h.gameId[i], dates[i], g, starters[i]))
    return np.array(X), np.array(y), pd.DataFrame(meta, columns=["team", "i", "gameId", "date", "cand", "actual"])


def old_rule(dates, starters, i, rest):
    """The previous heuristic: most-used goalie over last 10 games, other if he started yesterday on a B2B."""
    win = [g for g in starters[max(0, i - 10):i] if g == g]
    if not win:
        return None
    order = [g for g, _ in pd.Series(win).value_counts().items()]
    gid = order[0]
    if rest == 0 and starters[i - 1] == gid and len(order) > 1:
        gid = order[1]
    return gid


def accuracy(meta, score):
    m = meta.assign(s=score)
    best = m.loc[m.groupby(["team", "i"]).s.idxmax()]
    truth = m.drop_duplicates(["team", "i"])[["team", "i", "gameId", "date", "actual"]]
    out = truth.merge(best[["team", "i", "cand"]], on=["team", "i"], how="left")
    out["ok"] = out.cand == out.actual
    return out


def fit(X, y):
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=250, l2_regularization=1.0, random_state=0)
    return gb.fit(X, y)


_STATE = {}


def _prepare(cutoff=None):
    """Train once per process on history (optionally only games before `cutoff`, for honest backtests)."""
    key = str(cutoff)
    if key in _STATE:
        return _STATE[key]
    hist = team_history()
    X, y, meta = build_training(hist)
    keep = (meta.date < pd.Timestamp(cutoff)).values if cutoff is not None else np.ones(len(meta), bool)
    _STATE[key] = (hist, fit(X[keep], y[keep]))
    return _STATE[key]


def team_context(team, asof):
    """(rest days, plays_tomorrow) from the played history and the published schedule."""
    res = pd.read_csv(PROC / "nhl_schedule_results.csv", parse_dates=["date"])
    res = res[(res.season == CUR) & ((res.home == team) | (res.away == team))]
    past = res[res.date < asof]
    rest = min((asof - past.date.max()).days - 1, 4) if len(past) else 4
    tomorrow = float(((res.date - asof).dt.days == 1).any())
    return rest, tomorrow


def starter_probs(team, asof, roster_ids=None, cutoff=None):
    """[(goalie_id, name, prob)] best guess first. roster_ids (optional) drops goalies who left the club."""
    hist, model = _prepare(cutoff)
    asof = pd.Timestamp(asof)
    h = hist[team]
    h = h[h.date < asof]
    if h.empty:
        return []
    rest, tmrw = team_context(team, asof)
    dates, starters = list(h.date) + [asof], list(h.goalie_id) + [np.nan]
    season = list(h.gameId // 1_000_000) + [CUR]
    gsax, sa = list(h.gsax) + [np.nan], list(h.sa) + [np.nan]
    ids, F = cand_rows(dates, starters, len(h), rest, tmrw, asof=asof, season=season, gsax=gsax, sa=sa)
    if not len(ids):
        return []
    if roster_ids:
        keep = [k for k, g in enumerate(ids) if int(g) in roster_ids]
        if keep:
            ids, F = [ids[k] for k in keep], F[keep]
    p = model.predict_proba(F)[:, 1]
    p = p / p.sum()
    names = {int(r.goalie_id): r.goalie for r in h.dropna(subset=["goalie_id"]).itertuples()}
    return sorted([(int(g), names[int(g)], float(q)) for g, q in zip(ids, p)], key=lambda x: -x[2])


if __name__ == "__main__":
    hist = team_history()
    X, y, meta = build_training(hist)
    season = meta.gameId // 1_000_000
    print(f"{meta.drop_duplicates(['team', 'i']).shape[0]} team-games, {len(meta)} candidate rows")
    # old rule baseline over every team-game
    base = []
    for team, h in hist.items():
        dates, st = list(h.date), list(h.goalie_id)
        for i in range(1, len(h)):
            if st[i] != st[i]:
                continue
            rest = min((dates[i] - dates[i - 1]).days - 1, 4)
            base.append((int(h.gameId[i]) // 1_000_000, old_rule(dates, st, i, rest) == st[i]))
    base = pd.DataFrame(base, columns=["season", "ok"])
    print("\nold rule accuracy by season:", base.groupby("season").ok.mean().round(3).to_dict())
    for test_season in (CUR - 1, CUR):
        tr = (season < test_season) & (season >= CUR - 3)
        te = season == test_season
        for name, mdl in (("logistic", LogisticRegression(C=1.0, max_iter=2000)), ("boosted", None)):
            m = mdl.fit(X[tr], y[tr]) if mdl is not None else fit(X[tr], y[tr])
            r = accuracy(meta[te].reset_index(drop=True), m.predict_proba(X[te])[:, 1])
            print(f"test {test_season}: {name:9s} accuracy {r.ok.mean():.3f} on {len(r)} team-games")

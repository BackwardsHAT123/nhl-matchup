"""Same-game-parlay engine: simulate a game's joint outcomes, then measure how legs co-move.

Structure (parameters estimated from 2024-26 shot data, see sgp_validate.py):
  * Team shots on goal ~ negative binomial; team goals ~ Poisson.
  * Joint dependence via a Gaussian copula on (own SOG, own goals, opp SOG, opp goals) using the
    *empirical* correlations (own SOG<->own goals +.17, own SOG<->opp SOG -.18, ...). There is no
    shared "pace" factor: the data show the two teams' goals are ~uncorrelated.
  * Team SOG are split among players by multinomial shot shares; team goals are split by
    share x shooting%. So a player's goals/shots stay consistent with his team's totals.
  * Goal means are tilted so P(home win) equals the main model's probability.
"""
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm, poisson

from features import PROC
from odds import norm as norm_name

SHARE_WINDOW = 20          # team games for player shot shares / team shot rates
SHOOT_WINDOW = 60          # team games for player shooting%
SHOOT_PRIOR = 80.0         # pseudo-shots at the positional average
Q0 = {"F": 0.119, "D": 0.054}
SHARE_SUM = 0.84           # total shot share carried by the listed lineup (rest = unmodelled depth/new players)
SHARE_ALPHA = 100.0       # Dirichlet concentration of per-game shot shares (lower = more game-to-game variation)
SHARE_SHRINK = 0.0         # pull player shot shares toward their position-group mean (tuned in sgp_tune.py)
TEAM_PRIOR = 10.0          # pseudo-games at league average for team rates
OT_GOAL_PROB = 0.58
REG_SCALE = 0.974          # team goal rates already contain OT goals (~0.15/game); strip them before re-adding
EN_PROB = (0.65, 0.39, 0.13)   # P(leading team adds an empty-net goal | regulation margin 1, 2, 3+); fits ~0.35 EN goals/game        # share of tied-after-60 games decided in OT (rest: shootout)
# order: sog_home, goals_home, sog_away, goals_away
CORR = np.array([[1, .172, -.184, .042],
                 [.172, 1, .042, .027],
                 [-.184, .042, 1, .172],
                 [.042, .027, .172, 1.]])
SOG_R = 75.5               # NB size parameter; var = mu + mu^2/R (refit in sgp_validate.py)


class Store:
    """Historical shot tables with fast 'before this date' lookups."""

    def __init__(self):
        self.tg = pd.read_csv(PROC / "team_games.csv", parse_dates=["date"])[["gameId", "team", "date", "opposingTeam"]]
        ts = pd.read_csv(PROC / "team_shots.csv")
        res = pd.read_csv(PROC / "nhl_schedule_results.csv", parse_dates=["date"])
        res = res[res.state.isin(["OFF", "FINAL"])]
        h = res[["gameId", "date", "home", "home_score", "away_score"]].rename(columns={"home": "team", "home_score": "gf", "away_score": "ga"})
        a = res[["gameId", "date", "away", "away_score", "home_score"]].rename(columns={"away": "team", "away_score": "gf", "home_score": "ga"})
        g = pd.concat([h, a])  # official goals incl. OT/SO winner goal -> use for GF/GA strength only
        t = self.tg.merge(ts, on=["gameId", "team"], how="left")
        opp = ts.rename(columns={"team": "opposingTeam", "sog": "sog_against", "goals": "goals_against"})
        t = t.merge(opp, on=["gameId", "opposingTeam"], how="left").dropna(subset=["sog", "sog_against"])
        self.t = t.sort_values(["team", "date"]).reset_index(drop=True)
        self.by_team = {k: v.reset_index(drop=True) for k, v in self.t.groupby("team")}
        self.pg = pd.read_csv(PROC / "player_games.csv")
        self.pg_by_game = {k: v for k, v in self.pg.groupby("gameId")}
        self.league_sog = self.t.sog.tail(4000).mean()
        self.league_goals = self.t.goals.tail(4000).mean()

    def prev_lineup(self, team, asof, lineups):
        """Dressed skater ids from the team's previous game boxscore (what live predictions use)."""
        d = self.by_team[team]
        d = d[d.date < asof]
        if d.empty:
            return None
        return lineups.get(str(int(d.iloc[-1].gameId)), {}).get(team)

    def team_rates(self, team, asof):
        """Shots/goals for and against per game over the last SHARE_WINDOW games, shrunk to league."""
        d = self.by_team[team]
        d = d[d.date < asof].tail(SHARE_WINDOW)
        n = len(d)
        k = TEAM_PRIOR
        r = lambda col, lg: (d[col].sum() + k * lg) / (n + k)
        return dict(sog_for=r("sog", self.league_sog), sog_against=r("sog_against", self.league_sog),
                    gf=r("goals", self.league_goals), ga=r("goals_against", self.league_goals), n=n)

    def players(self, team, asof, lineup=None):
        """Per-player shot share and shooting% for `team` as of `asof`.

        lineup: iterable of player_ids dressed tonight; None -> everyone who shot in the
        team's previous game (backtest proxy for the lineup).
        """
        d = self.by_team[team]
        d = d[d.date < asof]
        recent = d.tail(SHOOT_WINDOW)
        if recent.empty:
            return pd.DataFrame()
        ids = recent.gameId.tolist()
        rows = pd.concat([self.pg_by_game[g].assign(order=i) for i, g in enumerate(ids) if g in self.pg_by_game])
        rows = rows[rows.team == team]
        win = rows[rows.order >= len(ids) - SHARE_WINDOW]
        team_sog = recent.tail(SHARE_WINDOW).sog.sum()
        if lineup is None:
            last = ids[-1]
            lineup = self.pg_by_game[last].query("team == @team").player_id.tolist() if last in self.pg_by_game else []
        s = win.groupby("player_id").agg(player=("player", "last"), pos=("pos", "last"), sog20=("sog", "sum"))
        q = rows.groupby("player_id").agg(sog_all=("sog", "sum"), g_all=("goals", "sum"))
        out = s.join(q, how="outer")
        out["sog20"] = out.sog20.fillna(0)
        out = out[out.index.isin(list(lineup))].copy()
        if out.empty:
            return out
        out["pos"] = out.pos.fillna("C")
        out["grp"] = np.where(out.pos == "D", "D", "F")
        q0 = out.grp.map(Q0)
        out["q"] = (out.g_all.fillna(0) + SHOOT_PRIOR * q0) / (out.sog_all.fillna(0) + SHOOT_PRIOR)
        raw = out.sog20 / max(team_sog, 1)
        gmean = raw.groupby(out.grp).transform("mean")
        share = (1 - SHARE_SHRINK) * raw + SHARE_SHRINK * gmean
        # tonight's shots can only come from tonight's lineup, so shares must (almost) sum to 1
        out["share"] = share / max(share.sum(), 1e-9) * SHARE_SUM
        out["name"] = out.player.fillna("")
        return out.reset_index().rename(columns={"index": "player_id"})[["player_id", "name", "pos", "share", "q"]]


@dataclass
class GameInputs:
    home: str
    away: str
    p_home: float
    mu_sog_h: float
    mu_sog_a: float
    lam_h: float
    lam_a: float
    players_h: pd.DataFrame = field(default_factory=pd.DataFrame)
    players_a: pd.DataFrame = field(default_factory=pd.DataFrame)


def build_inputs(store, home, away, asof, p_home, lineup_h=None, lineup_a=None):
    h, a = store.team_rates(home, asof), store.team_rates(away, asof)
    L = store.league_sog
    mu_h = float(np.clip(h["sog_for"] + a["sog_against"] - L, 20, 38))
    mu_a = float(np.clip(a["sog_for"] + h["sog_against"] - L, 20, 38))
    lam_h = 0.5 * (h["gf"] + a["ga"])
    lam_a = 0.5 * (a["gf"] + h["ga"])
    return GameInputs(home, away, p_home, mu_h, mu_a, lam_h, lam_a,
                      store.players(home, asof, lineup_h), store.players(away, asof, lineup_a))


def _nb_ppf(u, mu, r=None):
    r = SOG_R if r is None else r
    return nbinom.ppf(u, r, r / (r + mu))


def _win_prob_from(z_sog_u, z_g_u, lam_h, lam_a, delta):
    gh = poisson.ppf(z_g_u[:, 0], lam_h * np.exp(delta))
    ga = poisson.ppf(z_g_u[:, 1], lam_a * np.exp(-delta))
    return gh, ga, (gh > ga).mean() + 0.5 * (gh == ga).mean()


def simulate(inp: GameInputs, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 4)) @ np.linalg.cholesky(CORR).T
    u = norm.cdf(z)
    sog_h = _nb_ppf(u[:, 0], inp.mu_sog_h)
    sog_a = _nb_ppf(u[:, 2], inp.mu_sog_a)
    gu = u[:, [1, 3]]
    lo, hi = -0.6, 0.6                          # bisect the goal tilt to hit the model's P(home win)
    for _ in range(14):
        mid = (lo + hi) / 2
        _, _, p = _win_prob_from(None, gu, inp.lam_h * REG_SCALE, inp.lam_a * REG_SCALE, mid)
        lo, hi = (mid, hi) if p < inp.p_home else (lo, mid)
    delta = (lo + hi) / 2
    gh, ga, _ = _win_prob_from(None, gu, inp.lam_h * REG_SCALE, inp.lam_a * REG_SCALE, delta)
    tie = gh == ga
    margin = np.abs(gh - ga).astype(int)
    en = (~tie) & (rng.random(n) < np.take(np.array((0,) + EN_PROB), np.minimum(margin, 3)))
    ot_goal = tie & (rng.random(n) < OT_GOAL_PROB)
    home_win = (gh > ga) | (tie & (rng.random(n) < 0.5))
    out = dict(home_win=home_win, total=gh + ga + ot_goal + en, gh=gh, ga=ga, sog_h=sog_h, sog_a=sog_a,
               saves_h=sog_a - ga, saves_a=sog_h - gh, delta=delta)
    for side, sog, g, pl in (("h", sog_h, gh, inp.players_h), ("a", sog_a, ga, inp.players_a)):
        out[f"p_{side}"] = _allocate(rng, sog, g, pl)
    return out


def _split(rng, n, P):
    """Vectorised multinomial: split counts n (N,) over per-row probabilities P (N, k)."""
    N, k = P.shape
    out = np.zeros((N, k), dtype=int)
    left = n.astype(int).copy()
    rem = np.ones(N)
    for j in range(k - 1):
        pj = np.clip(P[:, j] / np.maximum(rem, 1e-12), 0, 1)
        x = rng.binomial(left, pj)
        out[:, j] = x
        left -= x
        rem = rem - P[:, j]
    out[:, k - 1] = left
    return out


def _allocate(rng, sog, goals, pl):
    """Split team SOG/goals among players (+ a tiny 'rest' bucket). Returns dict id -> (sog, goals).

    Shot shares vary game to game (Dirichlet, concentration SHARE_ALPHA); goals are drawn from each
    player's simulated shots weighted by his shooting%, so player goals and shots stay consistent.
    """
    if pl.empty:
        return {}
    share = pl.share.values.clip(1e-4)
    p = np.append(share, max(1 - share.sum(), 1e-4))
    g = rng.gamma(SHARE_ALPHA * p, size=(len(sog), len(p)))
    P = g / g.sum(1, keepdims=True)
    S = _split(rng, sog, P)
    q = np.append(pl.q.values, 0.085)
    W = S * q
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    G = np.minimum(_split(rng, goals, W), S)
    return {int(pid): (S[:, i], G[:, i]) for i, pid in enumerate(pl.player_id)}


def make_legs(inp, sim, top_players=5, extra=None):
    """Candidate legs -> dict label -> (bool array, meta). Meta: market, team, player, line."""
    legs = {}
    add = lambda lab, arr, **m: legs.__setitem__(lab, (arr.astype(bool), m))
    H, A = inp.home, inp.away
    add(f"{H} ML", sim["home_win"], market="ml", team=H, kind="ml", side="h")
    add(f"{A} ML", ~sim["home_win"], market="ml", team=A, kind="ml", side="a")
    for line in (5.5, 6.5):
        add(f"Over {line}", sim["total"] > line, market="total", team=None, kind="total", dir="over", line=line)
        add(f"Under {line}", sim["total"] < line, market="total", team=None, kind="total", dir="under", line=line)
    for side, team, pl in (("h", H, inp.players_h), ("a", A, inp.players_a)):
        if pl.empty:
            continue
        top = pl.assign(eg=pl.share * pl.q).sort_values("eg", ascending=False).head(top_players)
        for r in top.itertuples():
            S, G = sim[f"p_{side}"][int(r.player_id)]
            add(f"{r.name} anytime goal", G >= 1, market="atg", team=team, player=int(r.player_id), kind="atg", side=side, name=r.name)
            for line in (1.5, 2.5):  # 3.5 dropped: backtest shows it is over-predicted by 5-7 pts
                pr = (S > line).mean()
                if 0.2 <= pr <= 0.85:
                    add(f"{r.name} over {line} SOG", S > line, market="sog", team=team, player=int(r.player_id), kind="sog", side=side, line=line, name=r.name)
            for line in sorted(set((extra or {}).get("sog", {}).get(norm_name(r.name), [])) - {1.5, 2.5}):
                if 0.5 <= line <= 4.5:   # lines posted by books that the default grid lacks
                    add(f"{r.name} over {line} SOG", S > line, market="sog", team=team, player=int(r.player_id), kind="sog", side=side, line=line, name=r.name)
    for key, team in (("saves_h", H), ("saves_a", A)):
        gs = key[-1]
        mu = sim[key].mean()
        main = np.floor(mu) + 0.5
        for line in (main - 2, main, main + 2):
            add(f"{team} goalie over {line:g} saves", sim[key] > line, market="saves", team=team, kind="saves", side=gs, dir="over", line=line)
        add(f"{team} goalie under {main:g} saves", sim[key] < main, market="saves", team=team, kind="saves", side=gs, dir="under", line=main)
        for line in (extra or {}).get("saves", {}).get(gs, []):
            for d in ("over", "under"):
                arr = sim[key] > line if d == "over" else sim[key] < line
                add(f"{team} goalie {d} {line:g} saves", arr, market="saves", team=team, kind="saves", side=gs, dir=d, line=line)
    for line in (extra or {}).get("totals", []):
        if line not in (5.5, 6.5) and line % 1 == 0.5:
            add(f"Over {line}", sim["total"] > line, market="total", team=None, kind="total", dir="over", line=line)
            add(f"Under {line}", sim["total"] < line, market="total", team=None, kind="total", dir="under", line=line)
    return legs


def rank_combos(legs, sizes=(2, 3), min_joint=0.04, min_leg=0.15, max_leg=0.88):
    """All compatible 2-3 leg combos with joint prob, independent product, and lift."""
    names = [k for k, (a, m) in legs.items() if min_leg <= a.mean() <= max_leg]
    M = np.array([legs[k][0] for k in names], dtype=np.float32)
    meta = [legs[k][1] for k in names]
    p = M.mean(1)
    n = len(names)

    def compatible(idx):
        mk = [meta[i]["market"] for i in idx]
        if len(set(mk)) < len(mk):          # one leg per market (no over+under / two lines)
            return False
        pl = [meta[i].get("player") for i in idx if meta[i].get("player") is not None]
        if len(set(pl)) < len(pl):          # no two legs on the same player (books often block these)
            return False
        return True

    rows = []
    N = M.shape[1]
    if 2 in sizes:
        J = (M @ M.T) / N
        for i, j in combinations(range(n), 2):
            if J[i, j] >= min_joint and compatible((i, j)):
                rows.append(((i, j), J[i, j]))
    if 3 in sizes:
        for i, j in combinations(range(n), 2):
            if not compatible((i, j)):
                continue
            both = M[i] * M[j]
            if both.mean() < min_joint:
                continue
            tri = (M @ both) / N
            for k in range(j + 1, n):
                if tri[k] >= min_joint and compatible((i, j, k)):
                    rows.append(((i, j, k), tri[k]))
    out = []
    for idx, pj in rows:
        prod = float(np.prod(p[list(idx)]))
        mech = any(meta[i]["kind"] == "sog" for i in idx) and any(
            meta[j]["kind"] == "saves" and meta[j]["team"] != meta[i]["team"]
            for i in idx if meta[i]["kind"] == "sog" for j in idx)
        out.append(dict(legs=[names[i] for i in idx], idx=[names[i] for i in idx], p_joint=float(pj), p_indep=prod,
                        lift=float(pj / prod), leg_probs=[float(p[i]) for i in idx], mechanical=bool(mech)))
    return pd.DataFrame(out)


def load_lineups():
    import json
    f = PROC / "lineups.json"
    return json.loads(f.read_text()) if f.exists() else {}


def american(p):
    """Fair American odds for win probability p."""
    p = min(max(p, 1e-4), 1 - 1e-4)
    return round(-100 * p / (1 - p)) if p >= .5 else round(100 * (1 - p) / p)

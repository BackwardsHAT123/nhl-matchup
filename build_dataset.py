"""Build per-game team and goalie tables from MoneyPuck raw files.

Outputs (data/processed):
  games.csv       one row per regular-season game (home/away teams, date, winner)
  team_games.csv  one row per team-game: 5v5 xGF / xGA, home flag, date
  goalie_games.csv one row per team-game: starter id/name, shots faced, xGA, goals, GSAx
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
SEASONS = [2023, 2024, 2025, 2026]  # MoneyPuck season = start year (2026 = 2026-27)


def load_team_games() -> pd.DataFrame:
    cols = ["team", "season", "gameId", "playerTeam", "opposingTeam", "home_or_away",
            "gameDate", "situation", "xGoalsFor", "xGoalsAgainst", "goalsFor",
            "goalsAgainst", "playoffGame"]
    d = pd.read_csv(RAW / "all_teams.csv", usecols=cols)
    d = d[d.season.isin(SEASONS) & (d.playoffGame == 0)]
    five = d[d.situation == "5on5"].rename(
        columns={"xGoalsFor": "xgf", "xGoalsAgainst": "xga"})
    tot = d[d.situation == "all"][["gameId", "team", "goalsFor", "goalsAgainst"]]
    five = five.drop(columns=["goalsFor", "goalsAgainst"]).merge(tot, on=["gameId", "team"])
    five["date"] = pd.to_datetime(five.gameDate.astype(str), format="%Y%m%d")
    five["home"] = (five.home_or_away == "HOME").astype(int)
    five["win"] = (five.goalsFor > five.goalsAgainst).astype(int)
    return five[["season", "gameId", "date", "team", "opposingTeam", "home",
                 "xgf", "xga", "goalsFor", "goalsAgainst", "win"]].sort_values(
        ["date", "gameId", "team"]).reset_index(drop=True)


def load_goalie_games() -> pd.DataFrame:
    """GSAx per goalie-game from shot-level data: xGoals of on-goal shots minus goals.

    All strengths, empty-net shots excluded. Starter = goalie who faced the first
    shot for that team in the game.
    """
    rows = []
    for s in SEASONS:
        sh = pd.read_csv(RAW / f"shots_{s}" / f"shots_{s}.csv", usecols=[
            "game_id", "season", "goalieIdForShot", "goalieNameForShot", "xGoal", "goal",
            "shotWasOnGoal", "xShotWasOnGoal", "shotOnEmptyNet", "isPlayoffGame", "isHomeTeam",
            "homeTeamCode", "awayTeamCode", "period", "time"])
        sh = sh[(sh.isPlayoffGame == 0) & (sh.goalieIdForShot > 0)
                & (sh.shotWasOnGoal == 1) & (sh.shotOnEmptyNet == 0)].copy()
        # xGoal = P(goal | unblocked attempt); condition on the shot reaching the net
        sh["xGoal"] = (sh.xGoal / sh.xShotWasOnGoal.clip(lower=0.2)).clip(upper=1)
        sh["gameId"] = sh.season * 1_000_000 + sh.game_id
        # shooter home -> goalie belongs to away team
        sh["team"] = np.where(sh.isHomeTeam == 1, sh.awayTeamCode, sh.homeTeamCode)
        sh = sh.sort_values(["gameId", "period", "time"])
        for (gid, team), g in sh.groupby(["gameId", "team"], sort=False):
            starter = g.goalieIdForShot.iloc[0]
            gs = g[g.goalieIdForShot == starter]
            rows.append((gid, team, int(starter), g.goalieNameForShot.iloc[0],
                         len(gs), gs.xGoal.sum(), gs.goal.sum()))
    gg = pd.DataFrame(rows, columns=["gameId", "team", "goalie_id", "goalie",
                                     "sa", "xga_g", "ga_g"])
    # Calibrate xG to each season's league-wide goals so league-average GSAx is 0
    # (removes small season-to-season model drift that would bias the shrinkage prior).
    season = gg.gameId // 1_000_000
    k = gg.groupby(season).ga_g.transform("sum") / gg.groupby(season).xga_g.transform("sum")
    gg["gsax"] = gg.xga_g * k - gg.ga_g
    return gg


if __name__ == "__main__":
    tg = load_team_games()
    gg = load_goalie_games()
    # Team codes can differ between files; check overlap, then join on gameId + home/away.
    print("team codes only in team file:", sorted(set(tg.team) - set(gg.team)))
    print("team codes only in goalie file:", sorted(set(gg.team) - set(tg.team)))
    tg.to_csv(OUT / "team_games.csv", index=False)
    gg.to_csv(OUT / "goalie_games.csv", index=False)
    print(tg.groupby("season").gameId.nunique(), len(gg))

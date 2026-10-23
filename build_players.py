"""Per-game player and team shot tables from MoneyPuck shot-level data.

Outputs (data/processed):
  player_games.csv  gameId, team, player_id, player, pos, sog, goals  (only games where the player shot)
  team_shots.csv    gameId, team, sog, goals                      (shots on goal, empty net excluded)
Empty-net shots are excluded so shooting% reflects normal play.
"""
import pandas as pd

from build_dataset import RAW, OUT
from seasons import SEASONS

if __name__ == "__main__":
    pg, tg = [], []
    for s in SEASONS:
        f = RAW / f"shots_{s}" / f"shots_{s}.csv"
        if not f.exists():
            continue
        sh = pd.read_csv(f, usecols=["game_id", "season", "shooterPlayerId", "shooterName", "teamCode",
                                     "playerPositionThatDidEvent", "goal", "shotWasOnGoal", "shotOnEmptyNet", "isPlayoffGame"])
        sh = sh[(sh.isPlayoffGame == 0) & (sh.shotWasOnGoal == 1) & (sh.shotOnEmptyNet == 0)
                & (sh.shooterPlayerId > 0)].copy()
        sh["gameId"] = sh.season * 1_000_000 + sh.game_id
        g = sh.groupby(["gameId", "teamCode", "shooterPlayerId"]).agg(
            player=("shooterName", "last"), pos=("playerPositionThatDidEvent", "last"), sog=("goal", "size"), goals=("goal", "sum")).reset_index()
        pg.append(g.rename(columns={"teamCode": "team", "shooterPlayerId": "player_id"}))
        t = sh.groupby(["gameId", "teamCode"]).agg(sog=("goal", "size"), goals=("goal", "sum")).reset_index()
        tg.append(t.rename(columns={"teamCode": "team"}))
    pd.concat(pg).to_csv(OUT / "player_games.csv", index=False)
    pd.concat(tg).to_csv(OUT / "team_shots.csv", index=False)
    print(len(pd.concat(pg)), "player-games;", len(pd.concat(tg)), "team-games")

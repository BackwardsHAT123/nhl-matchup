"""Cache dressed-skater lineups (NHL boxscores) for backtesting: data/processed/lineups.json
{gameId: {team: [playerId, ...]}}. Live predictions fetch the few boxscores they need directly."""
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import predict
from features import PROC
from seasons import CUR

F = PROC / "lineups.json"


def fetch(gid):
    try:
        b = predict.api(f"https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore", tries=3)
        out = {}
        for side, key in (("homeTeam", "homeTeam"), ("awayTeam", "awayTeam")):
            t = b["playerByGameStats"][side]
            out[b[key]["abbrev"]] = [p["playerId"] for p in t["forwards"] + t["defense"]]
        return str(gid), out
    except Exception as e:
        return str(gid), None


if __name__ == "__main__":
    have = json.loads(F.read_text()) if F.exists() else {}
    res = pd.read_csv(PROC / "nhl_schedule_results.csv")
    ids = res[(res.season >= CUR - 1) & res.state.isin(["OFF", "FINAL"])].gameId.astype(str)
    todo = [g for g in ids if g not in have]
    print(len(todo), "boxscores to fetch", flush=True)
    with ThreadPoolExecutor(12) as ex:
        for i, (gid, out) in enumerate(ex.map(fetch, todo)):
            if out:
                have[gid] = out
            if i % 100 == 0:
                F.write_text(json.dumps(have))
                print(i, len(have), flush=True)
    F.write_text(json.dumps(have))
    print("done", len(have))

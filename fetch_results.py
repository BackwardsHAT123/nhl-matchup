"""Official final scores (incl. OT/shootout winners) and schedule from the NHL API."""
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

PROC = Path(__file__).parent / "data" / "processed"
TEAMS = ("ANA BOS BUF CAR CBJ CGY CHI COL DAL DET EDM FLA LAK MIN MTL NJD NSH NYI NYR OTT PHI "
         "PIT SEA SJS STL TBL TOR UTA VAN VGK WPG WSH").split()


def get(url, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1 + i)


def season_games(season: int) -> list[dict]:
    seen, rows = set(), []
    for t in TEAMS:
        try:
            data = get(f"https://api-web.nhle.com/v1/club-schedule-season/{t}/{season}{season + 1}")
        except Exception as e:  # team may not exist in older seasons
            print("skip", t, season, e)
            continue
        for g in data["games"]:
            if g["gameType"] != 2 or g["id"] in seen:
                continue
            seen.add(g["id"])
            h, a = g["homeTeam"], g["awayTeam"]
            rows.append(dict(gameId=g["id"], season=season, date=g["gameDate"], state=g["gameState"],
                             home=h["abbrev"], away=a["abbrev"],
                             home_score=h.get("score"), away_score=a.get("score"),
                             last_period=g.get("gameOutcome", {}).get("lastPeriodType")))
        time.sleep(0.15)
    return rows


if __name__ == "__main__":
    rows = []
    for s in (2023, 2024, 2025, 2026):
        rows += season_games(s)
    d = pd.DataFrame(rows)
    d.to_csv(PROC / "nhl_schedule_results.csv", index=False)
    print(d.groupby(["season", "state"]).size())

"""Post-game facts for one finished game (NHL boxscore + play-by-play + MoneyPuck luck measures).

Used by review.py to grade logged bets and explain why they hit or missed. Facts are frozen on disk
(log/facts/<date>.json) once complete, so each game is fetched once.
"""
import pandas as pd

import predict
from features import PROC

BASE = "https://api-web.nhle.com/v1/gamecenter"


def secs(mmss):
    try:
        m, s = str(mmss).split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return 0


def luck_tables():
    """5v5 xG share and starter GSAx per (gameId, team), from the pipeline's MoneyPuck tables."""
    tg = pd.read_csv(PROC / "team_games.csv")
    xg = {(r.gameId, r.team): r.xgf / (r.xgf + r.xga) for r in tg.itertuples() if (r.xgf + r.xga) > 0}
    gg = pd.read_csv(PROC / "goalie_games.csv")
    gs = {(r.gameId, r.team): (r.goalie, r.gsax) for r in gg.itertuples()}
    return xg, gs


def game_facts(gid, row, xg, gs):
    """row: the game's line from nhl_schedule_results (home, away, scores, last_period, date)."""
    b = predict.api(f"{BASE}/{gid}/boxscore")
    pbp = predict.api(f"{BASE}/{gid}/play-by-play")
    side_of = {b["homeTeam"]["id"]: "home", b["awayTeam"]["id"]: "away"}
    team = {"home": b["homeTeam"]["abbrev"], "away": b["awayTeam"]["abbrev"]}
    f = dict(gameId=int(gid), date=str(row["date"])[:10], home=team["home"], away=team["away"],
             hs=int(row["home_score"]), as_=int(row["away_score"]), period=row["last_period"],
             dressed={"home": [], "away": []}, skaters={}, goalies={"home": [], "away": []},
             goals=[], pen={s: dict(minor=0, major=0, fight=0, misc=0, pim=0) for s in ("home", "away")},
             team_sog={"home": 0, "away": 0})
    for side, key in (("home", "homeTeam"), ("away", "awayTeam")):
        st = b["playerByGameStats"][key]
        for p in st["forwards"] + st["defense"]:
            f["dressed"][side].append(p["playerId"])
            f["skaters"][str(p["playerId"])] = dict(n=p["name"]["default"], t=team[side], pos=p["position"], toi=secs(p["toi"]),
                                                   sog=p["sog"], g=p["goals"], a=p["assists"], pim=p["pim"])
            f["team_sog"][side] += p["sog"]
        for g in st["goalies"]:
            if g["toi"] != "00:00" or g.get("starter"):
                f["goalies"][side].append(dict(id=g["playerId"], n=g["name"]["default"], starter=bool(g.get("starter")),
                                               toi=secs(g["toi"]), sa=g["shotsAgainst"], sv=g["saves"], ga=g["goalsAgainst"]))
    for p in pbp["plays"]:
        d, t = p.get("details", {}), p["typeDescKey"]
        side = side_of.get(d.get("eventOwnerTeamId"))
        if t == "goal" and side:
            code = str(p.get("situationCode", "1551")).zfill(4)    # away G, away skaters, home skaters, home G
            sk_for, sk_opp = (int(code[2]), int(code[1])) if side == "home" else (int(code[1]), int(code[2]))
            g_for, g_opp = (code[3], code[0]) if side == "home" else (code[0], code[3])
            f["goals"].append(dict(team=side, scorer=d.get("scoringPlayerId"), per=p["periodDescriptor"]["number"],
                                   en=g_opp == "0", pp=(sk_for > sk_opp and g_for == "1" and g_opp == "1")))
        elif t == "penalty" and side:
            tc, dk = d.get("typeCode"), d.get("descKey")
            pn = f["pen"][side]
            if dk == "fighting":
                pn["fight"] += 1
            elif tc == "MAJ":
                pn["major"] += 1
            elif tc in ("MIS", "GAM"):
                pn["misc"] += 1
            elif tc in ("MIN", "BEN"):
                pn["minor"] += 1
            pn["pim"] += d.get("duration") or 0
    f["xg"] = {s: xg.get((int(gid), team[s])) for s in ("home", "away")}
    f["gsax"] = {s: (gs[(int(gid), team[s])][1] if (int(gid), team[s]) in gs else None) for s in ("home", "away")}
    f["complete"] = all(v is not None for v in f["xg"].values())    # MoneyPuck can lag a day
    return f

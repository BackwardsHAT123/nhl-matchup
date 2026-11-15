"""Automatic injury tracking: ESPN's public NHL injury feed -> docs/data/injuries.json (+ daily snapshot in log/injuries/).

  python3 injuries.py            # refresh (run daily, before export_dashboard.py / sgp_slate.py)

Players are matched to NHL ids through docs/data/team_stats.json (same team, normalised name), which also gives
the production numbers behind each player's weight. A player counts as OUT when ESPN lists him Out / IR and any
listed return date is after the game day; Day-To-Day counts half. When the feed drops a player (he is back) or his
return date passes, the penalty disappears by itself, so nothing needs to be remembered.

Win-model effect: team xGF% (the rolling average the model uses) is lowered by
    K * sum_i weight_i * max(share_i - REPLACEMENT, 0)         (capped at MAX_PENALTY)
where share_i is the player's blended share of team scoring (points) and ice time. K, REPLACEMENT and MAX_PENALTY are
priors, NOT fitted: ESPN keeps no history, so we snapshot the feed daily (log/injuries/) to fit them later.
"""
import json
import sys
import time
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

from odds import norm

ROOT = Path(__file__).parent
OUT = ROOT / "docs" / "data" / "injuries.json"
SNAP = ROOT / "log" / "injuries"
TEAM_STATS = ROOT / "docs" / "data" / "team_stats.json"
URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries"

K = 16.0             # xGF% points lost per 1.00 of team-production share (a 12%-share star out ~ -1.4 pts)
REPLACEMENT = 0.03   # share a replacement-level call-up supplies
MAX_PENALTY = 4.0    # cap per team, xGF% points
DEFAULT_SHARE = 0.045  # for listed players we could not match to team_stats (usually depth / long-term IR)
GAMES_PER_DAY = 0.45   # ~ team games per calendar day, to turn "listed N days ago" into games already missed
PTS_W = 0.7          # blend of points share vs ice-time share
OUT_STATUSES = {"out", "injured reserve"}
ESPN_ABBR = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL", "UTAH": "UTA", "WSH": "WSH", "VGS": "VGK"}


def fetch(url=URL, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception:                 # truncated chunked reads happen; just retry
            if i == tries - 1:
                raise
            time.sleep(1 + i)


def shares(team):
    """{norm name: (id, pos, share)} for a team's skaters, from team_stats.json."""
    t = team
    gp = max(t["summary"]["gp"], 1)
    team_goals = max(t["summary"]["gf_pg"] * gp, 1)
    sk = t["skaters"]
    toi_tot = sum(s["toi"] * s["gp"] for s in sk) or 1
    out = {}
    for s in sk:
        # per-game basis so a player who already missed games is not undervalued; ~2.7 point-credits per goal
        per_game = (s["points"] / max(s["gp"], 1)) / (2.7 * team_goals / gp)
        toi_pg = s["toi"] / (toi_tot / gp)
        out[norm(s["name"])] = (s["id"], s["pos"], round(PTS_W * per_game + (1 - PTS_W) * toi_pg, 4))
    return out


def build():
    feed = fetch()
    ts = json.loads(TEAM_STATS.read_text())["teams"]
    by_team = {abbr: shares(t) for abbr, t in ts.items() if t.get("skaters")}
    teams = {}
    unmatched = []
    for t in feed["injuries"]:
        for i in t["injuries"]:
            a = i["athlete"]
            ab = (a.get("team") or {}).get("abbreviation") or ""
            ab = ESPN_ABBR.get(ab, ab)
            if ab not in by_team:
                unmatched.append((a["displayName"], ab))
                continue
            hit = by_team[ab].get(norm(a["displayName"]))
            d = i.get("details") or {}
            rec = dict(name=a["displayName"], id=hit[0] if hit else None,
                       pos=(a.get("position") or {}).get("abbreviation") if not hit else hit[1],
                       status=i["status"], type=d.get("type"), return_date=d.get("returnDate"),
                       updated=i.get("date"), share=hit[2] if hit else None, note=i.get("shortComment"))
            if rec["pos"] == "G":
                continue                      # goalies are handled by the starter model
            teams.setdefault(ab, []).append(rec)
    if unmatched:
        print("unmapped team rows:", unmatched[:5], file=sys.stderr)
    return dict(generated_at=datetime.now(timezone.utc).isoformat(timespec="minutes"),
                source="ESPN", gp={a: t["summary"]["gp"] for a, t in ts.items()}, teams=teams)


# ---- consumers -----------------------------------------------------------------------------------------------

def load(path=OUT):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {"teams": {}}


def weight(rec, day):
    """Probability a listed player misses the game on `day` (YYYY-MM-DD): 1 out, 0.5 day-to-day, 0 once back."""
    rd = rec.get("return_date")
    if rd and rd <= day:
        return 0.0
    s = (rec.get("status") or "").lower()
    if s in OUT_STATUSES:
        return 1.0
    return 0.5 if s == "day-to-day" else 0.0


def team_impact(inj, team, day):
    """-> (xGF% points to subtract, [injured player dicts that matter]). Players with unknown production use a
    modest default share so an unmatched name still registers."""
    pen, rows = 0.0, []
    win = max(min(inj.get("gp", {}).get(team, 20), 40), 1)       # games in the model's rolling xGF% window
    today = date.today().isoformat()
    for rec in inj.get("teams", {}).get(team, []):
        w = weight(rec, day)
        if w == 0:
            continue
        # games already played without him are baked into the rolling xGF%, so only the not-yet-reflected part counts
        days = max((date.fromisoformat(today) - date.fromisoformat((rec.get("updated") or today)[:10])).days, 0)
        w *= max(0.0, 1 - GAMES_PER_DAY * days / win)
        if w == 0:
            continue
        share = rec.get("share")
        loss = K * max((share if share is not None else DEFAULT_SHARE) - REPLACEMENT, 0) * w
        pen += loss
        rows.append(dict(name=rec["name"], pos=rec.get("pos"), status=rec["status"], return_date=rec.get("return_date"),
                         p_out=w, xgf_pts=round(-loss, 2)))
    rows.sort(key=lambda r: r["xgf_pts"])
    return min(pen, MAX_PENALTY), rows


def out_ids(inj, team, day, min_p=1.0):
    """NHL ids of skaters expected out (p >= min_p) -- removed from parlay lineups."""
    return {r["id"] for r in inj.get("teams", {}).get(team, []) if r.get("id") and weight(r, day) >= min_p}


def main():
    inj = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(inj, indent=1))
    SNAP.mkdir(parents=True, exist_ok=True)
    (SNAP / f"{date.today().isoformat()}.json").write_text(json.dumps(inj))   # history for fitting K later
    n = sum(len(v) for v in inj["teams"].values())
    print(f"injuries: {n} skaters across {len(inj['teams'])} teams -> {OUT}")


if __name__ == "__main__":
    main()

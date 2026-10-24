"""Live NHL odds from The Odds API (https://the-odds-api.com) -> per-bookmaker leg prices.

Needs env ODDS_API_KEY (GitHub secret in CI). Credit cost per game = markets x regions = 5, so a
12-game slate is ~60 credits per refresh. Never raises: no key / API error -> empty result.

The API returns single-leg prices only (no same-game-parlay prices), so callers get leg prices per
bookmaker and must treat "legs multiplied" as an upper bound on what a book's SGP pays.
"""
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

BASE = "https://api.the-odds-api.com/v4/sports/icehockey_nhl"
MARKETS = "h2h,totals,player_goal_scorer_anytime,player_shots_on_goal,player_total_saves"
REGIONS = "us"
MAX_CREDITS_PER_RUN = int(os.environ.get("ODDS_MAX_CREDITS", "120"))
HOURS_AHEAD = 30

# distinctive substring of each club's name as The Odds API spells it (accent-stripped, lowercase)
TEAM_KEY = {
    "ANA": "ducks", "BOS": "bruins", "BUF": "sabres", "CGY": "flames", "CAR": "hurricanes", "CHI": "blackhawks",
    "COL": "avalanche", "CBJ": "blue jackets", "DAL": "stars", "DET": "red wings", "EDM": "oilers",
    "FLA": "panthers", "LAK": "kings", "MIN": "wild", "MTL": "canadiens", "NSH": "predators",
    "NJD": "devils", "NYI": "islanders", "NYR": "rangers", "OTT": "senators", "PHI": "flyers",
    "PIT": "penguins", "SJS": "sharks", "SEA": "kraken", "STL": "blues", "TBL": "lightning",
    "TOR": "maple leafs", "UTA": "utah", "VAN": "canucks", "VGK": "golden knights", "WSH": "capitals",
    "WPG": "jets",
}


def norm(name: str) -> str:
    """Accent/punctuation-insensitive name key for matching players across sources."""
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii)\b\.?", "", s)
    return " ".join(re.sub(r"[^a-z ]", "", s).split())


def team_abbr(full_name: str):
    n = norm(full_name)
    hits = [a for a, k in TEAM_KEY.items() if k in n]
    return hits[0] if len(hits) == 1 else None


def _get(path, params, tries=4):
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "nhl-matchup"}), timeout=30) as r:
                return json.loads(r.read()), r.headers.get("x-requests-remaining")
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 422, 429):   # auth / quota / bad request: retrying will not help
                raise
        except Exception:
            pass
        time.sleep(1 + i)
    raise RuntimeError("odds API unreachable")


def parse_event(ev):
    """One event payload -> {bookmaker_key: {title, ml, totals, atg, sog, saves}} (decimal prices)."""
    books = {}
    for b in ev.get("bookmakers", []):
        d = dict(title=b.get("title", b["key"]), ml={}, totals={}, atg={}, sog={}, saves={})
        for m in b.get("markets", []):
            k = m["key"]
            for o in m.get("outcomes", []):
                price, pt, desc = o.get("price"), o.get("point"), o.get("description")
                if price is None:
                    continue
                if k == "h2h":
                    a = team_abbr(o["name"])
                    if a:
                        d["ml"][a] = price
                elif k == "totals":
                    d["totals"].setdefault(pt, {})[o["name"].lower()] = price
                elif k == "player_goal_scorer_anytime" and o.get("name", "").lower() == "yes":
                    d["atg"][norm(desc)] = price
                elif k in ("player_shots_on_goal", "player_total_saves") and pt is not None:
                    tgt = d["sog"] if k == "player_shots_on_goal" else d["saves"]
                    tgt.setdefault(norm(desc), {}).setdefault(pt, {})[o["name"].lower()] = price
        books[b["key"]] = d
    return books


def fetch_slate():
    """Fetch odds for games starting within HOURS_AHEAD. Returns {(away, home): books} or {}."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("odds: no ODDS_API_KEY, skipping (EV vs market will show n/a)")
        return {}
    out, spent, remaining = {}, 0, None
    try:
        events, _ = _get("/events", {"apiKey": key})
        now = datetime.now(timezone.utc)
        for ev in events:
            t = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            if not (now - timedelta(hours=1) <= t <= now + timedelta(hours=HOURS_AHEAD)):
                continue
            if spent + 5 > MAX_CREDITS_PER_RUN:
                print(f"odds: credit guard hit ({spent}/{MAX_CREDITS_PER_RUN}); remaining games skipped")
                break
            data, remaining = _get(f"/events/{ev['id']}/odds", {
                "apiKey": key, "regions": REGIONS, "markets": MARKETS, "oddsFormat": "decimal"})
            spent += 5
            away, home = team_abbr(ev["away_team"]), team_abbr(ev["home_team"])
            if away and home:
                out[(away, home)] = parse_event(data)
            else:
                print("odds: unmatched teams", ev["away_team"], ev["home_team"])
    except Exception as e:
        print("odds: error", repr(e)[:120], "- continuing without market prices")
    print(f"odds: {len(out)} games priced, ~{spent} credits used, {remaining} remaining")
    return out


def fair_prob(over, under):
    """Vig-free probability of the 'over' side when both sides are quoted, else raw implied."""
    if over and under:
        return (1 / over) / (1 / over + 1 / under)
    return 1 / over if over else None

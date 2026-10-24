"""Results review: log parlay candidates pre-game, grade legs after, explain misses.

  python review.py                       # daily: log today's/upcoming slate, grade finished games
  python review.py backfill 2026-10-20 2026-10-23   # reconstruct pre-game logs for past days (marked backfill)

Writes log/sgp/<date>.json (pre-game parlay log, frozen once a game starts), log/facts/<date>.json (post-game
facts) and docs/data/results_review.json (dashboard).
"""
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import odds
from export_dashboard import et_today
from features import PROC
from review_facts import game_facts, luck_tables
from seasons import CUR

ROOT = Path(__file__).parent
LOG, FACTS = ROOT / "log" / "sgp", ROOT / "log" / "facts"
OUT = ROOT / "docs" / "data" / "results_review.json"
DAYS_SHOWN, PLAYS_SHOWN = 10, 6
LABEL = {
    "goalie_change": "Goalie change", "goalie_pulled": "Goalie pulled/changed", "scratch": "Key player out",
    "ot_so": "OT/shootout", "en_goal": "Empty-net goal", "penalties": "Penalty-heavy", "fight": "Fight",
    "major": "Major/misconduct", "pp_swing": "Special-teams swing", "hot_goalie": "Goalie stole it",
    "shaky_goalie": "Goalie shaky", "xg_mismatch": "Result vs. play (xG)", "blowout": "Blowout",
    "dnp": "Player didn't play", "toi_drop": "Ice time down", "shots_low": "Team shots low", "shots_high": "Team shots high",
    "no_finish": "Shots, no goal", "few_shots": "Few shots", "goal_pp": "Goal on power play", "goal_en": "Empty-net goal (his)",
    "sa_low": "Few shots faced", "sa_high": "Many shots faced",
}
DESCRIPTORS = {"no_finish", "few_shots", "goal_pp", "goal_en"}   # describe the outcome itself, so excluded from the "cause" table
GOALIE_TAGS = {"goalie_change", "goalie_pulled", "hot_goalie", "shaky_goalie"}


def jdump(o, **kw):
    return json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x), **kw)


def gnorm(name):
    p = odds.norm(name).split()
    return (p[-1], p[0][0]) if p else ("", "")


# ---------------------------------------------------------------- pre-game logging
def log_games(games, today, force=False, backfill=False):
    """Write/refresh log/sgp/<date>.json for dates >= today. Games already started keep their logged version."""
    LOG.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="minutes")
    by_date = defaultdict(list)
    for g in games:
        by_date[g["date"]].append(g)
    for d, gs in by_date.items():
        if d < today.isoformat() and not force:
            continue
        f = LOG / f"{d}.json"
        old = {e["gameId"]: e for e in (json.loads(f.read_text())["games"] if f.exists() else [])}
        for g in gs:
            o = old.get(g["gameId"])
            if o and (o.get("start_utc") or "9") <= now and not force:
                continue    # puck already dropped: never overwrite the pre-game record
            old[g["gameId"]] = dict(g, logged_at=now, backfill=backfill)
        f.write_text(jdump(dict(date=d, games=list(old.values()))))


def backfill(dates):
    import predict
    import sgp_slate as S
    import sgp_model as M
    cal = S.load_calibration()
    M.SOG_R = cal.get("sog_r", M.SOG_R)
    st = M.Store()
    res = pd.read_csv(PROC / "nhl_schedule_results.csv", parse_dates=["date"])
    res = res[(res.season == CUR) & res.state.isin(["OFF", "FINAL"])]
    cache = {}
    for d in dates:
        entries = []
        for i, g in enumerate(predict.compute_slate(d)):
            e, _ = S.build_game(g, i, st, res, cache, cal)
            if e:
                entries.append(e)
        log_games(entries, et_today(), force=True, backfill=True)
        print("backfilled", d, len(entries), "games")


# ---------------------------------------------------------------- grading
def game_tags(entry, f, usual):
    """Game-level explanatory tags: list of dict(k, t, team)."""
    T, tags = {"home": f["home"], "away": f["away"]}, []
    res_side = "home" if f["hs"] > f["as_"] else "away"
    for s, key in (("home", "h"), ("away", "a")):
        st = entry.get("starters") or {}
        gl = f["goalies"][s]
        starter = next((g for g in gl if g["starter"]), gl[0] if gl else None)
        if starter and st.get(key) and gnorm(starter["n"]) != gnorm(st[key]):
            tags.append(dict(k="goalie_change", team=T[s], t=f"{T[s]}: {starter['n']} started (model assumed {st[key]})"))
        if starter and len(gl) > 1 and starter["toi"] < 3300:
            tags.append(dict(k="goalie_pulled", team=T[s], t=f"{T[s]}: {starter['n']} out after {starter['toi'] // 60} min ({starter['ga']} GA)"))
        lineup = entry.get("lineup_h" if key == "h" else "lineup_a") or []
        out = [(n, sh) for pid, n, sh in lineup if pid not in set(f["dressed"][s]) and sh >= 0.06]
        if out:
            tags.append(dict(k="scratch", team=T[s], t=f"{T[s]}: {', '.join(f'{n} ({sh:.0%} of shots)' for n, sh in out)} not in lineup"))
        gs = f["gsax"].get(s)
        if gs is not None:   # this team's goalie
            nm = next((g["n"] for g in gl if g["starter"]), "goalie")
            if gs >= 1.5:
                tags.append(dict(k="hot_goalie", team=T[s], t=f"{T[s]} {nm}: {gs:+.1f} goals saved above expected"))
            elif gs <= -1.5:
                tags.append(dict(k="shaky_goalie", team=T[s], t=f"{T[s]} {nm}: {gs:+.1f} goals above expected allowed"))
        xs = f["xg"].get(s)
        if xs is not None and ((xs >= 0.58 and res_side != s) or (xs <= 0.42 and res_side == s)):
            tags.append(dict(k="xg_mismatch", team=T[s], t=f"{T[s]} had {xs:.0%} of 5v5 expected goals and {'won' if res_side == s else 'lost'}"))
    if f["period"] in ("OT", "SO"):
        tags.append(dict(k="ot_so", team=None, t="Went to " + ("overtime" if f["period"] == "OT" else "a shootout")))
    en = [g for g in f["goals"] if g["en"]]
    if en:
        tags.append(dict(k="en_goal", team=None, t=f"{len(en)} empty-net goal(s): " + ", ".join(T[g["team"]] for g in en)))
    pens = {s: f["pen"][s]["minor"] + f["pen"][s]["major"] for s in ("home", "away")}
    ppg = sum(g["pp"] for g in f["goals"])
    if max(pens.values()) >= 5 or sum(pens.values()) >= 9:
        tags.append(dict(k="penalties", team=None, t=f"Penalty-heavy: {f['home']} {pens['home']}, {f['away']} {pens['away']} penalties; {ppg} PP goal(s)"))
    elif ppg >= 3:
        tags.append(dict(k="pp_swing", team=None, t=f"{ppg} power-play goals"))
    fights = sum(f["pen"][s]["fight"] for s in ("home", "away"))
    if fights:
        tags.append(dict(k="fight", team=None, t=f"{fights // 2 or 1} fight(s)"))
    majors = sum(f["pen"][s]["major"] + f["pen"][s]["misc"] for s in ("home", "away"))
    if majors:
        tags.append(dict(k="major", team=None, t=f"{majors} major/misconduct penalt{'y' if majors == 1 else 'ies'}"))
    if abs(f["hs"] - f["as_"]) >= 4:
        tags.append(dict(k="blowout", team=None, t=f"{max(f['hs'], f['as_'])}-{min(f['hs'], f['as_'])} blowout"))
    return tags


def fmt_toi(sec):
    return f"{sec // 60}:{sec % 60:02d}"


def grade_leg(spec, label, entry, f, usual, gtags):
    k, tags = spec["kind"], []
    tot = f["hs"] + f["as_"] - (1 if f["period"] == "SO" else 0)
    home_won = f["hs"] > f["as_"]
    opp = lambda team: f["away"] if team == f["home"] else f["home"]
    if k == "ml":
        won = home_won == (spec["team"] == f["home"])
        status, actual = ("hit" if won else "miss"), f"{f['away']} {f['as_']} – {f['hs']} {f['home']}" + (f" ({f['period']})" if f["period"] != "REG" else "")
        rel = [t for t in gtags if t["k"] not in ("fight", "major")]
    elif k == "total":
        status = "hit" if (tot > spec["line"] if spec["dir"] == "over" else tot < spec["line"]) else "miss"
        actual = f"{tot} total goals"
        rel = [t for t in gtags if t["k"] in GOALIE_TAGS | {"en_goal", "ot_so", "penalties", "pp_swing", "fight", "blowout"}]
    elif k in ("atg", "sog"):
        sk = f["skaters"].get(str(spec["player"]))
        if not sk:
            return dict(label=label, status="void", actual="did not play", tags=[dict(k="dnp", t=f"{spec.get('name', 'Player')} did not play")])
        status = "hit" if (sk["g"] >= 1 if k == "atg" else sk["sog"] > spec["line"]) else "miss"
        actual = f"{sk['g']} G, {sk['sog']} SOG, {fmt_toi(sk['toi'])} TOI"
        side = "home" if sk["t"] == f["home"] else "away"
        rel = [t for t in gtags if (t["k"] == "scratch" and t["team"] == sk["t"]) or (t["k"] in GOALIE_TAGS and t["team"] == opp(sk["t"]))
               or t["k"] in ("penalties", "pp_swing")]
        u = usual.get(str(spec["player"]))
        if u and u >= 10 * 60 and sk["toi"] < 0.75 * u:
            tags.append(dict(k="toi_drop", t=f"{sk['n']} played {fmt_toi(sk['toi'])} vs usual {fmt_toi(int(u))}"))
        exp = (entry.get("exp_sog") or {}).get("h" if side == "home" else "a")
        if exp and f["team_sog"][side] < 0.8 * exp and exp - f["team_sog"][side] >= 4:
            tags.append(dict(k="shots_low", t=f"{sk['t']} had {f['team_sog'][side]} SOG (expected ~{exp:.0f})"))
        elif exp and f["team_sog"][side] > 1.2 * exp and f["team_sog"][side] - exp >= 4:
            tags.append(dict(k="shots_high", t=f"{sk['t']} had {f['team_sog'][side]} SOG (expected ~{exp:.0f})"))
        if k == "atg" and status == "miss":
            if sk["sog"] >= 3:
                tags.append(dict(k="no_finish", t=f"{sk['n']}: {sk['sog']} shots on goal, no goal"))
            elif sk["sog"] <= 1:
                tags.append(dict(k="few_shots", t=f"{sk['n']}: only {sk['sog']} shot(s) on goal"))
        if k == "sog" and status == "miss" and sk["sog"] <= spec["line"] - 1.5:
            tags.append(dict(k="few_shots", t=f"{sk['n']}: {sk['sog']} shot(s) on goal vs line {spec['line']:g}"))
        if k == "atg" and status == "hit":
            for g in f["goals"]:
                if g["scorer"] == spec["player"] and g["pp"]:
                    tags.append(dict(k="goal_pp", t="Scored on the power play"))
                    break
                if g["scorer"] == spec["player"] and g["en"]:
                    tags.append(dict(k="goal_en", t="Scored into an empty net"))
                    break
    elif k == "saves":
        sidef = "home" if spec["side"] == "h" else "away"
        gl = f["goalies"][sidef]
        starter = next((g for g in gl if g["starter"]), gl[0] if gl else None)
        assumed = (entry.get("starters") or {}).get(spec["side"])
        if not starter or (assumed and gnorm(starter["n"]) != gnorm(assumed)):
            return dict(label=label, status="void", actual=f"{starter['n'] if starter else 'different goalie'} started",
                        tags=[dict(k="goalie_change", t=f"{spec['team']}: leg was for {assumed}")])
        status = "hit" if (starter["sv"] > spec["line"] if spec["dir"] == "over" else starter["sv"] < spec["line"]) else "miss"
        actual = f"{starter['n']}: {starter['sv']} saves on {starter['sa']} shots"
        rel = [t for t in gtags if t["k"] in ("goalie_pulled",) and t["team"] == spec["team"]]
        exp = (entry.get("exp_sog") or {}).get("a" if spec["side"] == "h" else "h")
        if exp and starter["sa"] < 0.8 * exp and exp - starter["sa"] >= 4:
            tags.append(dict(k="sa_low", t=f"Faced {starter['sa']} shots (expected ~{exp:.0f})"))
        elif exp and starter["sa"] > 1.2 * exp and starter["sa"] - exp >= 4:
            tags.append(dict(k="sa_high", t=f"Faced {starter['sa']} shots (expected ~{exp:.0f})"))
    else:
        return dict(label=label, status="void", actual="unknown leg", tags=[])
    return dict(label=label, status=status, actual=actual, tags=[dict(k=t["k"], t=t["t"]) for t in rel] + tags)


def grade_play(play, entry, f, usual, gtags):
    legs = [grade_leg(sp, lb, entry, f, usual, gtags) for sp, lb in zip(play["specs"], play["legs"])]
    sts = [l["status"] for l in legs]
    result = "miss" if "miss" in sts else ("void" if all(s == "void" for s in sts) else "hit")
    return dict(legs=legs, result=result, has_void="void" in sts, p_joint=play["p_joint"], lift=play["lift"],
                sig=play["sig"], fair_odds=play["fair_odds"], market=play.get("market"))


def load_logged():
    out = {}
    for f in sorted(LOG.glob("*.json")) if LOG.exists() else []:
        for e in json.loads(f.read_text())["games"]:
            out[e["gameId"]] = e
    return out


def get_facts(date, gid, row, xg, gs, today):
    FACTS.mkdir(parents=True, exist_ok=True)
    f = FACTS / f"{date}.json"
    cache = json.loads(f.read_text()) if f.exists() else {}
    c = cache.get(str(gid))
    recent = (today - pd.Timestamp(date).date()).days <= 3
    if c and (c.get("complete") or not recent):
        return c
    try:
        c = game_facts(gid, row, xg, gs)
    except Exception as e:
        print("facts failed", gid, repr(e)[:80])
        return c
    cache[str(gid)] = c
    f.write_text(jdump(cache))
    return c


def main():
    today = et_today()
    full = PROC / "sgp_full.json"
    if full.exists():
        log_games(json.loads(full.read_text())["games"], today)
    res = pd.read_csv(PROC / "nhl_schedule_results.csv")
    res = res[(res.season == CUR) & res.state.isin(["OFF", "FINAL"])].set_index("gameId")
    usual = {}
    ts = ROOT / "docs" / "data" / "team_stats.json"
    if ts.exists():
        usual = {k: v * 60 for k, v in json.loads(ts.read_text()).get("toi", {}).items()}
    xg, gs = luck_tables()
    logged = load_logged()
    picks = pd.read_csv(ROOT / "docs" / "data" / "prediction_log.csv")
    gids = sorted(set(logged) | set(picks.gameId))
    games, tagstats, legstats = {}, defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0, 0])
    plays_all, starter_n, starter_wrong = [], 0, 0
    for gid in gids:
        if gid not in res.index:
            continue
        row = res.loc[gid]
        date = str(row["date"])[:10]
        if date >= today.isoformat():
            continue
        f = get_facts(date, gid, row, xg, gs, today)
        if not f:
            continue
        pr = picks[picks.gameId == gid]
        entry = logged.get(gid) or ({"starters": dict(h=pr.iloc[0].h_goalie, a=pr.iloc[0].a_goalie)} if len(pr) else {})
        gtags = game_tags(entry, f, usual)
        if (entry.get("starters") or {}).get("h"):
            starter_n += 2
            starter_wrong += sum(t["k"] == "goalie_change" for t in gtags)
        g = dict(gameId=gid, date=date, away=f["away"], home=f["home"], score=f"{f['as_']}-{f['hs']}", period=f["period"],
                 tags=[dict(k=t["k"], t=t["t"]) for t in gtags], backfill=bool(entry.get("backfill")))
        if len(pr):
            p = float(pr.iloc[0].p_home)
            pick = f["home"] if p >= 0.5 else f["away"]
            g["pick"] = dict(team=pick, conf=round(max(p, 1 - p), 3), correct=bool(pick == (f["home"] if f["hs"] > f["as_"] else f["away"])))
        g["plays"] = []
        for play in entry.get("plays", []) if entry.get("plays") else []:
            gp = grade_play(play, entry, f, usual, gtags)
            g["plays"].append(gp)
            plays_all.append(gp)
            for l, sp in zip(gp["legs"], play["specs"]):
                if l["status"] == "void":
                    legstats[sp["kind"]][2] += 1
                    continue
                legstats[sp["kind"]][0] += 1
                legstats[sp["kind"]][1] += l["status"] == "hit"
                for k in {t["k"] for t in l["tags"]} - DESCRIPTORS:
                    tagstats[k][0] += 1
                    tagstats[k][1] += l["status"] == "miss"
        games[gid] = g
    # ---------------- aggregates
    graded = [p for p in plays_all if p["result"] != "void"]
    n_miss_all = sum(1 for p in graded for l in p["legs"] if l["status"] == "miss")
    n_legs_all = sum(1 for p in graded for l in p["legs"] if l["status"] != "void")
    by_sig = defaultdict(lambda: [0, 0, 0.0])
    by_lift = defaultdict(lambda: [0, 0, 0.0])
    for p in graded:
        for d, k in ((by_sig, p["sig"]), (by_lift, "1.15-1.3" if p["lift"] < 1.3 else "1.3-1.5" if p["lift"] < 1.5 else "1.5+")):
            d[k][0] += 1
            d[k][1] += p["result"] == "hit"
            d[k][2] += p["p_joint"]
    mk = lambda d: [dict(key=k, n=v[0], hit=round(v[1] / v[0], 4), pred=round(v[2] / v[0], 4)) for k, v in sorted(d.items(), key=lambda x: -x[1][0])]
    pk = [g["pick"] for g in games.values() if "pick" in g]
    base_miss = n_miss_all / max(n_legs_all, 1)
    tag_rows = [dict(tag=k, label=LABEL.get(k, k), n_legs=v[0], n_miss=v[1], miss_rate=round(v[1] / v[0], 3))
                for k, v in sorted(tagstats.items(), key=lambda x: -x[1][0]) if v[0] >= 1]
    days = defaultdict(list)
    for g in sorted(games.values(), key=lambda g: (g["date"], g["gameId"]), reverse=True):
        g2 = dict(g, plays=sorted(g["plays"], key=lambda p: -p["lift"])[:PLAYS_SHOWN])
        days[g["date"]].append(g2)
    out = dict(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="minutes"),
        labels=LABEL,
        goalies=dict(n=starter_n, wrong=starter_wrong),
        picks=dict(n=len(pk), acc=round(float(np.mean([p["correct"] for p in pk])), 4) if pk else None,
                   avg_conf=round(float(np.mean([p["conf"] for p in pk])), 4) if pk else None),
        plays=dict(n=len(graded), hit=round(float(np.mean([p["result"] == "hit" for p in graded])), 4) if graded else None,
                   pred=round(float(np.mean([p["p_joint"] for p in graded])), 4) if graded else None,
                   voided=len(plays_all) - len(graded), by_sig=mk(by_sig), by_lift=mk(by_lift)),
        legs=dict(by_kind=[dict(kind=k, n=v[0], hit=round(v[1] / max(v[0], 1), 3), void=v[2]) for k, v in sorted(legstats.items())],
                  base_miss_rate=round(base_miss, 3), tags=tag_rows),
        days=[dict(date=d, games=gs_) for d, gs_ in list(days.items())[:DAYS_SHOWN]])
    OUT.write_text(jdump(out, separators=(",", ":")))
    print(f"graded {len(games)} games, {len(graded)} plays ({len(plays_all) - len(graded)} voided), wrote {OUT} {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "backfill":
        backfill(pd.date_range(sys.argv[2], sys.argv[-1]).strftime("%Y-%m-%d"))
    else:
        main()

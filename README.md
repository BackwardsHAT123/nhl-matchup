# NHL matchup model

Daily use:

    python3 update_data.py          # refresh MoneyPuck + NHL API data (needs network)
    python3 predict.py              # today's slate; add --date, --goalie TOR="Woll", --w1 0.6

Backtest: `python3 backtest.py` (outputs `data/processed/backtest_predictions.csv`).

Pipeline: `build_dataset.py` (MoneyPuck 5v5 xG per game, goalie GSAx per start from shot data)
-> `fetch_results.py` (official winners incl. shootouts, schedule) -> `features.py` -> `backtest.py` / `predict.py`.

Model: composite = 0.75*z(xGF% diff) + 0.15*z(goalie GSAx diff) + 0.10*z(rest/B2B diff);
P(home) = sigmoid(a + c*composite). Home ice is in the intercept `a`.
xGF%: this season's games only (resets each season; window grows to the last 40), shrunk to 50%. GSAx: last 25 starts, shrunk to 0,
calibrated so league-average = 0. Starter = goalie facing the first shot.

Caveats: starters are unknown pre-game (predict.py guesses; override them). No odds are used, so
this measures win-prediction skill, not betting edge.

## Daily automation + dashboard

`.github/workflows/daily.yml` runs twice a day (10:00 and 17:30 ET) and on demand: refreshes data, reruns the
backtest, runs `export_dashboard.py`, and commits `docs/data/*`. GitHub Pages serves `docs/` as the dashboard.

- `docs/data/prediction_log.csv`: every pre-game prediction (frozen once game day starts) -> the live, out-of-sample record.
- Confirm starters: edit `overrides.json` (`{"2026-10-20": {"TOR": "Woll"}}`) and run the workflow manually.
- Setup: push to GitHub, Settings -> Pages -> Deploy from branch `main`, folder `/docs`.

## Same-game parlays

`sgp_model.py` simulates each game's joint outcomes (team shots/goals via a copula using the *measured* correlations,
shots split among the dressed lineup, goals from each player's shots x shooting%) and ranks 2-3 leg combos by
**correlation lift** = P(all legs) / product of leg probabilities. `sgp_slate.py` writes `docs/data/sgp.json` daily.
Legs: team ML, game total 5.5/6.5, top-5 shooters' anytime goal and shots over 1.5/2.5, goalie saves.
Validate/recalibrate (slow, ~7 min, needs `python3 fetch_lineups.py` first): `sgp_validate.py` -> `sgp_calibrate.py`
(writes `sgp_calibration.json`, `docs/data/sgp_validation.json`). EV needs the book's SGP price: type it into the dashboard.

### Live odds (optional)
Add a repo secret `ODDS_API_KEY` (The Odds API key; Settings -> Secrets and variables -> Actions). Each run fetches
moneyline, totals, anytime-goal, shots-on-goal and saves for games starting within 30h: ~5 credits/game, capped by
`ODDS_MAX_CREDITS` (default 120/run). The API has leg prices only (no SGP prices), so the dashboard shows EV at 0% / 50% / 100% of the correlation priced in
by the book (price = legs multiplied / lift^f) and the share of correlation the play survives. Scheduled runs skip odds
(~70 credits/run) unless repo variable `ODDS_SCHEDULED=true`; manual runs always fetch. `tests/test_odds_mock.py`
checks the pricing path against a mock response.

## Matchup drill-down
Click a matchup on the dashboard to expand team and player stats. `team_stats.py` (daily) pulls NHL club-stats
(goals, assists, points, shots) for all 32 teams and writes `docs/data/team_stats.json`, with league ranks so top-10 players get a star.

## Results review
`review.py` runs daily: it logs every parlay candidate (with the lineups and starters the model assumed) to `log/sgp/<date>.json`
before puck drop (never overwritten once a game starts), then grades finished games using NHL boxscore + play-by-play facts
(`review_facts.py`, cached in `log/facts/`). Each leg gets tags explaining the outcome: goalie change/pulled, key player out,
ice-time drop, penalties/fights/majors, empty-net goals, OT/shootout, hot/shaky goalie, and result-vs-xG mismatch.
`python3 review.py backfill 2026-10-19 2026-10-22` rebuilds past days from pre-game data (marked "reconstructed").
Output: `docs/data/results_review.json` (dashboard "Results review"). Injuries are inferred from lineup and ice-time changes, not read from a feed.

## Starting-goalie model
`starters.py` predicts who starts: a boosted classifier scores each recent goalie on usage, tandem alternation, rest and
back-to-backs, whether the team plays tomorrow, this-season starts, schedule density and last-start performance (candidates are
filtered to the current roster). Backtest, next-game accuracy: 2025-26 69.9% vs 63.8% for the old "most-used" rule; 2026-27 to date 71.3% vs 61.4%;
on the 54 logged team-games 75.9% vs 63.0% (92% when it says >=70%). The NHL API publishes no probable starter before game time, so
overrides (`overrides.json`) remain the way to enter a confirmed starter. When uncertain, the main model uses the probability-weighted GSAx.

## xGF% vs. goalie scatter
Dashboard chart: each club's logo at (5v5 xGF%, likely starter's GSAx per start). Default = season to date (starter's starts this
season); toggle = last 5 team games / the starter's last 5 starts. Data: `team_goalies` in `docs/data/summary.json` (export_dashboard.py).
Logos are hot-linked from the NHL's asset host and fall back to the team abbreviation if unavailable.

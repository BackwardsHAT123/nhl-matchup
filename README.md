# NHL matchup model

Daily use:

    python3 update_data.py          # refresh MoneyPuck + NHL API data (needs network)
    python3 predict.py              # today's slate; add --date, --goalie TOR="Woll", --w1 0.6

Backtest: `python3 backtest.py` (outputs `data/processed/backtest_predictions.csv`).

Pipeline: `build_dataset.py` (MoneyPuck 5v5 xG per game, goalie GSAx per start from shot data)
-> `fetch_results.py` (official winners incl. shootouts, schedule) -> `features.py` -> `backtest.py` / `predict.py`.

Model: composite = 0.75*z(xGF% diff) + 0.15*z(goalie GSAx diff) + 0.10*z(rest/B2B diff);
P(home) = sigmoid(a + c*composite). Home ice is in the intercept `a`.
xGF%: last 40 team games (across seasons), shrunk to 50%. GSAx: last 25 starts, shrunk to 0,
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
`ODDS_MAX_CREDITS` (default 120/run). The API has leg prices only, so "EV*" = model hit rate x legs' prices multiplied - 1,
an upper bound; "cushion" = payout haircut the book can apply before the play stops being +EV. `tests/test_odds_mock.py`
checks the pricing path against a mock response.

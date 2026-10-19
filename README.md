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

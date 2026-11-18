# NHL matchup model

Keep this file under ~15 KB. Dated entries, review results and feature write-ups go at the END of NOTES-LOG.md, never here.

## Purpose and owner
- Daily NHL win-probability model, same-game-parlay finder and results review, published as a static dashboard (GitHub Pages from `docs/`). Repo: github.com/BackwardsHAT123/nhl-matchup.
- Owner: Tom. Explain in plain terms; he reads the dashboard and decides features, not the code details.
- Model and feature detail lives in `README.md` (read it before changing the model, parlays, injuries or starters).

## Build, test, run
- Python 3 (CI uses 3.12); deps: `pip install -r requirements.txt` (pandas, numpy, scikit-learn).
- Refresh data (network: moneypuck.com, peter-tanner.com, NHL API): `python3 update_data.py`
- Today's slate: `python3 predict.py` (`--date`, `--goalie TOR="Woll"`, `--w1 0.6`)
- Backtest: `python3 backtest.py` -> `data/processed/backtest_predictions.csv`
- Dashboard data: `python3 export_dashboard.py` (after `injuries.py`), then `team_stats.py`, `sgp_slate.py`, `review.py`.
- Test: `python3 -m pytest tests` (mock odds -> pricing -> slate path).
- View the dashboard locally: `python3 -m http.server 8765 --directory docs` (the `dashboard` config in `.claude/launch.json`).
- Verify a change: rerun the step it touches, check the JSON in `docs/data/`, and look at `docs/index.html` in the local server.

## Language / tooling rules
- `data/raw/` and `data/processed/` are gitignored and rebuilt by `update_data.py`; past-season shot files download once (CI caches them per season).
- Never hand-edit `docs/data/*` or `log/*`: the workflow regenerates and commits them, and its rebase takes the fresh run's copy (`-X theirs`).
- Cron is UTC-only and kept off the top of the hour (GitHub delays on-the-hour runs).

## Architecture map
- Pipeline: `update_data.py` -> `build_dataset.py` (MoneyPuck 5v5 xG per game, goalie GSAx from shots) -> `fetch_results.py` -> `features.py` -> `backtest.py` / `predict.py`.
- `seasons.py`: derives the current season (no hard-coded year).
- `starters.py`: starting-goalie classifier; `overrides.json` holds confirmed starters (`{"2026-10-20": {"TOR": "Woll"}}`).
- `injuries.py`: ESPN injury feed -> `docs/data/injuries.json`, snapshot `log/injuries/<date>.json`.
- Parlays: `sgp_model.py` (joint-outcome simulator), `sgp_slate.py` (-> `docs/data/sgp.json`), `odds.py` + `sgp_pricing.py` (live odds, optional), `sgp_validate.py` / `sgp_calibrate.py` / `sgp_tune.py` / `sgp_report.py` (slow validation, needs `fetch_lineups.py` first), `build_players.py`.
- `team_stats.py` -> `docs/data/team_stats.json` (matchup drill-down).
- `review.py` + `review_facts.py`: log parlay picks pre-game to `log/sgp/<date>.json`, grade finished games (facts cached in `log/facts/`) -> `docs/data/results_review.json`.
- `quadrant_lookback.py`: quadrant matchup lookback table.
- `docs/index.html`: the whole dashboard (single page); data in `docs/data/` (`summary.json`, `predictions.json`, `prediction_log.csv`, ...).
- `.github/workflows/daily.yml`: runs 10:17 and 21:43 UTC plus manual; commits `docs/data` and `log`.

## Conventions
- `docs/data/prediction_log.csv` is the frozen out-of-sample record: a prediction is never changed once game day starts; past dates are never adjusted (injuries included).
- Optional steps in the workflow use `continue-on-error` so they never block the main predictions.
- Odds API credits are rationed: scheduled runs skip odds unless repo variable `ODDS_SCHEDULED=true`; capped by `ODDS_MAX_CREDITS` (default 120/run).
- Dashboard follows the TAH design system (tokens, Outfit font, pills, no shadows).
- Commit messages: short imperative summary of the change; the bot's commits are "Update predictions <UTC time>".

## Standing facts
- **Where history lives:** dated entries go at the end of `NOTES-LOG.md` (append-only, grep it before re-fixing something); git log has the history before 2026-11-18.
- **Status:** live on GitHub Pages, workflow running twice daily. Season 2026-27 in progress.
- **Habits:** pull before pushing; the bot commits to `main` twice a day, so local `main` is often behind.
- **Decisions that stand:** no odds in the win model (it measures win-prediction skill, not betting edge); xGF% uses current-season games only (window grows to 40); morning run moved to 10:17 UTC (2026-11-15).
- **Still unconfirmed:** injury adjustment constants `K` / `REPLACEMENT` / `MAX_PENALTY` in `predict.py` are priors, not fitted; the daily `log/injuries/` snapshots are there to fit them later.

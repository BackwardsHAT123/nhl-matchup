# NOTES-LOG

Dated history for this project (created 2026-11-18 with CLAUDE.md; earlier history is in `git log`).
Append new dated entries at the END; do not add them to CLAUDE.md. Grep this file before re-fixing something.

## 2026-11-26 Best bet row on upcoming games
- Each game card now has a "Best bet" row and each day has a "Strongest" quick list (links jump to the card). Logic is in `bestBet()` in `docs/index.html`.
- Win: model favourite at 60%+ = Lean, 65%+ = Strong (backtest 65%+ bin won ~70%).
- Over/Under: `sgp_slate.py` now writes `p_over55` / `p_over65` per game into `sgp.json` (the JS falls back to the leg probabilities in the plays until a run writes these). The simulator averages more goals than history, so compare against the slate median of P(over 5.5), not the historical rate. A gap of 4+ points = Lean, 6+ = Strong; matching matchup history (quadrant chips) moves it up a step and history pointing the other way moves it down one; a starred history result alone gives a Lean. Show 6.5 for an over (5.5 for an under) only at 55%+.
- Parlay: the most likely lift >= 1.25 play that includes the best-bet legs, with no leg against them.
- Caveat: totals and the parlay hint have no backtest or odds behind them; the cutoffs (4/6 pts, 55%) are priors. Worth grading against results once a few weeks of picks exist.

## 2026-11-26 Win-model recalibration tested and rejected
- Tried two fixes for the 55-65% overconfidence seen last season: (a) 50/50 blend of sigmoid + bounded isotonic, (b) logit shrink toward 50% (temperature t = 0.6-0.9). Both were tested out of sample: trained on earlier seasons, scored on last season (1,312 games) and this season to Nov 25 (399 games).
- (a) Log loss 0.6802 -> 0.6795 last season, worse this season, and it flipped 26 picks (its comment claimed picks were unchanged). Discarded.
- (b) Best last season was t=0.75 (0.6802 -> 0.6789, within noise). This season every t was worse (t=1 0.6622, t=0.8 0.6646), and the model is slightly *under*-confident this season (60-65% picks won 62.5%, 65%+ won 73%). A uniform shrink also makes the 65%+ bucket underconfident.
- Decision: keep the plain sigmoid. Revisit only if the live 55-65% bucket stays overconfident over a full season.
- Local `update_data.py` inside the Claude sandbox: the ~130 MB MoneyPuck `all_teams.csv` gets cut off at 128.6 MB every retry (proxy limit); run it unsandboxed.

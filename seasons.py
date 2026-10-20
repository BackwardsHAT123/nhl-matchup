"""Single source of truth for which seasons the pipeline uses.

Seasons are identified by start year (MoneyPuck convention): 2026 = 2026-27.
"""
from datetime import datetime, timedelta, timezone


def current_season(today=None) -> int:
    """Start year of the in-progress (or just-starting) NHL season.

    The regular season can open in late September, so September onward counts as the new season;
    July-August still belong to the season that just ended.
    """
    today = today or (datetime.now(timezone.utc) - timedelta(hours=5)).date()  # US/Eastern-ish
    return today.year if today.month >= 9 else today.year - 1


CUR = current_season()
BURN_IN = CUR - 3                          # earliest season; only warms up rolling features
SEASONS = list(range(BURN_IN, CUR + 1))    # 4 seasons: burn-in, train, last full season, current


def label(season: int) -> str:
    return f"{season}-{str(season + 1)[2:]}"


if __name__ == "__main__":
    print(CUR, label(CUR), SEASONS)

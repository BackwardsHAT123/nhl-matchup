"""Refresh all data and rebuild features. Run before predict.py.

Downloads MoneyPuck team game-by-game + current-season shots, then NHL API results.
"""
import io
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
CUR = 2026  # MoneyPuck season = start year


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "all_teams.csv").write_bytes(fetch("https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv"))
    for y in range(2023, CUR + 1):  # past seasons never change: download once, refresh the current one
        if y == CUR or not (RAW / f"shots_{y}" / f"shots_{y}.csv").exists():
            z = zipfile.ZipFile(io.BytesIO(fetch(f"https://peter-tanner.com/moneypuck/downloads/shots_{y}.zip")))
            (RAW / f"shots_{y}").mkdir(parents=True, exist_ok=True)
            z.extractall(RAW / f"shots_{y}")
    for script in ("build_dataset.py", "fetch_results.py", "features.py"):
        subprocess.run([sys.executable, str(ROOT / script)], check=True, cwd=ROOT)

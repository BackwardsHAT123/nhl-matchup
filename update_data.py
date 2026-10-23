"""Refresh all data and rebuild features. Run before predict.py.

Downloads MoneyPuck team game-by-game + current-season shots, then NHL API results.
"""
import io
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
from seasons import BURN_IN, CUR  # MoneyPuck season = start year


def fetch(url, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError:
            raise  # 404 etc. is handled by the caller
        except Exception:  # dropped connections on large downloads
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "all_teams.csv").write_bytes(fetch("https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv"))
    for y in range(BURN_IN, CUR + 1):  # past seasons never change: download once, refresh the current one
        if y == CUR or not (RAW / f"shots_{y}" / f"shots_{y}.csv").exists():
            try:
                z = zipfile.ZipFile(io.BytesIO(fetch(f"https://peter-tanner.com/moneypuck/downloads/shots_{y}.zip")))
            except urllib.error.HTTPError as e:
                if y == CUR and e.code == 404:  # new season's file not published yet
                    print(f"shots_{y}.zip not published yet; continuing without it")
                    continue
                raise
            (RAW / f"shots_{y}").mkdir(parents=True, exist_ok=True)
            z.extractall(RAW / f"shots_{y}")
    for script in ("build_dataset.py", "build_players.py", "fetch_results.py", "features.py"):
        subprocess.run([sys.executable, str(ROOT / script)], check=True, cwd=ROOT)

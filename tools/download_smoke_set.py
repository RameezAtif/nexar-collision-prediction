"""Download the 12-video quick test. Run from any directory; no packages required.

Creates a private Kaggle environment, signs in via the browser, and downloads
only missing files. Accept the competition rules on Kaggle before running.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from make_smoke_set import SMOKE_IDS

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ENV = ROOT / ".venv-kaggle"
COMPETITION = "nexar-collision-prediction"
RULES = f"https://www.kaggle.com/competitions/{COMPETITION}/rules"


def run(command):
    subprocess.run([str(part) for part in command], check=True, cwd=ROOT)


def kaggle_command():
    scripts = ENV / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    kaggle = scripts / ("kaggle.exe" if os.name == "nt" else "kaggle")
    if kaggle.is_file():
        return kaggle
    if not python.is_file():
        candidates = [[sys.executable], ["py", "-3"], ["python3"]]
        for candidate in candidates:
            try:
                result = subprocess.run(
                    candidate + ["-c", "import sys; sys.exit(sys.version_info < (3, 11))"],
                    capture_output=True, timeout=15,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if result.returncode == 0:
                print("Setting up Kaggle (first run only)...", flush=True)
                run(candidate + ["-m", "venv", ENV])
                break
        else:
            raise RuntimeError("Kaggle needs Python 3.11 or newer. Install it, then rerun this command.")
    run([python, "-m", "pip", "install", "kaggle>=2.2.5,<3"])
    return kaggle


def download(kaggle, name, destination):
    # A failed transfer never leaves a partial file at the final destination.
    with tempfile.TemporaryDirectory(prefix="kaggle-", dir=RAW) as temp:
        staging = Path(temp)
        run([kaggle, "competitions", "download", "-c", COMPETITION,
             "-f", name, "-p", staging])
        downloaded = staging / name
        if not downloaded.is_file():
            downloaded = staging / Path(name).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if downloaded.is_file() and downloaded.stat().st_size:
            downloaded.replace(destination)
            return
        # Some CLI versions return a zip. Extract only the requested member.
        for archive in staging.rglob("*.zip"):
            with zipfile.ZipFile(archive) as zipped:
                members = [m for m in zipped.infolist()
                           if m.filename in (name, Path(name).name) and m.file_size > 0]
                if len(members) == 1:
                    extracted = staging / "extracted"
                    with zipped.open(members[0]) as source, extracted.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    extracted.replace(destination)
                    return
        raise RuntimeError(f"Kaggle did not return {name}. Rerun to try again.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-login", action="store_true",
                        help="use existing Kaggle credentials instead of browser login")
    args = parser.parse_args()
    files = ["train.csv"] + [f"train/{i:05d}.mp4" for i in SMOKE_IDS]
    missing = [name for name in files
               if not (RAW / name).is_file() or (RAW / name).stat().st_size == 0]
    if missing:
        print(f"Downloading {len(missing)} missing files (full quick test: about 190 MB).", flush=True)
        print(f"First, accept the competition rules: {RULES}", flush=True)
        kaggle = kaggle_command()
        if not args.skip_login:
            print("Sign in to Kaggle in the browser, then return here.", flush=True)
            run([kaggle, "auth", "login"])
        RAW.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(missing, 1):
            print(f"[{index}/{len(missing)}] {name}", flush=True)
            download(kaggle, name, RAW / name)
    print("All 12 videos and train.csv are in data/raw/.")
    print("Next, from the project root: python tools/make_smoke_set.py --build")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        print(f"Download stopped: {exc}", file=sys.stderr)
        print(f"For access errors, check your Kaggle login and accept the rules: {RULES}", file=sys.stderr)
        print("Rerun the same command to retry; completed files will be skipped.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped. Rerun to download the remaining files.", file=sys.stderr)
        sys.exit(130)

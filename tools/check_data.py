"""Check that the raw Nexar data is where the code expects it.

Expected layout (from the project root):

    data/raw/train.csv          columns: id, time_of_event, time_of_alert, target
    data/raw/train/00000.mp4    one file per video id, 5 digits, zero padded

Usage:

    python tools/check_data.py                    # data/raw
    python tools/check_data.py --raw D:/nexar     # raw data elsewhere

A partial download is fine for tools/compare_id_blocks.py --available-only; notebooks 01-05 need all 1,500 videos.
Exit code 0 if train.csv is valid and at least one video opens, 1 otherwise.
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

COLUMNS = ["id", "time_of_event", "time_of_alert", "target"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"), help="folder holding train.csv and train/")
    parser.add_argument("--open", type=int, default=3, help="how many videos to open and read a frame from")
    args = parser.parse_args()

    csv_path, video_dir = args.raw / "train.csv", args.raw / "train"
    if not csv_path.exists():
        print(f"MISSING {csv_path}\nPut train.csv from the Kaggle competition there (see README, Data).")
        return 1
    labels = pd.read_csv(csv_path)
    if labels.columns.tolist() != COLUMNS:
        print(f"UNEXPECTED columns in {csv_path}: {labels.columns.tolist()} (expected {COLUMNS})")
        return 1
    print(f"train.csv: {len(labels)} videos, {int(labels['target'].sum())} collision / {int((labels['target'] == 0).sum())} normal")

    if not video_dir.is_dir():
        print(f"MISSING folder {video_dir}\nPut the {{id:05d}}.mp4 files there (see README, Data).")
        return 1
    present = [i for i in labels["id"] if (video_dir / f"{i:05d}.mp4").exists()]
    extra = {p.name for p in video_dir.glob("*.mp4")} - {f"{i:05d}.mp4" for i in labels["id"]}
    print(f"videos found: {len(present)} of {len(labels)} labelled" + (f"; {len(extra)} files not in train.csv" if extra else ""))
    if not present:
        print("No video matches an id in train.csv. Files must be named like 00000.mp4.")
        return 1

    opened = 0
    for i in present[: args.open]:
        cap = cv2.VideoCapture(str(video_dir / f"{i:05d}.mp4"))
        ok, frame = cap.read()
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"  {i:05d}.mp4: {'ok' if ok else 'CANNOT READ'}" + (f", {frame.shape[1]}x{frame.shape[0]}, {fps:.0f} fps, {n} frames" if ok else ""))
        opened += ok
    if opened == 0:
        return 1
    print("Complete set." if len(present) == len(labels) else f"Partial set: notebooks 01-05 need all {len(labels)}; compare_id_blocks.py --available-only works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

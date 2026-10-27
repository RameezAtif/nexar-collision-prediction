"""Set up a 12-video smoke test of notebooks 01-05, so the raw-data pipeline can be tried without the full 31 GB download.

Why these videos: notebook 03 splits train.csv 80/20 stratified (it needs at least about 10 videos), and notebook 05 has a demo
cell that looks up video 01734 in the TRAINING clips. With these 12 videos the seed-42 split puts 01734 in the training set.

Steps (from the project root, ideally in a throwaway clone because the notebooks overwrite files in data/processed/ and results/):

    python tools/make_smoke_set.py --commands     # print the Kaggle commands that download the 12 videos
    python tools/make_smoke_set.py --build        # trim data/raw/train.csv to the 12 videos and clear the cached ResNet features
    python tools/run_notebooks.py 01_dataset_exploration 02_video_inspection 03_clip_sampling 04_baselines 05_learned_temporal_model
    python tools/make_smoke_set.py --restore      # put the full train.csv back

Only notebooks 01-05 are meant for the small set. Notebooks 06-09 need the committed full-size data and fail or are meaningless on 12 videos.
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
SMOKE_IDS = [0, 3, 13, 14, 24, 1040, 1041, 1042, 1043, 1045, 1046, 1734]  # 5 collision, 7 normal
FEATURES = Path("data/processed/cnn_features")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--commands", action="store_true", help="print the Kaggle download commands")
    group.add_argument("--build", action="store_true", help="trim train.csv to the 12 videos (the full one is saved as train_full.csv)")
    group.add_argument("--restore", action="store_true", help="restore the full train.csv")
    args = parser.parse_args()

    csv, full = RAW / "train.csv", RAW / "train_full.csv"

    if args.commands:
        print("kaggle competitions download -c nexar-collision-prediction -f train.csv -p data/raw")
        for i in SMOKE_IDS:
            print(f"kaggle competitions download -c nexar-collision-prediction -f train/{i:05d}.mp4 -p data/raw")
        return 0

    if args.restore:
        if not full.exists():
            print(f"Nothing to restore: {full} does not exist.")
            return 1
        shutil.move(str(full), str(csv))
        print(f"Restored the full {csv}.")
        return 0

    missing = [i for i in SMOKE_IDS if not (RAW / "train" / f"{i:05d}.mp4").exists()]
    if missing:
        print(f"Missing videos in {RAW / 'train'}: {', '.join(f'{i:05d}.mp4' for i in missing)}\nRun --commands to see how to download them.")
        return 1
    if not csv.exists():
        print(f"{csv} not found; download train.csv first (see --commands).")
        return 1
    if full.exists():
        print(f"{full} already exists, so {csv} is probably already trimmed. Run --restore first.")
        return 1
    labels = pd.read_csv(csv)
    shutil.copy(csv, full)
    small = labels[labels["id"].isin(SMOKE_IDS)]
    small.to_csv(csv, index=False)
    removed = list(FEATURES.glob("*.npy"))
    for p in removed:
        p.unlink()
    print(f"{csv} now lists {len(small)} videos ({int(small['target'].sum())} collision, {int((small['target'] == 0).sum())} normal); full copy: {full}.")
    print(f"Removed {len(removed)} cached ResNet feature files from {FEATURES} (restore them with: git checkout data/processed).")
    print("Next: python tools/run_notebooks.py 01_dataset_exploration 02_video_inspection 03_clip_sampling 04_baselines 05_learned_temporal_model")
    return 0


if __name__ == "__main__":
    sys.exit(main())

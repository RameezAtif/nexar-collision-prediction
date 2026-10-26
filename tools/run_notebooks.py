"""Execute notebooks from the project root and store their outputs in place.

Why a runner: the notebooks use ./data/... and ./results/... paths, which only resolve when the working directory is the project root.
This script always runs them from there, so results reproduce from a clean checkout without any Jupyter configuration.

Usage (from the project root):

    python tools/run_notebooks.py                  # the notebooks that need only the committed data: 06, 07, 08, 09
    python tools/run_notebooks.py 07_final_analysis 09_id_block_confound
    python tools/run_notebooks.py --list           # show the notebooks and what each needs

Notebooks 06-09 use only the committed, cached features in data/processed/ (no raw videos, no GPU). Together they take about 5 minutes on a CPU.
Notebooks 01-05 need the raw Nexar videos in data/raw/ (see README, "Data"); run them by name if you have the data.
All random seeds are set inside the notebooks (42 for splits and sampling; 42-46 for training seeds), so a rerun reproduces the committed result files.
"""

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK_DIR = Path("notebooks")
DEFAULT = ["06_temporal_ablation", "07_final_analysis", "08_failure_analysis", "09_id_block_confound"]
NEEDS = {
    "01_dataset_exploration": "raw data (train.csv)",
    "02_video_inspection": "raw videos",
    "03_clip_sampling": "raw videos; rewrites data/processed/{train,val}_split.csv (seed 42)",
    "04_baselines": "raw videos (optical flow); writes results/baseline_results.csv",
    "05_learned_temporal_model": "raw videos (ResNet features, cached afterwards); writes results/cnn_lstm_results.csv",
    "06_final_analysis_results": "initial analysis, hard-coded numbers (reference only)",
    "06_temporal_ablation": "committed data only; ~2 min",
    "07_final_analysis": "results of notebook 06; seconds",
    "08_failure_analysis": "committed data only; ~2 min",
    "09_id_block_confound": "committed data only; ~1 min",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="*", help="notebook names without .ipynb (default: 06_temporal_ablation 07 08 09)")
    parser.add_argument("--list", action="store_true", help="list notebooks and their requirements")
    parser.add_argument("--timeout", type=int, default=3600, help="per-cell timeout in seconds")
    args = parser.parse_args()

    if not NOTEBOOK_DIR.is_dir():
        print("Run this script from the project root (the folder that contains notebooks/ and data/).")
        return 1
    if args.list:
        for name in sorted(p.stem for p in NOTEBOOK_DIR.glob("*.ipynb")):
            print(f"{name:32s} {NEEDS.get(name, '')}")
        return 0

    names = args.notebooks or DEFAULT
    for name in names:
        path = NOTEBOOK_DIR / f"{name}.ipynb"
        if not path.exists():
            print(f"Notebook not found: {path}. Use --list to see the available names.")
            return 1
        nb = nbformat.read(path, as_version=4)
        start = time.time()
        print(f"running {name} ...", flush=True)
        NotebookClient(nb, timeout=args.timeout, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
        nbformat.write(nb, path)
        print(f"  done in {(time.time() - start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

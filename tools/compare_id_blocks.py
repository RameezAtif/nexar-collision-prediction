"""Compare the two video-id blocks of the Nexar training set using the RAW videos.

Why: every collision video has id <= 1039 and every normal video has id >= 1040 (see EXPERIMENT_LOG.md, E7/E8).
If the two blocks were recorded under different conditions, a classifier can score well by recognising the block
instead of the collision. The feature-based tests in notebooks/09_id_block_confound.ipynb are suggestive but cannot say
WHAT differs. This script measures simple, interpretable video properties for every labelled video and compares the blocks.

It needs the raw videos (data/raw/train/{id:05d}.mp4), which are not stored in the repo.

Usage (from the project root):

    python tools/compare_id_blocks.py                       # all 1,500 labelled videos
    python tools/compare_id_blocks.py --limit 60            # quick smoke test on a subset
    python tools/compare_id_blocks.py --raw D:/nexar/train  # raw videos elsewhere

Outputs (default results/id_blocks/):
    video_properties.csv      one row per video: size, fps, duration, bitrate, brightness, saturation, sharpness, ...
    block_comparison.csv      per property: median/IQR by class, single-property AUC for class, and a within-class id-drift test
    properties_by_id.png      each property against video id, with the class boundary marked
    contact_sheet_*.png       example frames: collision vs normal, boundary collisions, and persistent-error videos
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BOUNDARY = 1039.5          # collision ids <= 1039, normal ids >= 1040
FRACTIONS = (0.25, 0.5, 0.75)

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#8a8985"

PROPERTIES = [
    ("width", "Frame width (px)"), ("height", "Frame height (px)"), ("fps", "Frame rate (fps)"),
    ("duration_s", "Duration (s)"), ("bitrate_kbps", "File bitrate (kbit/s)"),
    ("luminance", "Mean luminance (0-255)"), ("luminance_std", "Luminance contrast (std)"),
    ("dark_fraction", "Share of dark pixels (<40)"), ("saturation", "Mean colour saturation"),
    ("sharpness", "Sharpness (log variance of Laplacian)"),
]


def video_properties(path: Path) -> dict | None:
    """Container properties plus appearance statistics averaged over a few frames. None if unreadable."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or float("nan")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / fps if fps and fps == fps and n_frames > 0 else float("nan")
    size_mb = path.stat().st_size / 1e6
    stats = []
    for frac in FRACTIONS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n_frames - 1, int(frac * n_frames))))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        stats.append((gray.mean(), gray.std(), float((gray < 40).mean()), hsv[..., 1].mean(),
                      float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var()))))
    cap.release()
    if not stats:
        return None
    lum, lstd, dark, sat, sharp = np.mean(stats, axis=0)
    return {"width": width, "height": height, "fps": fps, "frame_count": n_frames, "duration_s": duration,
            "size_mb": size_mb, "bitrate_kbps": size_mb * 8000 / duration if duration == duration and duration > 0 else float("nan"),
            "luminance": lum, "luminance_std": lstd, "dark_fraction": dark, "saturation": sat, "sharpness": sharp}


def read_frame(path: Path, fraction: float = 0.5):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n - 1, int(fraction * n))))
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def contact_sheet(rows, raw_dir: Path, out_path: Path, per_row: int = 6, title: str = ""):
    """rows: list of (row_label, [video ids]). One frame per video (middle of the video)."""
    fig, axes = plt.subplots(len(rows), per_row, figsize=(2.6 * per_row, 1.9 * len(rows) + 0.6), squeeze=False)
    for r, (label, ids) in enumerate(rows):
        for c in range(per_row):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(ids):
                frame = read_frame(raw_dir / f"{ids[c]:05d}.mp4")
                if frame is not None:
                    ax.imshow(frame)
                    ax.set_title(f"id {ids[c]}", fontsize=8, color=INK_2)
        axes[r][0].text(-0.06, 0.5, label, transform=axes[r][0].transAxes, ha="right", va="center", fontsize=9, color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0.08, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/train"), help="folder with {id:05d}.mp4 files")
    parser.add_argument("--splits", type=Path, nargs="+", default=[Path("data/processed/train_split.csv"), Path("data/processed/val_split.csv")],
                        help="split CSVs with columns id, target")
    parser.add_argument("--errors", type=Path, default=Path("results/failure_persistent_errors.csv"),
                        help="optional list of persistent-error videos (column id) for an extra contact sheet")
    parser.add_argument("--out", type=Path, default=Path("results/id_blocks"))
    parser.add_argument("--limit", type=int, default=0, help="only process this many videos (balanced, for a smoke test)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--available-only", action="store_true",
                        help="only use videos whose file exists in --raw (for a partial download); otherwise absent files are reported as missing")
    args = parser.parse_args()

    if not args.raw.is_dir():
        print(f"Raw video folder not found: {args.raw}\nPoint --raw at the folder containing 00000.mp4 ... (the Kaggle 'train' folder).")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    labelled = pd.concat([pd.read_csv(p)[["id", "target"]] for p in args.splits], ignore_index=True)
    labelled["id"] = labelled["id"].astype(float).astype(int)
    labelled = labelled.drop_duplicates("id").sort_values("id").reset_index(drop=True)
    if args.available_only:
        labelled = labelled[[(args.raw / f"{v:05d}.mp4").exists() for v in labelled["id"]]].reset_index(drop=True)
        print(f"--available-only: {len(labelled)} labelled videos have a file in {args.raw}")
    if args.limit:
        labelled = pd.concat([g.sample(min(len(g), args.limit // 2), random_state=args.seed) for _, g in labelled.groupby("target")])
        labelled = labelled.sort_values("id").reset_index(drop=True)

    rows, missing = [], []
    for i, (vid, target) in enumerate(zip(labelled["id"], labelled["target"]), 1):
        path = args.raw / f"{vid:05d}.mp4"
        props = video_properties(path) if path.exists() else None
        if props is None:
            missing.append(vid)
        else:
            rows.append({"id": vid, "target": target, **props})
        if i % 100 == 0 or i == len(labelled):
            print(f"  {i}/{len(labelled)} videos processed ({len(missing)} missing/unreadable)", flush=True)
    if not rows:
        print("No videos could be read; check --raw and the file naming ({id:05d}.mp4).")
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "video_properties.csv", index=False)
    if missing:
        pd.Series(missing, name="id").to_csv(args.out / "missing_or_unreadable_ids.csv", index=False)

    # --- block comparison -------------------------------------------------------------------------
    cmp_rows = []
    for key, label in PROPERTIES:
        v = df[key].astype(float)
        ok = v.notna()
        pos, neg = df[(df["target"] == 1) & ok][key], df[(df["target"] == 0) & ok][key]
        if len(pos) < 3 or len(neg) < 3:
            continue
        auc = roc_auc_score(df.loc[ok, "target"], v[ok]) if df.loc[ok, "target"].nunique() == 2 else float("nan")
        drift = {}
        for target, name in [(1, "collision"), (0, "normal")]:
            sub = df[(df["target"] == target) & ok]
            drift[name] = spearmanr(sub["id"], sub[key]).statistic if len(sub) > 5 and sub[key].nunique() > 1 else float("nan")
        boundary = df[(df["target"] == 1) & (df["id"] >= 1000) & ok][key]
        cmp_rows.append({
            "property": key, "label": label,
            "collision_median": pos.median(), "collision_q25": pos.quantile(0.25), "collision_q75": pos.quantile(0.75),
            "normal_median": neg.median(), "normal_q25": neg.quantile(0.25), "normal_q75": neg.quantile(0.75),
            "AUC_property_predicts_class": auc,
            "n_distinct_values": int(v.nunique()),
            "spearman_id_within_collision": drift["collision"], "spearman_id_within_normal": drift["normal"],
            "boundary_collision_(id1000-1039)_median": boundary.median() if len(boundary) else float("nan"),
        })
    if not cmp_rows:
        print("Fewer than 3 readable videos in one of the classes; skipping the comparison. "
              f"Readable: {int((df['target'] == 1).sum())} collision, {int((df['target'] == 0).sum())} normal "
              f"(see {args.out / 'video_properties.csv'}).")
        return 1
    comparison = pd.DataFrame(cmp_rows)
    comparison.to_csv(args.out / "block_comparison.csv", index=False)

    print("\nProperty comparison (AUC = how well this single property predicts the class; 0.5 = no information):")
    show = comparison[["property", "collision_median", "normal_median", "AUC_property_predicts_class", "n_distinct_values"]]
    print(show.round(3).to_string(index=False))
    strong = comparison[(comparison["AUC_property_predicts_class"] - 0.5).abs() > 0.15]
    print("\nProperties that separate the blocks well (|AUC - 0.5| > 0.15):",
          ", ".join(strong["property"]) if len(strong) else "none")

    # --- can the properties together predict the class, or the id half within a class? -----------
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    feats = [k for k, _ in PROPERTIES if df[k].nunique() > 1]
    Xp = df[feats].fillna(df[feats].median()).to_numpy()
    prng = np.random.default_rng(args.seed)

    def cv_auc(X, y, shuffle_labels=False):
        y = prng.permutation(y) if shuffle_labels else y
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        p = cross_val_predict(GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=args.seed), X, y,
                              cv=folds, method="predict_proba")[:, 1]
        return roc_auc_score(y, p)

    clf_rows = []
    tasks = [("class (collision vs normal)", np.ones(len(df), bool), df["target"].to_numpy())]
    for target, name in [(1, "collision"), (0, "normal")]:
        mask = (df["target"] == target).to_numpy()
        tasks.append((f"lower vs upper id half, within {name}", mask, (df.loc[mask, "id"] > df.loc[mask, "id"].median()).astype(int).to_numpy()))
    for name, mask, y in tasks:
        if len(np.unique(y)) < 2 or np.bincount(y).min() < 10:
            continue
        null = [cv_auc(Xp[mask], y, shuffle_labels=True) for _ in range(10)]
        clf_rows.append({"task": name, "n": int(mask.sum()), "AUC_from_video_properties": cv_auc(Xp[mask], y),
                         "shuffled_label_AUC_mean": float(np.mean(null)), "shuffled_label_AUC_max": float(np.max(null))})
    clf_df = pd.DataFrame(clf_rows)
    clf_df.to_csv(args.out / "property_classifier.csv", index=False)
    print("\nCan the video properties above (not the pixels) predict the class / the id half? 5-fold CV AUC, gradient boosting:")
    print(clf_df.round(3).to_string(index=False))

    # --- figure: property vs id ---------------------------------------------------------------
    n_prop = len(comparison)
    ncols = 3
    nrows = int(np.ceil(n_prop / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 2.9 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, row) in zip(axes.flat, comparison.iterrows()):
        ax.axis("on")
        ax.set_facecolor(SURFACE)
        for target, colour in [(1, BLUE), (0, NEUTRAL)]:
            s = df[df["target"] == target]
            ax.scatter(s["id"], s[row["property"]], s=7, color=colour, alpha=0.5, edgecolors="none")
        ax.axvline(BOUNDARY, color=ORANGE, linewidth=1.4, linestyle=(0, (4, 3)))
        ax.set_title(row["label"], fontsize=9, fontweight="bold")
        ax.set_xlabel("Video id", fontsize=8)
        ax.grid(color=GRID, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.suptitle("Video properties by id (blue = collision, grey = normal, dashed = class boundary)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out / "properties_by_id.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)

    # --- contact sheets ------------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)

    def pick(mask, n=6):
        ids = df[mask]["id"].to_numpy()
        return sorted(rng.choice(ids, min(n, len(ids)), replace=False).tolist()) if len(ids) else []

    contact_sheet([("Collision\n(random)", pick(df["target"] == 1)), ("Normal\n(random)", pick(df["target"] == 0))],
                  args.raw, args.out / "contact_sheet_random.png", title="Random videos, middle frame")
    contact_sheet([("Collision,\nid 1000-1039", pick((df["target"] == 1) & (df["id"] >= 1000))),
                   ("Collision,\nid < 100", pick((df["target"] == 1) & (df["id"] < 100))),
                   ("Normal,\nid 1040-1099", pick((df["target"] == 0) & (df["id"] < 1100))),
                   ("Normal,\nid > 2000", pick((df["target"] == 0) & (df["id"] > 2000)))],
                  args.raw, args.out / "contact_sheet_boundary.png", title="Videos next to the class boundary vs far from it")
    if args.errors.exists():
        err = pd.read_csv(args.errors)
        err["id"] = err["id"].astype(int)
        have = set(df["id"])
        kinds = [("False alarms\n(normal flagged)", err[err["error_type"] == "false alarm"]),
                 ("Missed collisions", err[err["error_type"] == "missed collision"])]
        sheet = [(name, [i for i in sub["id"].tolist() if i in have][:6]) for name, sub in kinds]
        contact_sheet(sheet, args.raw, args.out / "contact_sheet_persistent_errors.png",
                      title="Videos the model got wrong at every horizon (most confident first)")

    print(f"\nWrote results to {args.out}/ (video_properties.csv, block_comparison.csv, properties_by_id.png, contact_sheet_*.png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

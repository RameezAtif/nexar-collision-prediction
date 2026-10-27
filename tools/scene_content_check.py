"""Quick scene-content check of the two video-id blocks, using zero-shot CLIP on the raw videos that are available.

Question (EXPERIMENT_LOG.md, E12): do collision videos (ids <= 1039) and normal videos (ids >= 1040) differ in WHAT they show
(time of day, weather, road type, traffic), beyond the recording properties already compared in E9?

Method: 3 frames per video (at 25%, 50%, 75% of its length) are scored by the pretrained CLIP model (openai/clip-vit-base-patch32,
MIT licence) against plain-English descriptions ("a dashcam photo of night", ...). The 71 persistent-error videos of notebook 08 are left
out because they were selected for being hard for our model. Four analyses:
    A  per-description comparison, collision vs normal (permutation test, Benjamini-Hochberg corrected), plus the dominant label per group
    B  drift of each description with the video id INSIDE each class (a slow change in recording conditions would show up here)
    C  can the classes be told apart from scene appearance alone (5-fold cross-validated AUC), near the class boundary vs far from it

Extra requirement (not in requirements.txt):  pip install transformers      (tested with 5.20.2; downloads about 600 MB of model files once)

Usage (from the project root; needs raw videos in data/raw/train/, a partial download is fine):

    python tools/scene_content_check.py                  # all available videos except the persistent-error ones
    python tools/scene_content_check.py --raw D:/nexar   # raw data elsewhere

Outputs (default results/scene_content/): videos_used.csv, A_attributes_collision_vs_normal.csv, A2_dominant_label_by_group.csv,
B_within_class_drift.csv, C_scene_classifier_auc.csv (and clip_embeddings.npz, a cache that is not committed).
The numbers depend on WHICH videos are available (the committed results use the 431-video sample of E9, minus the 71 error videos = 360).
Seeds: 42 for all permutations and subsamples.
"""

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2_contingency, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

GROUPS = {
    "time of day": ["daytime", "night", "dusk or dawn"],
    "weather": ["clear sunny weather", "rain on the windshield", "snow", "fog or an overcast sky"],
    "road type": ["a highway or freeway", "a busy city street", "a quiet residential street", "a rural road",
                  "a parking lot", "a tunnel or underpass", "a bridge"],
    "traffic": ["heavy traffic with many cars", "light traffic with a few cars", "an empty road with no other cars"],
}
PROMPTS = [(g, p) for g, ps in GROUPS.items() for p in ps]
NAMES = [f"{g}: {p}" for g, p in PROMPTS]
TEMPLATE = "a dashcam photo of {}"
MODEL = "openai/clip-vit-base-patch32"
N_PERM = 2000


def frames_of(path, fractions=(0.25, 0.5, 0.75)):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for f in fractions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * f))
        ok, frame = cap.read()
        if ok:
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def as_tensor(x):
    """transformers versions differ in what get_*_features returns."""
    if torch.is_tensor(x):
        return x
    for key in ("image_embeds", "text_embeds", "pooler_output"):
        if getattr(x, key, None) is not None:
            return getattr(x, key)
    return x[0]


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p)
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate((p[order] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def embed(meta, raw_dir, cache):
    """CLIP image embeddings (videos, 3 frames, 512) and text embeddings (prompts, 512)."""
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["ids"], meta["id"].values):
            return z["emb"], z["txt"]
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(MODEL).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)
    norm = torch.nn.functional.normalize
    with torch.no_grad():
        tok = proc(text=[TEMPLATE.format(p) for _, p in PROMPTS], return_tensors="pt", padding=True)
        txt = norm(as_tensor(model.get_text_features(**dict(tok.items()))), dim=-1).numpy()
        emb = []
        for i, vid in enumerate(meta["id"], 1):
            x = proc(images=frames_of(raw_dir / "train" / f"{vid:05d}.mp4"), return_tensors="pt")
            e = norm(as_tensor(model.get_image_features(pixel_values=x["pixel_values"])), dim=-1).numpy()
            emb.append(e if len(e) >= 3 else np.vstack([e] * 3)[:3])
            if i % 60 == 0:
                print(f"  embedded {i}/{len(meta)} videos", flush=True)
    emb = np.stack([e[:3] for e in emb])
    np.savez(cache, ids=meta["id"].values, emb=emb, txt=txt)
    return emb, txt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"), help="folder with train.csv and train/")
    parser.add_argument("--out", type=Path, default=Path("results/scene_content"))
    parser.add_argument("--errors", type=Path, default=Path("results/failure_persistent_errors.csv"),
                        help="videos to leave out (the persistent errors of notebook 08)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    if not (args.raw / "train").is_dir() or not (args.raw / "train.csv").exists():
        print(f"Raw data not found under {args.raw}. See README, Data (python tools/check_data.py).")
        return 1
    try:
        import transformers  # noqa: F401
    except ImportError:
        print("This tool needs the transformers package:  pip install transformers")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    labels = pd.read_csv(args.raw / "train.csv", dtype={"id": int})[["id", "target"]]
    have = {int(p.stem) for p in (args.raw / "train").glob("*.mp4")}
    errors = set(pd.read_csv(args.errors)["id"].astype(int)) if args.errors.exists() else set()
    meta = labels[labels["id"].isin(have) & ~labels["id"].isin(errors)].sort_values("id").reset_index(drop=True)
    y, ids = meta["target"].values, meta["id"].values
    if len(set(y)) < 2 or min((y == 1).sum(), (y == 0).sum()) < 20:
        print("Need at least 20 videos of each class (excluding the error videos); download more (see README, Data).")
        return 1
    print(f"videos: {len(meta)} ({int((y == 1).sum())} collision, {int((y == 0).sum())} normal); excluded {len(errors)} persistent-error videos")
    meta.to_csv(args.out / "videos_used.csv", index=False)

    emb, txt = embed(meta, args.raw, args.out / "clip_embeddings.npz")

    # zero-shot probabilities: softmax inside each description group, averaged over the 3 frames
    logits = 100.0 * emb @ txt.T
    P = np.zeros(logits.shape)
    start = 0
    for _, ps in GROUPS.items():
        sl = slice(start, start + len(ps))
        e = np.exp(logits[:, :, sl] - logits[:, :, sl].max(-1, keepdims=True))
        P[:, :, sl] = e / e.sum(-1, keepdims=True)
        start += len(ps)
    P = P.mean(1)

    # A. per-description comparison, collision vs normal
    rows = []
    for k, name in enumerate(NAMES):
        a, b = P[y == 1, k], P[y == 0, k]
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or 1e-9
        null = []
        for _ in range(N_PERM):
            yy = rng.permutation(y)
            null.append(P[yy == 1, k].mean() - P[yy == 0, k].mean())
        p = (np.sum(np.abs(null) >= abs(a.mean() - b.mean())) + 1) / (len(null) + 1)
        rows.append((name, a.mean(), b.mean(), (a.mean() - b.mean()) / sd, p))
    A = pd.DataFrame(rows, columns=["attribute", "mean_collision", "mean_normal", "cohens_d", "perm_p"])
    A["q_BH"] = bh(A["perm_p"].values)
    A.to_csv(args.out / "A_attributes_collision_vs_normal.csv", index=False)

    # A2. dominant label per group
    rows, start = [], 0
    for g, ps in GROUPS.items():
        lab = P[:, start:start + len(ps)].argmax(1)
        tab = np.array([[np.sum((lab == j) & (y == c)) for c in (1, 0)] for j in range(len(ps))])
        keep = tab.sum(1) > 0
        chi = chi2_contingency(tab[keep], correction=False)[0]
        null = []
        for _ in range(N_PERM):
            yy = rng.permutation(y)
            t2 = np.array([[np.sum((lab == j) & (yy == c)) for c in (1, 0)] for j in range(len(ps))])
            null.append(chi2_contingency(t2[keep], correction=False)[0])
        p = (np.sum(np.array(null) >= chi) + 1) / (len(null) + 1)
        for j, label in enumerate(ps):
            rows.append((g, label, tab[j, 0] / (y == 1).sum(), tab[j, 1] / (y == 0).sum(), chi, p))
        start += len(ps)
    pd.DataFrame(rows, columns=["group", "label", "share_collision", "share_normal", "chi2", "perm_p"]).to_csv(
        args.out / "A2_dominant_label_by_group.csv", index=False)

    # B. drift with the id inside each class
    rows = []
    for c, cname in ((1, "collision"), (0, "normal")):
        m = y == c
        for k, name in enumerate(NAMES):
            r, p = spearmanr(ids[m], P[m, k])
            rows.append((cname, name, r, p))
    B = pd.DataFrame(rows, columns=["class", "attribute", "spearman_id", "p"])
    B["q_BH"] = B.groupby("class")["p"].transform(lambda s: bh(s.values))
    B.to_csv(args.out / "B_within_class_drift.csv", index=False)

    # C. can scene appearance alone separate the classes? (mean CLIP embedding, logistic regression, 5-fold CV AUC)
    V = emb.mean(1)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)

    def cv_auc(idx, labels_, seed):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        pr = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), V[idx], labels_, cv=skf, method="predict_proba")[:, 1]
        return roc_auc_score(labels_, pr)

    every = np.arange(len(y))
    auc_all = cv_auc(every, y, 0)
    null = [cv_auc(every, yy, 1) for yy in (rng.permutation(y) for _ in range(100))]
    near = np.where((ids >= 940) & (ids <= 1139))[0]
    far = np.where((ids < 940) | (ids > 1139))[0]
    auc_near = cv_auc(near, y[near], 0)
    far_aucs = []
    for r in range(30):
        sub = rng.choice(far, size=len(near), replace=False)
        if len(set(y[sub])) == 2:
            far_aucs.append(cv_auc(sub, y[sub], r))
    C = pd.DataFrame([("all videos", len(every), auc_all, np.mean(null), np.max(null)),
                      ("near boundary (ids 940-1139)", len(near), auc_near, np.nan, np.nan),
                      ("far from boundary, same size (mean of 30 draws)", len(near), np.mean(far_aucs), np.nan, np.nan)],
                     columns=["subset", "n", "cv_auc", "null_mean_auc", "null_max_auc"])
    C.to_csv(args.out / "C_scene_classifier_auc.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nA. largest differences, collision vs normal (standardised mean difference d):")
    print(A.reindex(A["cohens_d"].abs().sort_values(ascending=False).index).head(8).round(3).to_string(index=False))
    print(f"   {int((A['q_BH'] < 0.05).sum())} of {len(A)} descriptions differ at q < 0.05")
    print("\nA2. dominant label per group (share of collision vs normal videos):")
    for g, sub in pd.read_csv(args.out / "A2_dominant_label_by_group.csv").groupby("group", sort=False):
        print(f"   {g:12s} perm p = {sub['perm_p'].iloc[0]:.3f}   " + " | ".join(
            f"{r.label}: {r.share_collision:.0%} vs {r.share_normal:.0%}" for r in sub.itertuples()))
    print(f"\nB. descriptions drifting with id inside a class (q < 0.05): {int((B['q_BH'] < 0.05).sum())} of {len(B)}")
    print("\nC. separable by scene appearance alone?")
    print(C.round(3).to_string(index=False))
    print(f"\nWrote results to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

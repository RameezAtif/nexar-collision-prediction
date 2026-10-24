"""Extract the frames used on the slides from the raw Nexar videos (needs data/raw/train/*.mp4; not committed, see README).

Frames come from the Nexar Dashcam Collision Prediction dataset (Moura, Zhu, Zvitia, CVPR Workshops 2025) and are used with attribution under the
Nexar open data licence. Output: slides/frames/*.jpg (gitignored). Run from the project root:  python slides/extract_frames.py
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAW = Path("data/raw/train")
OUT = Path("slides/frames")
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv("data/raw/train.csv")
train["id"] = train["id"].astype(int)
EVENT = train.set_index("id")["time_of_event"].to_dict()
WIDTH = 1280


def read(vid, sec):
    cap = cv2.VideoCapture(str(RAW / f"{vid:05d}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(max(round(sec * fps), 0), n - 1)))
    ok, frame = cap.read()
    cap.release()
    assert ok, (vid, sec)
    return frame


def save(name, frame, width=WIDTH):
    h = int(frame.shape[0] * width / frame.shape[1])
    cv2.imwrite(str(OUT / f"{name}.jpg"), cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])


# --- hero storyline: a car cuts in on the bridge (id 1024), relative to the annotated event
ev = EVENT[1024]
for tag, off in [("m3.0", -3.0), ("m1.5", -1.5), ("m1.0", -1.0), ("m0.5", -0.5), ("ev", 0.0)]:
    save(f"hero1024_{tag}", read(1024, ev + off))

# --- title / closing backgrounds
save("bg_title", read(502, EVENT[502] - 1.5))
save("bg_close", read(540, EVENT[540] - 1.0))

# --- data mosaic: day, rain, night, autumn; collision (c) and normal (n) videos
for vid, off, kind in [(872, -1.0, "c"), (677, -0.5, "c"), (540, -1.5, "c"), (13, -1.5, "c")]:
    save(f"mos_{kind}{vid}", read(vid, EVENT[vid] + off))
for vid in (1736, 1770):
    save(f"mos_n{vid}", read(vid, 19.0))

# --- single-frame position story (id 329): far and near
save("pos329_far", read(329, EVENT[329] - 3.0))
save("pos329_near", read(329, EVENT[329] - 0.5))

# --- optical flow illustration: two sampled frames 0.133 s apart at the end of the TTE = 1.0 s window of id 1024
tp = ev - 1.0
t1, t0 = tp - 2.0 / 15, tp
f0, f1 = read(1024, t1), read(1024, t0)
save("flow_a", f0)
save("flow_b", f1)
g0, g1 = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (f0, f1))
flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv = np.zeros_like(f0)
hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 1] = 255
strength = np.clip(mag / max(np.percentile(mag, 99), 1e-6), 0, 1) ** 0.6          # brighten weak motion so it is visible on a slide
hsv[..., 2] = (strength * 255).astype(np.uint8)
colour = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
base = (cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR).astype(np.float32)) * 0.35               # dimmed frame for context
alpha = (0.25 + 0.75 * strength)[..., None]
save("flow_vis", np.clip(base * (1 - alpha) + colour * alpha, 0, 255).astype(np.uint8))

# --- the 16 sampled frames of that clip (small thumbnails)
for k, t in enumerate(np.linspace(tp - 2.0, tp, 16)):
    save(f"clip16_{k:02d}", read(1024, t), width=360)

# --- failure cases at TTE 1.0 s: last frame of the clip, with the ensemble probability
pred = pd.read_csv("results/val_predictions.csv")
pred = pred[pred["tte"] == 1.0].set_index("id")
errs = pd.read_csv("results/failure_persistent_errors.csv")
rows = []
for kind, ids in [("fa", [1694, 1950, 1295]), ("miss", [362, 403, 478])]:
    for vid in ids:
        r = pred.loc[vid]
        save(f"err_{kind}_{vid}", read(vid, float(r["prediction_time"])))
        rows.append({"kind": kind, "id": vid, "p_lstm_tte1.0": round(float(r["p_lstm"]), 2)})
pd.DataFrame(rows).to_csv(OUT / "failure_examples.csv", index=False)
print("wrote", len(list(OUT.glob("*.jpg"))), "frames to", OUT)
print(pd.DataFrame(rows).to_string(index=False))

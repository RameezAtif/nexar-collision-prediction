# Anticipating Dashcam Collisions

**Can a model see a collision coming from dashcam video — and does learned temporal modelling beat simple motion cues?**
A controlled comparison on the [Nexar](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) dashcam collision dataset, at 0.5, 1.0 and 1.5 seconds before the annotated event.

COMS4036A Computer Vision, University of the Witwatersrand · Muhammad Hoosen (2552770) · Rameez Atif (2612521)

**Quick links:** [Experiment log](EXPERIMENT_LOG.md) · [Notebook guide](notebooks/README.md) · [Original design document](docs/PROJECT_DESIGN.md)

---

## Overview

We compare three models under one protocol (same video-level split, same clips, same metric):

| Model | Description |
|---|---|
| **0 — Naive** | Predicts "collision" for every clip (the AP floor). |
| **1 — Optical flow** | Farnebäck flow between sampled frames → 21 motion statistics → logistic regression. |
| **2 — CNN + LSTM** | Frozen ImageNet ResNet-18 features for 16 frames → one-layer LSTM → collision probability. |

Controls on the same features (last frame only, mean-pooling, shuffled frame order) test whether *temporal* modelling is what helps.

## Results at a glance

Average precision (AP) on the 300 held-out test videos; mean ± std over 5 training seeds. Chance = 0.5.

| Model | 0.5 s | 1.0 s | 1.5 s |
|---|---:|---:|---:|
| Naive | 0.500 | 0.500 | 0.500 |
| Optical flow | 0.575 | 0.576 | 0.569 |
| **CNN + LSTM** | **0.762 ± 0.019** | **0.749 ± 0.011** | **0.675 ± 0.021** |
| *Control: last frame only* | *0.759 ± 0.005* | *0.752 ± 0.007* | *0.709 ± 0.018* |

![AP and recall versus time to event](results/figures/final_fig1_main_result.png)

**What we found**

- **Learned features beat motion cues** at every horizon (AP +0.197 / +0.180 / +0.117; all paired 95% bootstrap CIs exclude zero).
- **Temporal modelling is not demonstrated.** A single frame does as well as the LSTM, as do mean-pooling and an LSTM trained on shuffled frames — the gain comes from the pretrained visual features.
- **Performance falls with horizon**, but only the 0.5 s → 1.5 s drop (0.086 AP, 95% CI 0.03–0.14) is distinguishable from zero. The proposal's hypothesis (the learned model degrades *more slowly* than optical flow) is **not supported**.
- **Much of the signal is present long before the event** (a single frame from 3+ s before it already gives AP ≈ 0.69, range 0.68–0.70; chance is 0.5), so we describe the model as recognising high-risk scenes rather than detecting imminent collisions.

![Temporal ablation](results/figures/final_fig2_temporal_ablation.png)

![Single-frame AP by frame position](results/figures/failure_fig1_frame_position.png)

## Quick start

Tested from a fresh `git clone` on Windows 10 with Python 3.10, on a CPU. No GPU is needed: notebooks 06–09, which produce the reported results, deliberately run on the CPU. Notebook 05 (ResNet features and the original CNN+LSTM) uses an Apple GPU (MPS) or an NVIDIA GPU (CUDA) automatically when one is available, and the optical-flow notebook (04) is CPU only.

```bash
git clone https://github.com/RameezAtif/nexar-collision-prediction.git
cd nexar-collision-prediction
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # a few minutes (includes PyTorch)

python tools/run_notebooks.py      # reproduces all results (notebooks 06–09), about 5 minutes
```

No raw videos are needed for this: the split, clip definitions and cached features are committed in `data/processed/`.

## Reproducibility

- **Seeds:** 42 for the split and sampling; 42–46 for training; fixed seeds for bootstrap and permutation tests.
- **Verified:** a fresh clone with no raw data regenerated all 24 result CSVs identically (difference < 1e-9) on the same machine. Across operating systems and library versions expect small numerical differences (for example ≤ 0.0007 AP for the scaled optical-flow baseline).

### Where each reported number comes from

| Report item | File | Produced by |
|---|---|---|
| Main results (AP, recall at fixed false-alarm rate), mean ± std over 5 seeds | `results/temporal_ablation_summary.csv` | notebook 06 |
| Paired bootstrap confidence intervals | `results/temporal_ablation_bootstrap.csv` | notebook 06 |
| Single-frame AP by frame position; annotated-alert timing | `results/failure_frame_position_ap.csv`, `results/failure_alert_timing.csv` | notebook 08 |
| Video-id block tests | `results/id_block_test_*.csv`, `results/id_blocks/*.csv` | notebook 09, `tools/compare_id_blocks.py` |
| Scene-content check of the id blocks | `results/scene_content/*.csv` | `tools/scene_content_check.py` |
| Selection-optimism check | `results/selection_check_fulldata.csv` | `tools/fulldata_selection_check.py` |
| Figures and tables of the written report (submitted separately) | generated from the CSVs above | `tools/make_report_assets.py` |

## Step-by-step: rebuilding from the raw videos

The quick start reuses the committed features. To rebuild everything from the Nexar videos, follow these steps. Do this in a separate clone if you want to keep the committed files untouched, because the notebooks overwrite them.

1. **Install.** Clone the repo and install the requirements, as in the quick start.
2. **Download the data.** Accept the Kaggle competition rules, then fetch `train.csv` and the training videos into `data/raw/` (commands and exact layout in [Get the data](#get-the-data) below).
3. **Check it.** Run `python tools/check_data.py`. You want `videos found: 1500 of 1500 labelled` and `Complete set.`
4. **Optional: re-extract the ResNet features.** Notebook 05 reuses `data/processed/cnn_features/*.npy` if they exist. Delete them to compute the features again.
5. **Rebuild the inputs and the original models** (slow: hours on a CPU; the optical-flow notebook prints nothing until it finishes):
   ```bash
   python tools/run_notebooks.py 03_clip_sampling 04_baselines 05_learned_temporal_model
   ```
6. **Regenerate the reported results** (about 5 minutes): run the quick-start command again, `python tools/run_notebooks.py`.
7. **See what changed.** `git status` and `git diff --stat` list the files that differ from the committed ones. Expect small numerical differences across machines and library versions.

### Get the data

The raw Nexar videos (about 31 GB; 25.5 GB of training videos) are **not** in this repository. Only notebooks 01–05 and `tools/compare_id_blocks.py` need them.
The main results reproduce without them. Both ways start on the [Kaggle competition page](https://www.kaggle.com/competitions/nexar-collision-prediction/data), where you accept the rules.

**Everything, in the browser.** On that page click **Download All**. It is a zip of about 31 GB, so you need about twice that much free disk space to unzip it. Unzip it into `data/raw/` so that you get `data/raw/train.csv` and `data/raw/train/00000.mp4`, and so on. Then run `python tools/check_data.py` to confirm.

**Or with the command line** (also for a partial download, such as the 12-video quick test). Use the Kaggle command-line tool, **version 2.x**. It needs Python 3.11 or newer; check with `kaggle --version`. Setup details are in [Kaggle's instructions](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md).

```bash
pip install kaggle
kaggle auth login        # opens your browser, once
kaggle competitions download -c nexar-collision-prediction -f train.csv -p data/raw
kaggle competitions download -c nexar-collision-prediction -f train/00822.mp4 -p data/raw   # one video; repeat per id
kaggle competitions download -c nexar-collision-prediction -p data/raw   # the whole competition, same zip as Download All
```

For the 12-video quick test, `python tools/make_smoke_set.py --commands` prints the exact download commands.

**Where it goes.** The code reads this exact layout (relative to the project root):

```text
data/raw/
├── train.csv              columns: id, time_of_event, time_of_alert, target
└── train/
    ├── 00000.mp4          one file per video id, 5 digits, zero padded
    └── ...                1,500 files for the full set
```

A partial download is enough for `python tools/compare_id_blocks.py --available-only`; notebooks 01–05 need all 1,500 videos, or the 12-video quick test described below. The [Hugging Face copy](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) is gated and uses a different folder layout (`train/positive/`, `train/negative/`); we did not use it, so its files would have to be rearranged into the layout above.

### Quick test of notebooks 01–05 on 12 videos

To try the raw-video notebooks without the 31 GB download, use a small set of 12 videos (about 190 MB). Do this in a throwaway clone, because the notebooks overwrite the committed files in `data/processed/` and `results/`.

```bash
python tools/make_smoke_set.py --commands    # prints the Kaggle commands for train.csv and the 12 videos; run them from the project root
python tools/make_smoke_set.py --build       # trims data/raw/train.csv to the 12 videos and clears the cached ResNet features
python tools/run_notebooks.py 01_dataset_exploration 02_video_inspection 03_clip_sampling 04_baselines 05_learned_temporal_model   # about 4 minutes
python tools/make_smoke_set.py --restore     # puts the full train.csv back
```

The 12 videos are the smallest set that works: notebook 03 needs about 10 videos for its stratified 80/20 split, and notebook 05 has a demo cell that looks up video 01734 in the training clips. Only notebooks 01–05 are meant for this set; the results (AP and so on) are meaningless on 12 videos, and notebooks 06–09 need the committed full-size data.

The downloads need the Kaggle command-line tool and a login: see [Get the data](#get-the-data) above.

## Commands

| Command | What it does | Needs raw data |
|---|---|:---:|
| `python tools/run_notebooks.py` | Runs notebooks 06–09 and rewrites the files in `results/` | no |
| `python tools/make_report_assets.py` | Generates the figures and LaTeX tables used in the written report from `results/*.csv` (writes to a local, untracked `report/` folder) | no |
| `python tools/run_notebooks.py 03_clip_sampling 04_baselines 05_learned_temporal_model` | Rebuilds the split, clips and optical-flow features from the raw videos and retrains the original baselines and CNN+LSTM (slow: hours on a CPU). Notebook 05 reuses `data/processed/cnn_features/*.npy` if they exist; delete them to re-extract the ResNet features. Run `python tools/run_notebooks.py` afterwards to regenerate the reported results (notebooks 06–09) | yes |
| `python tools/compare_id_blocks.py --available-only` | Raw-video comparison of the two video-id blocks | yes |
| `python tools/fulldata_selection_check.py` | Quantifies the cost of choosing the epoch on the test videos | no |
| `python tools/scene_content_check.py` | Zero-shot CLIP check of whether the two id blocks differ in scene content; needs `pip install transformers` and raw videos (a partial set is fine) | yes |
| `python tools/run_notebooks.py --list` | Lists the notebooks and what each needs | no |
| `python tools/check_data.py` | Checks that `data/raw/train.csv` and `data/raw/train/*.mp4` are in place | yes |
| `python tools/make_smoke_set.py --build` | Sets up the 12-video quick test of notebooks 01–05 (see above); `--commands` lists the downloads, `--restore` undoes it | 12 videos |

## Repository layout

```text
.
├── README.md
├── EXPERIMENT_LOG.md        every experiment, decision and open item
├── requirements.txt
├── data/processed/          splits, clip definitions, cached ResNet and optical-flow features (committed)
├── notebooks/               01–09: exploration → baselines → CNN+LSTM → ablation → analysis (see notebooks/README.md)
├── tools/                   run_notebooks.py, check_data.py, make_smoke_set.py, make_report_assets.py, scene_content_check.py, compare_id_blocks.py, fulldata_selection_check.py
├── results/                 result CSVs and figures/
└── docs/                    200notok_proposal.tex (project proposal), PROJECT_DESIGN.md (original design document)
```

## Experimental protocol

- **Split:** 1,200 training / 300 test videos, stratified 50/50, made at video level *before* any clip is cut (seed 42, zero overlap checked).
- **Clips:** for a collision video with event time *t*ₑ and a chosen TTE, the input is 16 frames sampled evenly from the 2 s ending at *t*ₑ − TTE (no frame after that time). Normal videos get a prediction time drawn once from the collision distribution.
- **Training:** Adam, lr 1e-3, batch 32, 10 epochs; epoch chosen on 20% of the *training* videos, never on the test videos; 5 seeds.
- **Metrics:** average precision (primary); recall at ≤ 10% false-alarm rate; ROC-AUC, precision, recall, F1 in the CSVs. Uncertainty: seed std plus a paired bootstrap over the 300 test clips (2,000 resamples).

## Limitations

- **Video-id blocks.** Every collision video has id ≤ 1039 and every normal video id ≥ 1040. No model uses the id, but the blocks might differ in recording conditions. On a 431-video sample we found no difference in resolution, frame rate, duration, brightness or overlay. A quick scene check on 360 of these videos (zero-shot CLIP, `tools/scene_content_check.py`) found no difference in time of day or weather and no drift with the id, but the blocks differ in traffic: collision videos are more often busy city traffic (49% vs 27%) and normal videos more often empty roads (34% vs 16%). So we say the model separates the two groups, plausibly by scene type, not that it detects imminent collisions.
- **No separate untouched test set** (the 300 held-out videos informed earlier choices), and only 300 clips, so an AP has a 95% interval of about ±0.07.
- **One frozen backbone and one clip design;** the optical-flow baseline uses frames about 0.13 s apart, so classical cues may be underestimated.
- **Earlier numbers.** Notebook 05 and `results/final_analysis/` pick the best epoch on the test videos and are optimistic by about 0.04–0.07 AP; they are kept for reference only (see `EXPERIMENT_LOG.md`).

## Data attribution and licence

This project uses the **Nexar Dashcam Collision Prediction Dataset**: D. C. Moura, S. Zhu and O. Zvitia, "Nexar Dashcam Collision Prediction Dataset and Challenge," *CVPR Workshops*, 2025, pp. 2608–2616 (Nexar Inc.).
The Nexar open data licence permits free use, copying and redistribution **with attribution to Nexar Inc. and the copyright notice retained**, and prohibits resale, re-identification of people or vehicles, weapons use, and systems intended to cause harm
([licence text](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/LICENSE)).
This repository holds derived data only (labels, event times, clip windows, features) with this attribution. It contains no raw videos and no dataset frames; `tools/compare_id_blocks.py` can write example-frame contact sheets locally, but they are not committed.

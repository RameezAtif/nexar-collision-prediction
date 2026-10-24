# Predicting Vehicle Collisions from Dashcam Video

## Anticipating the Crash — A Temporal Analysis

Computer Vision Project — University of the Witwatersrand

---

## Current Status (updated 2026-10-24)

> **Where to look.** This README is the long-form design and background document; parts of it describe the *plan* and were written before the experiments.
> What has actually been run, the numbers, the decisions and the open items are in **`EXPERIMENT_LOG.md`**. A notebook-by-notebook guide is in **`notebooks/README.md`**.

All three required models (naive, optical flow, CNN + LSTM) are implemented and evaluated at TTE = 0.5 / 1.0 / 1.5 s, together with the proposal's secondary temporal ablation.
The remaining work is the written report.

**Headline results** (notebooks 06 and 07; mean ± std over 5 training seeds; epoch chosen on held-out *training* videos; scored on the 300 held-out videos, which act as the test set):

| Model | AP @ 0.5 s | AP @ 1.0 s | AP @ 1.5 s |
|---|---:|---:|---:|
| Naive | 0.500 | 0.500 | 0.500 |
| Optical flow (scaled) | 0.575 | 0.576 | 0.569 |
| CNN + LSTM | 0.762 ± 0.019 | 0.749 ± 0.011 | 0.675 ± 0.021 |

**What the experiments show**

* The learned model beats optical flow at every horizon (AP +0.197 / +0.180 / +0.117; all bootstrap 95% CIs exclude zero).
* **Temporal modelling is not demonstrated.** A classifier on the single last frame matches the LSTM (AP 0.759 / 0.752 / 0.709), as do mean-pooling and an LSTM trained on shuffled frames. The gain over optical flow comes mainly from the pretrained ResNet features.
* Performance falls with horizon, but only the 0.5 s to 1.5 s drop (0.086 AP, 95% CI 0.03 to 0.14) is distinguishable from zero. The proposal's hypothesis (the learned model degrades more slowly than optical flow) is **not supported**: optical flow is flat only because it is near chance everywhere.
* Much of the signal is present long before the event (a single frame taken 3+ seconds before it already gives AP about 0.70), so the model is better described as recognising high-risk driving scenes than as detecting imminent collisions.
* **Dataset caveat.** Video ids perfectly separate the classes (every collision video has id <= 1039, every normal video id >= 1040). No model uses the id. On a 431-video sample no difference in resolution, frame rate, duration, brightness or overlay was found between the id blocks, but a difference in scene content was not tested.

**Numbers not to cite.** Notebook 05's CNN + LSTM results (AP 0.793 / 0.774 / 0.739) and the partner's `results/final_analysis/` use the best epoch *chosen on the evaluation videos* and the unscaled optical-flow variant. They are optimistic by about 0.04 / 0.03 / 0.07 AP (see `EXPERIMENT_LOG.md`, E3 and E10) and are kept for reference only.

**Quick start:** see *Reproducing the results* below.

---

## Reproducing the results

Tested from a fresh `git clone` (no raw data present) on Windows 10, Python 3.10, CPU only. No GPU is needed.

### 1. Setup

```text
git clone https://github.com/RameezAtif/nexar-collision-prediction.git
cd nexar-collision-prediction
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data

| What | Where | In the repo? |
|---|---|---|
| Train/validation split, clip definitions, cached ResNet-18 features, optical-flow features | `data/processed/` | **Yes** (small, derived data). Notebooks 06-09 and the report assets run from this alone. |
| Raw Nexar videos and `train.csv` (about 31 GB in total, 25.5 GB of train videos) | `data/raw/train/*.mp4`, `data/raw/train.csv` | **No** (too large, and not ours to redistribute). Needed only to regenerate `data/processed/` (notebooks 01-05) and for `tools/compare_id_blocks.py`. |

To get the raw data: accept the rules of the Kaggle competition <https://www.kaggle.com/competitions/nexar-collision-prediction/data> (or use the Hugging Face dataset `nexar-ai/nexar_collision_prediction`),
then place `train.csv` and the `train/` folder under `data/raw/`.

### 3. Commands

```text
python tools/run_notebooks.py            # notebooks 06, 07, 08, 09 from the committed data (about 5 minutes on a CPU)
python tools/make_report_assets.py       # report figures and LaTeX tables, from results/*.csv
python tools/run_notebooks.py --list     # what each notebook needs
```

With the raw data in `data/raw/` (slow: feature extraction takes hours on a CPU):

```text
python tools/run_notebooks.py 03_clip_sampling 04_baselines 05_learned_temporal_model
python tools/compare_id_blocks.py --available-only      # raw-video comparison of the id blocks (needs only the videos you downloaded)
python tools/fulldata_selection_check.py                # E10: selection-optimism check
```

Notebooks can also be opened in Jupyter directly; each starts by moving to the project root if the kernel was launched inside `notebooks/`.

### 4. Seeds and determinism

* Train/validation split and negative sampling: seed 42 (notebook 03), saved once to `data/processed/*.csv` and reused by every model.
* Model training (notebooks 06, 08): seeds 42, 43, 44, 45, 46, set with `torch.manual_seed` and `numpy.random.seed` before each run.
* Bootstrap, permutation and shuffling in notebooks 06, 08, 09: fixed seeds (`default_rng`), so the confidence intervals reproduce too.
* A rerun on the same machine reproduces the committed result files exactly. Across operating systems and library versions, expect small numerical differences (optical-flow baseline: at most 0.0007 AP).

### 5. Where each reported number comes from

| Report item | File | Produced by |
|---|---|---|
| Main results table (AP, recall at fixed false-alarm rate), mean ± std over 5 seeds | `results/temporal_ablation_summary.csv` | notebook 06 |
| Paired bootstrap confidence intervals | `results/temporal_ablation_bootstrap.csv` | notebook 06 |
| Single-frame AP by frame position | `results/failure_frame_position_ap.csv` | notebook 08 |
| Annotated-alert timing analysis | `results/failure_alert_timing.csv` | notebook 08 |
| Video-id block tests | `results/id_block_test_*.csv`, `results/id_blocks/*.csv` | notebook 09, `tools/compare_id_blocks.py` |
| Selection-optimism check (notebook 05 vs clean protocol) | `results/selection_check_fulldata.csv` | `tools/fulldata_selection_check.py` |
| Report tables and figures | `report/tables/*.tex`, `report/figures/*.pdf`, `report/tables/numbers_quoted_in_text.csv` | `tools/make_report_assets.py` |

### 6. Build the report (needs LaTeX or Docker)

```text
docker run --rm -v "%cd%/report:/work" -w /work texlive/texlive:latest-medium sh -c "pdflatex main && bibtex main && pdflatex main && pdflatex main"
```

(macOS/Linux: replace `%cd%` with `$(pwd)`.) The output is `report/main.pdf`.

### 7. Data attribution and licence

This project uses the **Nexar Dashcam Collision Prediction Dataset**: D. C. Moura, S. Zhu and O. Zvitia, "Nexar Dashcam Collision Prediction Dataset and Challenge," CVPR Workshops 2025, pp. 2608-2616 (Nexar Inc.).
It is released under the Nexar open data licence, which permits free use, copying and redistribution **with attribution to Nexar Inc. and the copyright notice retained**, and prohibits resale, attempts to re-identify people or vehicles,
weapons use, and building systems intended to cause harm. This repository holds only derived data (labels, event times, clip windows, features) and a few small contact sheets of example frames, with this attribution; it contains no raw videos.
Licence text: <https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/LICENSE>.

---

## 1. Project Overview

This project investigates whether vehicle collisions or near-miss events can be predicted from dashcam video **before the annotated event occurs**.

The central question is not simply whether a model can recognise a collision after it happens. Instead, the project asks:

> **How far in advance can collision-related events be predicted from dashcam video, and do learned temporal representations provide an advantage over simpler motion-based cues?**

The project uses the **Nexar Collision Prediction Dataset** from the Kaggle competition:

`Nexar Collision Prediction`

Dataset source:

https://www.kaggle.com/competitions/nexar-collision-prediction/data

The experiments compare progressively more sophisticated approaches:

1. A trivial/naive baseline.
2. A classical motion-based baseline using optical flow and simple motion statistics.
3. A learned temporal model using CNN frame representations followed by temporal modelling such as an LSTM/GRU.

The main experimental variable is **Time To Event (TTE)**.

The initial TTE conditions are:

* 0.5 seconds before the event
* 1.0 second before the event
* 1.5 seconds before the event

The project therefore evaluates not only whether an event can be predicted, but how prediction performance changes as the prediction point moves further away from the event.

---

# 2. Research Motivation

Collision prediction is a temporal computer-vision problem.

A single image may contain useful information about:

* vehicles
* roads
* intersections
* traffic
* pedestrians
* vehicle positions
* scene geometry

However, collision anticipation depends heavily on **how the scene changes over time**.

For example, a single frame may show two vehicles approaching an intersection without making it obvious whether a collision is imminent.

A sequence may reveal:

```text
Frame 1
Vehicle A approaches

        ↓

Frame 2
Vehicle A continues

        ↓

Frame 3
Vehicle B enters trajectory

        ↓

Frame 4
Relative motion becomes dangerous

        ↓

Prediction
Collision may occur
```

This makes temporal information an important part of the problem.

The project therefore compares:

```text
Simple motion information
          vs
Learned temporal representations
```

while also testing how early the event can be anticipated.

---

# 3. Research Question

The current primary research question is:

> **How far in advance can collision-related events be predicted from dashcam video, and do learned temporal representations improve over classical motion cues?**

The project can be expressed through the following experimental questions.

### RQ1 — Early prediction

How does prediction performance change as the prediction point moves further away from the annotated event?

We test:

```text
TTE = 0.5 s
TTE = 1.0 s
TTE = 1.5 s
```

### RQ2 — Representation

Does a learned temporal model provide useful predictive information beyond simple motion cues?

We compare:

```text
Optical-flow / motion baseline
              vs
CNN + temporal model
```

### RQ3 — Temporal dependence

Does the learned model actually depend on temporal ordering?

This can later be investigated through an optional ablation where the temporal order of frames is shuffled.

---

# 4. Project Type

This project follows a **Compare/Probe** experimental structure.

The project is not intended to develop an entirely new neural-network architecture.

Instead, it establishes controlled comparisons between:

* a trivial baseline,
* a classical motion baseline,
* a learned temporal approach.

The goal is to produce a measurable and defensible experimental result within the available project scope.

---

# 5. Dataset

## 5.1 Dataset

The project uses the Nexar Collision Prediction Dataset.

The dataset was obtained from the Kaggle competition:

https://www.kaggle.com/competitions/nexar-collision-prediction/data

The local dataset contains:

```text
data/raw/
├── train/
├── test/
├── train.csv
├── test.csv
└── sample_submission.csv
```

The raw dataset is approximately 30 GB.

It must **not** be committed to Git.

---

# 6. Dataset Structure

The training metadata file is:

```text
data/raw/train.csv
```

It contains exactly four columns:

```text
id
time_of_event
time_of_alert
target
```

### `id`

The video identifier.

The corresponding MP4 file is stored in:

```text
data/raw/train/
```

Video filenames use five-digit formatting.

For example:

```text
id = 780
```

corresponds to:

```text
data/raw/train/00780.mp4
```

The mapping is therefore:

```python
video_path = TRAIN_DIR / f"{video_id:05d}.mp4"
```

---

## 6.1 `time_of_event`

For positive examples, this represents the annotated event time in seconds.

For negative examples, this value is missing.

---

## 6.2 `time_of_alert`

For positive examples, this represents the annotated alert time.

For negative examples, this value is missing.

The difference:

```text
time_of_event - time_of_alert
```

represents the annotated lead time between the alert and event.

Notebook 1 found, for the 750 positive training videos:

```text
Mean lead time   ≈ 1.60 seconds
Median lead time ≈ 1.43 seconds
```

These values are useful contextual information when interpreting the selected TTE values.

They should not be interpreted as a universal human reaction-time threshold.

---

## 6.3 `target`

The target label is binary:

```text
0 = negative
1 = positive
```

The local training dataset contains:

```text
750 negative
750 positive
```

Therefore:

```text
1500 total
50% negative
50% positive
```

---

# 7. Important Dataset Observation

Missing timestamps for negative videos are expected.

Do **not** remove negative rows because:

```text
time_of_event = NaN
time_of_alert = NaN
```

These missing values are part of the dataset's label structure.

Positive examples have annotated event/alert times.

Negative examples do not.

---

# 8. Current Dataset Statistics

Notebook 1 established the following training metadata statistics.

## Class distribution

```text
Negative: 750
Positive: 750
Total:    1500
```

## Positive event times

For the 750 positive training examples:

```text
Mean:   19.101628 s
Std:     3.565120 s
Minimum: 3.032 s
Median: 19.802 s
Maximum: 56.800 s
```

## Positive alert times

```text
Mean:   17.501271 s
Std:     3.658975 s
Minimum: 1.966 s
Median: 18.259 s
Maximum: 55.467 s
```

## Alert lead time

```text
Mean:   1.600357 s
Std:    0.866063 s
Minimum: 0.033 s
Median: 1.433 s
Maximum: 4.466 s
```

These statistics should be regenerated if the dataset or preprocessing pipeline changes.

---

# 9. Repository Structure

The intended repository architecture is:

```text
nexar-collision-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   ├── test/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   │
│   └── processed/
│       ├── train_split.csv
│       └── val_split.csv
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_video_inspection.ipynb
│   ├── 03_clip_sampling.ipynb
│   └── 04_baselines.ipynb
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── clips.py
│   │   └── splits.py
│   │
│   ├── baselines/
│   │   ├── naive.py
│   │   └── optical_flow.py
│   │
│   ├── models/
│   │   └── cnn_temporal.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluate.py
│   │
│   └── utils/
│       ├── seed.py
│       └── config.py
│
├── experiments/
│   ├── configs/
│   └── runs/
│
└── results/
    ├── figures/
    ├── tables/
    ├── logs/
    └── checkpoints/
```

The project should be built incrementally.

Not every directory needs to exist immediately.

> **Actual layout (2026-10-24).** `src/` and `experiments/` from the plan above were **not** created: the implementation lives in the notebooks, plus two small scripts in `tools/`.
>
> ```text
> notebooks/    01-09 (see notebooks/README.md; two notebooks are numbered 06)
> tools/        compare_id_blocks.py (raw-video id-block comparison), fulldata_selection_check.py
> data/processed/   splits, clip definitions, cached CNN and optical-flow features
> results/      baseline and model result CSVs, temporal_ablation_*, final_*, failure_*, id_block_*, figures/
> results/final_analysis/   the partner's analysis (reference only, see its README)
> EXPERIMENT_LOG.md         running record of experiments, decisions and open items
> requirements.txt
> ```

---

# 10. Data Storage Rules

The raw dataset is large and should remain local.

Do not commit:

```text
data/raw/
```

Do not commit:

```text
data/processed/
```

if it contains large generated frame/image/video data.

Small metadata files such as the train/validation split CSVs can be committed if desired.

The principle is:

```text
Raw videos
    ↓
remain outside Git

Small metadata
    ↓
can be version controlled

Generated models/checkpoints
    ↓
remain outside Git
```

---

# 11. `.gitignore`

The repository should contain:

```gitignore
# Dataset
data/raw/
data/processed/

# Python
__pycache__/
*.py[cod]

# Virtual environment
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# Model files
*.pt
*.pth
*.ckpt

# Experiment outputs
results/checkpoints/
results/logs/

# OS
.DS_Store
```

If small processed metadata files need to be committed, the `data/processed/` rule can later be adjusted.

---

# 12. Local Environment Setup

The project is currently developed locally using:

* macOS
* VS Code
* Python
* Jupyter Notebook

The project should also be compatible with Google Colab later if GPU acceleration is required.

---

# 13. Python Environment

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

---

# 14. Install Initial Dependencies

The current notebooks require:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter ipykernel opencv-python
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user \
    --name nexar-collision \
    --display-name "Python (Nexar Collision)"
```

In VS Code/Jupyter, select:

```text
Python (Nexar Collision)
```

as the notebook kernel.

---

# 15. Requirements File

The project should maintain a `requirements.txt`.

At the current stage it can contain:

```text
numpy
pandas
matplotlib
scikit-learn
opencv-python
jupyter
ipykernel
```

Additional deep-learning dependencies should only be added when the learned model is implemented.

Likely future dependencies include PyTorch and torchvision.

Do not add unnecessary packages simply because they might eventually be useful.

> **Update (2026-10-24).** `requirements.txt` now exists and includes PyTorch and torchvision (the CNN + LSTM was implemented), SciPy (bootstrap and
> rank tests) and Pillow. It lists minimum versions; the exact versions used for notebooks 06-09 are recorded in its header comment.
> The unscaled optical-flow logistic regression does not converge in 1,000 iterations and its results shift slightly between environments, which is one reason the scaled variant is the reference baseline.

---

# 16. Dataset Installation

Download the dataset from the Kaggle competition:

https://www.kaggle.com/competitions/nexar-collision-prediction/data

Place the extracted files into:

```text
data/raw/
```

The resulting structure should be:

```text
data/
└── raw/
    ├── train/
    ├── test/
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

Verify that:

```text
data/raw/train.csv
```

exists.

Also verify that the corresponding MP4 files exist in:

```text
data/raw/train/
```

---

# 17. Notebook Pipeline

The notebooks are intended to be run in order.

```text
01_dataset_exploration
          ↓
02_video_inspection
          ↓
03_clip_sampling
          ↓
04_baselines
          ↓
model development
          ↓
experiments
          ↓
evaluation
```

The notebooks are primarily for:

* exploration,
* debugging,
* visualisation,
* understanding the data,
* validating individual components.

Reusable final functionality should eventually move into `src/`.

> **Update (2026-10-24).** The pipeline now continues past the baselines:
>
> ```text
> 04_baselines -> 05_learned_temporal_model -> 06_temporal_ablation -> 07_final_analysis -> 08_failure_analysis -> 09_id_block_confound
> ```
>
> A second notebook numbered 06 (`06_final_analysis_results`) is the partner's parallel analysis and is kept for reference. See `notebooks/README.md` for the purpose and status of each notebook.

---

# 18. Notebook 1 — Dataset Exploration

File:

```text
notebooks/01_dataset_exploration.ipynb
```

Purpose:

* Load `train.csv`
* Inspect columns
* Inspect missing values
* Check class balance
* Analyse event/alert times
* Calculate alert lead times
* Verify ID → MP4 mappings

Important findings from the completed notebook:

```text
1500 training videos
750 positive
750 negative
```

The positive event and alert timestamp distributions were also analysed.

Notebook 1 does not train a model.

It establishes what the dataset actually contains.

---

# 19. Notebook 2 — Video Inspection

File:

```text
notebooks/02_video_inspection.ipynb
```

Purpose:

* Open real MP4 files
* Inspect FPS
* Inspect frame count
* Inspect resolution
* Calculate duration
* Read individual frames
* Convert BGR → RGB
* Inspect event and alert frames
* Inspect TTE frames
* Check temporal indexing

A tested example showed:

```text
FPS = 30
```

for the inspected video.

The notebook uses the video's actual FPS rather than assuming every video has the same FPS.

The basic conversion is:

```text
frame_number ≈ time_seconds × FPS
```

---

# 20. Notebook 3 — Clip Sampling

File:

```text
notebooks/03_clip_sampling.ipynb
```

This notebook establishes the temporal sampling protocol.

The current observation window is:

```text
2 seconds
```

The model receives a sequence of frames covering the two seconds immediately before the prediction point.

---

# 21. Time To Event

For positive videos:

```text
prediction_time = time_of_event - TTE
```

For example:

```text
event = 20.0 s
TTE   = 1.0 s

prediction_time = 19.0 s
```

The model may therefore use frames up to approximately:

```text
19.0 s
```

but must not use frames after that point.

---

# 22. Positive Clip Structure

For a 2-second observation window and TTE = 1.0:

```text
                    prediction             event
                         ↓                    ↓
                         |                    |
17.0 s ---------------- 19.0 s ------------ 20.0 s
|-----------------------|
      2-second clip

                 1-second TTE
```

The model sees:

```text
17.0 s → 19.0 s
```

It does not see:

```text
19.0 s → 20.0 s
```

and it does not see the event frame itself.

This prevents future-information leakage.

---

# 23. Main TTE Conditions

The current experiment uses:

```python
TTE_VALUES = [0.5, 1.0, 1.5]
```

These should remain the primary experimental conditions unless the project scope is explicitly changed.

Potential additional TTE values should be treated as optional extensions.

---

# 24. Clip Sampling

The current sampler uses:

```text
2-second observation window
16 sampled frames
```

The 16 frames are approximately uniformly distributed across the observation window.

For example:

```text
Frame 1
   ↓
Frame 2
   ↓
...
Frame 16
   ↓
prediction point
```

This reduces computational cost compared with processing every frame.

The exact number of frames can later be treated as an experimental configuration.

It should not be changed casually between models.

---

# 25. Train/Validation Split

The dataset is split **before clip generation**.

Current split:

```text
Training:
1200 videos
600 positive
600 negative

Validation:
300 videos
150 positive
150 negative
```

The split uses:

```python
train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["target"]
)
```

The resulting overlap is:

```text
0 videos
```

This is critical.

---

# 26. Why Split by Video?

Suppose a single video produces:

```text
Clip A
Clip B
Clip C
Clip D
```

If:

```text
A, B → training
C, D → validation
```

the model may effectively see the same underlying scene in both sets.

This creates leakage.

Instead:

```text
Video A
 ├── Clip A
 ├── Clip B
 └── Clip C

all → TRAIN
```

while:

```text
Video B
 ├── Clip D
 ├── Clip E
 └── Clip F

all → VALIDATION
```

This ensures validation videos remain unseen.

---

# 27. TTE Validity Results

The completed Notebook 3 produced:

```text
TTE = 0.5s → 0 invalid positive training videos
TTE = 1.0s → 0 invalid positive training videos
TTE = 1.5s → 1 invalid positive training video
```

For validation:

```text
TTE = 0.5s → 0 invalid positive validation videos
TTE = 1.0s → 0 invalid positive validation videos
TTE = 1.5s → 0 invalid positive validation videos
```

Therefore:

```text
0.5 s → fully supported
1.0 s → fully supported
1.5 s → fully supported in validation
```

Only one training video lacks enough pre-event footage for the 1.5-second condition.

That video should be excluded from that specific TTE condition rather than padded with artificial frames.

---

# 28. Processed Split Files

Notebook 3 saves:

```text
data/processed/train_split.csv
data/processed/val_split.csv
```

These files define the fixed video-level experimental split.

Future notebooks and scripts should load these files rather than repeatedly generating new splits.

---

# 29. Negative Sampling

Negative videos do not have:

```text
time_of_event
```

Therefore the positive-video equation:

```text
event - TTE
```

cannot be used directly.

A negative clip must instead use a valid prediction point within the video.

The current implementation demonstrates random sampling of a valid prediction point.

However:

> **The exact negative sampling policy must be finalised before the main experiments are run.**

This is an important experimental-design decision.

The final policy must ensure that positive and negative clips are comparable in:

* observation duration,
* number of frames,
* preprocessing,
* temporal structure,
* availability of information.

The negative sampling decisions should be generated once using a fixed seed and saved.

The same sampled clips must then be used by every model.

---

# 30. Why Negative Sampling Matters

A poorly designed negative sampling strategy could produce misleading results.

For example, if positive clips are concentrated near events while negative clips are sampled from arbitrary locations, the model might exploit unintended differences.

The desired comparison is:

```text
Positive:
2 seconds of history
        ↓
prediction point
        ↓
event occurs later

Negative:
2 seconds of history
        ↓
prediction point
        ↓
no annotated event
```

The difference should primarily be the predictive information in the video, not an accidental difference in sampling procedure.

---

# 31. Baseline Strategy

The project requires a baseline before the more sophisticated model.

The baseline pipeline will eventually contain:

```text
Baseline 0
Naive classifier
      ↓
Baseline 1
Motion / optical flow
      ↓
Main model
CNN + temporal model
```

This provides progressively stronger references.

---

# 32. Baseline 0 — Naive Classifier

Because the training data is balanced:

```text
50% positive
50% negative
```

a trivial majority classifier provides a reference point.

Possible naive references include:

* majority class,
* random prediction,
* constant probability.

The exact metrics used for the baseline must match those used for the real models.

---

# 33. Baseline 1 — Optical Flow

The classical motion baseline will use optical flow.

Conceptually:

```text
Frame t
   +
Frame t+1
   ↓
Optical Flow
   ↓
Motion vectors
   ↓
Motion statistics
   ↓
Classifier
   ↓
Prediction
```

Possible features include:

* mean flow magnitude,
* standard deviation of flow magnitude,
* maximum motion,
* directional statistics,
* spatial distribution of motion.

The first implementation should remain simple.

The objective is not to engineer a large handcrafted feature system.

The baseline should answer:

> How much predictive information can we obtain from relatively simple motion cues?

---

# 34. Optical Flow Principle

Optical flow estimates apparent pixel motion between consecutive frames.

For example:

```text
Frame 1             Frame 2

  CAR                 CAR
   ↓                   ↓
 [###]               [###]
```

The movement can be represented by vectors:

```text
→ → →
 → → →
  → → →
```

These vectors provide information about motion without requiring a learned neural network.

---

# 35. Main Learned Model

The main learned approach is planned as a CNN + temporal model.

The architecture is conceptually:

```text
Video clip
    ↓
16 frames
    ↓
CNN feature extractor
    ↓
16 feature vectors
    ↓
LSTM / GRU
    ↓
Temporal representation
    ↓
Fully connected layer
    ↓
Collision probability
```

The CNN processes spatial information.

The LSTM/GRU processes temporal information.

---

# 36. CNN Feature Extraction

Each frame:

```text
H × W × 3
```

is passed through a pretrained CNN.

The CNN converts each image into a feature vector:

```text
Frame 1 → feature vector 1
Frame 2 → feature vector 2
...
Frame 16 → feature vector 16
```

The resulting structure is approximately:

```text
16 × feature_dimension
```

This becomes the input sequence for the temporal model.

---

# 37. Temporal Model

The temporal model receives:

```text
f1, f2, f3, ..., f16
```

and learns relationships across time.

For example:

```text
f1 → f2 → f3 → ... → f16
```

The final temporal representation is used to predict:

```text
P(positive)
```

---

# 38. Why Use a Temporal Model?

A frame-only model may recognise visual context such as:

* roads,
* vehicles,
* intersections,
* traffic density.

But the project is specifically interested in whether **temporal evolution** provides additional predictive information.

A temporal model can potentially learn patterns such as:

```text
relative movement
approach speed
trajectory changes
interaction between vehicles
```

without requiring all such features to be manually engineered.

---

# 39. Potential Temporal Ablation

If time allows, the learned model can be tested with shuffled frame order.

Normal:

```text
1 → 2 → 3 → 4 → ... → 16
```

Shuffled:

```text
7 → 2 → 15 → 4 → ... → 9
```

The same frames are present, but their temporal order is destroyed.

If performance decreases substantially after shuffling, this provides evidence that temporal ordering contributes useful information.

This is an optional experiment and should not be allowed to delay the core project.

---

# 40. Main Experimental Matrix

The minimum viable experiment is:

| Model                 | 0.5s TTE | 1.0s TTE | 1.5s TTE |
| --------------------- | -------: | -------: | -------: |
| Naive baseline        |        ✓ |        ✓ |        ✓ |
| Optical-flow baseline |        ✓ |        ✓ |        ✓ |
| CNN + temporal model  |        ✓ |        ✓ |        ✓ |

This gives:

```text
3 models × 3 TTE conditions
```

The primary result can then be visualised as performance versus TTE.

---

# 41. Primary Evaluation Metric

The final metric must be chosen based on the actual challenge/task evaluation protocol and the project's class-balanced experimental design.

A strong candidate is:

> **Average Precision / Area Under the Precision-Recall Curve (AP/AUPRC)**

This is useful because collision prediction is fundamentally a binary detection problem where the quality of positive predictions matters.

Other metrics should also be reported where appropriate:

* precision,
* recall,
* F1,
* ROC-AUC,
* confusion matrix.

If a fixed false-alarm operating point is used, report the corresponding recall as an additional operational measure.

The exact primary metric must be locked before final experiments.

---

# 42. Evaluation Principle

Every model must be evaluated under the same conditions.

That means:

```text
Same validation videos
        ↓
Same temporal clips
        ↓
Same TTE values
        ↓
Same preprocessing
        ↓
Same metric
```

Only the model should change.

This is essential for a defensible comparison.

---

# 43. Experimental Reproducibility

Every experiment should record:

* random seed,
* train/validation split,
* TTE,
* clip duration,
* number of frames,
* model configuration,
* CNN backbone,
* temporal model,
* optimiser,
* learning rate,
* batch size,
* number of epochs,
* checkpoint selection method,
* evaluation metric.

A future researcher should be able to determine exactly how a result was produced.

---

# 44. Random Seeds

The current split uses:

```text
random_state = 42
```

Other random operations should also use explicitly defined seeds.

A future utility file can provide:

```text
src/utils/seed.py
```

for setting:

* Python random seed,
* NumPy seed,
* PyTorch seed.

---

# 45. Avoiding Data Leakage

The following rules are mandatory.

### Rule 1

Split videos before generating clips.

### Rule 2

Never allow validation videos into training.

### Rule 3

For positive clips, never include frames after the prediction time.

### Rule 4

Do not use event information as a model input.

The model may use the event timestamp to determine where to sample the training clip, but it must not receive the event timestamp as a feature.

### Rule 5

Do not use `time_of_alert` as a predictive input unless explicitly designed as a separate experiment.

### Rule 6

Use the same validation clips for every model.

---

# 46. Important Distinction: Sampling vs Prediction

The event timestamp is used to construct the experimental example.

It does **not** mean the model knows when the event will occur.

For example:

```text
Known during dataset preparation:

event = 20.0 seconds
TTE = 1.0 second

therefore:

prediction point = 19.0 seconds
```

The model then receives:

```text
17.0 → 19.0 seconds
```

and must predict the label.

The event timestamp is therefore part of the **experimental sampling procedure**, not the model's input.

---

# 47. Computational Strategy

The raw dataset is approximately 30 GB.

Do not preprocess all videos into thousands of image files unless there is a demonstrated need.

The preferred approach is:

```text
Original MP4
    ↓
On-demand frame extraction
    ↓
Temporal clip
    ↓
Model
```

This reduces:

* disk usage,
* duplicated data,
* preprocessing time,
* repository complexity.

---

# 48. Caching

If repeated frame decoding becomes a performance bottleneck, caching can be introduced later.

Possible approaches:

```text
MP4
 ↓
cached resized frames
```

or:

```text
MP4
 ↓
cached CNN features
```

CNN feature caching may be particularly useful if the CNN backbone is frozen.

However, caching should only be introduced after measuring whether it is actually necessary.

---

# 49. GPU Strategy

Development is currently local.

Training the learned model may benefit substantially from GPU acceleration.

Google Colab can be used if necessary.

The project should remain structured so that:

```text
local CPU/GPU
```

and:

```text
Google Colab GPU
```

use the same source code and configuration wherever possible.

The raw 30 GB dataset should not automatically be uploaded to Google Drive simply because Colab is available.

---

# 50. Training Strategy

The learned model should initially use a pretrained CNN.

A sensible development progression is:

### Stage 1

Freeze the CNN.

Train only:

```text
LSTM/GRU
+
classification head
```

This is computationally cheaper and establishes whether the temporal component works.

### Stage 2

If successful, optionally fine-tune part of the CNN.

This should be treated as an extension rather than a requirement.

---

# 51. Development Order

Do not begin by implementing the entire deep-learning system.

Use this order:

```text
1. Dataset
   ↓
2. Video inspection
   ↓
3. Clip sampler
   ↓
4. Negative sampling
   ↓
5. Naive baseline
   ↓
6. Optical-flow baseline
   ↓
7. Evaluation pipeline
   ↓
8. CNN feature extraction
   ↓
9. Temporal model
   ↓
10. Main experiments
   ↓
11. Ablations
   ↓
12. Failure analysis
```

This makes debugging much easier.

---

# 52. Current Project Status

As of the current project state:

### Completed

* Dataset downloaded locally.
* Dataset structure verified.
* Training metadata inspected.
* Class balance verified.
* Event/alert timestamps analysed.
* MP4 ID mapping verified.
* Video inspection notebook developed.
* Actual MP4 frame extraction tested.
* FPS/frame indexing tested.
* TTE calculation implemented.
* Two-second temporal clip sampling implemented.
* Video-level train/validation split implemented.
* Train/validation leakage check performed.
* TTE validity checked.

### Current split

```text
Train:      1200
Validation: 300

Train:
600 positive
600 negative

Validation:
150 positive
150 negative
```

### Current TTE values

```text
0.5 s
1.0 s
1.5 s
```

### Current clip duration

```text
2.0 seconds
```

### Current sampled frames

```text
16 frames per clip
```

> **Update (2026-10-24).** The list above describes the state when notebook 3 was complete. Since then: negative sampling was frozen and saved
> (`data/processed/{train,val}_clips.csv`); the naive and optical-flow baselines were built (notebook 04); ResNet-18 features were cached and a CNN + LSTM trained
> (notebook 05); a temporal ablation with 5 seeds, bootstrap CIs and clean epoch selection was run (notebook 06); final tables and figures were produced (notebook 07);
> a failure analysis and an investigation of the class-specific video-id blocks were carried out (notebooks 08 and 09). One positive training video was too short for the
> 2-second window at TTE 1.5 s and is excluded there, so that training set has 1,199 clips.

---

# 53. Current Stage

As of 2026-10-24:

```text
DATA PREPARATION          ✓
VIDEO INSPECTION          ✓
TEMPORAL SAMPLING         ✓
BASELINES                 ✓   (naive, optical flow)
CNN + LSTM                ✓
TTE EXPERIMENTS           ✓   (0.5 / 1.0 / 1.5 s, 5 seeds)
TEMPORAL ABLATION         ✓
FINAL ANALYSIS            ✓
FAILURE ANALYSIS          ✓
ID-BLOCK INVESTIGATION    ✓   (scene content between blocks still untested)
       ↓
WRITTEN REPORT            ← CURRENT
```

---

# 54. Immediate Next Task

The next task is the **written report**, built from `EXPERIMENT_LOG.md`, `notebooks/07_final_analysis.ipynb` and `results/figures/final_fig*.png`.
It should answer the three research questions (horizon, learned vs classical, temporal dependence), state that the proposal's hypothesis is not supported,
use the clean-protocol numbers (not notebook 05's), and report the video-id block structure as a limitation.

Optional, only if time allows: a GRU in place of the LSTM, dropout or weight decay (the models overfit after a few epochs), extending the raw-video id-block comparison to all 1,500 videos
or testing scene content, and merging or renumbering the two notebooks numbered 06.

---

# 55. Recommended Source Architecture

> **Status (2026-10-24): not implemented.** Sections 55 to 65 describe a planned `src/` package and experiment configs. The work was done in notebooks and two scripts in `tools/`
> instead. Moving the code into `src/` remains an optional refactor and is not needed for the experiments to be reproducible.

Once the notebook implementation stabilises, reusable functions should move into:

```text
src/
```

Recommended structure:

```text
src/
├── data/
│   ├── dataset.py
│   ├── clips.py
│   └── splits.py
│
├── baselines/
│   ├── naive.py
│   └── optical_flow.py
│
├── models/
│   └── cnn_temporal.py
│
├── evaluation/
│   ├── metrics.py
│   └── evaluate.py
│
└── utils/
    ├── seed.py
    └── config.py
```

---

# 56. `src/data/dataset.py`

This module should eventually handle:

* loading metadata,
* loading split files,
* mapping IDs to video paths.

Example responsibilities:

```text
load_metadata()
get_video_path()
load_train_split()
load_validation_split()
```

It should not contain model code.

---

# 57. `src/data/clips.py`

This module should contain the temporal sampling logic.

Responsibilities:

```text
get_clip_times()
time_to_frame()
read_clip()
sample_positive_clip()
sample_negative_clip()
```

This becomes the single source of truth for temporal sampling.

The same code should be used by every model.

---

# 58. `src/data/splits.py`

This module should handle:

```text
create_train_val_split()
load_train_val_split()
check_split_overlap()
```

The split should be generated once and saved.

---

# 59. `src/baselines/naive.py`

This module should contain simple reference classifiers.

For example:

```text
majority prediction
random prediction
constant probability
```

These are not intended to be sophisticated.

Their purpose is to establish a lower reference point.

---

# 60. `src/baselines/optical_flow.py`

This module should contain:

```text
frame pair
    ↓
optical flow
    ↓
motion statistics
    ↓
feature vector
```

The final classifier can initially be something simple such as logistic regression.

Do not immediately build a complicated classifier.

The important variable is whether simple motion contains useful predictive information.

---

# 61. `src/models/cnn_temporal.py`

This module will eventually contain the learned model.

Conceptually:

```text
CNN
 ↓
frame features
 ↓
sequence
 ↓
LSTM/GRU
 ↓
classification head
```

The implementation should make it possible to configure:

```text
CNN backbone
hidden dimension
number of recurrent layers
dropout
```

without rewriting the entire model.

---

# 62. `src/evaluation/metrics.py`

This module should contain all evaluation calculations.

For example:

```text
average_precision()
roc_auc()
precision()
recall()
f1()
confusion_matrix()
```

This prevents different notebooks from accidentally calculating metrics differently.

---

# 63. `src/evaluation/evaluate.py`

This module should eventually handle:

```text
model predictions
        ↓
metric calculation
        ↓
result dictionary
        ↓
CSV/JSON result
```

A result should contain enough information to identify the experiment.

For example:

```text
model
tte
clip_duration
num_frames
seed
metric
value
```

---

# 64. Experiment Configuration

Experiment parameters should eventually be stored separately from source code.

Possible structure:

```text
experiments/
└── configs/
    ├── baseline.yaml
    ├── optical_flow.yaml
    └── cnn_temporal.yaml
```

This prevents parameters from being buried inside notebooks.

---

# 65. Results Structure

Results should eventually be organised as:

```text
results/
├── figures/
│   ├── performance_vs_tte.png
│   ├── confusion_matrix.png
│   └── ...
│
├── tables/
│   ├── main_results.csv
│   └── ...
│
├── logs/
│   └── ...
│
└── checkpoints/
    └── ...
```

Model checkpoints should not be committed to Git.

---

# 66. Main Result

The most important planned result is a plot showing performance as a function of TTE.

Conceptually:

```text
Performance
     ↑
     |
     |       ●
     |     ●
     |   ●
     | ●
     |
     +--------------------→ TTE
       0.5   1.0   1.5
```

Separate curves can represent:

```text
Optical flow
CNN + temporal model
```

This directly answers the project's early-prediction question.

---

# 67. Reporting Results

Do not report only the best result.

Report all primary TTE conditions.

For example:

```text
Model                  TTE       AP
------------------------------------
Optical Flow           0.5       ...
Optical Flow           1.0       ...
Optical Flow           1.5       ...

CNN + Temporal        0.5       ...
CNN + Temporal        1.0       ...
CNN + Temporal        1.5       ...
```

This allows the reader to see how performance changes as prediction becomes earlier.

---

# 68. Statistical Reporting

If computational resources permit, experiments should use multiple random seeds.

For example:

```text
seed 42
seed 43
seed 44
```

Then report:

```text
mean ± standard deviation
```

This is preferable to presenting a single lucky run.

However, if the project scope or computational budget prevents repeated deep-learning runs, document that limitation rather than hiding it.

---

# 69. Failure Analysis

Failure analysis is optional but useful.

Examples:

### False positive

```text
Model predicts collision
but target = 0
```

Investigate whether:

* strong vehicle motion,
* sudden camera movement,
* dense traffic,
* braking,
* unusual scene geometry

could explain the prediction.

### False negative

```text
Model predicts normal
but target = 1
```

Investigate whether:

* event develops very rapidly,
* visual evidence is weak,
* event is partly occluded,
* motion is ambiguous

could explain the failure.

Failure analysis should be descriptive rather than speculative.

---

# 70. Optional Temporal Ablation

If the core experiments are complete, perform:

```text
normal temporal order
vs
shuffled temporal order
```

The purpose is to test whether temporal ordering contributes to performance.

This should not replace the main comparison.

---

# 71. Optional TTE Extensions

The primary TTE values are:

```text
0.5
1.0
1.5
```

Additional values such as:

```text
2.0
2.5
3.0
```

can be investigated later.

Do not expand the TTE range until the primary experiment is working.

The project should prioritise completing the core experiment over adding more conditions.

---

# 72. Optional Architecture Extensions

Potential extensions include:

* fine-tuning the CNN,
* different CNN backbones,
* GRU instead of LSTM,
* attention-based temporal aggregation,
* video transformers.

These are **stretch goals**.

A transformer should not be introduced before the baseline and CNN-temporal pipeline are complete.

---

# 73. Minimum Viable Project

If time becomes limited, the project should stop expanding and complete this:

```text
Dataset
  ✓

Train/validation split
  ✓

Temporal clip sampler
  ✓

Naive baseline
  ↓
Optical-flow baseline
  ↓
CNN + temporal model
  ↓
TTE 0.5 / 1.0 / 1.5
  ↓
Evaluation
  ↓
Comparison
  ↓
Report
```

This is the minimum experiment necessary to answer the core research question.

---

# 74. What Not to Do

Do not:

* upload the entire 30 GB dataset to GitHub,
* commit raw videos,
* generate every possible frame unnecessarily,
* split clips randomly across train and validation,
* use future frames,
* feed event timestamps into the model,
* change TTE definitions between models,
* evaluate models on different validation clips,
* compare models using different preprocessing,
* tune the model on the final validation set repeatedly,
* add complex architectures before establishing baselines.

---

# 75. Common Debugging Checks

If a model produces suspiciously high performance, immediately check:

### Check 1

Are train and validation videos completely separate?

```text
intersection(train_ids, val_ids) == 0
```

### Check 2

Are any frames after the prediction point included?

### Check 3

Are event frames accidentally included?

### Check 4

Are positive and negative clips sampled differently?

### Check 5

Was validation data used during training?

### Check 6

Are labels correctly aligned with clips?

### Check 7

Are all models evaluated using the same validation set?

Unexpectedly high performance should trigger leakage checks before celebrating the result.

---

# 76. Reproducibility Checklist

Before declaring an experiment complete, record:

```text
[ ] Dataset version/source
[ ] Train/validation split
[ ] Random seed
[ ] TTE
[ ] Clip duration
[ ] Number of frames
[ ] Frame preprocessing
[ ] CNN backbone
[ ] Temporal architecture
[ ] Optimiser
[ ] Learning rate
[ ] Batch size
[ ] Epochs
[ ] Model checkpoint
[ ] Evaluation metric
[ ] Final results
```

A person who was not involved in the original experiment should be able to reproduce it from these details.

---

# 77. How a New Contributor Should Start

If someone new joins the project, they should:

### Step 1

Clone the repository.

### Step 2

Create the virtual environment.

### Step 3

Install requirements.

### Step 4

Obtain the Nexar dataset from Kaggle.

### Step 5

Place the dataset in:

```text
data/raw/
```

### Step 6

Open:

```text
notebooks/01_dataset_exploration.ipynb
```

and run it.

### Step 7

Run:

```text
02_video_inspection.ipynb
```

### Step 8

Run:

```text
03_clip_sampling.ipynb
```

### Step 9

Verify:

```text
data/processed/train_split.csv
data/processed/val_split.csv
```

exist.

### Step 10

Continue with:

```text
04_baselines.ipynb
```

Do not skip the earlier notebooks unless the corresponding functionality has already been verified.

### Step 11 onwards

After the baselines, continue with:

```text
05_learned_temporal_model.ipynb      (needs the raw videos to cache ResNet features; the cached features are already in data/processed/cnn_features/)
06_temporal_ablation.ipynb           (cached features only)
07_final_analysis.ipynb              (reads the notebook 06 results)
08_failure_analysis.ipynb            (cached features only)
09_id_block_confound.ipynb           (cached features only)
```

Run all notebooks from the project root. `EXPERIMENT_LOG.md` explains what each experiment found and which numbers to use.

---

# 78. Current Known Good Configuration

At the current project stage:

```text
Dataset:
Nexar Collision Prediction

Training videos:
1500

Train:
1200

Validation:
300

Class balance:
50 / 50

Train positive:
600

Train negative:
600

Validation positive:
150

Validation negative:
150

Split seed:
42

Observation window:
2.0 seconds

Sampled frames:
16

Primary TTE:
0.5 s
1.0 s
1.5 s

FPS:
Read from each video's metadata

Temporal rule:
No frames after prediction time

Video leakage:
None detected
```

**Additional frozen configuration used by notebooks 05 to 09**

```text
CNN backbone:               ResNet-18 (ImageNet weights), frozen, final FC removed -> 512-d per frame
Temporal model:             LSTM, 1 layer, hidden size 128, final hidden state -> linear logit
Optimiser / loss:           Adam, lr 1e-3, BCE with logits
Batch size / epochs:        32 / 10
Seeds (notebook 06 onward): 42, 43, 44, 45, 46
Epoch selection:            best AP on a stratified 20% inner split of the training videos (seed 42);
                            the 300 evaluation videos are never used for selection
Inner split:                960 fit / 240 selection clips per TTE (959 / 240 at TTE 1.5 s)
Optical flow baseline:      7 motion statistics per frame pair -> 21-d features -> StandardScaler + logistic regression (scaled variant is the reference)
Metrics:                    AP (primary), ROC-AUC, recall at <=10% and <=20% false-alarm rate, precision / recall / F1 at threshold 0.5
Uncertainty:                mean ± std over seeds; paired bootstrap 95% CIs (2,000 resamples of the 300 evaluation clips)
```

---

# 79. Important Current Decision

> **Resolved (2026-10-24).** Negative prediction-point sampling was reviewed and frozen: the sampled clips are saved once in `data/processed/{train,val}_clips.csv`
> and every model uses exactly those clips. The text below is kept as the original rationale.

The following is **not yet final**:

```text
Negative prediction-point sampling
```

The current notebook demonstrates random sampling, but before running the main experiments this policy should be reviewed and then frozen.

Once frozen:

1. Generate the negative sampling metadata.
2. Save it.
3. Use the exact same samples for all models.
4. Do not regenerate them between experiments.

---

# 80. Project Development Philosophy

The project should follow this principle:

> **Simple, controlled, measurable experiments are more valuable than a complicated model with an unclear experimental setup.**

The project is ultimately trying to establish a defensible comparison.

The priority is therefore:

```text
Correct experiment
      >
Reproducibility
      >
Clear baseline
      >
Reliable evaluation
      >
Model complexity
```

---

# 81. Final Intended Architecture

The completed project should eventually resemble:

```text
                         NEXAR DATASET
                              |
                              v
                     Dataset Metadata
                              |
                              v
                  Video-Level Train/Val Split
                              |
                 +------------+------------+
                 |                         |
                 v                         v
             TRAIN VIDEOS            VALIDATION VIDEOS
                 |                         |
                 +------------+------------+
                              |
                              v
                     Temporal Clip Sampler
                              |
                    +---------+---------+
                    |         |         |
                    v         v         v
                 TTE 0.5   TTE 1.0   TTE 1.5
                    |         |         |
                    +---------+---------+
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
       Naive Baseline   Optical Flow      CNN Features
                                             |
                                             v
                                       LSTM / GRU
                                             |
                                             v
                                     Classification
                                             |
                                             v
                                       Probability
                                             |
                                             v
                                         Metrics
                                             |
                         +-------------------+-------------------+
                         |                   |                   |
                         v                   v                   v
                       TTE 0.5             TTE 1.0             TTE 1.5
                         |                   |                   |
                         +-------------------+-------------------+
                                             |
                                             v
                                      Final Comparison
                                             |
                                             v
                                    Performance vs TTE
                                             |
                                             v
                                       Conclusions
```

---

# 82. Final Project Objective

The completed project should be able to make a clear empirical statement about:

1. whether collision-related events can be predicted before the annotated event,
2. how prediction performance changes with increasing TTE,
3. how a simple motion-based approach compares with a learned temporal representation,
4. whether temporal information appears to contribute useful predictive information.

The final conclusion must be based on the measured experimental results.

The project should not assume beforehand that the learned model will outperform optical flow.

The experiment is designed to determine that empirically.

---

# 83. Current Next Step

As of 2026-10-24 the experimental work is complete:

```text
Baselines -> CNN feature extraction -> CNN + LSTM -> TTE experiments -> Temporal ablation -> Final analysis -> Failure analysis
```

The next step is the written report (see section 54 and `EXPERIMENT_LOG.md`, section 7).

This README should be treated as the project's background and design handoff document, with the **Current Status** section at the top and `EXPERIMENT_LOG.md` as the source of truth for results.
Whenever the experimental protocol changes, both should be updated so that they remain consistent with the actual implementation.

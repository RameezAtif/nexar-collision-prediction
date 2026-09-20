# Predicting Vehicle Collisions from Dashcam Video

## Anticipating the Crash — A Temporal Analysis

Computer Vision Project — University of the Witwatersrand

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

---

# 53. Current Stage

The project is currently at:

```text
DATA PREPARATION
       ✓
VIDEO INSPECTION
       ✓
TEMPORAL SAMPLING
       ✓
       ↓
BASELINE DEVELOPMENT
       ← CURRENT NEXT STAGE
       ↓
OPTICAL FLOW
       ↓
CNN + TEMPORAL MODEL
       ↓
MAIN EXPERIMENTS
```

---

# 54. Immediate Next Task

The next notebook should be:

```text
notebooks/04_baselines.ipynb
```

Its purpose is to establish the baseline and evaluation pipeline.

It should not immediately implement the full CNN-LSTM.

First establish:

1. validation loading,
2. negative sampling,
3. naive prediction,
4. metric calculation,
5. baseline reporting,
6. reproducible result storage.

Then implement optical flow.

---

# 55. Recommended Source Architecture

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

---

# 79. Important Current Decision

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

The immediate next development task is:

```text
Notebook 4 — Baselines and Evaluation
```

The next notebook should establish the evaluation machinery before the main deep-learning model is implemented.

The recommended order is:

```text
Notebook 4
    ↓
Naive baseline
    ↓
Finalise negative sampling
    ↓
Optical-flow baseline
    ↓
Evaluation framework
    ↓
CNN feature extraction
    ↓
CNN + LSTM/GRU
    ↓
TTE experiments
    ↓
Temporal ablation
    ↓
Final analysis
```

This README should be treated as the project's current technical handoff document. Whenever the experimental protocol changes, the README should be updated so that it remains consistent with the actual implementation.

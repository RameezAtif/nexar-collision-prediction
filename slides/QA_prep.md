# Presentation and Q&A prep

Presentation: **Tuesday 27 October 2026** (time to be confirmed). Both of us must be able to explain every part. The deck has speaker notes on every slide; the numbers below are all in `results/*.csv` (see the README table "Where each reported number comes from").

## Suggested timing (about 12 minutes; cut slides 5 and 14 first if short)

| Slides | Topic | Suggested speaker |
|---|---|---|
| 1-4 | Question, data, how "before the crash" is measured | A |
| 5-6 | Protocol, the three models | B |
| 7-8 | Results: learned vs flow; degradation and the hypothesis | A |
| 9-10 | Probes: single frame = LSTM; signal is there long before the event | B |
| 11-12 | Failures; the id-block caveat | A |
| 13-15 | Evaluation lesson; limits; take-aways | B |

Swap as you like, but each of you should be able to take any slide.

## The ten numbers to know

| What | Number |
|---|---|
| Data | 1,500 videos, 750 / 750, about 40 s, 1280x720, about 30 fps |
| Split | 1,200 train / 300 test, seed 42, zero overlap |
| Clip | 16 frames from the 2 s ending at t_p = event - TTE |
| CNN+LSTM AP | 0.762 / 0.749 / 0.675 at 0.5 / 1.0 / 1.5 s |
| Optical flow AP | 0.575 / 0.576 / 0.569; naive 0.500 |
| Gap CNN+LSTM minus flow | +0.197 / +0.180 / +0.117 (CIs exclude 0) |
| Drop 0.5 s to 1.5 s | 0.086 (95% CI 0.03 to 0.14); flow 0.007 |
| Last frame only | 0.759 / 0.752 / 0.709 (matches the LSTM) |
| Single frame 3+ s before event | AP about 0.69 (chance 0.5) |
| Selection inflation | about 0.04 / 0.03 / 0.07 (first run 0.793 / 0.774 / 0.739 vs 0.754 / 0.753 / 0.676 with no selection) |

## Likely questions, and honest answers

**Why is average precision your main metric?** The proposal fixed it. It summarises the precision-recall trade-off without picking a threshold. We also report recall at a false-alarm rate of at most 10% of normal clips, because false alarms matter for warning systems.

**What does AP = 0.5 mean for the naive model?** The classes are balanced (150/150 in the test set), so predicting "collision" for everything gives AP equal to the prevalence, 0.5. It is the floor.

**How do you know there is no leakage?** The split is by video before any clip is cut (zero overlap checked); every sampled frame is at or before the prediction time (checked); negatives' prediction times are drawn from the positives' distribution, and a classifier on prediction time alone scores AUC 0.50-0.53; no model sees the event time or the video id.

**Your single frame matches your LSTM. Doesn't that mean the project failed?** It means we found something: on frozen ResNet-18 features, in this data, temporal modelling adds nothing we can measure. The learned features still beat optical flow clearly. The negative result on the temporal part is the finding, with intervals.

**Could the LSTM just be badly trained?** We tried 10 and 30 epochs (30 epochs overfit and did slightly worse), five seeds, and an LSTM trained on shuffled frames does as well, so it is not that the LSTM fails to learn; the features carry the signal. We did not tune the architecture, which is a limitation.

**Why is optical flow so weak?** It is a deliberately light baseline: 7 statistics per frame pair, on frames about 0.13 s apart (not adjacent), summarised into 21 numbers with logistic regression. Classical cues may be underestimated; we say so.

**What is the id-block problem and does it invalidate your results?** Every collision video has id <= 1039 and every normal video id >= 1040, so the id alone separates the classes. No model sees the id, so the numbers are valid as measured. But if the blocks differ in recording conditions the model could be recognising that. On 431 raw videos we found no difference in resolution, frame rate, duration, brightness or overlay; the features show a weak id-ordered trace (collision score drifts with id, rho -0.13, p 0.0005; separation AUC 0.59 near the boundary vs 0.73 far). We could not test scene content. So we say the model separates the two groups, plausibly by recognising high-risk scenes, not that it detects imminent collisions.

**Why do you say "high-risk scenes"?** Because a frame from 3+ s before the event already gives AP about 0.69, the persistent false alarms are dense night traffic with a close vehicle, and clips ending after the annotated alert are detected somewhat better (AUC 0.72-0.77) than clips ending before it (0.57-0.66). It is the most consistent reading, but it is a hypothesis.

**Why did your numbers change from your first notebook?** The first run kept the best epoch on the evaluation videos, then reported that score. We re-ran with the epoch chosen on 20% of the training videos, and checked against a full-data fixed-epoch run (no selection). The selection inflated AP by about 0.04-0.07, most at 1.5 s, which also flattened the apparent degradation.

**Is the 300-video set really a test set?** It never influenced epoch selection or tuning in the final protocol, so we call it the test set. But earlier notebooks looked at it and we chose the scaled optical-flow variant after seeing it (about 0.01 AP). There is no separately untouched set.

**What did you write and what did you reuse?** Reused: the Nexar data, ImageNet ResNet-18 weights (torchvision), Farneback optical flow (OpenCV), LSTM/Adam (PyTorch), logistic regression and metrics (scikit-learn). Written by us: the video-level split, the clip/TTE sampler and leakage checks, negative sampling, the flow-statistic features, feature caching, training and evaluation code, the ablation variants, the bootstrap and permutation analyses, the failure and id-block analyses, and the notebooks and scripts.

**How would someone reproduce this?** From a fresh clone: `pip install -r requirements.txt`, then `python tools/run_notebooks.py` (about 5 minutes, CPU, no raw data needed because the cached features are committed). We verified all 24 result files regenerate identically (difference below 1e-9) on the same machine. Seeds: 42 for the split, 42-46 for training.

**What would you do with another week?** Compare the id blocks on scene content using all 1,500 videos and evaluate on a split that holds out id ranges; choose epoch and regularisation by nested cross-validation; compute flow on adjacent frames and try a fine-tuned or two-stream model; add seeds and intervals to every ablation.

## Things not to say

- Do not say the model "predicts collisions" or "detects imminent collisions"; say it separates the two video groups and recognises high-risk scenes (hypothesis).
- Do not quote the first-run numbers (0.793 / 0.774 / 0.739) as results; they are the optimistic ones (and the partner's `results/final_analysis/` uses them).
- Do not claim the LSTM "uses temporal information"; the evidence is that a single frame does as well.

## Slide credits

All frames are from the Nexar Dashcam Collision Prediction dataset (Moura, Zhu, Zvitia, CVPR Workshops 2025), used for research and teaching with attribution under the Nexar open data licence. The deck itself is not committed to the repository, to avoid redistributing dataset frames.

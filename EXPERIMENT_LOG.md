# Experiment Log

A running record of what has been tried in this project, why, what came out, and what is still open.
Written so that someone (or a future session) can pick the project up without re-deriving anything.

- **Spec:** `200notok_proposal.tex` (COMS4036A, Compare/Probe). It wins over the README and any informal notes where they disagree.
- **Overview, results and how to run things:** `README.md`. The original long-form design document is `docs/PROJECT_DESIGN.md` (kept for reference).
- **Last updated:** 2026-10-07, after notebooks 07-09 (final analysis, failure analysis, id-block confound tests).

---

## 1. What the proposal actually requires

| Item | Requirement |
|---|---|
| Research question | How does collision-prediction performance change as time-to-event (TTE) increases, and does a lightweight learned temporal model beat classical motion cues? |
| Hypothesis | Performance drops from 0.5 s to 1.5 s, but the learned model degrades **more slowly** than the classical baseline. |
| Data | Nexar Dashcam Collision Prediction, ~1,500 videos (750 pos / 750 neg). Split at **video level before clip generation**. |
| TTE conditions | Clips ending 0.5 s, 1.0 s, 1.5 s before the event. |
| Primary metric | **AUPRC (AP)**, plus precision and recall. |
| Also requested | Performance-vs-TTE curve; **recall at a fixed false-alarm rate** "where practical". |
| Secondary experiment | Temporal ablation: normal frame order vs simplified temporal inputs (optical flow or **shuffled frames**). |
| Models | 0 Naive; 1 Optical flow + simple classifier; 2 Pretrained-CNN frame features + GRU/LSTM. |
| Optional | Weather/lighting/scene metadata **only for failure analysis**. |
| Minimum viable result | The 3 models at the 3 TTEs. |

**Not in the proposal** (do not add without a reason): inference-time measurement, environment-breakdown experiments, transformers/3D CNNs/other backbones, extra TTEs. A partner's summary claimed inference time was proposed; it is not.
The proposal says "training and test sets", but the repo only has **train/validation** (no held-out test), see §6 and §7.

---

## 2. Shared experimental protocol (fixed across all experiments)

- **Split:** 1,500 videos -> 1,200 train / 300 val, stratified 50/50, seed 42, video-level, zero overlap (checked). Files: `data/processed/{train,val}_split.csv`.
- **Clips:** 2.0 s observation window ending at `prediction_time`; 16 frames sampled per clip; no frame after `prediction_time` (checked, incl. a frame-boundary fix for video 1734, which has 570 frames).
- **Positives:** `prediction_time = time_of_event - TTE`.
- **Negatives:** `prediction_time` sampled (seeded) from the matching positive distribution, so negatives are not trivially identifiable by clip position. Files: `data/processed/{train,val}_clips.csv` (one row per video x TTE).
- **Sizes:** 1,200 train clips per TTE, **except TTE 1.5 which has 1,199** (one clip dropped upstream; the reason has not been documented, worth checking). Validation is 300 clips per TTE, the same 300 videos in the same order at every TTE (asserted in notebook 06).
- **CNN features:** frozen ImageNet ResNet-18, final FC removed -> 512-d per frame -> cached as `(N, 16, 512)` in `data/processed/cnn_features/{train,val}_tte_{TTE}.npy`.
- **Flow features:** 21-d per clip (7 motion statistics per frame-pair, summarised), cached as `data/processed/{X,y}_{train,val}_flow_tte_{TTE}.npy`.

---

## 3. Experiment log

### E0 - Dataset, video inspection, split, clip sampling (notebooks 01-03) - DONE
Verified metadata, class balance, event/alert times (mean alert lead time about 1.6 s), MP4 id mapping, FPS/frame indexing; built the leak-free split and TTE clip sampler. No open issues except the 1,199 count above.

### E1 - Naive baseline (notebook 04) - DONE
Predict-positive-for-all. On the balanced val set: AP 0.5, ROC-AUC 0.5, precision 0.5, recall 1.0, F1 0.667 at every TTE. It is a reference, not a model.

### E2 - Optical-flow baseline (notebook 04) - DONE
Seven motion statistics per frame pair -> 21-d clip features -> logistic regression.
Two variants were run:

| Variant | AP @ 0.5 / 1.0 / 1.5 s | Status |
|---|---|---|
| Unscaled LR | 0.5821 / 0.5692 / 0.5680 | Only in git history (`f2bec14:results/baseline_results.csv`). Quoted by the partner. |
| **Standardised (StandardScaler) + LR** | **0.5745 / 0.5758 / 0.5683** | **Final baseline**; this is what `results/baseline_results.csv` and every later comparison use. |

The two differ by about 0.01 AP; the story is unaffected, but **reports must quote the scaled version consistently**.
When refitted on this machine (Python 3.10, sklearn 1.7.2) the scaled model gives AP 0.5753 / 0.5756 / 0.5685, within 0.0007 of the saved numbers (solver/BLAS differences across environments); notebook 06 uses the refit values so all models are scored by the same code.

### E3 - CNN + LSTM (notebook 05) - DONE, but numbers superseded by E4
ResNet-18 (frozen) -> LSTM (hidden 128, 1 layer) -> linear logit. Adam, lr 1e-3, batch 32, 10 epochs, BCE-with-logits, one run per TTE, seed 42. **The epoch with the best *validation* AP was kept and that same validation AP was reported.**
Reported: AP 0.7934 / 0.7737 / 0.7392 (ROC-AUC 0.781 / 0.747 / 0.709). These are **optimistic** (selection on the evaluation set, single seed); do not use them in the report. The code and cached features remain valid.

### E4 - Temporal ablation + multi-seed + clean model selection (notebook 06) - DONE
**Question:** is the CNN+LSTM gain from temporal modelling, or just from strong static ResNet features?
**Protocol changes vs E3:** 5 seeds (42-46); 20% of *training* videos held out (stratified, fixed, seed 42) as an inner set to pick the epoch; the official val set is scored once per run and never used for selection; recall @ fixed FPR added; bootstrap CIs over val clips.
Same features, clips, optimiser and hyper-parameters as E3 (10 epochs).

Variants (all on the same cached `(16, 512)` features):

| Variant | What it tests |
|---|---|
| `lstm` | The E3 model, order preserved |
| `lstm_test_shuffled` | The trained LSTM fed shuffled frames at test time (10 shuffles averaged) |
| `lstm_train_shuffled` | LSTM trained **and** tested on shuffled frames |
| `mean_pool` | Mean of the 16 frame features -> MLP (all frames, no order) |
| `last_frame` | Only the last frame -> MLP (a single static image) |
| `optical_flow`, `naive` | Re-scored with the same metric code |

Parameter counts: LSTM variants 328,833; MLP variants 65,793.

**Results, 10 epochs (primary).** Mean ± std over 5 seeds (flow/naive deterministic, one run). Recall@FPR for naive is the chance value, not a measurement.

| Variant | TTE | AP | ROC-AUC | Recall@FPR≤10% | Recall@FPR≤20% | Selected epoch | Oracle val AP (nb05 protocol) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive | 0.5 | 0.500 | 0.500 | 0.100 | 0.200 |  |  |
| Optical flow (scaled) | 0.5 | 0.575 | 0.589 | 0.173 | 0.260 |  |  |
| Last frame only | 0.5 | 0.759 ± 0.005 | 0.750 ± 0.005 | 0.363 ± 0.024 | 0.527 ± 0.034 | 9.4 | 0.788 |
| Mean-pool (no order) | 0.5 | 0.737 ± 0.007 | 0.713 ± 0.005 | 0.425 ± 0.018 | 0.515 ± 0.015 | 9.0 | 0.748 |
| LSTM, shuffled train+test | 0.5 | 0.736 ± 0.012 | 0.717 ± 0.007 | 0.387 ± 0.026 | 0.521 ± 0.040 | 8.2 | 0.752 |
| LSTM, shuffled at test only | 0.5 | 0.728 ± 0.013 | 0.707 ± 0.009 | 0.367 ± 0.046 | 0.497 ± 0.030 | 9.4 |  |
| CNN + LSTM | 0.5 | 0.762 ± 0.019 | 0.750 ± 0.013 | 0.383 ± 0.034 | 0.547 ± 0.050 | 9.4 | 0.790 |
| Naive | 1.0 | 0.500 | 0.500 | 0.100 | 0.200 |  |  |
| Optical flow (scaled) | 1.0 | 0.576 | 0.592 | 0.153 | 0.267 |  |  |
| Last frame only | 1.0 | 0.752 ± 0.007 | 0.729 ± 0.006 | 0.404 ± 0.027 | 0.483 ± 0.021 | 8.4 | 0.775 |
| Mean-pool (no order) | 1.0 | 0.725 ± 0.008 | 0.698 ± 0.007 | 0.360 ± 0.033 | 0.485 ± 0.032 | 9.0 | 0.739 |
| LSTM, shuffled train+test | 1.0 | 0.719 ± 0.017 | 0.699 ± 0.012 | 0.360 ± 0.038 | 0.479 ± 0.028 | 8.8 | 0.739 |
| LSTM, shuffled at test only | 1.0 | 0.725 ± 0.014 | 0.695 ± 0.017 | 0.377 ± 0.038 | 0.477 ± 0.036 | 6.8 |  |
| CNN + LSTM | 1.0 | 0.749 ± 0.011 | 0.726 ± 0.013 | 0.425 ± 0.015 | 0.511 ± 0.045 | 6.8 | 0.771 |
| Naive | 1.5 | 0.500 | 0.500 | 0.100 | 0.200 |  |  |
| Optical flow (scaled) | 1.5 | 0.569 | 0.574 | 0.207 | 0.253 |  |  |
| Last frame only | 1.5 | 0.709 ± 0.018 | 0.690 ± 0.013 | 0.352 ± 0.011 | 0.469 ± 0.022 | 7.0 | 0.747 |
| Mean-pool (no order) | 1.5 | 0.707 ± 0.004 | 0.677 ± 0.005 | 0.359 ± 0.019 | 0.479 ± 0.010 | 9.4 | 0.727 |
| LSTM, shuffled train+test | 1.5 | 0.709 ± 0.017 | 0.681 ± 0.017 | 0.344 ± 0.048 | 0.459 ± 0.043 | 8.0 | 0.728 |
| LSTM, shuffled at test only | 1.5 | 0.676 ± 0.013 | 0.651 ± 0.011 | 0.299 ± 0.015 | 0.419 ± 0.025 | 8.8 |  |
| CNN + LSTM | 1.5 | 0.675 ± 0.021 | 0.662 ± 0.016 | 0.304 ± 0.026 | 0.416 ± 0.030 | 8.8 | 0.734 |

Paired bootstrap (2,000 resamples of the 300 val clips; probabilities averaged across seeds first, so these CIs are for the seed-ensemble):

| Comparison | TTE | Estimate | 95% CI | CI excludes 0 |
|---|---:|---:|---|:--:|
| Δ AP: CNN+LSTM − Optical flow | 0.5 | +0.197 | [+0.111, +0.266] | yes |
| Δ AP: CNN+LSTM − Optical flow | 1.0 | +0.180 | [+0.103, +0.251] | yes |
| Δ AP: CNN+LSTM − Optical flow | 1.5 | +0.117 | [+0.032, +0.192] | yes |
| Δ AP: CNN+LSTM − Last frame (no temporal) | 0.5 | +0.009 | [-0.008, +0.026] | no |
| Δ AP: CNN+LSTM − Last frame (no temporal) | 1.0 | +0.002 | [-0.018, +0.023] | no |
| Δ AP: CNN+LSTM − Last frame (no temporal) | 1.5 | -0.025 | [-0.044, -0.001] | yes |
| Δ AP: CNN+LSTM − Mean-pool (no order) | 0.5 | +0.030 | [-0.001, +0.062] | no |
| Δ AP: CNN+LSTM − Mean-pool (no order) | 1.0 | +0.028 | [-0.002, +0.056] | no |
| Δ AP: CNN+LSTM − Mean-pool (no order) | 1.5 | -0.022 | [-0.058, +0.017] | no |
| Δ AP: CNN+LSTM − LSTM, shuffled train+test | 0.5 | +0.032 | [-0.002, +0.066] | no |
| Δ AP: CNN+LSTM − LSTM, shuffled train+test | 1.0 | +0.023 | [-0.003, +0.050] | no |
| Δ AP: CNN+LSTM − LSTM, shuffled train+test | 1.5 | -0.030 | [-0.062, +0.007] | no |
| Δ AP: CNN+LSTM − LSTM, shuffled at test only | 0.5 | +0.031 | [+0.008, +0.057] | yes |
| Δ AP: CNN+LSTM − LSTM, shuffled at test only | 1.0 | +0.023 | [+0.003, +0.043] | yes |
| Δ AP: CNN+LSTM − LSTM, shuffled at test only | 1.5 | -0.006 | [-0.032, +0.022] | no |
| AP drop 0.5s→1.0s: CNN + LSTM | 1.0 | +0.016 | [-0.021, +0.055] | no |
| AP drop 0.5s→1.5s: CNN + LSTM | 1.5 | +0.086 | [+0.030, +0.138] | yes |
| AP drop 0.5s→1.0s: Mean-pool (no order) | 1.0 | +0.014 | [-0.018, +0.045] | no |
| AP drop 0.5s→1.5s: Mean-pool (no order) | 1.5 | +0.034 | [-0.010, +0.076] | no |
| AP drop 0.5s→1.0s: Last frame (no temporal) | 1.0 | +0.010 | [-0.035, +0.053] | no |
| AP drop 0.5s→1.5s: Last frame (no temporal) | 1.5 | +0.052 | [-0.005, +0.103] | no |
| AP drop 0.5s→1.0s: Optical flow | 1.0 | -0.000 | [-0.050, +0.056] | no |
| AP drop 0.5s→1.5s: Optical flow | 1.5 | +0.007 | [-0.052, +0.063] | no |

### E5 - 30-epoch check for under-training (notebook 06, `NUM_EPOCHS=30`) - DONE
**Why:** E4's learning curves were still rising at epoch 10 and selected epochs were often 8-10.
**Result:** not under-trained. Inner-validation AP peaks at about epochs 8-15 and then **declines** for every variant (overfitting). Mean AP is *lower* at 30 epochs, because the longer schedule gives the small (240-clip) inner set more chances to pick a bad epoch. Conclusions unchanged. **Decision: keep 10 epochs.**

| Variant | TTE | AP | ROC-AUC | Recall@FPR≤10% | Recall@FPR≤20% | Selected epoch | Oracle val AP (nb05 protocol) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive | 0.5 | 0.500 | 0.500 | 0.100 | 0.200 |  |  |
| Optical flow (scaled) | 0.5 | 0.575 | 0.589 | 0.173 | 0.260 |  |  |
| Last frame only | 0.5 | 0.726 ± 0.012 | 0.723 ± 0.010 | 0.327 ± 0.012 | 0.475 ± 0.036 | 16.6 | 0.788 |
| Mean-pool (no order) | 0.5 | 0.717 ± 0.014 | 0.692 ± 0.010 | 0.359 ± 0.040 | 0.471 ± 0.032 | 15.4 | 0.748 |
| LSTM, shuffled train+test | 0.5 | 0.694 ± 0.027 | 0.674 ± 0.021 | 0.297 ± 0.042 | 0.423 ± 0.056 | 14.4 | 0.752 |
| LSTM, shuffled at test only | 0.5 | 0.705 ± 0.021 | 0.686 ± 0.014 | 0.336 ± 0.048 | 0.464 ± 0.027 | 13.0 |  |
| CNN + LSTM | 0.5 | 0.741 ± 0.021 | 0.729 ± 0.018 | 0.355 ± 0.036 | 0.500 ± 0.046 | 13.0 | 0.790 |
| Naive | 1.0 | 0.500 | 0.500 | 0.100 | 0.200 |  |  |
| Optical flow (scaled) | 1.0 | 0.576 | 0.592 | 0.153 | 0.267 |  |  |
| Last frame only | 1.0 | 0.735 ± 0.015 | 0.714 ± 0.013 | 0.355 ± 0.035 | 0.452 ± 0.026 | 12.0 | 0.775 |
| Mean-pool (no order) | 1.0 | 0.680 ± 0.014 | 0.663 ± 0.012 | 0.265 ± 0.013 | 0.387 ± 0.044 | 19.2 | 0.739 |
| LSTM, shuffled train+test | 1.0 | 0.671 ± 0.027 | 0.664 ± 0.015 | 0.249 ± 0.064 | 0.388 ± 0.042 | 20.0 | 0.739 |
| LSTM, shuffled at test only | 1.0 | 0.695 ± 0.015 | 0.665 ± 0.017 | 0.313 ± 0.031 | 0.408 ± 0.030 | 14.0 |  |
| CNN + LSTM | 1.0 | 0.712 ± 0.029 | 0.692 ± 0.022 | 0.331 ± 0.055 | 0.445 ± 0.050 | 14.0 | 0.771 |
| Naive | 1.5 | 0.500 | 0.500 | 0.100 | 0.200 |  |  |
| Optical flow (scaled) | 1.5 | 0.569 | 0.574 | 0.207 | 0.253 |  |  |
| Last frame only | 1.5 | 0.705 ± 0.022 | 0.687 ± 0.016 | 0.345 ± 0.023 | 0.467 ± 0.025 | 7.6 | 0.747 |
| Mean-pool (no order) | 1.5 | 0.690 ± 0.018 | 0.662 ± 0.018 | 0.328 ± 0.047 | 0.453 ± 0.026 | 13.2 | 0.727 |
| LSTM, shuffled train+test | 1.5 | 0.707 ± 0.019 | 0.679 ± 0.021 | 0.344 ± 0.048 | 0.449 ± 0.042 | 8.4 | 0.728 |
| LSTM, shuffled at test only | 1.5 | 0.667 ± 0.023 | 0.643 ± 0.020 | 0.281 ± 0.037 | 0.403 ± 0.050 | 10.2 |  |
| CNN + LSTM | 1.5 | 0.662 ± 0.031 | 0.654 ± 0.025 | 0.283 ± 0.058 | 0.400 ± 0.055 | 10.2 | 0.734 |

| Comparison | TTE | Estimate | 95% CI | CI excludes 0 |
|---|---:|---:|---|:--:|
| Δ AP: CNN+LSTM − Optical flow | 0.5 | +0.175 | [+0.089, +0.242] | yes |
| Δ AP: CNN+LSTM − Optical flow | 1.0 | +0.146 | [+0.068, +0.218] | yes |
| Δ AP: CNN+LSTM − Optical flow | 1.5 | +0.101 | [+0.014, +0.179] | yes |
| Δ AP: CNN+LSTM − Last frame (no temporal) | 0.5 | +0.023 | [-0.003, +0.050] | no |
| Δ AP: CNN+LSTM − Last frame (no temporal) | 1.0 | -0.019 | [-0.043, +0.003] | no |
| Δ AP: CNN+LSTM − Last frame (no temporal) | 1.5 | -0.037 | [-0.059, -0.009] | yes |
| Δ AP: CNN+LSTM − Mean-pool (no order) | 0.5 | +0.031 | [-0.003, +0.066] | no |
| Δ AP: CNN+LSTM − Mean-pool (no order) | 1.0 | +0.039 | [+0.000, +0.079] | yes |
| Δ AP: CNN+LSTM − Mean-pool (no order) | 1.5 | -0.022 | [-0.059, +0.022] | no |
| Δ AP: CNN+LSTM − LSTM, shuffled train+test | 0.5 | +0.043 | [+0.012, +0.077] | yes |
| Δ AP: CNN+LSTM − LSTM, shuffled train+test | 1.0 | +0.027 | [-0.009, +0.061] | no |
| Δ AP: CNN+LSTM − LSTM, shuffled train+test | 1.5 | -0.049 | [-0.082, -0.005] | yes |
| Δ AP: CNN+LSTM − LSTM, shuffled at test only | 0.5 | +0.027 | [+0.002, +0.052] | yes |
| Δ AP: CNN+LSTM − LSTM, shuffled at test only | 1.0 | +0.012 | [-0.016, +0.039] | no |
| Δ AP: CNN+LSTM − LSTM, shuffled at test only | 1.5 | -0.012 | [-0.035, +0.017] | no |
| AP drop 0.5s→1.0s: CNN + LSTM | 1.0 | +0.028 | [-0.014, +0.075] | no |
| AP drop 0.5s→1.5s: CNN + LSTM | 1.5 | +0.081 | [+0.018, +0.137] | yes |
| AP drop 0.5s→1.0s: Mean-pool (no order) | 1.0 | +0.037 | [-0.002, +0.079] | no |
| AP drop 0.5s→1.5s: Mean-pool (no order) | 1.5 | +0.028 | [-0.016, +0.072] | no |
| AP drop 0.5s→1.0s: Last frame (no temporal) | 1.0 | -0.013 | [-0.065, +0.040] | no |
| AP drop 0.5s→1.5s: Last frame (no temporal) | 1.5 | +0.021 | [-0.046, +0.082] | no |
| AP drop 0.5s→1.0s: Optical flow | 1.0 | -0.000 | [-0.050, +0.056] | no |
| AP drop 0.5s→1.5s: Optical flow | 1.5 | +0.007 | [-0.052, +0.063] | no |

### E6 - Final analysis (notebook 07) - DONE
No training; reads the notebook 06 CSVs (10-epoch protocol) and produces the report figures and tables.
**Framing:** the 300 official-validation videos are called the **held-out test set** from here on. Notebook 06 chose epochs on an inner split of the training videos and never used these 300 for selection or tuning, so they are clean for that protocol. Minor exception: the scaled-vs-unscaled optical-flow choice was made after seeing their results (about 0.01 AP effect on the baseline only).

| Output | File |
|---|---|
| Fig 1: AP and recall@FPR<=10% vs TTE (naive / flow / CNN+LSTM) | `results/figures/final_fig1_main_result.png` |
| Fig 2: temporal ablation, per TTE | `results/figures/final_fig2_temporal_ablation.png` |
| Fig 3: paired AP differences, CNN+LSTM minus each other model (95% bootstrap CI) | `results/figures/final_fig3_paired_differences.png` |
| Fig 4: AP lost relative to 0.5 s, with CIs | `results/figures/final_fig4_degradation.png` |
| Main table (3 models x 3 TTEs, all metrics) | `results/final_results_table.csv` |
| Ablation table | `results/final_ablation_table.csv` |
| Notebook-05 selection-optimism table | `results/final_selection_optimism.csv` |

Colours follow the dataviz method (blue = CNN+LSTM, orange = flow, grey = controls; blue/orange validated for CVD separation and contrast). Notebook 05 optimism measured: 0.028 / 0.022 / 0.059 AP at 0.5 / 1.0 / 1.5 s.

### E7 - Failure analysis (notebook 08) - DONE
**Question:** the single-frame result (E4) implies the CNN features carry most of the signal. Is that an imminent-collision cue, or scene/video-level appearance? Uses only cached features and metadata (**the raw videos are not on this machine**, so no frames were inspected).
It first retrains the 3 CNN models with the identical protocol (5 seeds) to recover per-clip probabilities, and asserts the seed-ensemble APs match notebook 06 to 1e-6. Predictions are saved to `results/val_predictions.csv`.

| Analysis | Result |
|---|---|
| **Frame-position AP:** single-frame classifier on each of the 16 positions, x-axis = seconds before the event (`failure_fig1_frame_position.png`, `failure_frame_position_ap.csv`) | AP is about 0.70 (chance 0.5) even for frames 3+ s before the event; it rises only to about 0.75 within about 1.2 s of the event. The three TTE curves overlap on this axis. Most of the signal is **not** a late pre-collision cue. |
| **Alert timing:** AUC of positives vs all negatives, split by whether the clip ends before or after `time_of_alert` (`failure_fig2_alert_timing.png`, `failure_alert_timing.csv`) | After-alert positives: AUC about 0.72-0.77. Before-alert: about 0.62-0.66 (n = 46 at 1.0 s, 82 at 1.5 s; only 9 at 0.5 s). Both above chance; a real but smaller hazard-related component. CNN+LSTM and last-frame behave the same; optical flow shows no such pattern. |
| **Metadata shortcuts** (`failure_metadata_shortcuts.csv`) | `prediction_time` alone: AUC 0.50-0.53 (negative sampling worked). **Video id alone: AUC 1.000** (see below). |
| **Error vs position in video** (`failure_by_prediction_time.csv`) | AP lower when the prediction point is early in the video (about 0.58-0.68 in the early third) than in the middle third (about 0.78-0.88). Small groups. |
| **Persistent errors** (`failure_persistent_errors.csv`, `failure_fig3_score_distributions.png`) | 152 / 300 test videos correct at all three horizons; 71 wrong at all three (36 false alarms, 35 missed collisions). Ids and clip windows are exported for visual inspection when the raw videos are available. |

**Dataset-structure finding (important):** across all 1,500 labelled videos, **every collision video has id <= 1039 and every normal video has id >= 1040** (zero overlap; 729 of 750 positives are below 1000). The models never see the id (they receive only ResNet activations; the id only locates the video file), so no reported number is invalid. But the two classes were evidently assembled in separate blocks. If the blocks differ in camera, device, region, time of day or recording batch, a classifier can score well by recognising that, which is consistent with the strong static-frame result. It **cannot be confirmed or ruled out without the raw videos**.

### E8 - Is the id-block structure a confound? Feature-based tests (notebook 09) - DONE, inconclusive but suggestive
**Why:** E7 found that video id perfectly separates the classes. The direct check needs the raw videos, which are **not on this machine** (searched `D:`, Downloads, Documents, Desktop; no Kaggle credentials either). This notebook does what the cached ResNet features allow, and `tools/compare_id_blocks.py` (below) is the direct check for wherever the dataset is.
**Method:** all 1,500 labelled videos, one clip each at TTE 1.0 s, mean-pooled ResNet features, regularised logistic regression, **out-of-fold** scores (10-fold), so no video is scored by a model that saw it. Collision-vs-normal out-of-fold AUC = 0.72.

| Test | Prediction if features encode an id-ordered recording batch | Result |
|---|---|---|
| A. Within one class, predict lower vs upper id half | above chance | collision AUC 0.563 (shuffled-label 95th pct 0.522); normal AUC 0.501 (null 0.54) |
| B. Boundary groups (21 collisions with id 1000-1039; 40 normals with id 1040-1099) | shift toward the neighbouring class | collisions -0.078 (CI -0.17..+0.02, p=0.10); normals +0.055 (CI -0.02..+0.13, p=0.15). Right direction, not significant |
| C. Within-class trend of the score along id | monotone drift | collision: Spearman -0.128, permutation p = 0.0005 (mean score 0.64 -> 0.57 from lowest to highest id quartile); normal: -0.058, p = 0.10 |
| D. Class separation (AUC) near the boundary vs far | lower near the boundary | AUC 0.594 within +/-100 ids vs 0.730 farther away (diff -0.136, 95% CI -0.236..-0.036); +/-200 and +/-300 windows agree (CIs exclude 0) |

**Reading:** consistent with the features carrying some information about *position in the id ordering* in addition to collision content: the score drifts down with id and the classes become harder to separate where their ids are close. Not the whole story (the classes are still separable far from the boundary, AUC 0.73; the within-class drift is about 0.07 vs a class gap of about 0.2). It does **not** identify what changes with id, and an effect that is constant within each class would pass every test; ids could also be ordered by something event-related (e.g. severity). Evidence the confound is plausible, not proof.
Outputs: `results/id_block_test_{a,b,c,d}_*.csv`, `results/id_block_test_c_id_quartiles.csv`, `results/figures/id_block_fig1_scores_by_id.png`.

**Raw-video tool (`tools/compare_id_blocks.py`).** For every available labelled video it measures resolution, fps, duration, bitrate, brightness, contrast, dark-pixel share, saturation and sharpness (3 sampled frames), then reports per-property medians by class, the single-property AUC for predicting the class, within-class id drift, a property-only classifier (class and id-half, with a shuffled-label baseline), a properties-vs-id figure and contact sheets. Run: `python tools/compare_id_blocks.py` from the project root with `data/raw/train/{id:05d}.mp4` present (`--available-only` for a partial download, `--limit 60` for a quick check, `--raw PATH` if the videos live elsewhere). Its results are E9.

### E9 - Raw-video comparison of the id blocks (`tools/compare_id_blocks.py`) - DONE on a 431-video sample
**Data:** the raw videos were downloaded on 2026-10-07 through the Kaggle REST API (the account token was supplied by the user; the CLI available for Python 3.10, 1.7.4.5, does not read the newer `access_token` file, so a small downloader was used; it lives only in the scratchpad, not in the repo). **Only a sample was downloaded (7.4 GB of the 25.5 GB train set):** all 61 videos next to the boundary (21 collisions with id 1000-1039, 40 normals with id 1040-1099), all 71 persistent-error videos, and 150 random collision + 150 random normal videos; 431 with `00000.mp4` from the probe (206 collision, 225 normal). All file sizes match Kaggle's listing. The videos are in `data/raw/train/` (gitignored). Test videos and the remaining ~1,070 train videos were not downloaded (disk: `D:` had 47 GB free).

| Check | Result |
|---|---|
| Resolution, frame rate, duration | identical: every video is 1280x720; fps about 30 (21 distinct values around 30); duration about 40 s in both classes (a short cluster near 18-19 s in both: 10% of collisions, 16% of normals) |
| Single-property AUC for the class (10 properties) | all within 0.46-0.56 (largest: sharpness 0.563, duration 0.556); none passes the |AUC - 0.5| > 0.15 threshold |
| Day/night, brightness, contrast, saturation | similar in both classes (luminance median 98 vs 95; both bimodal day/night) |
| Property-only classifier (gradient boosting, 5-fold CV) | class: AUC 0.579 vs shuffled-label mean 0.49 (max 0.61), i.e. not distinguishable from chance. Id half within collision: 0.604 (null max 0.67); within normal: 0.603 (null max 0.60) |
| Overlay (Nexar logo top-left, "GETNEXAR.COM" badge top-right) | present on videos from every id range in both classes (36 top-strip crops checked); not a block difference. One normal video (id 1988) also has a burned-in dashcam date stamp |
| Weak id-ordered traces | bitrate rises with id inside the collision block (Spearman +0.241, p = 0.0005); duration rises with id inside the normal block (+0.176, p = 0.008). Boundary groups do not differ significantly (bitrate medians: collision <1000 3,494; collision 1000-1039 4,055; normal 1040-1099 3,894; normal >=1100 3,671 kbit/s; Mann-Whitney p = 0.38, 0.29, and 0.98 for boundary-collision vs boundary-normal) |
| Contact sheets (`results/id_blocks/contact_sheet_*.png`, 12 persistent-error videos viewed) | Both blocks are the same kind of footage: US dashcam, urban/suburban/highway, day and night. Persistent false alarms are mostly dense night city traffic with a vehicle close ahead (ids 1344, 1694, 1950, 1295); persistent misses are rain, overcast, overexposed or empty-road scenes (362, 403, 457). Qualitative, from a dozen videos. |

**Reading:** no simple recording property, format or overlay separates the two id blocks, and these properties together do not predict the class above chance, so a *device / format / watermark* confound is not supported at the level measured. A small id-ordered drift exists (bitrate, duration) and the feature tests (E8) also saw one, but it is weak and not explained by these properties. What remains possible, and cannot be tested with these measurements, is a difference in **scene content** between blocks (location, season, traffic density, collection period). The error examples hint that the model keys on scene context (dense, close-following night traffic reads as risky), which would make it a "risky-looking scene" detector rather than an imminent-collision detector; that is a hypothesis from a handful of videos, not a result.
**Limits:** a sample of 431 of 1,500 videos (the boundary groups and error videos are complete); sampled frames at 25/50/75% of each video; 10 simple properties.
Outputs: `results/id_blocks/` (`video_properties.csv`, `block_comparison.csv`, `property_classifier.csv`, `properties_by_id.png`, three contact sheets).

### E10 - Which CNN+LSTM estimate is closer to the truth? Full-data, no-selection check (`tools/fulldata_selection_check.py`) - DONE
**Why:** the partner's analysis reports notebook 05's numbers; this log reports notebook 06's clean-protocol numbers. Notebook 06 uses only 80% of the training videos, so part of the gap could be lost training data rather than selection bias. This isolates the two.
**Method:** same CNN+LSTM, **all** training clips, fixed 10 epochs, 5 seeds, validation AP recorded each epoch.

| CNN+LSTM AP | 0.5 s | 1.0 s | 1.5 s |
|---|---|---|---|
| Notebook 05 as reported (best epoch picked on the evaluation videos, 1 seed) | 0.793 | 0.774 | 0.739 |
| Same protocol, 5-seed mean of the per-run best epoch ("oracle") | 0.793 | 0.780 | 0.742 |
| **Full data, fixed 10 epochs, no selection** (mean ± std, 5 seeds) | 0.754 ± 0.015 | 0.753 ± 0.009 | 0.676 ± 0.021 |
| Mean over epochs 1-10 (no selection, all epochs) | 0.778 | 0.765 | 0.714 |
| **Notebook 06 clean protocol** (80% data, epoch chosen on an inner split) | 0.762 | 0.749 | 0.675 |

**Reading:** selection on the evaluation videos inflates AP by about 0.039 / 0.027 / 0.066 (oracle minus no-selection); losing 20% of the training data costs essentially nothing (clean 0.762 / 0.749 / 0.675 vs no-selection 0.754 / 0.753 / 0.676). So the notebook-06 numbers are a sound estimate and notebook 05's are optimistic. The inflation is **not uniform**: it is largest at 1.5 s, so the optimistic numbers also flatten the degradation curve (0.5 s to 1.5 s drop: 0.054 optimistic vs about 0.08 clean). Caveat: validation AP peaks around epochs 3-5 (mean best epoch 4.6 / 3.2 / 3.0) and is lower at epoch 10, so a fixed 10-epoch schedule is itself slightly conservative; a properly tuned epoch count (chosen on held-out training data, not the evaluation set) could land between the two, but not at the optimistic level.
Outputs: `results/selection_check_fulldata.csv` (per run).

### E11 - Report and reproducibility check (2026-10-24) - DONE
**Report:** `report/main.tex` -> `report/main.pdf` (IEEE two-column, 4 pages; limit is 6), built with Docker (`report/build.sh`). Figures and tables come from `tools/make_report_assets.py`, which reads only the result CSVs; the numbers typed in the text were checked against those CSVs. The text states the Nexar licence terms. AI use is declared on the separate form the course requires, not in the report (the section was removed on request).
**Clean checkout:** a fresh clone with no raw data ran notebooks 06-09 via `python tools/run_notebooks.py` and regenerated all 24 result CSVs identically (difference below 1e-9; same machine, Python 3.10). Fixes this needed: notebooks now start by moving to the project root when launched from `notebooks/`, and the runner script was added.
**Course requirements** (from the project brief): report at most six pages, IEEE two-column, repo link on page 1 and reachable logged out; repo marked on README with setup, data and commands, seeds set, results reproducible from a clean checkout; third-party data licence recorded in the report; AI use declared in the report. Presentation and live Q&A on Tue 27 Oct (every member must be able to explain every part); report due 07:00 Wed 28 Oct.
**Open before submission:** repo must be public (it is currently private); the working branch must be merged to `main` or linked directly; make sure the separate AI declaration form is attached and accurate (confirm with Rameez, and against the course AI policy).

---

## 4. Findings so far (test set = the 300 held-out videos, descriptive, not final conclusions)

1. **Learned model > classical.** CNN+LSTM beats optical flow by +0.197 / +0.180 / +0.117 AP (10 ep; +0.175 / +0.146 / +0.101 at 30 ep); all CIs exclude zero. Flow is only slightly above chance (AP about 0.57).
2. **Temporal benefit not demonstrated.** A single last-frame classifier ties the LSTM at 0.5 s and 1.0 s and beats it at 1.5 s (0.709 vs 0.675, CI of the difference just excludes zero). Mean-pooling and an LSTM trained on shuffled frames are within noise of the LSTM. The static ResNet features carry most of the signal.
3. **Weak sign of order use.** Test-time shuffling lowers the trained LSTM by about 0.03 AP at 0.5 s and about 0.02 at 1.0 s (CIs only just exclude 0), not at 1.5 s. Yet a model trained without order does as well.
4. **Hypothesis not supported.** The proposal predicted the learned model degrades more slowly than flow. CNN+LSTM drops 0.086 AP from 0.5 s to 1.5 s (CI 0.03-0.14); flow is flat (-0.007) only because it is near chance everywhere. The 0.5 s -> 1.0 s drop is not distinguishable from zero for any model.
5. **Notebook 05 was optimistic by about 0.03 (0.5 s, 1.0 s) and about 0.06 (1.5 s) AP**, from selecting the epoch on the evaluation set. The oracle column reproduces its numbers (0.790 / 0.771 / 0.734).
6. **False-alarm view:** at <=10% false-alarm rate the CNN-based models recover roughly 30-43% of collisions; flow recovers 15-21% (chance 10%).

7. **The signal is mostly static (E7).** A single frame from 3+ s before the event gives AP about 0.70; frames near the event add only about +0.05. So the model is largely not detecting an imminent-collision cue.
8. **A hazard-related component exists but is smaller (E7).** Positives whose clip ends after the annotated alert are detected better (AUC about 0.72-0.77) than those ending before it (about 0.62-0.66).
9. **Video id perfectly separates the classes (E7)** (collision <= 1039, normal >= 1040). Models do not use it, but the blocks may differ in recording conditions, which would partly explain finding 7. Unverified without the raw videos.
10. **The id structure leaves a visible trace in the features (E8).** Collision score drifts down with id within the collision block (rho -0.128, p = 0.0005), and class separation is weaker near the id boundary (AUC 0.59 vs 0.73). Moderate evidence that part of the signal is id-ordered recording/collection structure; cause unknown until the raw videos are compared.
11. **The raw videos show no device/format/overlay difference between the id blocks (E9).** Same resolution, fps, duration, overlay and day/night mix; simple properties do not predict the class (AUC 0.58, inside the shuffled-label range). A scene-content difference between blocks remains possible and untested; error examples suggest the model keys on dense, close-following traffic scenes.

Suggested wording for the report: *"Pretrained CNN features separate the collision and normal videos substantially better than optical flow at every horizon. A single-frame ablation shows this comes mainly from the visual representation rather than from temporal modelling, and a frame-position analysis shows much of the signal is present well before the event, so the model should be described as recognising high-risk driving scenes, not as detecting imminent collisions. The two classes occupy disjoint video-id blocks; we found no difference in resolution, frame rate, duration or overlay between blocks, but a scene-content difference between blocks cannot be excluded."*

---

## 5. Decisions made (and why)

| Decision | Reason |
|---|---|
| Scaled optical flow is the baseline | It is what the code/CSVs produce; difference to unscaled is about 0.01 AP. |
| Use notebook 06 numbers, not notebook 05, for the report | Notebook 05 selects the epoch on the evaluation set. |
| Select epochs on an inner split of the *training* videos | Avoids tuning on the evaluation set; costs 20% of training data. |
| Keep 10 epochs | 30-epoch run (E5) showed overfitting, no gain. |
| Fixed inner split, seeds vary only init/batch order | Isolates training noise; the bootstrap covers val-sampling noise. |
| Bootstrap on seed-averaged probabilities, paired across models | Seed std alone understates uncertainty; 300 val clips give about +/-0.07 on a single AP. |
| Treat the 300 official-validation videos as the held-out test set | Never used for epoch selection or tuning in notebook 06 (selection used an inner training split); no re-split needed. Caveat: flow-variant choice was made after seeing them. |
| Figures use seed mean ± std; inference uses paired bootstrap CIs | Seed std understates uncertainty; paired CIs on the same resampled clips answer "is model A better than B". |
| Failure analysis uses cached features + metadata only | Raw videos are not available on this machine. |
| Do not add inference timing / environment breakdown / new architectures | Not in the proposal; lecturer warned against PhD-sized scope. |

---

## 6. Known issues / caveats

- **Class-specific video-id blocks (E7-E9).** Collision ids <= 1039, normal ids >= 1040. Not used by any model. Features show an id-ordered trace (E8); raw videos (E9, a 431-video sample) show no resolution/fps/duration/overlay/brightness difference between blocks and the simple properties do not predict the class, but scene content (location, season, traffic) could still differ and was not tested. Report this as a limitation.
- **Test set is a relabelled validation set.** The 300 videos were never used for selection in notebook 06, but earlier notebooks (04/05) reported on them and the scaled-flow variant was chosen by looking at them. No separate untouched set exists.
- **Single backbone, frozen features, one clip configuration** (16 frames over 2 s). Conclusions about temporal modelling apply to this setup only.
- **Small val set.** Differences of about 0.03 AP are near the noise floor; a single AP has a 95% interval of about +/-0.07.
- **Epoch selection noise.** 240-clip inner set; selected epochs vary across seeds.
- **Recall@FPR resolution.** With 150 val negatives, FPR<=10% means at most 15 false alarms, so the metric is coarse. Naive's value is the theoretical chance level.
- **Unexplained 1,199 clips** at TTE 1.5 train (see §2).
- **Notebooks 06-08 were executed end-to-end in Jupyter** (nbclient, Python 3.10 kernel, 2026-10-07) and their outputs are stored in the `.ipynb` files; they were first developed as scripts, and the notebook runs reproduce the script numbers exactly (CNN+LSTM AP 0.762 / 0.749 / 0.675). Notebooks 04 and 05 were run elsewhere (Python 3.13).
- **Environment drift:** earlier notebooks ran under Python 3.13 (likely a Mac with MPS); notebook 06 was run under Python 3.10 CPU (torch 2.13.0, sklearn 1.7.2, numpy 2.2.6, pandas 2.3.3). Flow refit differs from saved by <=0.0007 AP.
- **Why last-frame works** has not been investigated: it may reflect real pre-crash visual cues (close lead vehicle, braking scene) or dataset/sampling regularities. Worth a failure analysis before over-interpreting.

---

## 7. Open items / next steps (priority order)

1. **Optional extension of E9:** the sample covers 431 of 1,500 train videos. If more certainty is wanted, download the rest (about 18 GB, `D:` has about 40 GB free) and rerun `python tools/compare_id_blocks.py`; or quantify the scene-content hypothesis (e.g. traffic density / lead-vehicle proximity proxies on the persistent-error videos). Not required for the project.
2. ~~**README refresh**~~ Done 2026-10-24: the README is now a short project overview with results, quick start and reproducibility notes; the old design document moved to `docs/PROJECT_DESIGN.md`.
3. **Report writing**, using notebook 06/07 numbers, the figures in `results/figures/final_fig*.png`, and the wording and caveats in §4. Cover: RQ1 (horizon), RQ2 (learned vs classical), RQ3 (temporal ablation), hypothesis not supported, id-block caveat, notebook-05 optimism.
4. **Optional, only if time allows:** GRU instead of LSTM (proposal allows "GRU/LSTM or similar"); dropout / weight decay (overfitting is evident); find which clip was dropped at TTE 1.5; a stronger generalisation test that holds out id blocks (only meaningful if the raw videos show the blocks are comparable).

Done since the last revision: final analysis notebook (E6), failure analysis (E7). The "held-out test split" item is resolved by treating the 300 validation videos as the test set (see §5).

Explicitly **not planned**: transformers, 3D CNNs, other backbones, extra TTE values, fine-tuning the CNN (the design document lists these as stretch goals only).

---

## 8. How to reproduce

From the project root with the dataset in `data/raw/` and processed files present:

```text
notebooks/01 -> 02 -> 03   # split, clips, TTE sampling   (data/processed/*.csv)
notebooks/04               # naive + optical flow          (results/baseline_results.csv)
notebooks/05               # ResNet-18 features + CNN+LSTM (data/processed/cnn_features/*, results/cnn_lstm_results.csv)
notebooks/06               # ablation, 5 seeds, bootstrap  (results/temporal_ablation_*.csv, results/figures/*)
notebooks/07               # final figures/tables from the notebook-06 CSVs (no training)
notebooks/08               # failure analysis; retrains the CNN models (about 3 min on CPU)
notebooks/09               # id-block confound tests on the cached features (under 1 min)
tools/compare_id_blocks.py # raw-video block comparison (only where data/raw/train exists)
```

Notebook 06 only needs the cached features and flow arrays (no raw videos), and runs in about 1.5 min on CPU for 10 epochs (about 4 min for 30).
To test a different epoch budget, set the environment variable `NUM_EPOCHS` (outputs get an `_{N}ep` suffix; the default of 10 has none).
Model checkpoints (`*.pt`) are gitignored; notebook 05 writes them to `results/` locally.

---

## 9. File index

| Path | What |
|---|---|
| `200notok_proposal.tex` | Project proposal (spec) |
| `README.md` | Project overview, results, quick start, reproducibility |
| `docs/PROJECT_DESIGN.md` | Original long-form design document (reference) |
| `EXPERIMENT_LOG.md` | This file |
| `notebooks/01-03` | Exploration, video inspection, clip sampling and split |
| `notebooks/04_baselines.ipynb` | Naive + optical flow |
| `notebooks/05_learned_temporal_model.ipynb` | ResNet-18 features + CNN+LSTM (optimistic selection, see E3) |
| `notebooks/06_temporal_ablation.ipynb` | E4/E5: ablation, seeds, bootstrap, findings |
| `notebooks/07_final_analysis.ipynb` | E6: final figures, tables, hypothesis check (reads CSVs, no training) |
| `notebooks/08_failure_analysis.ipynb` | E7: frame-position, alert-timing, shortcut and error analysis |
| `notebooks/09_id_block_confound.ipynb` | E8: feature-based tests of the id-block confound |
| `tools/compare_id_blocks.py` | Raw-video comparison of the two id blocks (needs the dataset; untested on real data) |
| `results/baseline_results.csv` | Naive + scaled flow (final baselines) |
| `results/cnn_lstm_results.csv`, `baseline_and_cnn_lstm_results.csv` | Notebook 05 results (superseded) |
| `results/temporal_ablation_runs.csv` | Every run (variant x TTE x seed), 10 ep |
| `results/temporal_ablation_summary.csv` | Mean/std over seeds, 10 ep |
| `results/temporal_ablation_bootstrap.csv` | Bootstrap CIs, 10 ep |
| `results/temporal_ablation_*_30ep.csv` | Same, 30-epoch check |
| `results/final_results_table.csv`, `final_ablation_table.csv`, `final_selection_optimism.csv` | Report tables (E6) |
| `results/figures/final_fig{1-4}_*.png` | Report figures (E6) |
| `results/val_predictions.csv` | Per-clip test probabilities (CNN+LSTM, last-frame, mean-pool, flow) at all TTEs |
| `results/failure_*.csv` | Frame-position AP, alert-timing AUC, metadata shortcuts, error-by-position, persistent errors (E7) |
| `results/figures/failure_fig{1-3}_*.png` | Failure-analysis figures (E7) |
| `results/id_block_test_*.csv`, `results/figures/id_block_fig1_scores_by_id.png` | Id-block confound tests (E8) |
| `results/figures/temporal_ablation_ap_vs_tte[_30ep].png` | AP and recall@FPR vs TTE |
| `results/figures/temporal_ablation_learning_curves[_30ep].png` | Inner-validation AP per epoch |

---

## 10. Corrections to earlier informal notes

- **Partner's parallel final analysis (`origin/main`, commit `6142e6f`, 2026-10-07, "add 06 final analysis").** Adds `notebooks/06_final_analysis_results.ipynb` and `results/final_analysis/` (plots and CSVs for AP, ROC-AUC, precision/recall/F1 vs TTE). It hard-codes the **unscaled** optical-flow numbers (0.582 / 0.569 / 0.568) and the **notebook-05** CNN+LSTM numbers (0.793 / 0.774 / 0.739) as the primary results, so its headline figures and "CNN minus flow = 0.211 / 0.205 / 0.171" differ from this log's clean-protocol numbers (CNN+LSTM 0.762 / 0.749 / 0.675 vs scaled flow 0.575 / 0.576 / 0.569). It also modifies `03_clip_sampling.ipynb` cosmetically (stripped stray instruction text, added a split-saving cell, a validation-invalid check, a protocol write-up and stored outputs; code and the split CSVs are unchanged). It explains the 1,199 train clips at TTE 1.5: **one positive training video was too short for the 2-second window at that TTE and was excluded**. No file overlaps with the analysis branch (a merge dry run is conflict-free), but the two sets of final results must be reconciled before the report: use one baseline definition and the clean-protocol CNN+LSTM numbers. Notebook numbering also clashes (his 06 vs `06_temporal_ablation`).

- Optical-flow figures quoted as 0.5821 / 0.5692 / 0.5680 are the **unscaled** variant; the repo's final baseline is **scaled** (0.5745 / 0.5758 / 0.5683).
- "Inference time" and "environment breakdown" were described as proposal items; the proposal lists neither as a requirement (metadata is optional failure analysis only).
- The CNN+LSTM headline numbers (0.793 / 0.774 / 0.739) should be replaced by notebook 06's (0.762 / 0.749 / 0.675) because of the model-selection issue in E3.
- "Learned model degrades more slowly than flow" (the proposal hypothesis) is not supported by the data (§4, item 4).

# `results/final_analysis/` — partner's analysis (kept for reference)

**Source:** commit `6142e6f` "add 06 final analysis" by Rameez Atif, produced by `notebooks/06_final_analysis_results.ipynb`.
The files in this folder are **unmodified**; nothing here was edited when the branches were merged.

## Status: kept, but do not cite these as the final performance numbers

The notebook presents the results of notebooks 04 and 05 exactly as they were originally reported. They are faithful copies
(checked against `results/cnn_lstm_results.csv` and the original unscaled baseline CSV in git history), but they differ from the
results used in the report-ready analysis for two reasons:

| | This folder | Report-ready analysis (`notebooks/07_final_analysis.ipynb`, `results/final_results_table.csv`) |
|---|---|---|
| CNN + LSTM AP (0.5 / 1.0 / 1.5 s) | 0.793 / 0.774 / 0.739 | 0.762 / 0.749 / 0.675 |
| Why | Best epoch chosen **on the evaluation videos**, then reported on the same videos; single seed | Epoch chosen on held-out *training* videos; 5 seeds; bootstrap CIs |
| Optical-flow AP (0.5 / 1.0 / 1.5 s) | 0.582 / 0.569 / 0.568 (**unscaled** logistic regression) | 0.575 / 0.576 / 0.569 (**scaled**, as declared final in notebook 04) |
| Why | The unscaled model did not converge in 1,000 iterations and shifts by about 0.006 AP across environments | The scaled model is stable to about 0.0007 AP |

An independent check (`tools/fulldata_selection_check.py`, results in `results/selection_check_fulldata.csv`, logged as E10 in
`EXPERIMENT_LOG.md`) trained the same model on the full training set for a fixed 10 epochs with no epoch selection: AP 0.754 / 0.753 / 0.676.
That is close to the report-ready numbers and about 0.04 / 0.03 / 0.07 below the numbers in this folder, so the numbers here are optimistic by roughly that much.
The optimism is largest at 1.5 s, so it also flattens the apparent degradation with horizon (0.5 s to 1.5 s drop: 0.054 here vs about 0.08).

## What is still useful here

* Plots of precision, recall, F1 and ROC-AUC against TTE (`*_vs_tte.png`) and the CNN + LSTM training curves. These can be regenerated from the report-ready CSVs.
* The explanation of the 1,199 training clips at TTE 1.5 s: one positive video was too short for the 2-second window.
* The documented optical-flow variant comparison (`optical_flow_scaled_results.csv`).

Missing relative to the project proposal: recall at a fixed false-alarm rate, uncertainty estimates, and the temporal ablation (all in notebooks 06 and 07 of the analysis branch).

See `EXPERIMENT_LOG.md` (sections E6, E10 and "Corrections to earlier informal notes") for the full comparison.

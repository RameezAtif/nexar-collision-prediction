# Notebook index

Run from the project root (the notebooks use `./data/...` and `./results/...` paths). `EXPERIMENT_LOG.md` is the running record of every experiment, decision and open item.

| Notebook | Purpose | Status |
|---|---|---|
| `01_dataset_exploration.ipynb` | Metadata, class balance, event/alert times | Done |
| `02_video_inspection.ipynb` | MP4 inspection, FPS/frame indexing | Done |
| `03_clip_sampling.ipynb` | TTE clips, video-level train/validation split | Done (cosmetic cleanup merged from the partner's commit `6142e6f`; code unchanged) |
| `04_baselines.ipynb` | Naive and optical-flow baselines | Done; scaled flow declared final |
| `05_learned_temporal_model.ipynb` | ResNet-18 features + CNN + LSTM | Done; **epoch was chosen on the evaluation videos, so its numbers are optimistic** (see E3, E10) |
| `06_temporal_ablation.ipynb` | Temporal ablation, 5 seeds, clean epoch selection, bootstrap CIs | Done (E4, E5) |
| `06_final_analysis_results.ipynb` | **Initial analysis**: presentation of the notebook 04/05 results | **Kept for reference; uses the original (optimistic / unscaled) numbers. Do not cite as final.** See `results/final_analysis/README.md` |
| `07_final_analysis.ipynb` | Report-ready figures and tables from notebook 06 | Done (E6) |
| `08_failure_analysis.ipynb` | Frame-position, alert-timing, shortcut and error analysis | Done (E7) |
| `09_id_block_confound.ipynb` | Feature-based tests of the class-specific video-id blocks | Done (E8) |

**Numbering note:** two notebooks are numbered 06 because two analyses were developed in parallel. They have different file names and write to different
result folders (`results/final_analysis/` for the initial analysis; `results/temporal_ablation_*` and `results/final_*` for the analysis branch), so they do not overwrite each other.
Renumbering is deliberately left for the authors to agree on, to avoid conflicting edits.

Non-notebook tools (in `../tools/`): `compare_id_blocks.py` (raw-video comparison of the id blocks; needs the dataset) and `fulldata_selection_check.py` (selection-optimism check, E10).

# Pitch v1 (Hann Qian): pitch correction model and pyin pitch tracks

**Status: pitch v1 was left out of the final pipeline (no gain).** The pyin pitch tracks made here
(`f0_tracks.npz`) **are still used by the final pipeline**, so do not move, rename or delete that file.

The notebook for this model is [hann_qian_pitch_starter.ipynb](../hann_qian_pitch_starter.ipynb) (section "Pitch model").
This folder holds the scripts and saved outputs behind it.

## What the model does

For each Basic Pitch note, an **XGBoost classifier** predicts one of three actions: move the pitch **down 1 semitone,
keep it, or move it up 1 semitone**. A note is changed only when the model's confidence passes a cutoff, and that cutoff
is chosen by inner cross-validation on the training folds.

- Training data: proposed v2 labels (`basic_pitch_runs/correction_handover/proposed_v2/train_targets_v2.csv`),
  3,697 matched notes, of which only **292 (7.9%) have the wrong pitch**.
- Evaluation: song-grouped 5-fold cross-validation on the 60 training songs (`folds.csv`). Test songs were never used.
- Inputs: the shared note features, the neighbouring-note intervals, and pyin features from the audio. The pyin
  features measure how far the sung pitch is from the predicted MIDI pitch, in cents.

## Results

| Setup (train, out-of-fold) | Notes fixed | Notes broken | Pitch accuracy | Song-macro F1 |
|---|---:|---:|---:|---:|
| No change | 0 | 0 | 0.9210 | 0.2628 |
| Note features only | 0 | 0 | 0.9210 | 0.2628 |
| + neighbouring notes | 8 | 5 | 0.9218 | 0.2632 |
| + neighbouring notes + pyin | 53 | 19 | 0.9302 | 0.2655 |
| pyin only | 54 | 23 | 0.9294 | 0.2652 |

Source: [results.txt](results.txt). The model in the notebook scores 0.2628 → 0.2652 song-macro F1 (train out-of-fold).
When pitch v1 was tested inside the chain after the timing step, validation F1 went from 0.3796 to 0.3795, so it was
left out (see [final_pipeline/README.md](../../../final_pipeline/README.md)).

**Why it did not help:** Basic Pitch already gets 92.1% of matched note pitches right, so only 292 notes could be
fixed. The best setup fixed 53 and broke 19 (net +34). pyin pointed the right way on 87% of wrong-pitch notes, but
too few notes were wrong for this to move the song-level score.

## Files

| File | What it is | Used in report / pipeline? |
|---|---|---|
| `pyin_tracks.py` | Runs `librosa.pyin` (C2–C6, 22.05 kHz, hop 256) once per train/val recording and saves the tracks to `f0_tracks.npz`. Test recordings are excluded. | Yes, it builds the cache below |
| `f0_tracks.npz` | Cached pitch track for 694 train/val recordings: `<recording_id>_t` (frame times, s) and `<recording_id>_f0` (Hz, NaN when unvoiced). | **Yes, final pipeline** (see below) |
| `pyin_tracks_log.txt` | Log of the pyin run (694 recordings, about 28 min on 6 cores). | Record only |
| `pyin_features.py` | Runs pyin with the same settings and builds the per-note pyin features (voiced share, median/IQR of cents offset, share of frames above/below). | Yes, pitch v1 |
| `train_pyin_features.csv`, `val_pyin_features.csv` | Output of `pyin_features.py` for train and val notes. | Yes, pitch v1 |
| `eval_pitch_pyin.py` | Main comparison: XGBoost with and without neighbouring-note and pyin features. Prints the table saved in `results.txt`. | Yes, results in the table above |
| `eval_pitch_middle.py` | Side check: pyin over the whole note vs the middle of the note (trim 10/20/30% each end). Prints the table saved in `results_middle.txt`. Trimming did not help. | Side experiment |
| `eval_pitch_tuning.py` | Side check: adding singer-tuning features (each recording's typical cents offset). No saved output; it did not change the decision. | Side experiment |

### Where the final pipeline uses `f0_tracks.npz`

The pitch model was dropped, but its pitch tracks are reused as input clues by later models. These training
scripts load `f0_tracks.npz` directly:

- `final_pipeline/experiments/delete_v2/features.py` (step 3, delete; see `final_pipeline/chain_v4/stage2_README.md`)
- `final_pipeline/experiments/start_snap_v3/common.py` (step 4, start snap)
- `final_pipeline/experiments/end_snap/build_candidates.py` (step 5, end snap)

For new audio (the live demo and the test run), `final_pipeline/inference/features.py` recomputes the same pyin track
with the same settings (C2–C6, hop 256), so those models get the same clues they were trained on.

## How to run

You don't need to rerun anything: the saved outputs above are the results. To reproduce them, run from the
repository root with `librosa`, `xgboost`, `joblib`, `pandas` and `numpy` installed, and with the audio under `data/`:

```bash
# 1. cached pitch tracks (slow, about 30 min; only needed if f0_tracks.npz is missing)
python notebooks/correction_handover/hann_qian_pitch_model/pyin_tracks.py .
# 2. per-note pyin features -> <out_dir>/train_pyin_features.csv, val_pyin_features.csv
#    (runs pyin itself, so it is also slow and does not need step 1)
python notebooks/correction_handover/hann_qian_pitch_model/pyin_features.py . notebooks/correction_handover/hann_qian_pitch_model
# 3. model comparison -> prints the results table (as in results.txt)
python notebooks/correction_handover/hann_qian_pitch_model/eval_pitch_pyin.py . notebooks/correction_handover/hann_qian_pitch_model
# optional side checks
python notebooks/correction_handover/hann_qian_pitch_model/eval_pitch_middle.py .
python notebooks/correction_handover/hann_qian_pitch_model/eval_pitch_tuning.py . notebooks/correction_handover/hann_qian_pitch_model
```

All scripts read train and validation data only. The test split stays locked.

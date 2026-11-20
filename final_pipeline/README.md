# final_pipeline: the v2 correction pipeline (the actual run)

This folder is the **final correction pipeline** that produced the numbers in the report and slides, and the one that
the live demo (`backend/`) runs. In the report's words: the **v1 models** are the four first models for the planned flow
(they live in `notebooks/correction_handover/`); the **v2 pipeline** is what we kept, replaced and added after
comparing them, and it is what is in this folder.

Basic Pitch (tuned) writes down the notes, then **5 correction steps (7 trained models)** fix them, then a sheet is made.

```
 audio (.wav)
   -> Basic Pitch (onset 0.99, frame 0.3, min note 127.7 ms)   raw notes
   -> 1. Timing       2 random forests        move each start and end a little
   -> 2. Split        RF + gradient boosting  cut a note that is really two
   -> 3. Delete       gradient boosting       remove notes nobody sang
   -> 4. Start snap   LightGBM ranker (x2)    move each start onto a real sound moment
   -> 5. End snap     random forest           same for each end        -> corrected notes (F1 is measured here)
   -> 6. Sheet        tempo -> MIDI -> round to 32nd notes -> music21 -> MusicXML
```

## The 5 steps (new models) and the 4 v1 models they came from

| # | Step | Final model (v2) | Folder | v1 model for the same job | Decision |
|---|---|---|---|---|---|
| 1 | Timing | 2 random forests (start, end), 31 clues | `chain_v4/`, `inference/` (notebook: `notebooks/correction_handover/timing_v1_and_final.ipynb`) | timing v1: random forest, 7 note features | Kept, improved with audio and pitch-track clues |
| 2 | Split | random forest + gradient boosting, averaged | `experiments/split_v2/` | split v1 (`split_v1.ipynb`) | Added |
| 3 | Delete | gradient boosting, 49 clues | `experiments/delete_v2/` | delete v1: Gaussian naive Bayes | Replaced |
| 4 | Start snap | LightGBM ranker, 2 rounds | `experiments/start_snap_v3/` | start snap v1 / v2 (`start_snap_v1_v2.ipynb`) | Added (v3 is the final one) |
| 5 | End snap | random forest | `experiments/end_snap/` | none | Added |
| - | Merge | none | - | merge v1: logistic regression | Removed (no gain) |
| - | Pitch | none | - | pitch v1: XGBoost | Removed (no gain) |

The 7 trained models: 4 random forests (timing start, timing end, split, end snap), 2 gradient boosting (split, delete)
and 1 ranking model (start snap). `pipeline.md` explains each step in plain words; `chain_v4/stageN_README.md` has the
exact recipe and numbers per step.

## Results (song-macro F1, onset + pitch / offset)

| After step | Train (out-of-fold) | Val (20 songs) | Test (20 songs, run once) |
|---|---|---|---|
| Basic Pitch alone | 0.2628 / 0.1539 | 0.2726 / 0.1602 | 0.2397 / 0.1340 |
| 1. Timing | 0.3541 / 0.2176 | 0.3755 / 0.2360 | 0.3407 / 0.2043 |
| 2. Split | 0.3622 / 0.2342 | 0.3844 / 0.2584 | 0.3620 / 0.2373 |
| 3. Delete | 0.3810 / 0.2475 | 0.4050 / 0.2730 | 0.3843 / 0.2524 |
| 4. Start snap | 0.4213 / 0.2772 | 0.4563 / 0.3159 | 0.4148 / 0.2773 |
| **5. End snap (final notes)** | **0.4213 / 0.2844** | **0.4563 / 0.3213** | **0.4148 / 0.2812** |
| 6. Sheet read back (32nd grid) | - | 0.4280 / 0.2937 | 0.3954 / 0.2480 |

All 20 test songs improve. Every model and threshold was chosen on training songs only; the test songs were run once
(`inference/run_test.py` refuses a second run on purpose).

## What is where

| Folder | What it is |
|---|---|
| `inference/` | **The deployable pipeline.** `pipeline.py`, the 7 saved models in `models/`, `features.py`, test-run scripts. Has its own README (how to call it from code, the sheet step, how the results were checked). |
| `chain_v4/` | One complete note table after every step, for training songs (out-of-fold) and the 20 val songs, an action history per step, a README per step and `val_score.py`. |
| `test_run/` | The one test run: note tables per step, scores, and the sheets (`sheets/`, with `which_song.csv` to tell the files apart). |
| `experiments/` | How each step was built and compared: `split_v2/`, `delete_v2/`, `start_snap_v3/`, `end_snap/` (each has a README or RESULTS.md). `relabel_not_used/` is a side experiment that gave no gain. |
| `tools/` | Helpers: `shim.py` (stands in for librosa and mir_eval if they will not install), `snap_bp_clues.py`, `bp_onset_features.py`. |
| `archive/` | Not in the report, kept for the record: `early/` (first exploratory scripts). |
| `bp_activations/` | Basic Pitch's saved outputs per recording (about 850 MB). Not in git; rebuild with `tools/snap_bp_clues.py` if missing. |

`sheets_val/` (160 val sheets) is inside `chain_v4/`.

## How to run it

Python 3.13 (see `inference/requirements.txt`; keep scikit-learn 1.9.1 and lightgbm 4.7.0, the saved models need them).

```bash
python -m venv venv
venv\Scripts\activate
pip install -r final_pipeline/inference/requirements.txt
```

Turn one recording into corrected notes, MIDI and sheet:

```bash
python final_pipeline/inference/pipeline.py song.wav --out results
```

From code (the back end does this): `Corrector().transcribe('song.wav', 'outdir')`, see `inference/README.md`.

Re-check the reported numbers (no test data involved):

```bash
python final_pipeline/chain_v4/val_score.py          # val score after every step, from the saved tables
python final_pipeline/inference/check_val.py         # rebuilds val from audio with the saved models
python final_pipeline/inference/score_test.py --check-val   # scorer gives the same val numbers as cu.evaluate_end_to_end
```

The test result is saved in `test_run/test_scores.csv`. Re-running the pipeline on another machine gives the same notes
and scores within about +-0.002 F1, because librosa's audio features differ in the last digits between machines.

## Rules followed
- `data/test` was opened once, for the single final run; never tuned on.
- The answer key is used only to make training labels and to score, never as a model input.
- TextGrid is not used (it lines up with the answer key almost exactly).
- Every output is a complete note table `note_id, recording_id, start, end, pitch`; source tables are never overwritten.

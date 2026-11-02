# Chain v4, stage 1: split v2 (training songs, out-of-fold by song)

**Score (cu.evaluate_end_to_end, all 534 training recordings): onset F1 0.3622 (0.362204), offset F1 0.2342 (0.234248).**
Input timing-only score: 0.3541 / 0.2176. Old split v1: 0.3585 / 0.2243.

## Files
- `stage1_split_v2_train_oof.csv`: complete note table (`note_id, recording_id, start, end, pitch`), 8,493 notes
  (7,623 in), all 534 training recordings.
- `stage1_split_v2_actions.csv`: one row per cut (`note_id` = original id, `recording_id`, `cut_time` s, `prob`
  = out-of-fold probability, `fold`, `threshold`). 870 cuts in 677 notes.
- `stage1_export.py`: builds both files from the split_v2 caches and prints the score.

**Note id scheme:** untouched notes keep their id. Every piece of a cut note gets a new id: `<orig>_a`, `<orig>_b`,
`<orig>_c`, ... in time order (the first piece is `_a`, so the original id disappears). Piece times: original start ->
cut 1 -> ... -> original end. Piece pitch = rounded median pyin pitch of ~150 ms on each side of the cut (first piece:
150 ms before the first cut; others: 150 ms after their cut), kept inside the note; falls back to the original pitch
if no pitch is found (same as the split notebook's `apply_splits`).

## Recipe (what was trained, and how to make the final model for val later)
Code: `le_xin_work/experiments/split_v2/` (`common.py`, `clues.py`, `s3_candidates.py`, `s4_clues.py`, `s5_models.py`).
1. **Audio tracks** (`s1_tracks.py`): librosa 22,050 Hz, hop 256; pyin (65-1100 Hz) MIDI + voiced + voicing prob,
   onset strength (/99th percentile), RMS dB, 13 MFCC. Basic Pitch activations (onset/note) per recording,
   time axis = basic_pitch `model_frames_to_time`.
2. **Candidates** (`s3_candidates.py`, `run`): the split notebook's finder unchanged (pitch change >= 0.5 st,
   onset-strength peak >= 0.25, voice-on; >= 60 ms from note edges; strongest per 45 ms cluster) **plus** extra
   frames > 3 frames from any old pick: pitch change >= 0.3 st, Basic Pitch onset peak (any pitch) >= 0.3,
   loudness valley >= 3 dB, MFCC timbre change >= 0.15 (local peaks, +-3 frames). Train: 36,096 candidates.
   Labels (train only): a true start inside the note, > 50 ms from its edges and > 80 ms from any other BP start,
   marks its nearest candidate within 50 ms.
3. **Clues** (`s4_clues.py` + `clues.add_comparison` + `common.add_clues`): 74 = `common.OLD_FEATS` (19) +
   `clues.NEW_BASE` (26) + `clues.COMP_FEATS` (19). Audio + Basic Pitch + predicted notes only.
4. **Model** (`s5_models.Avg`): mean probability of
   `RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.5, class_weight='balanced_subsample', random_state=0)`
   and `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, class_weight='balanced', random_state=0)`.
5. **Threshold**: nested inside the training folds on onset F1, grid 0.3-0.95; 0.6 chosen in all 5 folds.
   OOF probabilities from the outer folds; cut where prob >= that fold's threshold.
6. **Final model for val (NOT run):** fit `Avg` on all 36,096 training candidates (74 clues, labels as above);
   threshold = best training out-of-fold onset F1 over the same grid using the outer OOF probabilities
   (`split_v2/cache/prob_v2_avg_rich.npy`); expected 0.6. Val input = `exports/le_xin_timing_val_rf_audio_pitch.csv`;
   val tracks + candidates + clues built the same way (no labels; Basic Pitch val activations exist in the other
   `le_xin_work/bp_activations/val`); then `apply_splits` with the same id scheme.

Caveat: the later stages (delete, snap) were trained on the old split's output, so their inputs change.

## Validation (run once, 2026-10-26; script `val_stage1_apply.py`, log `logs/val_stage1.log`)
- Final model: `Avg` (RF + boosting) fit on ALL 36,096 training candidates (same training input as the OOF above).
  Threshold from the training OOF onset F1 only: 0.3 0.2783 | 0.4 0.3295 | 0.5 0.3547 | **0.6 0.3622** | 0.7 0.3595 |
  0.8 0.3557 | 0.9 0.3542 | 0.95 0.3541 -> **0.6**. Nothing was chosen on val.
- Val features: `le_xin_work/experiments/split_v2/s8_val_features.py` (no answer key). Val tracks, v2 candidates and the 74 clues were
  built the same way as for train, using the Basic Pitch val activations in `le_xin_work/bp_activations/val`.
  The old-finder part of the val candidates equals the notebook's `split_candidates_val_v1.csv` exactly (6,158 rows).
  11,508 val candidates in total.
- Val references: `notebooks/bp_experiment.references` (as in Le Xin's Step 6), used only for scoring.
- **Cuts: 258 in 211 notes; notes 2,329 -> 2,587.** Files: `val_stage1_split_v2.csv` (complete table, all 160 val
  recordings, same id scheme `<orig>_a`, `_b`, ...) and `val_stage1_split_v2_actions.csv` (note_id, recording_id,
  cut_time, prob, threshold).

| Val (20 songs, 160 recordings) | onset F1 | offset F1 |
|---|---|---|
| Input: timing final model | 0.3755 | 0.2360 |
| Old split v1 (for reference, notebook Step 3) | 0.3796 | 0.2427 |
| **Timing + split v2** | **0.3844** | **0.2584** |

Songs (onset F1 vs input): 12 better, 8 worse, 0 same. The gain over timing (+0.0089 onset / +0.0224 offset) is in line
with training OOF (+0.0081 / +0.0166), so split v2 generalises to new songs.

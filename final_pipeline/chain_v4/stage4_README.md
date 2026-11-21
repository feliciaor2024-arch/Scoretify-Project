# Chain v4, stage 4: end snap (training songs, out-of-fold by song)

Input: `stage3_snap_v3_train_oof.csv` (timing -> split v2 -> delete v2 -> snap v3), 7,096 notes, 534 recordings.
Only note ENDS change; starts, pitches and note IDs are kept, and every end stays > start (min length 10 ms).
Validation, test and TextGrid were not used. The answer key (`references_train.csv`) was used only for labels and scoring.

## Results (cu.evaluate_end_to_end, song-macro, all 534 training recordings)
| Version | ends moved | labelled ends right | onset F1 | offset F1 |
|---|---|---|---|---|
| Stage 3 input (snap v3) | 0 | 65.3% | 0.4213 | 0.2772 |
| Ceiling: always pick a right option (answer key) | 321 | 75.4% | 0.4213 | 0.3202 |
| **Stage 4: RF end-snap picker, nested margin** | **3,282** | **66.7%** | **0.4213** | **0.2844** |

- Offset F1 goes up by 0.0072. Onset F1 cannot change.
- Song by song (offset F1): 27 of 60 songs are better, 16 are worse and 17 are the same.
- Margins chosen per held-out fold (on the other 4 folds, by offset F1): fold 0 -0.05, 1 -0.05, 2 0.0, 3 -0.05, 4 -0.05.
- "Labelled ends right": of the 3,186 notes whose start matches a true note (same pitch, start within 50 ms), the
  share whose end is within max(50 ms, 20% of the true note's length) of the true end. For 75.4% of these notes,
  "stay" or one of the options is right.
- The picker gets about 17% of the gain that is possible (0.0072 of 0.0430).

## Files
- `stage4_end_snap_train_oof.csv`: the COMPLETE note table (note_id, recording_id, start, end, pitch).
- `stage4_end_snap_actions.csv`: one row per moved end: note_id, recording_id, old_end, new_end, prob (P(right) of
  the chosen option), stay_prob, fold, margin, shift_ms, model. A move happens when prob - stay_prob >= margin.
- Code: `final_pipeline/experiments/end_snap/`. The run order is `END_SNAP_TAG=v4 python build_candidates.py` (~2 min, candidates
  `end_candidates_train_v4.csv`), then `END_SNAP_TAG=v4 python end_snap_model.py rf`, then `python stage4_export.py`.

## Method
- **Options** = "stay" plus moments within +-150 ms of the current end:
  - the voice switches off
  - the voiced share drops
  - loudness falls >= 1 dB in one frame
  - the sung pitch leaves the note pitch
  - the pitch jumps >= 0.5 semitone
  - the next note's start
  - the Basic Pitch note activation at the note pitch (+-1 bin) falls below 0.3
- **Clues**: 45 per option, from the audio (librosa at 22.05 kHz, hop 256), Hann Qian's pyin tracks, the cached
  Basic Pitch activations, the neighbouring predicted notes, and a comparison with "stay".
- **Model**: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.5,
  class_weight='balanced_subsample', random_state=0), trained on the options of labelled notes only.

## Final-model recipe for val (not run)
1. Build the val candidates with the same `rows_for_recording` (no answer key needed) on the stage-3 val table.
2. Fit the RF above on ALL labelled training options (`end_candidates_train_v4.csv`, the 45 clue columns).
3. Score each val option. For each note, move the end to its best option if P(best) - P(stay) >= **0.0**, the margin
   with the best all-train out-of-fold offset F1 (saved in `end_snap/end_snap_rf_margins_v4.json`).
4. Set end = max(new end, start + 0.01), then score once with cu.evaluate_end_to_end.

## Validation (20 val songs, 160 recordings; run ONCE, approved by Le Xin, 2026-10-26)
- Input: `val_stage3_snap_v3.csv` (2,258 notes). The val candidates and clues were built with the same code as on
  training (audio, pyin, cached Basic Pitch val activations), without the answer key.
- Final model: the RF above, fitted on ALL labelled training options. Margin **0.0**, chosen on the all-train
  out-of-fold offset F1. Nothing was chosen on val.
- Ends moved: 804 of 2,258 notes (median +34.8 ms; 73% later).

| Val (scored with `val_score.py`) | notes | onset F1 | offset F1 |
|---|---|---|---|
| + snap v3 (stage 3) | 2,258 | 0.4563 | 0.3159 |
| **+ end snap (stage 4)** | 2,258 | **0.4563** | **0.3213** |

- Offset F1 +0.0054 on val (+0.0072 on train OOF).
- Song by song (offset F1): 12 songs better, 5 worse, 3 same. `val_score.py` counts by onset F1, which cannot change
  here, so it prints 0/0.
- Files: `val_stage4_end_snap.csv` (complete note table, only ends change, end > start) and
  `val_stage4_end_snap_actions.csv` (note_id, recording_id, old_end, new_end, prob, stay_prob, margin, shift_ms, model).
  The code is `final_pipeline/experiments/end_snap/val_stage4.py`, and the val candidates are cached in `end_snap/end_candidates_val_v4.csv`.
- Caveat: val was already used to choose the Basic Pitch settings and earlier stages, so it is continued
  development, not untouched evidence.

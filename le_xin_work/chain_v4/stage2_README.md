# Chain v4, stage 2: delete v2 (training songs, out-of-fold by song)

**Score (cu.evaluate_end_to_end, all 534 training recordings): onset F1 0.3810 (0.381007), offset F1 0.2475 (0.247541).**
1,397 of 8,493 notes deleted (7,096 kept). Input (stage 1, timing + split v2): 0.3622 / 0.2342.

| Version (same 8,493-note pool) | notes | deleted | onset F1 | offset F1 | precision | recall |
|---|---|---|---|---|---|---|
| Stage 1 input (timing + split v2) | 8,493 | 0 | 0.3622 | 0.2342 | 0.356 | 0.376 |
| Felicia's model (her out-of-fold keep/delete per original BP note, applied to all its pieces) | 8,062 | 431 | 0.3702 | 0.2395 | 0.371 | 0.376 |
| **Delete v2 GB (this stage)** | **7,096** | **1,397** | **0.3810** | **0.2475** | 0.406 | 0.367 |

So delete v2 adds +0.019 onset F1 to stage 1, vs +0.008 for Felicia's decisions on the same pool.
Felicia's baseline was cheap. Her decisions come from `exports/felicia_false_actions_train_oof_v2_exploratory.csv`,
are out-of-fold on the same folds, and were made on the raw Basic Pitch notes. They were not retrained on this pool.
Of the 1,397 deleted notes, 968 are "truly false": they overlap no reference note within 50 cents. The rest
match nothing under the official rule, usually because they are real notes whose onset is still more than 50 ms off.

## Files
- `stage2_delete_v2_train_oof.csv`: complete note table (`note_id, recording_id, start, end, pitch`) = stage 1
  minus the deleted notes. Ids and times are unchanged from stage 1. Written with `cu.export_corrections`.
- `stage2_delete_v2_actions.csv`: one row per stage-1 note. `prob` = out-of-fold probability that the note is
  false, `fold`, `threshold` = that fold's nested threshold, `deleted` = prob >= threshold.
- Code: `le_xin_work/delete_v2/stage2.py`, which calls `features.py <pool> <out>` to build
  `delete_v2/features_chain_v4_stage1.csv` and then `models.run`. Results are also in `delete_v2/stage2_results.csv`.

## Recipe
1. **Pool**: `stage1_split_v2_train_oof.csv`. There is no snap before this stage, Felicia's probability is not
   used, and there are no snap clues.
2. **Label** (answer key, labels only): `y_official_pool` = the note is matched to no reference note by
   mir_eval's one-to-one matching (onset 50 ms, pitch 50 cents, no offset), computed per recording on the whole pool.
3. **Clues (49)**, from predicted notes and audio only:
   - Note shape and neighbours (20): duration, log duration, pitch, BP amplitude (of the original note), piece
     flag/count, gaps, pitch jumps, overlaps with other notes, place in the recording, values relative to the recording.
   - pyin track `notebooks/correction_handover/hann_qian_pitch_model/f0_tracks.npz` (14): voiced share, share
     on/near the note pitch, octave share, median and spread, voiced/on-pitch share in the 100 ms before/after.
   - Basic Pitch note/onset activations at the note pitch (11): the same ONNX outputs as the frozen predictions,
     time axis = `model_frames_to_time`.
   - Loudness (6) from `le_xin_cache/audio_features_train_v1.csv`, per original BP note.
4. **Model**: `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
   l2_regularization=1.0, random_state=0)`. Missing values are set to -1.
5. **Threshold (nested)**: for each outer fold k, inner out-of-fold probabilities on the other 4 folds are used to
   pick the threshold with the best onset F1 on those training songs (ties go to fewer deletions). Grid: 0.30-0.55
   step 0.05, then 0.60-0.975 step 0.025, plus "delete nothing". The model is refit on the 4 folds and scores fold k.
   A note is deleted if prob >= threshold. **Fold thresholds 0.90 / 0.85 / 0.90 / 0.875 / 0.875** (folds 0-4).
6. **Final model for val (NOT run)**: fit the same GB on all 8,493 training notes of this pool (49 clues, label as
   above). Use **threshold 0.875**, which gives the best all-train onset F1 over the same grid using the outer
   out-of-fold probabilities. Val input = the val version of stage 1. Val clues are built the same way: pyin val
   tracks are in `f0_tracks.npz`, and Basic Pitch val activations and `audio_features_val_v1.csv` exist. Val needs no labels.

## Validation (run once, approved by Le Xin; nothing chosen on val)
- Final model: the GB above, fit on all 8,493 training notes of `stage1_split_v2_train_oof.csv` (49 clues,
  `y_official_pool`). **Threshold 0.875** = best all-train onset F1 over the stage-2 grid, using the training
  out-of-fold probabilities in `stage2_delete_v2_actions.csv`. At that threshold the train OOF onset F1 is 0.3822,
  slightly optimistic because the threshold was picked on the same scores; the nested value is 0.3810.
- Val input: `val_stage1_split_v2.csv` (2,587 notes, 160 recordings). Val clues were built by `delete_v2/features.py
  <pool> <out> val` with no answer key: pyin tracks from `f0_tracks.npz` (all 160 val recordings present),
  Basic Pitch val activations (`bpact_val`, same ONNX model), and loudness from `le_xin_cache/audio_features_val_v1.csv`.
- **329 of 2,587 notes deleted (12.7%; train OOF 16.4%), 2,258 kept.**
- Scored with `val_score.py` (song-macro, all 160 val recordings):

| Val stage | notes | onset F1 | offset F1 | songs better/worse vs previous |
|---|---|---|---|---|
| timing (val input) | 2,329 | 0.3755 | 0.2360 | - |
| + split v2 | 2,587 | 0.3844 | 0.2584 | 12/8 |
| **+ delete v2** | **2,258** | **0.4050** | **0.2730** | **19/1** |

  Gain from this stage: +0.021 onset / +0.015 offset on val, vs +0.019 / +0.013 on train OOF, so it carries over
  to new songs. Caveat (as before): val was used earlier to choose Basic Pitch settings.
- Files: `val_stage2_delete_v2.csv` (complete note table minus deleted notes, written with `cu.export_corrections`),
  `val_stage2_delete_v2_actions.csv` (`note_id, recording_id, prob, threshold, deleted`). Code:
  `delete_v2/val_stage2.py`; val clues cached in `delete_v2/features_chain_v4_val_stage1.csv`; log `delete_v2/val_stage2_log.txt`.

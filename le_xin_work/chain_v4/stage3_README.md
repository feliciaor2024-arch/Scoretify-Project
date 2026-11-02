# Chain v4, stage 3: snap v3 (training songs, out-of-fold by song)

Input: `stage2_delete_v2_train_oof.csv` (timing -> split v2 -> delete v2; 7,096 notes, 534 recordings).
The options, clues (including Basic Pitch clues from the cached activations, and neighbour clues) and labels were
rebuilt on this table. The old v2 candidate caches were not reused. Folds come from `folds.csv`. Each fold's margin
was chosen on the other 4 folds (inner out-of-fold). Scores come from `cu.evaluate_end_to_end` on all 534 training
recordings. Validation and test data were not opened. No TextGrid was used. The answer key was used only for labels and scoring.

| version | starts moved | labelled right* | onset F1 | offset F1 | margins (fold 0-4) |
|---|---|---|---|---|---|
| input (stage 2) | 0 | 67.7% | 0.3810 | 0.2475 | - |
| snap v2 (forest, +-100 ms, 45 clues, 1 round) | 1,922 | 73.6% | 0.4127 | 0.2702 | 0 / 0 / 0 / 0 / 0 |
| snap v3 round 1 | 2,161 | 74.6% | 0.4200 | 0.2761 | .05 / .10 / 0 / .10 / .15 |
| **snap v3 round 1 + round 2 (output)** | **2,184** (round 2 moved 141) | **74.8%** | **0.4213** | **0.2772** | round 2: .05 / .10 / .10 / .10 / .15 |

\* Share of the labelled notes of this input (true start within +-150 ms, same pitch) whose start ends up within 50 ms of the true start.

So snap v3 adds +0.040 onset / +0.030 offset over stage 2, and +0.009 / +0.007 over plain snap v2 on the same input.
Only starts change: ends, pitches and note IDs are kept, and every note keeps start <= end - 10 ms.

## Files
- `stage3_snap_v3_train_oof.csv`: COMPLETE note table (note_id, recording_id, start, end, pitch), written with `cu.export_corrections`.
- `stage3_snap_v3_actions.csv`: one row per move: note_id, recording_id, round (1/2), old_start, new_start, shift_ms,
  score (the chosen option's per-note softmax score), stay_score, fold, margin. There are 2,302 rows; some notes moved in both rounds.
- Code: `le_xin_work/experiments/snap_v3/stage3_v4.py` (run with `SNAP_CHAIN=<this stage2 csv> SNAP_CACHE=cache_v4`); log
  `snap_v3/logs/stage3_v4.log`; caches in `snap_v3/cache_v4/`.

## Final-model recipe for val
1. Options: `build_options(val_chain, win=0.15, extra_kinds=True)` from `snap_v3/common.py`. Moment kinds: attack,
   louder, voice on, pitch jump, plus Basic Pitch onset peak at the note's pitch (>= 0.3) and pitch-settle moment.
   Clues = the 45 v2 clues + `is_bp_onset`, `is_settle`. `common.py` is train-only today: point it at the val manifest,
   the val audio, and `bpact_val` (Basic Pitch activations). Do not compute labels on val.
2. Picker: `LGBMRanker(objective='lambdarank', n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=0)`. Group = one note's options.
   Train it on ALL labelled options of the training notes from this stage-2 table. Score = softmax of the ranker
   output within each note.
3. Round 1: move a note to its best move option if that option's score minus stay's score is >= **0.10**.
4. Round 2: rebuild the options around the round-1 starts, score them with the SAME trained ranker, and move if the
   difference is >= **0.05**.
   Both margins were chosen on all training songs' out-of-fold scores; val was not used.
5. Keep start <= end - 0.01 s. LightGBM is needed (`pip install lightgbm`; tested with 4.7.0).

## Validation (20 val songs, 160 recordings; run ONCE, nothing chosen on val)
Input: `val_stage2_delete_v2.csv` (2,258 notes). The final ranker (recipe above, seed 0) was trained on all 19,398
options of the 4,236 labelled training notes from the stage-3 training run (`snap_v3/cache_v4/options_w150x.pkl`).
Margins come from the all-train out-of-fold scores: round 1 **0.10**, round 2 **0.05**. Round 2 rebuilds the
options around the round-1 val output and scores them with the same ranker, as in the out-of-fold training run.
The val options were built from audio, pyin, the cached Basic Pitch val activations (`bpact_val`) and neighbours,
with no answer key. Scored with `val_score.py`.

| val stage | notes | onset F1 | offset F1 | songs better / worse vs previous |
|---|---|---|---|---|
| + delete v2 (input) | 2,258 | 0.4050 | 0.2730 | - |
| **+ snap v3** | 2,258 | **0.4563** | **0.3159** | 19 / 1 |

Starts moved: round 1 595, round 2 166 (707 notes moved in total; some notes moved in both rounds).
Gain on val is +0.051 onset / +0.043 offset, against +0.040 / +0.030 on train out-of-fold.
Note: round 2 moved more on val (7% of notes) than on train (2%), probably because the all-train round-2 margin (0.05) is lower
than most per-fold margins (0.05-0.15). It still helped overall. Caveat as before: val was used earlier in the project
(Basic Pitch settings, earlier stages), so this is continued development, not untouched evidence.
Files: `val_stage3_snap_v3.csv` (COMPLETE note table, only starts changed, start < end), `val_stage3_snap_v3_actions.csv`
(note_id, recording_id, round, old_start, new_start, shift_ms, score, stay_score, margin). Code: `snap_v3/val_stage3.py`, log `snap_v3/logs/val_stage3.log`.

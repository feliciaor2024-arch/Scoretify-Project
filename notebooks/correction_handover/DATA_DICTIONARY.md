# Data dictionary and leakage boundaries

All paths below are relative to `basic_pitch_runs/correction_handover/`.

## Features: allowed inputs and identifiers

`train_features.csv` and `val_features.csv` contain one row per predicted note. The validation file has **no correction labels**.

| Column | Meaning | Use as model feature? |
|---|---|---|
| note_id | Stable `<recording_id>:n<sorted-note-index>` for this frozen prediction version | No; join/provenance only |
| recording_id | Excerpt identity | No |
| sample_id | Song identity; grouping key for splitting | No |
| split | Original train/val membership | No |
| start, end | Predicted seconds relative to excerpt | Yes |
| pitch | Predicted MIDI pitch | Yes |
| amplitude | Basic Pitch note-event amplitude value, not calibrated correctness probability | Yes |
| duration | Predicted end minus start | Yes |
| prev_gap, next_gap | Gaps to neighbours in time-sorted predicted notes; overlap gives negative gap | Yes |
| audio_duration | Validated excerpt duration, useful for output bounds | Not in initial allowlist |

At missing-neighbour boundaries the gap is explicitly zero. This conflates a boundary with a true zero gap; adding separate boundary flags is a potential future feature improvement. All features are prediction/metadata-derived, not reference-derived.

`merge_candidates.csv`: pair_id, note_id_a, note_id_b, recording_id, sample_id, split plus gap, pitch_delta and duration_sum. Candidates are immediately neighbouring predicted events of the same pitch with gap between -0.02 and 0.12 seconds. Thus pitch_delta is zero in this version. This restrictive candidate generator can miss valid merges and yields zero candidates for the final onset-0.99 baseline; its suitability must be reviewed before supervised merge training. Pair labels are in a separate file.

## Targets: never use as X

`train_targets.csv` keeps a row for **every training prediction**, even unresolved notes. Empty is unknown/not-applicable, never implicitly zero.

| Column | Meaning |
|---|---|
| relationship | one_to_one, fragment, spurious_candidate, split_candidate, ambiguous |
| reference_index | Index in sorted excerpt reference list; evaluator/traceability only |
| false_note | 1 for isolated spurious proposal, 0 for accepted one-to-one proposal, blank otherwise |
| start_delta, end_delta | Reference minus predicted time, seconds; only one-to-one proposals |
| pitch_delta | Reference minus predicted MIDI pitch; only one-to-one proposals |
| reference_start/end/pitch | Absolute correction targets; only one-to-one proposals |
| wrong_start | Absolute start error > 0.05 s |
| wrong_end | Absolute end error > max(0.05 s, 20% reference duration) |
| wrong_pitch | Integer predicted/reference MIDI pitches differ |
| status | provisional_auto or unresolved; no manually approved labels yet |

`merge_targets.csv`: pair_id, merge (1/0/blank), status. A merge requires fragments of the same reference event; keep requires distinct one-to-one reference correspondences. Blank pairs need review. Pair merge labels do not automatically define transitive multi-pair actions.

The starter loading cells explicitly rename raw targets to helper names: false_note -> false_label, reference_start/end -> target_start/end, reference_pitch -> target_pitch, merge -> merge_label; note_id_a/b -> left_note_id/right_note_id. This is an interface rename, not new ground truth.

## Complete evaluator data

- `references_train.csv`: all training reference events, including ones with no usable correction label. Columns note_id (reference-local), recording_id, start, end, pitch.
- `training_recording_manifest.csv`: every training recording, even if no model predictions exist. Use this for full evaluation, not a manifest reconstructed from filtered notes.
- `recording_manifest.csv`: train/val recording metadata, including source-relative paths and duration. Reference source paths here are not features and are not automatically read by task starters.
- `no_correction_training_baseline.json` / `no_correction_song_metrics.csv`: scores on all training references and predictions, not only labelled examples.
- `training_diagnostics.csv`: counts of provisional relationships; counts are not true error prevalence.

## Review and provenance

- `review_queue.csv`: note examples with blank human decision, corrected relationship, reviewer and reason fields.
- `merge_review_queue.csv`: all proposed pairs plus blank review fields.
- `review_examples.json` and `*_review.png`: selected training excerpts shown with embedded playable audio in the audit notebook.
- `group_review_gate.json`: approval remains false.
- `input_version.json`: fixed prediction hashes and version.
- `label_version.json`: algorithm/data hashes for this provisional label snapshot, written by final verification.
- `prior_artifact_hashes.json`: original tuning_001 and top-level notebook/helper preservation baseline. This is an audit artifact, not model data.

## Corrected prediction format

`note_id, recording_id, start, end, pitch` is a COMPLETE replacement event table. Timestamps are seconds; pitch is MIDI. Deleted notes are omitted; merged notes retain one source ID. No reference columns. Keep source-to-output action history separately for audit. Use the original complete recording manifest for scoring so fully deleted/empty recordings still receive a score.

Timing and pitch can both be corrected for the same event. Do not assign overlapping merges or delete an absorbed note twice without an explicit conflict policy. Combined-system orchestration is not implemented because trained component models are not yet available.

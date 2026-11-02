# Relabel after correction: more training labels for the timing model?

Training songs only (534 recordings, 60 songs, 5 song-grouped folds). No val, no test, no TextGrid.
Scripts: `relabel.py` (main, ~52 min), `per_song.py` (paired song comparison), `why_unlabelled.py`.
Outputs: `results.csv`, `report.json`, `per_song_comparison.csv`, `why_unlabelled.csv`, `labels_*.csv`, `logs/run_log.txt`.

## Idea
The v2 labels were made by matching the RAW Basic Pitch notes to the answer key (mutual-best overlap, start within
200 ms, etc.). BP starts are late, so a badly timed note can fail the match. We move the notes with the timing model
first, then match again with the same v2 rule (incl. safety filter), then retrain the same RF (31 features, nested gate).

Leakage control: features are unchanged (computed from raw notes); corrected positions only decide *which* reference
a note belongs to; label values are still reference start/end. "Strict nested": for held-out fold k, the
first-pass correction of the other 4 folds comes from inner out-of-fold models that never saw fold k, fold k gets no
labels, and gates are chosen on the other 4 folds only.
Sanity: the matcher on raw positions reproduces v2 (3,696 of 3,697, all same reference); the old-label rerun
reproduces the notebook exactly (0.3524/0.2088 ungated, 0.3541/0.2176 nested gate).

## Label counts (matching on corrected positions, all 60 songs)
| Label set | labels | new vs v2 | v2 labels lost | same reference where both |
|---|---|---|---|---|
| v2 (raw matching) | 3,697 | - | - | - |
| rematch on gated OOF export | 3,877 | +367 | 187 | 3,510 / 3,510 (100%) |
| union (v2 + new) on gated OOF export | 4,059 | +362 | 0 | 3,697 / 3,697 (100%) |
| rematch on ungated OOF shifts | 3,898 | +403 | 202 | 100% |
| union on ungated OOF shifts | 4,094 | +397 | 0 | 100% |

New labels are the hard, very late notes: 93-95% have a start >50 ms off (v2: 51%), median start shift -209 ms.

## Retrained timing RF, train out-of-fold F1 (song-macro, all 534 recordings, real mir_eval)
| Labels | train labels / fold | onset F1 | offset F1 | onset MAE ms (v2 notes) | gates start/end ms |
|---|---|---|---|---|---|
| BP unchanged | - | 0.2628 | 0.1539 | 68.8 | - |
| v2 (old), nested gate | 2,841-3,032 | **0.3541** | **0.2176** | 48.6 | 30-40 / 60-100 |
| union, strict nested, nested gate | 3,147-3,351 | 0.3538 | 0.2192 | 49.0 | 30-50 / 80-140 |
| rematch-only, strict nested, nested gate | 3,000-3,202 | 0.3559 | 0.2192 | 48.8 | 30-50 / 100-160 |
| union from export (simple, weak leak), nested gate | 3,124-3,338 | 0.3552 | 0.2188 | 48.3 | 30-40 / 80-120 |
| (ungated: v2 / union / rematch / export) | | 0.3524 / 0.3516 / 0.3531 / 0.3539 | 0.2088 / 0.1992 / 0.2000 / 0.2060 | | |

Per-song paired change vs v2 (60 songs, bootstrap 95% CI): every gain is within +-0.002 and every CI contains 0
(e.g. rematch-only onset +0.0019 [-0.0059, 0.0093], 25 songs better / 28 worse). **No real improvement.**

## Why so little
Remaining unlabelled notes after correction (export, union): 1,603 are fragments/duplicates (their reference note is
covered more by another BP note), 859 still start >200 ms off, 412 cover a 2nd reference >30% (split cases), 294 are
v2 spurious/split candidates, 175 length >3x, 108 touch no reference (false notes), 74 pitch off. So most unlabelled
notes are not "timing too bad to match" but structural (fragments, splits, false notes): re-timing cannot label them.
The few new labels are extreme late notes; they make the RF predict bigger shifts (ungated offset F1 drops
0.2088 -> 0.1992), and the gate then has to be stricter (end gate 80-160 ms) to undo that, ending at a tie.

## Recommendation
- Keep the v2 labels for the timing model; relabelling is not worth the extra complexity (gain is noise-level).
- Do not retrain split or snap on relabelled data for this reason: those models' labels come from the same matching and
  the bottleneck is structural, not timing. (Retraining split/snap on the *corrected chain's inputs* is a separate
  question and still open.)
- If more labels are wanted, the large pool is the 1,603 fragment/duplicate notes: a rule that labels the
  best-covering fragment and marks the rest as delete-candidates would help Felicia's delete model more than timing.

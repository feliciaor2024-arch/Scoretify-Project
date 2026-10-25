# Delete v2 prototype: removing false notes from the CORRECTED notes (training songs only)

Separate prototype for the group to compare with Felicia's delete step. Nothing of hers was changed.
Validation and test were not touched. Every number is out-of-fold by song (folds.csv). The delete threshold was
chosen nested: inside each outer fold, it was picked on onset F1 of the 4 training folds only.

## What goes in
- Note pool = the chain just before the delete step (timing + split, `le_xin_split_train_oof_v1`, 7,982 notes),
  with the snapped start (`le_xin_snap_train_oof_v2`) for the 7,550 notes Felicia kept. The 432 notes she deleted
  were never snapped, so they keep their split-step start.
- Labels (answer key, used for labels and scoring only):
  - `official`: the note matches no reference note under the official rule (onset 50 ms, pitch 50 cents).
  - `loose` ("truly false"): the note overlaps no reference note within 50 cents. Nothing was sung at that pitch then.
- Clues (49 base, all from predictions/audio): note shape and neighbours (duration, gaps, overlaps with other notes,
  pitch jumps, place in recording); pyin pitch track over the note (`f0_tracks.npz`: voiced share, share on/near the
  note's pitch, octave share, spread); Basic Pitch's own note/onset activations at the note's pitch; loudness from
  Le Xin's timing cache. "More" setting extras: Felicia's out-of-fold probability (same folds) and 15 snap clues.
  The snap clues exist only for notes Felicia kept.

## Results (534 training recordings, song-macro, out-of-fold)
| Version | notes | deleted | of which truly false | onset F1 | offset F1 |
|---|---|---|---|---|---|
| Chain without any delete | 7,982 | 0 | - | 0.3942 | 0.2509 |
| **Felicia only (current chain)** | 7,550 | 432 | 417 | **0.4034** | **0.2568** |
| Rule: delete if little of the note is sung near its pitch (replace) | 7,437 | 545 | 528 | 0.4064 | 0.2588 |
| RF, official label, replace | 6,952 | 1,030 | 818 | 0.4087 | 0.2615 |
| RF, loose label, replace | 7,152 | 830 | 735 | 0.4097 | 0.2614 |
| RF, official label, Felicia prob as clue, replace | 7,109 | 873 | 747 | 0.4102 | 0.2620 |
| RF, official label, delete more after Felicia (+snap clues) | 6,741 | 809 | 483 | 0.4099 | 0.2623 |
| GB, official label, delete more after Felicia (+snap clues) | 7,067 | 483 extra | 333 | 0.4112 | 0.2620 |
| **GB, official label, replace (best)** | **7,063** | **919** | **765** | **0.4116** | **0.2627** |
| Ceiling: delete exactly the truly false notes | 5,588 | 2,394 | 2,394 | 0.4581 | 0.2901 |
| Ceiling: delete every unmatched note | 3,119 | 4,863 | - | 0.5628 | 0.3541 |

All runs are in `results.csv`. Best GB vs Felicia only: 48 songs better, 10 worse, 2 same. Its thresholds were
0.90/0.85/0.90/0.90/0.90. A finer grid up to 0.975 (`results_grid.csv`) picked the same values. The model deletes
414 of Felicia's 432 notes plus 505 more.
AUC (official label): GB/RF 0.70, Felicia's probability 0.61, rule 0.60. Strongest clues: share of the note sung on
its pitch (pyin), gap to the previous note, onset strength at the start, Basic Pitch's note activation.

## Plain-language summary
- About 59% of the notes we output still match nothing. Only about a quarter are "truly false": nothing was sung at
  that pitch. The rest are real notes whose timing is still more than 50 ms off.
- Deleting a note only pays off when we are very sure it is wrong. At F1 ~0.41, a note needs roughly an 80%+ chance
  of being wrong before removing it raises F1. So the models only delete their most confident ~900 notes.
- The new model beats Felicia's by +0.008 onset F1 / +0.006 offset F1 on the training songs. It improves 48 of 60
  songs. That is a solid but small step. Most of the 0.055 headroom is out of reach because the clues cannot
  separate a false note from a badly timed real one with enough certainty.
- Training on the corrected notes helps mostly through the pyin "is the note actually sung at this pitch" clues
  and Basic Pitch's own confidence. The snap clues and Felicia's probability added nothing measurable.

## Recommendation
Offer **GB, official label, replace** as a candidate replacement for the delete slot (one model, no Felicia
dependency, +0.008 onset F1 on train OOF). Before adopting it: train a final model on all training songs,
use threshold 0.90, and check once on val against Felicia's step. Note one caveat: in "replace", the 18 Felicia
deletions it keeps were never snapped. Rerunning snap after this delete step would fix that.
If the group prefers to keep Felicia's model, "GB delete more" gives almost the same gain (0.4112).

## Files
`common.py` (paths, pool, fast scorer = official scorer exactly), `features.py` -> `features_train.csv`,
`models.py` -> `results.csv`, `oof_*.csv` (scores + delete flag per note), `check_grid.py` -> `results_grid.csv`,
`rf_importance.csv`. Runtime: features ~4 min, models ~70 min (n_jobs=4).

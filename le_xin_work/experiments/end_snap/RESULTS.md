# End snap (mirror of the start snap), training songs, out-of-fold by song

Input: `exports/le_xin_snap_train_oof_v2.csv` (timing -> split -> delete -> start snap v2), 7,550 notes, 534 recordings.
Validation and test were NOT opened. No TextGrid. Answer key used only for labels and scoring.

## How it works
For every note: options = "stay" + moments within +-150 ms of its current end (voice switches off, voiced share
drops, loudness drops >= 1 dB/frame, sung pitch leaves the note pitch, pitch jump >= 0.5 st, next note's start,
Basic Pitch note activation at the note pitch (+-1 bin) falls below 0.3). Median 2 options per note.
Each option gets 45 clues (audio, pyin, Basic Pitch activations, neighbour notes, comparison with "stay").
Label: true note = reference note of the same pitch whose start is within 50 ms of this note's start (what mir_eval
matches); an option is right if it is within max(50 ms, 20% of that true note's length) of its end (mir_eval offset rule).
Random forest gives P(right) per option; move the end to the best option only if it beats "stay" by >= margin;
margin chosen per held-out fold on the other 4 folds (nested), by offset F1.

## Results (song-macro, all 534 training recordings)
| Version | ends moved | labelled ends right | onset F1 | offset F1 | margins (fold 0-4) |
|---|---|---|---|---|---|
| Snap v2 input (no end snap) | 0 | 63.2% | 0.4034 | 0.2568 | - |
| Ceiling: always pick a right option (answer key) | 333 | 73.9% | 0.4034 | 0.2989 | - |
| Rule: snap to nearest next note start | 2,015 | 62.6% | 0.4034 | 0.2540 | - |
| Rule: snap to nearest Basic Pitch off | 2,632 | 63.2% | 0.4034 | 0.2581 | - |
| Rule: snap to nearest voice off | 2,123 | 63.3% | 0.4034 | 0.2574 | - |
| Decision tree picker | 1,614 | 63.6% | 0.4034 | 0.2578 | 0.4/0.2/0.2/0.15/0.1 |
| **Random forest picker** | 2,676 | 65.1% | 0.4034 | **0.2643** | 0.0/0.05/0.0/-0.05/0.05 |

"labelled ends right" = share of the 3,121 notes that have a matched true note whose end is now within tolerance.
RF: 34 of 60 training songs better on offset F1, 11 worse, 15 same. Of 1,062 labelled ends moved: 126 fixed,
66 broken (the rest wrong before and after). Moves: median +34 ms (61% later), IQR -52..+53 ms.
Onset F1 cannot change (only ends move).

## Plain-language summary
Moving ends to a "real moment" in the audio helps a little: offset F1 0.2568 -> 0.2643 (+0.0075). The most this
idea could ever give is 0.2989, so the picker gets about 18% of the possible gain. Simple rules (always snap to
the next note, to where Basic Pitch stops, or to where the voice stops) do nothing or hurt; snapping to the next
note's start hurts again, as before. Why is it hard? Only 3,121 of 7,550 notes have a matched true start at all
(an end only counts if the start is right), and for most of them "stay" is already right (63%); only 10.7% have a
right move. The audio clues at ends are weak: every kind of moment is right about 26% of the time, same as chance.

## Recommendation
Keep it as an optional 5th step: small but consistent on training (34 vs 11 songs). Before adding it to the
chain, train the final RF on all training options, choose the margin on all-train OOF (folds picked ~0.0-0.05),
and check once on val. Bigger offset gains need better STARTS first (an end only scores when the start matches).

Files: `build_candidates.py` (Step 1, ~14 min), `end_candidates_train.csv`, `end_snap_model.py` (Step 2),
`results_train_oof_rules_dt_rf.csv`, `end_snap_train_oof_rf.csv` / `_dt.csv` (note_id -> new end, OOF),
`end_snap_rf_oof_prob.csv`.

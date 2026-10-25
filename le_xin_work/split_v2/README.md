# Split v2: experiments to improve the split step (training songs only)

Input = timing OOF export `exports/le_xin_timing_rf_audio_pitch_v2_nestedgate.csv` (as in the split notebook).
Everything is out-of-fold by song (5 folds, `folds.csv`), the probability threshold is chosen inside the training folds
(nested, on onset F1), and scores are `cu.evaluate_end_to_end` on all 534 training recordings.
No validation, no test, no TextGrid; the answer key is used only for labels and scoring.

## Results (training songs, OOF)

| Version | candidates (recall of true cuts) | cuts | real cuts (precision) | onset F1 | offset F1 |
|---|---|---|---|---|---|
| Timing only (no split) | - | 0 | - | 0.3541 | 0.2176 |
| **Old split v1** (RF, 19 clues) — rebuilt exactly | 19,729 (63%) | 359 | 133 (37%) | 0.3585 | 0.2243 |
| Old candidates, RF + richer clues (74) | 19,729 (63%) | 597 | 226 (38%) | 0.3587 | 0.2273 |
| Old candidates, gradient boosting + richer clues | 19,729 (63%) | 598 | 234 (39%) | 0.3599 | 0.2295 |
| Old candidates, at most one cut per note (RF rich) | 19,729 (63%) | 584 | 223 (38%) | 0.3585 | 0.2280 |
| Old candidates, two-stage (note needs a cut? x candidate) | 19,729 (63%) | 363 | 128 (35%) | 0.3569 | 0.2189 |
| More candidates (v2), gradient boosting, old 19 clues | 36,096 (89%) | 736 | 222 (30%) | 0.3584 | 0.2276 |
| More candidates (v2), RF + richer clues | 36,096 (89%) | 744 | 267 (36%) | 0.3615 | 0.2317 |
| More candidates (v2), gradient boosting + richer clues | 36,096 (89%) | 1,100 | 356 (32%) | 0.3608 | 0.2328 |
| More candidates (v2), two-stage | 36,096 (89%) | 461 | 137 (30%) | 0.3577 | 0.2210 |
| More candidates (v2), boosting + richer + consonant clues (80) | 36,096 (89%) | 1,126 | 364 (32%) | 0.3616 | 0.2323 |
| **More candidates (v2), RF + boosting average, richer clues (best)** | 36,096 (89%) | 870 | 310 (36%) | **0.3622** | **0.2342** |
| Ceiling, perfect picks, old candidates | 19,729 | 1,217 | 1,217 | 0.4194 | 0.2942 |
| Ceiling, perfect picks, v2 candidates | 36,096 | 1,710 | 1,710 | 0.4430 | 0.3326 |

Best (RF + boosting average) vs old split: onset F1 better in **all 5 folds** (0.4071/0.3216/0.3690/0.3544/0.3590 vs
0.4062/0.3165/0.3634/0.3541/0.3525); 35 songs better, 24 worse, 1 same (of 60). Threshold 0.6 chosen in every fold.
Gain over timing only: +0.0081 onset / +0.0166 offset (old split: +0.0044 / +0.0067), i.e. roughly double.

## What was tried
- **Candidate recall** (`s3_candidates.py`): the old finder catches 64% of the 1,924 hidden true starts. Missed ones
  have weaker attacks (median onset strength 0.19 vs 0.41) and smaller loudness valleys. Extra sources: smaller pitch
  change (>=0.3 st), Basic Pitch onset peak (any pitch, >=0.3), loudness valley (>=3 dB), timbre (MFCC) change (>=0.15).
  Recall 64% -> 89%, candidates 19.7k -> 36.1k. Real-cut share by source: timbre 12.4%, valley 11.6%, old 5.6%,
  small pitch 4.5%, Basic Pitch onset 2.5%. Old candidates are kept unchanged (rebuild matches the notebook cache).
- **Richer clues** (`clues.py`, 74 total): pitch before/after over 100 ms, steadiness, pitch settle time and whether it
  settles on a new pitch, timbre change, loudness valley, Basic Pitch onset at the old and NEW pitch, note activation
  rise/fall, note-level (number of steady pitches in the note, pitch spread), neighbours, and comparison with the
  other candidates of the same note (rank, gap to the note's max).
  Importance (RF, v2): old 19 clues 0.34, timbre+valley 0.22, comparison 0.18, pitch-100ms/settle 0.10,
  note-level 0.09, Basic Pitch 0.08. Single strongest clue: timbre change (AUC 0.70).
- **Models**: RF vs gradient boosting (HistGradientBoosting) vs their average; two-stage note/candidate; top-1 per note.
- **Consonant clues** (`s7_timbre.py`: high-frequency burst, zero-crossing rate, longer timbre window): no gain.

## Files
`common.py` (copied notebook logic + fast scorer that equals cu.evaluate_end_to_end), `exp.py` (nested OOF),
`s1_tracks.py` (audio tracks cache), `s2_baseline.py` (old model rebuilt: 0.3585/0.2243 exactly), `s3_candidates.py`,
`s4_clues.py`, `s5_models.py`, `s6_compare.py`, `s7_timbre.py`, `results.csv`. Caches in `cache/` (OOF probs/cuts per run).

## Not done
- Validation: needs val tracks, v2 candidates and clues for val (Basic Pitch val activations exist in the other
  session's scratchpad `bpact_val`). Final model = average on all train candidates, threshold 0.6.
- Effect on the full chain (timing -> split -> delete -> snap) not measured; later steps were trained on the old split.

## Validation (run once, chain v4 stage 1)
Final model on all train candidates, threshold 0.6 from train OOF. Val timing 0.3755/0.2360 -> + split v2
0.3844/0.2584 (old split v1: 0.3796/0.2427). 258 cuts in 211 notes; 12 songs better, 8 worse.
See `le_xin_work/chain_v4/stage1_README.md`.

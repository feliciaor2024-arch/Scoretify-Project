# Snap v3: improving the start "snap" step (training songs only)

All numbers come from the 534 training recordings (60 songs), scored with the shared scorer (same numbers as `cu.evaluate_end_to_end`).
They are out-of-fold by song (`folds.csv`, 5 folds). For each held-out fold, the margin was chosen on the other 4 folds
(inner out-of-fold), the same way as snap v2. Input = `exports/combined_final_train_oof_v1.csv`.
**labelled right** = share of the 4,285 labelled notes whose start lands within 50 ms of the true start.
Validation and test data were not opened. No TextGrid was used. The answer key was used only for labels and scoring.

## Main table

| version | starts moved | labelled right | onset F1 | offset F1 | margins (fold 0-4) |
|---|---|---|---|---|---|
| Final chain, no snap | 0 | 65.8% | 0.3667 | 0.2294 | - |
| **snap v2** (forest, 45 clues, +-100 ms), reproduced exactly | 2546 | 72.7% | 0.4034 | 0.2568 | 0 / 0 / 0 / -0.05 / 0 |
| 1. gradient boosting (HistGradientBoosting) | 2388 | 72.3% | 0.4005 | 0.2544 | .05 / .05 / 0 / 0 / 0 |
| 1. forest + boosting average | 2516 | 72.7% | 0.4029 | 0.2559 | .05 / 0 / 0 / -.05 / 0 |
| 1. ranker: LightGBM lambdarank (compares a note's options) | 2358 | 72.9% | 0.4047 | 0.2567 | .05 / .05 / .15 / .10 / .15 |
| 1. ranker: LightGBM xendcg (listwise softmax) | 1945 | 72.9% | 0.4048 | 0.2565 | 0 / .05 / .05 / .05 / .05 |
| 1. two-stage: move at all? then which? | 1833 | 70.5% | 0.3901 | 0.2469 | -.05 / -.10 / 0 / -.20 / -.10 |
| 2. forest, +-100 ms, rebuilt table (control) | 2577 | 72.4% | 0.4017 | 0.2551 | .05 / 0 / 0 / -.05 / -.05 |
| 2. forest, +-100 ms + 2 extra moment kinds | 2900 | 73.9% | 0.4096 | 0.2595 | -.05 / 0 / 0 / 0 / 0 |
| 2. forest, +-150 ms | 2350 | 72.9% | 0.4056 | 0.2569 | 0 / 0 / 0 / 0 / 0 |
| 2. forest, +-150 ms + extra kinds | 2829 | 73.1% | 0.4065 | 0.2577 | 0 / -.05 / 0 / 0 / 0 |
| 2. ranker, +-150 ms | 2837 | 73.3% | 0.4094 | 0.2611 | 0 / -.05 / .05 / 0 / .10 |
| 2. ranker, +-150 ms + extra kinds | 3311 | 74.1% | 0.4133 | 0.2637 | -.05 / .05 / 0 / .05 / -.10 |
| 3. snap v2 + repeat snap | 2554 | 73.0% | 0.4066 | 0.2586 | round 2: .05 / 0 / .05 / 0 / 0 |
| 3. forest +-100 ms + extra kinds + repeat | 2921 | 74.1% | 0.4137 | 0.2633 | round 2: -.05 / 0 / 0 / 0 / .05 |
| **3. ranker +-150 ms + extra kinds + repeat (best)** | **3333** | **74.1%** | **0.4163** | **0.2664** | round 2: .10 / .05 / .10 / .05 / -.05 |

Each option set has a ceiling: the share of labelled notes that have at least one right option. "Stay" alone is right for 65.8%.

| option set | options per note | ceiling |
|---|---|---|
| +-100 ms (v2) | 3.0 | 91.1% |
| +-100 ms + extra kinds | 3.35 | 93.5% |
| +-150 ms | 4.09 | 96.3% |
| +-150 ms + extra kinds | 4.61 | 97.3% |

Seed check for the ranker, +-150 ms + extra kinds (seeds 0 / 1 / 2): onset F1 0.4133 / 0.4147 / 0.4153 and offset F1
0.2637 / 0.2644 / 0.2637. Song by song against snap v2, it is better on 33 / 32 / 36 of the 60 songs and worse on 24 / 19 / 16.

## In plain words
- **A smarter model on the same options barely helps.** Gradient boosting and the forest/boosting average are slightly worse.
  The rankers, which compare a note's options with each other instead of judging each option alone, gain only about
  +0.001, which is noise. The two-stage model ("should it move?" then "where to?") is clearly worse, because the
  "should it move?" question is hard to answer without looking at the options.
- **Giving the model better options helps most.** We added two new kinds of moment: a Basic Pitch onset peak at the
  note's own pitch, and the moment the sung pitch settles on the note. With these, the forest gains +0.008 over the
  like-for-like control. Widening the search to +-150 ms lifts the ceiling from 91% to 97%, but a wider search also means
  more wrong options to choose from. The forest cannot use the wider window well; the ranker can (+0.006 to +0.010).
- **Snapping a second time adds about +0.003 to +0.004 every time.** Round 2 moves about 260-410 notes again. It fixes a few
  starts that round 1 only half-fixed.
- Combining all three (ranker + +-150 ms + extra kinds + repeat): onset F1 **0.4163** (+0.013 over snap v2) and offset
  F1 **0.2664** (+0.010).
- Noise level: when the same +-100 ms table is rebuilt from scratch, the forest scores 0.4017 instead of 0.4034 (two
  pitch-match clues differ slightly on about 2.5% of rows from the older cache). Ranker seeds also move the score by
  about 0.002. So differences under about 0.003 should not be trusted.

## Recommendation
Use **snap v3 = LightGBM lambdarank picker, options within +-150 ms including the 2 extra moment kinds, run twice
(repeat snap)**. Then check it once on validation against snap v2 (val 0.4281 / 0.2796).
Expect a gain of about +0.01 onset F1 there, not more. The gain on training songs is consistent but small: about 55% of songs better, 25-40% worse.
If the group wants fewer moving parts, the forest with +-100 ms + extra kinds + repeat scores almost as well
(0.4137 / 0.2633) and needs no LightGBM.

To run on val, you need: val options built with `build_options` (it now needs the val BP activation cache `bpact_val`
and val audio curves; `common.py` is train-only today), the final ranker trained on all labelled training options,
and the margins chosen on all-train out-of-fold scores.

## Files (all in this folder)
- `common.py`: rebuilds the notebook's Step 1/2/4 functions (options, clues, labels, nested margin) plus a memoised scorer.
  `step0_verify.py` checks that it reproduces snap v2 (0.4034 / 0.2568 / 72.7%) and the official scorer exactly.
- `pickers.py`: forest, boosting, LightGBM rankers, two-stage. LightGBM 4.7 is installed only in this session's scratchpad
  `pylib` (base Python is not touched); to use it elsewhere, run `pip install lightgbm`.
- `step1_pickers.py`, `step2_wider.py`, `step3_repeat.py`, `step4_seeds.py`, plus `*.log` and `results_step*.csv`.
- `results.csv`: every row above. `cache/`: audio curves, option tables, out-of-fold scores.
- Repeat snap is done fairly: for fold k, round 2 uses the round-1 model that never saw fold k, and the round-2 margin is chosen
  on the other 4 folds from their own inner out-of-fold round-1 output.

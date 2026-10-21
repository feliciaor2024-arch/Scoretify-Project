# Why the set-A extra-note models delete notes (exploratory)

**Status: EXPLORATORY.** Training numbers use the proposed v2 labels, which are automatic, not manually verified and not approved. The explanations below are the most likely reasons supported by the numbers. They have not been confirmed by listening.

This note covers the four models in `felicia_false_starter.ipynb` that use only the 7 starter features (set A): `start, end, pitch, amplitude, duration, prev_gap, next_gap`. For each model it gives what the model deletes, a likely reason, and the numbers on training and validation.

## How the numbers were made

- **Script:** every number comes from `felicia_false_explanations.py`. Run it with `.venv/Scripts/python notebooks/correction_handover/felicia_false_explanations.py`. It writes `exports/felicia_false_explanations_numbers_v2_exploratory.json`. The model code is copied from the notebook. Its held-out training results and the final model's validation result match the notebook's exported tables exactly.
- **Training (60 songs, 7,623 notes):** model decisions are **held-out**. Each fold's notes are decided by a model whose settings and cutoff were chosen on the other four folds (notebook section 3).
- **Validation (20 songs, 2,329 notes):** each model is refit on all 3,893 labelled training notes and applied once (notebook section 8). Validation has **no correction labels**, because the handover never labels validation.
- **Two measures of "wrong":**
  - **v2 false** (training only). The labelling rule marks a note false when no reference note overlaps it and every reference note is more than 0.2 s away. It marks a note real when it pairs up with one reference note: mutual best overlap, overlap ≥ 60%, starts within 0.2 s, pitch within 12 semitones.
  - **Unmatched** (both splits). The shared evaluator finds no reference note within 50 ms onset and 50 cents pitch (`onset_matched` in the notebook). 72.8% of training notes and 71.6% of validation notes are unmatched, so a signal matters only if it pushes the rate clearly above that.
- **Thresholds:** the decision tree's own splits. Low pitch is ≤ 44 (G#2 and below), high pitch is > 71 (C5 and above), and quiet is amplitude ≤ 0.425. "Late" means the last 20% of the clip, and "short" means under 0.2 s.
- **Harmonic interval:** a note sits exactly 12, 19 or 24 semitones from the loudest note it overlaps. These are the 2nd, 3rd and 4th harmonics of that note.

## Results at a glance

Unchanged Basic Pitch scores onset+pitch F1 **0.2628** on training and **0.2726** on validation.

| Model (refit settings) | Cutoff | Train precision / recall | Train deleted (v2 false / real / unlabelled) | Train BP-correct notes deleted | Train F1 (Δ) | Val deleted | Val BP-correct deleted | Val F1 (Δ) | Val songs better / worse |
|---|---:|---|---|---:|---|---:|---:|---|---|
| Decision tree (depth 5, min leaf 20) | 1.000 | 0.867 / 0.066 | 42 (13 / 2 / 27) | 0 | 0.2637 (+0.0009) | 20 | 0 | 0.2736 (+0.0010) | 6 / 0 |
| Random forest (300 trees, min leaf 1) | 0.667 | 0.788 / 0.209 | 253 (41 / 11 / 201) | 10 | 0.2655 (+0.0028) | 66 | 0 | 0.2764 (+0.0037) | 13 / 0 |
| Logistic regression (C 0.1, balanced) | 0.975 | 0.200 / 0.005 | 42 (1 / 4 / 37) | 2 | 0.2631 (+0.0003) | 2 | 0 | 0.2728 (+0.0001) | 2 / 0 |
| **Gaussian NB** (Yeo-Johnson, 1e-9), final | 0.901 | 0.857 / 0.184 | 431 (36 / 6 / 389) | 1 | 0.2692 (+0.0064) | 122 | 0 | 0.2806 (+0.0080) | 16 / 0 |

Cutoffs are from the refit on all labelled notes. Held-out training used a separate cutoff per fold. "BP-correct notes deleted" counts deleted notes that the evaluator had matched, so each one costs F1.

## The signals and why they point to extra notes

### 1. Very low pitch (≤ 44): noise or under-tones below the voice

| | Train | Val |
|---|---:|---:|
| Notes | 84 | 34 |
| Median pitch | 31 (G1) | 32 (G#1) |
| Below E2 (bottom of a bass voice) | 73.8% | 58.8% |
| Median distance from the recording's median pitch | −26 semitones | −23.5 semitones |
| **Unmatched** | **100%** (rest of notes: 72.4%) | **100%** (rest: 71.2%) |
| v2 labels | 43 false, 4 real, 37 unlabelled | — |
| Overlap another note | 44.0% (37); 83.8% of them sit below it | 47.1% (16); 68.8% sit below it |

**Explanation:** these notes sit about two octaves below the singer's own pitch, mostly below what any voice can sing. They are probably low-frequency noise or under-tones of a sung note. The evaluator matches **none** of them on either split. This is the most reliable signal in the data. The 4 notes labelled real are unmatched too; the label rule accepts a pitch difference of up to 12 semitones.

### 2. High pitch (> 71): mostly overtones of a sung note

| | Train | Val |
|---|---:|---:|
| Notes | 490 | 135 |
| Median pitch | 78 (F#5) | 75 (D#5) |
| Median distance from the recording's median pitch | +15 semitones | +8 semitones |
| **Unmatched** | **91.2%** (mid band: 71.1%) | **93.3%** (mid band: 69.8%) |
| Overlap another note | 66.7% (327) | 58.5% (79) |
| ...of those, the high note is **above** it | 96.3% | 96.2% |
| ...of those, at a **harmonic interval** (12 / 19 / 24) | **38.2%** (22.0 / 7.6 / 8.6) | **43.0%** (22.8 / 6.3 / 13.9) |
| Same rate for overlapping notes the evaluator matches | 13.5% | 11.6% |
| Share of each voice's notes that are high | Bass 11.5%, Tenor 7.0%, Alto 4.9%, Soprano 4.5% | Bass 8.1%, Soprano 7.6%, Tenor 4.7%, Alto 3.1% |

**Explanation:** most high notes are detected **at the same time as a sung note and above it**, very often exactly one octave, an octave plus a fifth, or two octaves higher. That pattern points to overtones (harmonics) of the sung note picked up as separate notes. Bass singers have the **largest** share of high notes on both splits even though they cannot sing there, which also fits overtones.

**What the labels can and cannot see:** the v2 rule cannot label an overtone false, because an overtone overlaps a reference note. Of the 490 high training notes, 393 are unlabelled, including all 117 from bass singers. The 42 labelled-false high notes are a different group. By the label rule they are more than 0.2 s from any reference note, and they come only from sopranos (31) and tenors (11), at a median of +4 semitones, so they are probably high sung sounds the reference does not mark. The models learn "high = suspicious" from those 42 notes, then mostly delete overtones (see Gaussian NB below).

### 3. Quiet (amplitude ≤ 0.425)

| | Train | Val |
|---|---:|---:|
| Quiet notes | 523 | 132 |
| **Unmatched** | **97.3%** (rest: 70.9%) | **98.5%** (rest: 70.0%) |
| Quiet share: unmatched vs matched notes | 9.2% vs 0.7% | 7.8% vs 0.3% |
| Quiet share: v2 false vs v2 real | 19.4% vs 1.2% | — |
| Median amplitude: v2 false vs v2 real | 0.634 vs 0.688 | — |

**Explanation:** a sung note carries energy, while breath, note tails and overtones are weaker. The *typical* false note is only slightly quieter (0.634 vs 0.688), so amplitude is not useful in the middle of the range. Below 0.425, though, almost nothing is matched on either split.

### 4. Late in the clip (last 20%): mostly "after the singing stopped"

| | Train | Val |
|---|---:|---:|
| Notes in the last 20% | 942 | 304 |
| **Unmatched** | 74.5% (rest: 72.5%) | 75.0% (rest: 71.1%) |
| Start after the recording's last reference note ends | 73 notes (7.5% of late notes); 100% unmatched | 25 notes (8.2%); 100% unmatched |
| v2 false in the last 20% vs v2 real | 35.2% vs 10.7% | — |
| Late v2-false notes that start after the last reference note | **94.2%** | — |
| All v2-false vs v2-real notes after the last reference note | 33.7% vs 0.0% | — |

**Explanation:** after the singer finishes, Basic Pitch still picks up notes, probably from breath or room sound. The label rule marks them false once they are more than 0.2 s past the last reference note, and it can never mark them real. So lateness looks strong in the **labels**. For the evaluator, a late note is barely more likely to be wrong than any other (+2.0 points on training, +3.9 on validation). Lateness is therefore mostly a **label effect**. Only the small group after the last reference note is reliably extra. The same holds for clip length: false notes come from longer clips (median 12.87 s vs 10.48 s for real notes), but unmatched and matched notes do not differ (10.75 vs 10.65 s on training, 10.52 vs 10.98 s on validation).

### 5. Short (< 0.2 s): fragments

| | Train | Val |
|---|---:|---:|
| Short notes | 1,673 | 505 |
| **Unmatched** | **83.8%** (rest: 69.6%) | **82.4%** (rest: 68.6%) |
| Short share: unmatched vs matched | 25.3% vs 13.0% | 24.9% vs 13.5% |
| At the length floor (< 0.14 s): unmatched vs matched | 4.2% vs 2.1% | 4.2% vs 1.1% |
| Short share: v2 false vs v2 real | 33.7% vs 17.6% (at floor: 10.2% vs 2.7%) | — |

**Explanation:** a sung note usually lasts longer than 0.2 s. Short detections are more often brief bursts or pieces of a note. The shortest note on both splits is 0.129 s, right at Basic Pitch's configured minimum note length (127.7 ms, `baseline_config.json`), so many fragments are as short as Basic Pitch allows.

### 6. Overlapping the previous note (prev_gap < 0): only matters together with pitch

| | Train | Val |
|---|---:|---:|
| Notes that overlap the previous one | 797 (10.5%) | 242 (10.4%) |
| **Unmatched** | 74.3% (rest: 72.6%) | 70.7% (rest: 71.7%) |
| Overlapping **matched** notes within 2 semitones of the other note | 69.0% | 76.1% |
| Overlapping **unmatched** notes at a harmonic interval (vs matched) | 32.2% (vs 13.5%) | 35.7% (vs 11.6%) |
| v2 false vs v2 real | 14.3% vs 5.6% | — |

**Explanation:** overlap on its own is not a sign of a wrong note: on validation, overlapping notes are matched slightly *more* often. Real notes overlap their neighbours too (legato singing, Basic Pitch's note tails), usually within 2 semitones. What marks an extra note is **what it overlaps**: a note sitting a harmonic interval above a sung note. That is why overlap works for the models in combination with high pitch (section 2).

## The models

### Decision tree: "pitch ≤ 44.5 and amplitude ≤ 0.425 → delete"

**Decision:** the refit tree deletes only one leaf: very low **and** quiet notes. That leaf holds 27 v2-false and 0 real notes. It contains 50 training notes and 20 validation notes, and the evaluator matches none of the 20.

**Why:** the cutoff came out at 1.000, so only a leaf with no real notes can delete. The other low leaf (amplitude > 0.425) has 16 false and 4 real notes. The best high-pitch leaf (pitch > 71.5, duration ≤ 0.384 s, start > 5.35 s) has 22 false and 8 real notes. Both fall short of the cutoff. Pitch makes up 0.852 of the tree's importance and duration 0.087.

**Held out vs refit:** trees fit on four folds sometimes also delete the high leaf. On training, the 42 held-out deletions are 15 low and 27 high notes. Of the 30 that overlap another note, 90.0% sit at a harmonic interval. On validation the refit tree deletes 20 notes, all low and none BP-correct, for a gain of +0.0010.

### Random forest: "≥ 2/3 of 300 trees vote delete"

**Decision:** there is no single rule. Notes are deleted when several weak signals agree. Training deletions are 66 low, 161 high and 26 mid notes. Their medians: amplitude 0.364, 75.5% of the way through the clip, duration 0.221 s, and 66.8% overlap the previous note. Validation deletions are 22 low, 41 high and 3 mid notes, with 74.2% overlapping the previous note.

**Why:** importance is spread across all features (pitch 0.276, amplitude 0.159, start 0.125, end 0.122, prev_gap 0.111, duration 0.106, next_gap 0.101). The forest can combine "high + overlapping + quiet" into overtone detection: 43.8% (training) and 50.9% (validation) of its overlapping deletions sit at harmonic intervals. The cost is that weak-signal combinations sometimes match real notes, so it deleted **10 BP-correct notes** on training, the most of any model. On validation it deleted 66 notes, none BP-correct, for +0.0037, with 13 songs better and 0 worse.

### Logistic regression: "weighted sum ≥ 0.975"

**Decision:** weights per 1 SD are amplitude −0.489, start +0.342, end +0.311, duration −0.234, pitch +0.230, next_gap +0.154 and prev_gap +0.092. It deletes quiet, late notes: median amplitude 0.342, 84.0% of the way through the clip. On training that is 42 notes (6 low, 15 high, 21 mid) with precision 0.200. On validation it is 2 notes, for +0.0001.

**Why it does worst:**
1. **Pitch can only count in one direction.** False notes are at both ends of the range, so the two ends partly cancel and pitch gets a weak +0.230. The model deletes 6 of the 84 always-unmatched low training notes and 0 of 34 on validation.
2. **It leans on lateness**, the weakest signal for the evaluator (74.5% vs 72.5% unmatched). Lateness only works through the 7–8% of late notes that come after the singing ends.

Reaching 0.90 precision then needs a 0.975 cutoff, which leaves almost nothing to delete.

### Gaussian naive Bayes (final model): "does not look like a normal note, ≥ 0.901"

**Decision:** after a Yeo-Johnson transform, each class is described by a mean and a spread per feature. The prior is 0.9497 retain to 0.0503 delete.

| Feature | Retain mean (sd) | Delete mean (sd) |
|---|---|---|
| start / end | −0.037 (0.987) / −0.036 (0.989) | +0.691 (1.000) / +0.671 (0.978) |
| amplitude | +0.030 (0.976) | −0.567 (1.247) |
| duration | +0.021 (0.996) | −0.397 (1.002) |
| pitch | −0.011 (**0.907**) | +0.200 (**2.074**) |
| prev_gap | 0.000 (**0.926**) | −0.005 (**1.924**) |
| next_gap | +0.001 (1.008) | −0.013 (0.840) |

**Why it works best:** the delete class is **2.3× wider in pitch** and **2.1× wider in prev_gap** than the retain class. A note far from normal pitch in *either* direction, or with an unusual overlap or gap before it, fits "delete" better, and when both happen the evidence adds up. That combination is the overtone signature from sections 2 and 6, and it also covers the low notes from section 1. Being quieter, shorter and later adds more evidence.

**Numbers:**
- **Training:** 431 deletions (50 low, 266 high, 115 mid). 74.9% overlap the previous note, compared with 10.5% of all notes. Of the overlapping deletions, 52.1% sit at a harmonic interval and only 1.8% within 2 semitones. 389 of the 431 are unlabelled, so most of what it removes is notes the label rule could not judge. One BP-correct note was deleted.
- **Validation:** 122 deletions (19 low, 66 high, 37 mid). 84.4% overlap the previous note, and 60.9% of the overlapping ones sit at a harmonic interval. **None** were BP-correct. F1 0.2806 (+0.0080), with 16 of 20 songs better and 0 worse.

## What this does not show

- **Not confirmed by listening.** The overtone and noise explanations rest on pitch, timing and interval patterns. `listen(note_id)` in the notebook (section 6) is the direct check.
- **Different measures per split.** Training numbers mix v2 labels and the evaluator match, while validation has only the evaluator match. They agree on low pitch, high pitch, quiet and short. They disagree on late and overlap, where the label rule itself shapes the result.
- **Not independent evidence.** Validation was already used to choose Basic Pitch's settings, so these validation numbers are continued development. Test was not opened.

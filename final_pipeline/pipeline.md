# The correction pipeline, step by step

This page explains the current (final) pipeline in plain words: what each step fixes, how it decides, and which model
it uses. The code and results are in `final_pipeline/chain_v4/`.

## The big picture

Basic Pitch listens to a singer and writes down a list of notes. Each note has a **start time**, an **end time** and a
**pitch**. Many of these notes are a bit wrong. The pipeline passes the notes through five steps, one after the other.
Each step fixes one kind of mistake and hands the notes on to the next step.

```
 Singing audio
      |
      v
 Basic Pitch  ->  list of notes (start, end, pitch)        onset F1 0.273
      |
      v
 1. Timing      move each start and end a bit               0.376
      |
      v
 2. Split       cut notes that are really two notes         0.384
      |
      v
 3. Delete      remove notes nobody sang                    0.405
      |
      v
 4. Start snap  move each start onto a real sound event     0.456
      |
      v
 5. End snap    move each end onto a real sound event       0.456 (offset F1 0.321)
      |
      v
 Corrected list of notes
```

(Scores = onset F1 on the 20 validation songs after each step. Basic Pitch alone: onset 0.273, offset 0.160.)

**Onset F1** counts a note as right if it starts within 50 ms (1/20 of a second) of a real note and has the right
pitch. **Offset F1** also needs the end to be close. Both go from 0 (nothing right) to 1 (perfect).

## The three kinds of model used

| Model | How it works, simply | Used in |
|---|---|---|
| **Random forest** | Builds hundreds of small decision trees, each on a slightly different sample of the data, and lets them vote. Hard to fool, works well with many clues. | Timing, split, end snap |
| **Gradient boosting** | Builds trees one after another; each new tree focuses on the mistakes the earlier trees made. | Split, delete |
| **Ranking model** (LightGBM) | A type of gradient boosting that looks at all the options for one note together and puts them in order, best first. | Start snap |

A **decision tree** asks a series of yes/no questions about the clues ("is the note shorter than 0.2 s?",
"does the voice get much louder here?") and ends with an answer.

In total: **5 steps, 7 trained estimators in 6 joblib containers** (4 random forests, 2 gradient boosting, 1 ranking model; the split container wraps two estimators).

## What the models are allowed to see

Only what a real user would have:
- **Basic Pitch's notes** (start, end, pitch, loudness, gaps to the notes before and after).
- **Clues measured from the audio:** how loud it is, how suddenly it gets louder, whether the voice is on or off,
  what pitch is being sung (measured with a pitch tracker called pYIN), and Basic Pitch's own inner signals.

The real notes (the answer key) are used only to **teach** the models and to **mark** them, never as a clue.
Lyric timings (TextGrid) are never used, because they line up with the answer key almost exactly.

## How every step decides, and why it is careful

Every step works the same way:
1. It **looks at the possible changes** (move by this much, cut here, delete this note, snap to this moment).
2. A model **gives each possible change a score**: how likely it is to be right.
3. The change is **only made if the model is sure enough**. How sure is "sure enough" was chosen on the training
   songs only, never on the songs being tested.

That last part matters. A change that is only slightly likely to be right breaks as many notes as it fixes.

(The examples in each step below are made up to show the idea; the results are real.)

---

## Step 1. Timing

**The mistake:** Basic Pitch starts most notes too late (by about 37 ms in the middle case), and its ends are often
off too. About half its starts are more than 50 ms off, so they count as wrong even when the pitch is right.

**What it does:** for every note, it predicts **how far to move the start** and **how far to move the end**.

**Models:** 2 random forests: one for the start, one for the end.

**Clues (31):**
- The note itself: start, end, pitch, loudness, length, gap to the previous and next note.
- Loudness around the start and end: how much it rises or drops within 100 ms.
- The pitch track around the start and end: does the voice switch on, does the pitch change, is the singer already on
  this note's pitch before Basic Pitch's start?

**Careful rule:** a start is only moved if the model wants to move it by more than about 40 ms, and an end by
more than about 80 ms. Smaller moves were mostly noise.

**Example:** a note starts at 1.12 s. The model predicts "move it 60 ms earlier", so it now starts at 1.06 s.

**Result:** 0.273 to **0.376** onset F1, the biggest single gain. All 20 validation songs got better.

## Step 2. Split

**The mistake:** sometimes Basic Pitch writes **one long note where the singer sang two or more** (about 15% of real
notes are hidden like this). The second note is then missing completely.

**What it does:** finds **places inside a note where it should be cut** into two.

**How it finds possible cut points:** places inside the note where the pitch changes, the loudness dips, the sound's
colour changes (for example a new vowel or consonant), or Basic Pitch's own "new note" signal fires.

**Models:** a random forest and a gradient boosting model. Their two scores are averaged, which was a little better
than either one alone.

**Clues (64 in the saved runtime):** the pitch before and after the cut point, how steady the pitch is, how deep the loudness dip is, how much
the sound's colour changes, Basic Pitch's signals, and how this cut point compares with the note's other possible cuts. The authoritative input list is [models/settings.json](inference/models/settings.json). Earlier stage recipes describe a richer set as “74”; that historical count is not the deployed feature count.

**Careful rule:** a note is only cut if the averaged score is at least 0.6.

**Example:** one note runs 2.00-2.80 s at pitch 62. Around 2.40 s the sung pitch jumps to 64 and the loudness dips.
The model is confident, so it becomes two notes: 2.00-2.40 s and 2.40-2.80 s, each with its own pitch.

**Result:** 0.376 to **0.384** onset F1. A small gain, but it finds notes that were completely missing.

## Step 3. Delete

**The mistake:** some notes match **nothing that was sung**: Basic Pitch heard something and wrote a note that is not
in the real song. Each one counts against the score.

**What it does:** gives every note a score for "this note matches nothing that was sung" and removes the notes it is
very sure about.

**Model:** gradient boosting.

**Clues (49):** how much of the note is actually sung at its pitch (from the pitch track), how spread out the pitch is,
the note's length and gaps, Basic Pitch's own confidence, and loudness.

**Careful rule:** a note is only deleted if the score is at least about **0.875**. That is high on purpose: removing a
real note costs as much as keeping a fake one, so the model only deletes when it is very sure.

**Example:** a very short note sits right after a long note, nothing is sung at its pitch, and Basic Pitch was unsure
about it. Score 0.93, so it is deleted.

**Result:** 0.384 to **0.405** onset F1. 329 notes deleted on validation, and 19 of 20 songs got better.

**Note:** this replaces Felicia's original delete model, which was trained before the timing and split fixes. This
version is retrained on the already-fixed notes and scores higher on the same notes (0.381 vs 0.370 on training
songs). Using it needs Felicia's agreement.

## Step 4. Start snap

**The mistake:** after timing, many starts are close but still not within 50 ms. The timing model guesses a time, and
a guess can land slightly off.

**What it does:** instead of guessing a time, it **picks a real moment in the sound** near the current start, the way
a shape in a drawing app "snaps" onto a grid line.

**How it finds the options:** within 150 ms of the current start, it lists every moment where:
- the voice gets suddenly louder,
- the voice switches on,
- the pitch jumps,
- Basic Pitch's own "a note starts here" signal fires at this note's pitch,
- the sung pitch settles on the note's pitch,

plus the option **"stay where you are"**.

**Model:** a ranking model (LightGBM). It looks at all of one note's options together and puts them in order.
Comparing the options with each other works better than judging each one on its own.

**Clues (47):** how far the option is from the current start, what kind of moment it is, how strong each sound clue is
there, Basic Pitch's signals just before and after, how well the sung pitch matches the note after that moment, the
neighbouring notes, and how this option compares with the note's other options.

**Careful rule:** the note only moves if the best option clearly beats "stay".

**Done twice:** after the first round, the options are looked for again around the new starts, and the same model picks
again. The second round fixes a few starts that the first round only half-fixed.

**Example:**

| | Time |
|---|---|
| Real start (answer key) | 1.00 s |
| After timing | 1.07 s (70 ms late, counts as wrong) |
| Moments found nearby | 0.99 s (voice gets louder), 1.12 s (pitch jump), or stay at 1.07 s |
| Snap picks | 0.99 s, now 10 ms off, counts as right |

**Result:** 0.405 to **0.456** onset F1, the second-biggest gain. 707 starts moved on validation, and 19 of 20 songs
got better.

## Step 5. End snap

**The mistake:** ends are the hardest part. The voice often fades out slowly or slides into the next note, so Basic
Pitch's ends are often far off.

**What it does:** the same idea as start snap, for the **end** of each note.

**How it finds the options:** within 150 ms of the current end, moments where the voice switches off, the voice
becomes less steady, the loudness drops, the sung pitch leaves the note, the pitch jumps, the next note starts, or
Basic Pitch's own "this note is sounding" signal fades. Plus "stay where you are".

**Model:** a random forest. It scores each option, and the best one is used if it beats "stay".

**Result:** onset F1 stays **0.456** (only ends change). Offset F1 goes from 0.316 to **0.321**. 804 ends moved on
validation.

---

## How we know it really works

- **Fair testing:** the 60 training songs are split into 5 groups. Each model is always marked on a group of songs it
  never learned from, and every "sure enough" level is chosen without looking at that group.
- **Validation is development evidence:** the 20 validation songs were already used for Basic Pitch decoder selection and later correction development. Their gain is not fresh independent evidence and does not alone rule out overfitting.
- **Final test completed:** the recorded chain-v4 test run on 27 October 2026 covers 20 songs / 164 recordings. [Saved test scores](test_run/test_scores.csv) give 0.4148 onset F1 / 0.2812 offset F1 for final notes. The [scientific chronology](../SCIENTIFIC_GUIDE.md#test-chronology-and-freeze-boundary) distinguishes this from the earlier locked-test stage. Do not rerun or tune on test results.

## All results in one table

| After step | Training onset F1 | Training offset F1 | Validation onset F1 | Validation offset F1 |
|---|---|---|---|---|
| Basic Pitch alone | 0.263 | 0.154 | 0.273 | 0.160 |
| 1. + timing | 0.354 | 0.218 | 0.376 | 0.236 |
| 2. + split | 0.362 | 0.234 | 0.384 | 0.258 |
| 3. + delete | 0.381 | 0.248 | 0.405 | 0.273 |
| 4. + start snap | 0.421 | 0.277 | 0.456 | 0.316 |
| **5. + end snap (final)** | **0.421** | **0.284** | **0.456** | **0.321** |

## Where each step lives

| Step | How it was built and tested | Final output tables |
|---|---|---|
| 1. Timing | [timing notebook](../notebooks/correction_handover/le_xin_timing_starter.ipynb) | `exports/le_xin_timing_*` |
| 2. Split | [split_v2](experiments/split_v2/) | `chain_v4/stage1_*`, `chain_v4/val_stage1_*` |
| 3. Delete | [delete_v2](experiments/delete_v2/) | `chain_v4/stage2_*`, `chain_v4/val_stage2_*` |
| 4. Start snap | [start_snap_v3](experiments/start_snap_v3/) | `chain_v4/stage3_*`, `chain_v4/val_stage3_*` |
| 5. End snap | [end_snap](experiments/end_snap/) | `chain_v4/stage4_*`, `chain_v4/val_stage4_*` |

The [stage 1](chain_v4/stage1_README.md), [stage 2](chain_v4/stage2_README.md), [stage 3](chain_v4/stage3_README.md) and [stage 4](chain_v4/stage4_README.md) recipes retain the historical construction settings; [runtime settings](inference/models/settings.json) are the current feature/threshold authority. Literal paths elsewhere on this page are repository-root relative, except output patterns in this table, which refer to the [handover exports](../notebooks/correction_handover/exports/) and [chain tables](chain_v4/).

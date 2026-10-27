# Le Xin's part: how I built the correction model

Basic Pitch listens to a singer and writes down the notes, but it gets a lot wrong. My part was to build the
correction steps that clean up its notes so they match the real notes (the answer key) more closely.
This page explains what I built, in what order, what worked, what didn't, and why the score stops well short of 0.8.
For the full story in the order I did it, see `le_xin_explainer.md`.

**Result in one line:** on the 20 validation songs, onset F1 went from **0.273** (Basic Pitch alone) to **0.456**,
and offset F1 from **0.160** to **0.321**.

---

## 1. What the score means

- A note counts as **right** if it starts within **50 ms** (one twentieth of a second) of a real note and has the
  right pitch (within half a semitone).
- **Onset F1** (0 to 1) combines two questions: of the notes we wrote, how many are right? And of the real notes,
  how many did we find?
- **Offset F1** is the same, but the note's **end** must also be close. It is harder, so it is always lower.
- Scores are worked out song by song and then averaged, with the shared scorer `cu.evaluate_end_to_end`.

## 2. The rules I followed

1. **Test songs are locked.** I never opened or scored `data/test`. They are kept for one final run after the group agrees.
2. **The models only see what a real user would have:** Basic Pitch's notes and the audio. The answer key is used only
   to teach (labels) and to mark (scores), never as a clue.
3. **No lyric timings (TextGrid).** 80% of the true note starts sit within 1 ms of a lyric boundary, so using them
   would be copying the answers. Real users don't have them either.
4. **Fair testing by song.** The training songs are split into 5 groups (`folds.csv`). A model is always scored on
   songs it never learned from, and every setting (for example "how sure before moving a note") is picked using the
   other groups only.
5. **Validation is checked once**, after everything has been chosen on training songs.
6. **Never overwrite.** Every step saves a complete new note table (`note_id, recording_id, start, end, pitch`) plus
   a history of what it changed. The original tables stay as they were.

## 3. What I built, step by step

Each step takes the notes from the step before it and fixes one kind of mistake.

| Step | What it fixes | Model |
|---|---|---|
| 1. **Timing** | Basic Pitch starts notes too late (by about 37 ms in the middle case) and ends them in the wrong place. This step works out how far to move each start and end. | Random forest |
| 2. **Split** | Sometimes one long note really covers two or more sung notes. This step finds where to cut it. | Random forest + gradient boosting (average of the two) |
| 3. **Delete** | Removes notes that match nothing that was sung (false notes). | Gradient boosting (replaces Felicia's model if she agrees, see below) |
| 4. **Snap starts** | Moves each start onto a real moment in the audio nearby (the voice getting louder, switching on, a pitch change, Basic Pitch's own onset signal, the pitch settling), or leaves it where it is. Done twice. | Ranking model (LightGBM) |
| 5. **Snap ends** | The same idea for note ends. | Random forest |

A **random forest** is many small decision trees that vote. **Gradient boosting** builds trees one after another, each
fixing the last one's mistakes. A **ranking model** looks at all the options for one note together and picks the best.

### How each step was made better
1. **Timing.** I started with only the note itself (start, end, pitch, loudness, length, gaps to its neighbours).
   Then I added clues from the **audio**: how much the loudness jumps near each start and end. That was the biggest
   single gain. Finally I added the **pitch track** (pyin): when the voice switches on and when the pitch changes.
   I also added a **gate**: only move a note if the model wants to move it by more than about 40 ms. Small moves
   were mostly noise.
2. **Split.** Version 1 found possible cut points from pitch changes and loudness dips. Version 2 looks in more places
   (timbre changes, loudness valleys, Basic Pitch onsets), catching 89% of the real cut points instead of 63%, and
   uses richer clues.
3. **Delete.** Felicia's model was trained on the original Basic Pitch notes. I retrained a delete model on the notes
   *after* my timing and split fixes, with clues such as "how much of this note is actually sung at its pitch".
4. **Snap.** Version 1 picked between "stay" and a few nearby audio moments. Version 2 added Basic Pitch's own signals
   and "where does the pitch settle". Version 3 looks a bit wider (±150 ms), uses a model that compares a note's
   options with each other, and runs the snap a second time.
5. **End snap.** Ends had been hard to fix (the timing step mostly leaves them alone), so I added the same snap idea
   for ends.

## 4. Scores after each step

All training numbers are on songs the model never learned from. Validation was run once.

| After step | Training onset F1 | Training offset F1 | Validation onset F1 | Validation offset F1 |
|---|---|---|---|---|
| Basic Pitch alone | 0.263 | 0.154 | 0.273 | 0.160 |
| + timing | 0.354 | 0.218 | 0.376 | 0.236 |
| + split | 0.362 | 0.234 | 0.384 | 0.258 |
| + delete | 0.381 | 0.248 | 0.405 | 0.273 |
| + snap starts | 0.421 | 0.277 | 0.456 | 0.316 |
| **+ snap ends (final)** | **0.421** | **0.284** | **0.456** | **0.321** |

Training and validation move together, so the system works on songs it has never seen; it isn't just memorising.

How the system improved over time (validation onset F1): first full chain **0.428**, then **0.456** after the
second round of improvements (split v2, new delete, snap v3, end snap).

## 5. What I used and what I didn't

| Idea | Used? | Why |
|---|---|---|
| Random forest for timing | Used | Best of what I tried (decision tree 0.290, forest 0.288 on note information only; forest with audio 0.341, with audio + pitch 0.354). |
| Clues from the audio (loudness, pitch track, Basic Pitch's own signals) | Used | The biggest single boost. |
| Only moving a note when the model is fairly sure (gate / margin) | Used | Small moves were mostly noise. Every gate was picked on training songs only. |
| Split instead of merge | Used | See section 6. |
| Ranking model for snap starts | Used | Better than the forest once the search was widened to ±150 ms. |
| Snapping twice | Used | Adds about +0.003 every time. |
| Decision tree (one tree instead of a forest) | Not used | Worked, but always a bit worse than the forest. |
| Simple rules ("move every note 37 ms earlier", "snap to the loudest jump") | Not used | Made things worse or did nothing. Each note needs its own fix. |
| Cutting each note's end at the next note's start | Not used | Lowered offset F1 (0.208 → 0.200). |
| "Two-stage" snap (first "should it move?", then "where?") | Not used | Clearly worse (0.390 vs 0.403). |
| Rebuilding the training labels from the corrected notes | Not used | 10% more labels, but no gain. |
| Lyric timings (TextGrid) | Not used | That would be copying the answer key. |
| Queenie's merge model | Not used | Lowered the score (0.263 → 0.252 on training). Even a perfect merge would add only +0.004. |
| Hann Qian's pitch model | Not used | Followed the rules, but made no difference on validation (0.3796 → 0.3795). |
| Felicia's delete model | Used in the first chain; replaced in the final chain if she agrees | On the same notes, the retrained delete model scores 0.381 vs 0.370 for hers (training songs). |

## 6. Why the order differs from the original plan

The original plan was four models in a row: **Le Xin (timing) → Queenie (merge) → Hann Qian (pitch) → Felicia (delete)**.
When we put them together and tested each one, two of them didn't help, so the plan changed:

- **Merge was swapped for split.** Merging joins notes that were wrongly broken in two, but with our Basic Pitch
  settings that mistake is rare. The common mistake is the opposite: one long note covering two or more sung notes
  (about 15% of real notes). So I built a model that splits notes instead.
- **The pitch step was left out.** Only about 5% of notes have the wrong pitch, and the pitch model fixed about as many
  as it broke.
- **Delete moved after split.** Deleting first or after the split scored almost the same (0.3657 vs 0.3667), so we
  kept the simpler order. Later the delete model was retrained on the split notes, which helped more.
- **Snap steps were added.** A "what if" check showed start times are still where most of the score is lost, so I built
  models that move starts (and then ends) onto real moments in the audio. Snapping starts became the second-biggest gain after timing.

The rule I used: **keep a step only if it raises the score on training songs, then confirm once on validation.**

## 7. Why we couldn't reach 0.8

0.8 is possible **in theory**: with perfect start times and perfect deleting, validation onset F1 would be about 0.82,
and fixing every mistake would give about 0.97. So Basic Pitch isn't the limit. The limit is that each fix has to be
almost perfect, and we are not there yet.

- **The 50 ms window is very tight.** To reach 0.8, about 9 out of 10 starts must land within one twentieth of a second.
  Now about 7.4 out of 10 do (6.6 before snapping).
- **Singing has no clear "bang" where a note starts.** In 79% of cases a note starts exactly where the last one ends:
  the singer slides from one note to the next. A piano note has a sharp hit; a sung note often doesn't.
- **The answer key marks starts by the lyrics, not by the sound.** Only lyric timings (which we may not use) line up with
  them exactly. From the sound alone, the true start is often a guess.
- **About half of the notes have no teaching label.** Only about half of Basic Pitch's notes could be matched to a real
  note automatically, so the models learned from half the data, and those labels were automatic too.
- **Not much data:** 100 songs for training.
- **Small mistakes add up:** notes Basic Pitch missed completely (about 6%), wrong pitch (about 5%), notes hidden inside
  a longer note. Each one costs a little.

A realistic target with this approach was 0.45 to 0.50. The final system reaches 0.456 on validation.

## 8. Where everything is

| What | Where |
|---|---|
| Timing model (steps 1-6, with explanations) | `le_xin_timing_starter.ipynb` |
| Split model v1 | `le_xin_split_model.ipynb` |
| Snap model v1 and v2 | `le_xin_snap_model.ipynb` |
| First combined chain (timing → split → Felicia's delete → snap) | `combined_correction_system.ipynb` |
| **Final chain** (all stages, training + validation tables, one README per stage) | `le_xin_work/chain_v4/` |
| How each new stage was built and tested | `le_xin_work/experiments/` (split_v2, delete_v2, snap_v3, end_snap, relabel) |
| Folder map and how to rerun | `le_xin_work/README.md` |
| Saved corrected note tables from the notebooks | `exports/le_xin_*` and `exports/combined_final_*` |

## 9. Still to do

- **Felicia's OK** to replace her delete step with the retrained one. If not, her step can go back in its place (the
  first chain used it).
- Update `combined_correction_system.ipynb` to the final chain.
- **Test songs:** run once only, on the finished system, after the group approves.

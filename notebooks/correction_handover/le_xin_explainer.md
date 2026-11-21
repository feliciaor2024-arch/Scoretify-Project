# Le Xin's part, step by step

This is the story of my part in the order I did it: what I tried at each step, why, and what the score did.
For a shorter summary (the tables of what was and wasn't used), see `le_xin_correction_explained.md`.

---

## Step 0. Understand the problem

**What Basic Pitch gives us.** Basic Pitch listens to a singing recording and writes down a list of notes.
Each note is three numbers: **when it starts** (seconds), **when it ends** (seconds) and **its pitch**
(a MIDI number: 60 = middle C, 61 = one semitone higher, ...). We keep these as a table
(`note_id, recording_id, start, end, pitch`). No sheet music is made at any point.

**What the right answer looks like.** The GTSinger dataset comes with the real notes for every recording, with exact
start and end times in seconds and the pitch (read from each recording's annotation file). That is the answer key.

**How we are marked.** The scorer pairs our notes with the real notes, one to one.
- A note is a **hit** if it **starts** within **50 ms** (one twentieth of a second) of a real note and has the right
  pitch (within half a semitone).
- **Onset F1** (0 to 1) balances two questions: *how much of what we wrote is right?* and *how much of the real song
  did we find?* It only looks at starts.
- **Offset F1** also needs the **end** to be close (within 50 ms or a fifth of the note's length). Ends are harder to
  hear, so this score is always lower.
- Scores are worked out per song, then averaged over songs.

**Starting point:** Basic Pitch alone scores **0.263** onset F1 and **0.154** offset F1 on the training songs.

## Step 1. Set the rules before touching anything

1. **Test songs are locked.** Never opened. They are for one final check after the group agrees.
2. **No cheating clues.** The models only see what a real user would have: Basic Pitch's notes and the audio.
   The answer key is only used to teach the model (labels) and to mark it (scores).
3. **No lyric timings (TextGrid).** 80% of the real note starts sit exactly on a lyric boundary, so using them would
   be copying the answers.
4. **Fair testing by song.** The 60 training songs are split into 5 groups. A model is always scored on a group of
   songs it never learned from, and every setting is chosen using the other groups only.
5. **Validation songs (20) are checked once**, only after everything is decided.
6. **Never overwrite.** Every step saves a complete new note table and a list of what it changed.

## Step 2. Look at what Basic Pitch gets wrong

Before building anything, I compared Basic Pitch's notes with the real notes:
- Its **starts are late**: in the middle case the real start is **37 ms earlier**.
- **About half (51%) of its starts are more than 50 ms off**, which means they score zero even when the pitch is right.
- Its **ends** are even messier (some are off by a third of a second).

So my part became: **fix the timing.**

## Step 3. Try the simplest fix first

Idea: move every note 37 ms earlier.
Result: the score went **down** (0.263 → 0.251). Some notes are late, some early, some already right,
so one fixed shift breaks as many notes as it fixes. **Each note needs its own fix.** That calls for a model.

## Step 4. A model that predicts how far to move each note

For each note, a model predicts "move the start by X ms" and "move the end by Y ms".
It learns from notes where we know the right answer.
Clues it gets: start, end, pitch, loudness, length, and the gaps to the notes before and after.

| Model | Onset F1 | Offset F1 |
|---|---|---|
| Basic Pitch alone | 0.263 | 0.154 |
| Decision tree (one tree) | 0.290 | 0.157 |
| Random forest (many trees that vote) | 0.288 | 0.164 |

Both helped. I kept the **random forest**, because it was better on ends and steadier.

**Added a "gate":** only move a note if the model wants to move it by more than about 30-40 ms. Tiny moves were
mostly noise. The gate size was chosen on the other song groups only. Result: **0.292 / 0.175.**

## Step 5. Let the model hear the audio

The note numbers alone can't say *where* the real start is. The audio can: when someone starts singing a note, the
voice usually gets louder. I added **loudness clues** around each start and end
(how much it jumps, how big the change is).

- Snapping straight to the loudest jump, as a simple rule: no gain (0.263).
- Giving the loudness clues **to the random forest**: **0.341**, the biggest single jump of the whole project.
- With the gate: **0.344 / 0.208.**

## Step 6. Let the model hear the pitch too

I added a **pitch track** (what pitch is being sung, moment by moment): when the voice switches on, where the pitch
changes, and whether the singer is already on the note's pitch before Basic Pitch's start.

Result with the gate: **0.354 / 0.218** on training songs.

I also tried cutting each note's end at the next note's start. That *lowered* offset F1 (0.208 → 0.200), so I
dropped it.

## Step 7. Check on validation songs

I trained the final timing model on all training songs and ran it once on the 20 validation songs:
**0.273 → 0.376** onset F1, **0.160 → 0.236** offset F1, and **all 20 songs got better.**
The gain on new songs (+0.10) was as big as on training songs (+0.09), so the model isn't just memorising.

## Step 8. Find where the rest of the score is lost

I ran "what if" checks using the answer key (only to measure, never inside a model):
- If pitch were ignored completely, Basic Pitch's score would rise from 0.27 to 0.32. If the start window were
  100 ms instead of 50 ms, it would rise to 0.39. So **timing costs more than pitch.**
- If every labelled note had a perfect start and end, the score would reach 0.48, so **timing is still the main lever.**
- One long note covering two sung notes is common (about 15% of real notes).
- Notes wrongly broken in two (what "merge" fixes) are rare.

## Step 9. Put the group's models together and test each one

The original plan: **my timing → Queenie's merge → Hann Qian's pitch → Felicia's delete.**
I tested each step by whether it raises the score:

| Step | Effect | Decision |
|---|---|---|
| My timing | Big gain | Keep |
| Queenie's merge | Lowered the score (0.263 → 0.252). Even a perfect merge would add only +0.004. | Leave out |
| Hann Qian's pitch | No change (0.3796 → 0.3795). Only about 5% of notes have the wrong pitch. | Leave out |
| Felicia's delete | Helped 16 of 20 validation songs, hurt none | Keep |

**New rule from here: a step stays only if it raises the score.** Queenie's and Hann Qian's results are what told us
those mistakes are rare, so their work shaped the final design.

## Step 10. Build the steps the data actually needs

- **Split.** The opposite of merge: find long notes that cover two sung notes and cut them.
  A model scores possible cut points (pitch changes, loudness dips). Small gain: 0.354 → **0.359.**
- **Delete.** Felicia's model, placed after the split: **0.367.**
- **Snap.** Step 8 showed starts still lose the most score. For each note I list a few real moments in the audio
  nearby (voice gets louder, voice switches on, pitch changes) plus "stay where you are", and a model picks the best one.
  - Snap version 1: **0.394**
  - Snap version 2 (added Basic Pitch's own signals and "where the pitch settles"): **0.403**

**First full system:** 0.403 / 0.257 on training songs and **0.428 / 0.280 on validation songs.**

## Step 11. Second round: improve every step

I tried five improvements, each scored the fair way on training songs:

| Idea | Result | Kept? |
|---|---|---|
| Split v2: look for cuts in more places, richer clues | Small gain | Yes |
| Delete v2: retrain the delete model on the already-fixed notes | Small gain (0.381 vs 0.370 for the old one on the same notes) | Yes, if Felicia agrees |
| Snap v3: search a bit wider (±150 ms), compare a note's options with each other, snap twice | +0.009 over snap v2 | Yes |
| End snap: the same snap idea for note ends | Raises offset F1 | Yes |
| Rebuild the teaching labels from the fixed notes | 10% more labels, no gain | No |

Each step was retrained, in order, on the notes coming out of the step before it, so they fit together.

## Step 12. Final results

| After step | Training onset F1 | Training offset F1 | Validation onset F1 | Validation offset F1 |
|---|---|---|---|---|
| Basic Pitch alone | 0.263 | 0.154 | 0.273 | 0.160 |
| + timing | 0.354 | 0.218 | 0.376 | 0.236 |
| + split | 0.362 | 0.234 | 0.384 | 0.258 |
| + delete | 0.381 | 0.248 | 0.405 | 0.273 |
| + snap starts | 0.421 | 0.277 | 0.456 | 0.316 |
| **+ snap ends (final)** | **0.421** | **0.284** | **0.456** | **0.321** |

Validation was run once, with every setting picked on training songs. Training and validation moved together all
the way, so the system works on songs it has never seen.

## Step 13. Why it stops well short of 0.8

- **Possible in theory:** perfect start times plus perfect deleting would give about 0.82; fixing everything about 0.97.
- **But 50 ms is very tight.** About 7 in 10 starts are now inside the window; 0.8 needs about 9 in 10.
- **Sung notes glide into each other.** In 79% of cases a note starts exactly where the last one ends, with no clear
  "hit" to hear, unlike a piano.
- **The answer key follows the lyrics,** which we are not allowed to use, so from the sound alone the exact start is
  often a guess.
- **Only about half the notes have a teaching label,** and there are only 60 training songs.

## Step 14. What is left

- Felicia's OK to use the retrained delete step (otherwise her original step goes back in).
- Update the combined notebook to the final system.
- Run the test songs **once**, on the finished system, after the group approves.

---

**Where the work is:** timing in `le_xin_timing_starter.ipynb`, split v1 in `le_xin_split_model.ipynb`,
snap v1/v2 in `le_xin_snap_model.ipynb`, first full system in `combined_correction_system.ipynb`,
final system in `final_pipeline/chain_v4/`, and how each new step was built in `final_pipeline/experiments/`.

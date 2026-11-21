# Correction pipeline for the app (v2, the actual run; handover for the back end)

One function takes a singing recording and gives back the corrected notes, plus a MIDI file and a sheet (MusicXML).
It is chain v4, the final system: the same 7 trained models and settings that scored **0.456 onset F1 / 0.321
offset F1** on the 20 validation songs (Basic Pitch alone: 0.273 / 0.160).

```
 audio (.wav)
   │
   ├─ 0. Basic Pitch (onset 0.99, frame 0.3, min note 127.7 ms)   raw notes
   ├─ 1. Timing        2 random forests      move each start / end
   ├─ 2. Split         RF + boosting         cut notes that are really two
   ├─ 3. Delete        boosting              remove notes nobody sang
   ├─ 4. Start snap    LightGBM ranker (x2)  put starts on real sound moments
   ├─ 5. End snap      random forest         same for ends
   │                                          ──> corrected notes  (this table is what F1 is measured on)
   └─ 6. Sheet         tempo (librosa) → MIDI (pretty_midi) → round to 32nd notes → music21 → MusicXML
```

## Files

| File | What it is |
|---|---|
| `pipeline.py` | The pipeline. `Corrector().correct(path)` or run it from the command line. |
| `features.py` | The clues each step measures from the audio (copied from the stage scripts, unchanged). |
| `models/` | The 7 trained models (`*.joblib`, 27 MB) and `settings.json` (clue lists, thresholds, margins). |
| `train_final_models.py` | Remakes `models/` from the training songs. Only needed if the models change. |
| `check_val.py` | Proves the pipeline gives exactly the chain v4 val tables, step by step. |
| `run_test.py`, `score_test.py` | The one test run (already done) and its scoring. Results in `../test_run/`. |
| `requirements.txt` | Package versions it was tested with (Python 3.13). |

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r final_pipeline/inference/requirements.txt
```

Keep **scikit-learn 1.9.1** and **lightgbm 4.7.0**: the saved models are loaded with these versions. Basic Pitch
needs `onnxruntime` (no TensorFlow needed; the warnings about TensorFlow/CoreML/TFLite can be ignored).

## Use

Command line:

```bash
python final_pipeline/inference/pipeline.py song.wav --out results
```

writes `results/song_basic_pitch.csv`, `results/song_corrected.csv`, `results/song.mid`, `results/song.musicxml` and
`results/song_sheet_notes.csv` (the notes exactly as they appear on the sheet, in seconds).

In the back end (one call per upload; load the Corrector once at server start):

```python
import sys; sys.path.insert(0, 'final_pipeline/inference')
from pipeline import Corrector

corrector = Corrector()                        # loads the 7 models; Basic Pitch loads on the first song, then stays
result = corrector.transcribe('upload.wav', 'outdir', recording_id='upload')
# result is plain data, ready for json.dumps:
# {'recording_id', 'notes': [{'start', 'end', 'pitch'}, ...], 'basic_pitch_notes', 'bpm',
#  'midi': 'outdir/upload.mid', 'musicxml': 'outdir/upload.musicxml', 'sheet_notes': [...]}
```

For the before/after tables use `corrector.correct(path)` instead: it returns DataFrames for every step.

- `start` / `end` are in seconds and `pitch` is a MIDI number (60 = middle C).
- `correct()` also returns every in-between table (`basic_pitch`, `timing`, `split`, `delete`, `start_snap`) and what
  was changed (`cuts`, `deletions`), in case the app wants to show a before/after.
- `sheet_notes` are the notes exactly as drawn on the sheet (rounded), `notes` are the corrected notes (exact). Play
  the MIDI for 'what we heard', show the MusicXML for 'the score' (e.g. with OpenSheetMusicDisplay in the browser).
- Speed (laptop CPU): loading the models takes about 8 s, so do it once. Then a 6-10 s clip takes about 6-8 s to
  correct and under 1 s for the sheet. The very first clip after starting Python takes about 10 s longer (librosa
  compiles its pitch tracker), and the first run on a new machine can take a few minutes. Most of the time goes to the
  two pitch trackers (pyin).
- Any audio file librosa can read works; it is converted to mono 22,050 Hz inside.

## The sheet (step 6, our replacement for PM2S)

- The tempo is estimated from the audio (`librosa.feature.tempo`) and rounded to a whole BPM (120 if it can't tell).
- `write_midi` saves the corrected notes (exact times) as MIDI at that tempo ('Voice Oohs' sound).
- `sheet_notes` rounds each note's start and length to the nearest **32nd note** (1/8 beat), keeps one melody line
  (a note that runs into the next is cut there; two notes on the same start keep the longer) and returns the rounded
  notes in beats and in seconds.
- music21 builds the score from those rounded notes (4/4, key guessed by `analyze('key')`, bars, rests and ties) and
  writes MusicXML. Any MusicXML viewer shows it (MuseScore, or OpenSheetMusicDisplay in a web page).
- Why not PM2S: in Felicia's notebook PM2S cut onset F1 by about 0.13, because it moves ~71% of starts by more than
  50 ms. Why not music21's own MIDI rounding (`quantizePost`): it turns the tiny fragments into overlapping extra
  voices; reading that sheet back gave val onset F1 0.280. Our rounding keeps **0.428 / 0.294** on val when the
  sheet is read back (corrected notes before rounding: 0.456 / 0.321). The grid was picked on training songs: 32nd
  0.398, 32nd + sextuplet 0.401, 16th + triplet 0.367, 16th 0.348 (onset F1); GTSinger's own scores use 16ths, 8ths
  and some 32nds, no triplets, all 4/4, so the 32nd grid matches them and stays readable.
- The reported F1 is still the step-5 note table's; the sheet is for showing the result.
- Known limits: the first beat is assumed to be at 0 s (lining the grid up with librosa's beat tracker scored worse on
  training songs), the time signature is always 4/4, and the tempo guess can be off by double or half on slow songs.

## How we know it matches the reported results

- `check_val.py` rebuilds all 160 val recordings from the audio with these saved models and compares every step with
  the chain v4 val tables. The timing step matches exactly. After that, **1 note out of 2,587 differs**: one split
  candidate had probability 0.601 in the original run (cutoff 0.6) and 0.581 now, so that note is not cut.
  The cause is librosa itself: its onset strength and MFCC differ by about 1e-7 from run to run, even in the
  same process. One tree in the split model splits right at that value. Val score: 0.4557 / 0.3206 now, against
  0.4563 / 0.3213 in the chain v4 tables. Expect the same tiny run-to-run wobble on any song.
- Full audio-to-sheet run on a val recording with fresh Basic Pitch: the same corrected notes as the chain v4 table.
  Basic Pitch itself gives exactly the same activations and notes as the frozen predictions (3 val recordings checked).
- The correction models were fitted on the 60 training songs. Validation was used during development, including the earlier Basic Pitch decoder selection; it is not fresh independent evidence. Test remained locked through the BP/initial-handover stage; the later run is recorded below and in `SCIENTIFIC_GUIDE.md` (repository root).

## Test result (the one test run, 2026-10-27)

`run_test.py` ran this exact pipeline (audio -> Basic Pitch -> steps 1-6) once on the 20 locked test songs
(164 recordings, 2,519 reference notes) in 5.4 minutes; `score_test.py` scored it. Nothing was changed afterwards.
Song-macro F1 (onset + pitch / offset):

| Stage | Test onset F1 | Test offset F1 | Val onset F1 (for comparison) |
|---|---|---|---|
| Basic Pitch unchanged | 0.240 | 0.134 | 0.273 |
| 1. Timing | 0.341 | 0.204 | 0.376 |
| 2. Split | 0.362 | 0.237 | 0.384 |
| 3. Delete | 0.384 | 0.252 | 0.404 |
| 4. Start snap | 0.415 | 0.277 | 0.456 |
| **5. End snap (final notes)** | **0.415** | **0.281** | **0.456** |
| 6. Sheet, read back (32nd grid) | 0.395 | 0.248 | 0.428 |

- The pipeline improves onset F1 by **+0.175** over Basic Pitch on test (+0.184 on val) and improves **20 of 20** test songs.
- Every step helps on test, in the same order as on val.
- Test songs start lower (Basic Pitch 0.240 vs 0.273 on val), so the absolute scores are lower, but the gain holds.
- Files: `../test_run/` (`test_<stage>.csv` note tables, `test_sheet_notes.csv`, `sheets/` MIDI + MusicXML,
  `test_scores.csv`, `test_scores_by_song.csv`, `test_run_log.csv`). `score_test.py --check-val` shows its scorer
  gives exactly `cu.evaluate_end_to_end`'s val numbers (that scorer refuses test on purpose).
- `run_test.py` refuses to run again. Don't tune anything on these numbers.

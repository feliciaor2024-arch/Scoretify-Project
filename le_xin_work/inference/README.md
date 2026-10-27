# Correction pipeline for the app (handover for the back end)

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
   └─ 6. Sheet         tempo (librosa) → MIDI (pretty_midi) → music21 rounds notes → MusicXML
```

## Files

| File | What it is |
|---|---|
| `pipeline.py` | The pipeline. `Corrector().correct(path)` or run it from the command line. |
| `features.py` | The clues each step measures from the audio (copied from the stage scripts, unchanged). |
| `models/` | The 7 trained models (`*.joblib`, 27 MB) and `settings.json` (clue lists, thresholds, margins). |
| `train_final_models.py` | Remakes `models/` from the training songs. Only needed if the models change. |
| `check_val.py` | Proves the pipeline gives exactly the chain v4 val tables, step by step. |
| `requirements.txt` | Package versions it was tested with (Python 3.13). |

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r le_xin_work/inference/requirements.txt
```

Keep **scikit-learn 1.9.1** and **lightgbm 4.7.0**: the saved models are loaded with these versions. Basic Pitch
needs `onnxruntime` (no TensorFlow needed; the warnings about TensorFlow/CoreML/TFLite can be ignored).

## Use

Command line:

```bash
python le_xin_work/inference/pipeline.py song.wav --out results
```

writes `results/song_basic_pitch.csv`, `results/song_corrected.csv`, `results/song.mid` and `results/song.musicxml`.

In the back end:

```python
import sys; sys.path.insert(0, 'le_xin_work/inference')
from pipeline import Corrector, write_sheet

corrector = Corrector()                       # load the models once, at server start
result = corrector.correct('upload.wav', recording_id='upload')
notes = result['notes']                       # DataFrame: note_id, recording_id, start, end, pitch
sheet = write_sheet(notes, 'upload.wav', 'outdir', 'upload')   # {'bpm', 'midi', 'musicxml'}
```

- `start` / `end` are in seconds and `pitch` is a MIDI number (60 = middle C).
- `result` also has every in-between table (`basic_pitch`, `timing`, `split`, `delete`, `start_snap`) and what was
  changed (`cuts`, `deletions`), in case the app wants to show a before/after.
- Speed (laptop CPU): loading the models takes about 8 s, so do it once. Then a 6-10 s clip takes about 6-8 s to
  correct and under 1 s for the sheet. The very first clip after starting Python takes about 10 s longer (librosa
  compiles its pitch tracker), and the first run on a new machine can take a few minutes. Most of the time goes to the
  two pitch trackers (pyin).
- Any audio file librosa can read works; it is converted to mono 22,050 Hz inside.

## The sheet (step 6)

- The tempo is estimated from the audio (`librosa.feature.tempo`) and rounded to a whole BPM (120 if it can't tell).
- The notes go into a MIDI file at that tempo ('Voice Oohs' sound). music21 then rounds each note to the nearest
  16th or triplet and writes MusicXML in 4/4.
- For the sheet only, a note that runs into the next one is cut where the next one starts, so the melody stays one line.
- **The sheet is not scored.** Rounding moves notes by tens of milliseconds, so F1 is always measured on the
  corrected note table from step 5. The sheet is for showing the result.
- Known limits: the first beat is assumed to be at 0 s, the time signature is always 4/4, and the tempo guess can be
  off by double or half on slow songs.

## How we know it matches the reported results

- `check_val.py` rebuilds all 160 val recordings from the audio with these saved models and compares every step with
  the chain v4 val tables. The timing step matches exactly. After that, **1 note out of 2,587 differs**: one split
  candidate had probability 0.601 in the original run (cutoff 0.6) and 0.581 now, so that note is not cut.
  The cause is librosa itself: its onset strength and MFCC differ by about 1e-7 from run to run, even in the
  same process. One tree in the split model splits right at that value. Val score: 0.4557 / 0.3206 now, against
  0.4563 / 0.3213 in the chain v4 tables. Expect the same tiny run-to-run wobble on any song.
- Full audio-to-sheet run on a val recording with fresh Basic Pitch: the same corrected notes as the chain v4 table.
  Basic Pitch itself gives exactly the same activations and notes as the frozen predictions (3 val recordings checked).
- All models were trained on the 60 training songs only. Val was used only for checking. The test songs were never opened.

## Rules for the final test run

Run the 20 test songs **once**, after the group agrees, on this exact pipeline. Score the step-5 note tables with
`cu.evaluate_end_to_end`. Don't change any model or setting after seeing test results.

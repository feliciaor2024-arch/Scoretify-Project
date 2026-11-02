# Scoretify back end

A small FastAPI server that connects the frontend (`frontend/`) to the chain v4 correction pipeline
(`le_xin_work/inference/`). It imports `pipeline.py` directly, so fixes there apply here.

```
browser                                   back end (port 8000)
 record / upload ─► decode, convert to ─► POST /api/transcribe
                    mono 16-bit WAV         ├─ Corrector.transcribe()   Basic Pitch → 5 correction steps → sheet
                                            ├─ synth.render()           notes as tones; tones + voice
                                            └─ JSON: notes, bpm, file links
 OpenSheetMusicDisplay ◄── GET /api/runs/{id}/score.musicxml | score.mid | synth.wav | mix.wav | input.wav
```

## Setup (once)

Python 3.13, from the repository root:

```bash
py -3.13 -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt
pip install --no-deps basic-pitch==0.4.0
```

`basic-pitch` goes in separately with `--no-deps`. It asks for TensorFlow, which has no Python 3.13 build, but the
pipeline only uses its ONNX model. After that, pip may say basic-pitch needs tensorflow and an older resampy. Both
warnings are safe to ignore: resampy 0.4.3 is needed on Python 3.13 (older versions import `pkg_resources`).

## Run the demo

Two terminals, both from the repository root:

```bash
venv\Scripts\python -m uvicorn app:app --app-dir backend --port 8000
```

```bash
cd frontend
npm ci          # first time only
npm run dev     # http://127.0.0.1:5173, /api is proxied to port 8000
```

Start-up loads the 7 models (a few seconds) and then transcribes the bundled sample once in the background. The
first song after Python starts is about 10 s slower, so the warm-up takes that cost. `GET /api/health` returns
`{"models_loaded": true, "warmed_up": true}` when the server is ready for the demo. After that, a 10 s clip takes
about 3-8 s.

To demo the old fixed sample without the back end: `set VITE_SCORE_PROVIDER=mock` before `npm run dev`.

## API

`POST /api/transcribe` takes multipart form field `audio`: a WAV (or FLAC/OGG) clip of at most 30 s and 20 MB.
It returns:

```json
{
  "id": "1d41cfc39142", "bpm": 136.0, "duration": 11.4, "seconds": 3.4, "basic_pitch_notes": 21,
  "notes":       [{"start": 0.52, "end": 0.81, "pitch": 65}, ...],
  "sheet_notes": [{"start": 0.55, "end": 0.83, "pitch": 65}, ...],
  "files": {"musicxml": "/api/runs/<id>/score.musicxml", "midi": "...score.mid",
            "synth": "...synth.wav", "mix": "...mix.wav", "input": "...input.wav"}
}
```

- `notes` are the corrected notes (exact times, what F1 is measured on). `sheet_notes` are the same notes rounded to
  the 32nd-note grid, as drawn on the score.
- `synth.wav` contains the corrected notes as soft tones. `mix.wav` is those tones over the singer's recording. Both
  start at 0 s of the recording, the same as the score.
- Errors return `{"detail": "<message for the user>"}`: 413/422 for clips that are too big or too long, or have no
  sung notes; 415 for audio the server cannot read; 500 for anything else (logged with a traceback).

Clips are processed one at a time (a lock around the models). Each run's files go to `backend/runs/<id>/`
(git-ignored), and only the last 20 runs are kept.

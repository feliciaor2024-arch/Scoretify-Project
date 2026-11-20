"""Scoretify back end: one recording in, a corrected score out.

Wraps the chain v4 pipeline in final_pipeline/inference (imported, not copied, so fixes there apply here).

    POST /api/transcribe        form field 'audio': a WAV clip (the browser converts recordings to WAV)
    GET  /api/runs/{id}/{file}  score.musicxml, score.mid, synth.wav, mix.wav, input.wav
    GET  /api/health            whether the models are loaded and warmed up

Run from the repository root (see backend/README.md):
    venv\\Scripts\\python -m uvicorn app:app --app-dir backend --port 8000
"""
import logging
import re
import shutil
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'final_pipeline' / 'inference'))
from pipeline import Corrector  # noqa: E402
import synth  # noqa: E402

RUNS = Path(__file__).resolve().parent / 'runs'
WARMUP_CLIP = ROOT / 'frontend' / 'public' / 'demo' / 'sample.wav'
MAX_BYTES = 20 * 1024 * 1024     # same limits as the frontend (src/lib/audio.ts)
MAX_SECONDS = 30
KEEP_RUNS = 20                   # older runs are deleted
FILES = {'score.musicxml': 'application/vnd.recordare.musicxml+xml', 'score.mid': 'audio/midi',
         'synth.wav': 'audio/wav', 'mix.wav': 'audio/wav', 'input.wav': 'audio/wav'}

log = logging.getLogger('uvicorn.error')
state = dict(corrector=None, warm=False)
lock = threading.Lock()          # one song at a time: the models are shared and the CPU is the bottleneck


def warm_up():
    """The first song after start-up is ~10 s slower (librosa compiles its pitch tracker), so pay that here."""
    with lock:
        state['corrector'].transcribe(WARMUP_CLIP, RUNS / '_warmup', 'score')
        state['warm'] = True


@asynccontextmanager
async def lifespan(app):
    state['corrector'] = Corrector()                 # ~8 s: loads the 7 models
    threading.Thread(target=warm_up, daemon=True).start()
    yield


app = FastAPI(title='Scoretify', lifespan=lifespan)


@app.get('/api/health')
def health():
    return dict(models_loaded=state['corrector'] is not None, warmed_up=state['warm'])


@app.post('/api/transcribe')
async def transcribe(audio: UploadFile = File(...)):
    data = await audio.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, 'Choose an audio file of 20 MB or less.')
    run_id = uuid.uuid4().hex[:12]
    out = RUNS / run_id
    out.mkdir(parents=True)
    wav = out / 'input.wav'
    wav.write_bytes(data)
    try:
        duration = sf.info(str(wav)).duration
        if not 0 < duration <= MAX_SECONDS + 0.5:
            raise HTTPException(422, 'Use a playable audio clip of 30 seconds or less.')
        result = await run_in_threadpool(run, wav, out, run_id)
    except Exception as e:
        shutil.rmtree(out, ignore_errors=True)
        if isinstance(e, HTTPException):
            raise
        if isinstance(e, sf.LibsndfileError):
            raise HTTPException(415, 'The server could not read this audio. Send a WAV file.')
        log.exception('transcription failed for run %s', run_id)
        raise HTTPException(500, 'The transcription failed on the server. Check the back-end log.') from e
    prune()
    return result


def run(wav, out, run_id):
    with lock:
        t0 = time.perf_counter()
        r = state['corrector'].transcribe(wav, out, 'score')
        if not r['notes']:
            raise HTTPException(422, 'No sung notes were found. Try singing a little louder or closer to the microphone.')
        duration = synth.render(r['notes'], wav, out / 'synth.wav', out / 'mix.wav')
        seconds = time.perf_counter() - t0
    url = f'/api/runs/{run_id}/'
    files = dict(musicxml=url + 'score.musicxml', midi=url + 'score.mid', synth=url + 'synth.wav',
                 mix=url + 'mix.wav', input=url + 'input.wav')
    return dict(id=run_id, bpm=r['bpm'], duration=duration, seconds=round(seconds, 1),
                basic_pitch_notes=r['basic_pitch_notes'], notes=r['notes'], sheet_notes=r['sheet_notes'], files=files)


@app.get('/api/runs/{run_id}/{name}')
def run_file(run_id: str, name: str):
    path = RUNS / run_id / name
    if not re.fullmatch(r'[0-9a-f]{12}', run_id) or name not in FILES or not path.is_file():
        raise HTTPException(404, 'Not found.')
    return FileResponse(path, media_type=FILES[name], filename=f'scoretify-{run_id}.{name.split(".")[1]}'
                        if name.startswith('score') else None)


def prune():
    runs = sorted((p for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith('_')),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for old in runs[KEEP_RUNS:]:
        shutil.rmtree(old, ignore_errors=True)

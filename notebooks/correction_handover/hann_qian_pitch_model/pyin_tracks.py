"""Run librosa.pyin once per train/val recording and cache the pitch tracks (never test).

Saves f0_tracks.npz next to this script: for each recording_id, '<id>_t' (frame times, s)
and '<id>_f0' (Hz, NaN where unvoiced). Features are then built from the cache by
eval_pitch_middle.py, so trying different settings does not rerun pyin.
Usage: python pyin_tracks.py <GTSinger_100_unique_sample>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed

ROOT = Path(sys.argv[1])
HERE = Path(__file__).resolve().parent
DATA = ROOT / 'basic_pitch_runs' / 'correction_handover'
SR, HOP = 22050, 256

manifest = pd.read_csv(DATA / 'recording_manifest.csv', dtype={'recording_id': str})
manifest = manifest[manifest.split.isin(['train', 'val'])]  # test stays locked
assert not manifest.audio_path.str.startswith('data/test').any()


def run(rid, audio_path):
    y, _ = librosa.load(ROOT / audio_path, sr=SR, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'),
                                 sr=SR, hop_length=HOP)
    return rid, librosa.times_like(f0, sr=SR, hop_length=HOP), np.where(voiced, f0, np.nan)


if __name__ == '__main__':
    # Compile librosa's numba code once here, so parallel workers don't race on its cache.
    librosa.pyin(np.zeros(SR, dtype=np.float32), fmin=65.0, fmax=1047.0, sr=SR, hop_length=HOP)
    tracks = Parallel(n_jobs=6, verbose=5)(delayed(run)(r.recording_id, r.audio_path)
                                           for r in manifest.itertuples())
    arrays = {}
    for rid, t, f in tracks:
        arrays[f'{rid}_t'], arrays[f'{rid}_f0'] = t, f
    np.savez_compressed(HERE / 'f0_tracks.npz', **arrays)
    print('saved', len(tracks), 'recordings')

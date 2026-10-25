"""Step 1: cache per-recording audio tracks for the TRAIN recordings (pyin pitch, voicing, onset strength, loudness, MFCC)."""
import numpy as np
from joblib import Parallel, delayed
import common as C

manifest = C.load()[0]
(C.CACHE / 'tracks').mkdir(parents=True, exist_ok=True)

def run(rid, path):
    f = C.CACHE / 'tracks' / f'{rid}.npz'
    if not f.exists():
        np.savez_compressed(f, **C.tracks(C.ROOT / path))
    return rid

if __name__ == '__main__':
    import librosa
    librosa.pyin(np.zeros(C.SR, dtype=np.float32), fmin=65, fmax=1100, sr=C.SR, hop_length=C.HOP)
    Parallel(n_jobs=4, verbose=2)(delayed(run)(r.recording_id, r.audio_path) for r in manifest.itertuples())
    print('done', len(manifest))

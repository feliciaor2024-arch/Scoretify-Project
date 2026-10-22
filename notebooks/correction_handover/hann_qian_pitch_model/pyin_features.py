"""Per-note audio pitch features from librosa.pyin (train + val only, never test).

For each predicted note, measures how far the sung pitch sits from the predicted
MIDI pitch, in cents, over the note's [start, end] window.
Usage: python pyin_features.py <GTSinger_100_unique_sample> <out_dir>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
DATA = ROOT / 'basic_pitch_runs' / 'correction_handover'
SR, HOP = 22050, 256

manifest = pd.read_csv(DATA / 'recording_manifest.csv', dtype={'recording_id': str})
manifest = manifest[manifest.split.isin(['train', 'val'])]  # test stays locked
assert not manifest.audio_path.str.startswith('data/test').any()


def run(rid, audio_path):
    y, _ = librosa.load(ROOT / audio_path, sr=SR, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'),
                                 sr=SR, hop_length=HOP)
    times = librosa.times_like(f0, sr=SR, hop_length=HOP)
    return rid, times, np.where(voiced, f0, np.nan)


def note_features(notes, times, f0):
    rows = []
    for n in notes.itertuples():
        w = f0[(times >= n.start) & (times < n.end)]
        v = w[np.isfinite(w)]
        if len(v) == 0:
            rows.append(dict(note_id=n.note_id, f0_voiced_frac=0.0, f0_cents_med=0.0, f0_cents_iqr=0.0,
                             f0_frac_up=0.0, f0_frac_down=0.0, f0_has=0))
            continue
        cents = 1200 * np.log2(v / librosa.midi_to_hz(n.pitch))
        cents = (cents + 600) % 1200 - 600  # fold octave errors into [-600, 600)
        rows.append(dict(note_id=n.note_id, f0_voiced_frac=len(v) / max(len(w), 1),
                         f0_cents_med=float(np.median(cents)),
                         f0_cents_iqr=float(np.subtract(*np.percentile(cents, [75, 25]))),
                         f0_frac_up=float((cents > 50).mean()), f0_frac_down=float((cents < -50).mean()),
                         f0_has=1))
    return rows


if __name__ == '__main__':
    tracks = Parallel(n_jobs=-1, verbose=5)(delayed(run)(r.recording_id, r.audio_path)
                                            for r in manifest.itertuples())
    tracks = {rid: (t, f) for rid, t, f in tracks}
    for split in ['train', 'val']:
        notes = pd.read_csv(DATA / f'{split}_features.csv', dtype={'note_id': str, 'recording_id': str})
        rows = []
        for rid, g in notes.groupby('recording_id'):
            rows += note_features(g, *tracks[rid])
        out = pd.DataFrame(rows).set_index('note_id').loc[notes.note_id].reset_index()
        out.to_csv(OUT / f'{split}_pyin_features.csv', index=False)
        print(split, len(out), 'notes; with voiced frames:', int(out.f0_has.sum()))

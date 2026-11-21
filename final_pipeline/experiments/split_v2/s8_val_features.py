"""Step 8: VAL features for the final split v2 model: audio tracks, v2 candidates (old finder + extra sources) and the
74 richer clues, built exactly as for train. The val answer key is NOT read here (no labels). Test is never opened."""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import common as C
import clues as K
from s3_candidates import frame_tracks, peaks

all_man = pd.read_csv(C.DATA / 'recording_manifest.csv')
man = all_man[all_man.split == 'val'].reset_index(drop=True)
assert len(man) == 160 and man.audio_path.str.startswith('data/val/').all()       # val only, never test
val_raw = pd.read_csv(C.DATA / 'val_features.csv', dtype={'note_id': str, 'recording_id': str, 'sample_id': str})
timed = pd.read_csv(C.HERE / 'exports' / 'le_xin_timing_val_rf_audio_pitch.csv', dtype={'note_id': str, 'recording_id': str})
timed = timed.merge(val_raw[['note_id', 'sample_id']], on='note_id', validate='one_to_one')
assert len(timed) == len(val_raw) and timed.recording_id.isin(man.recording_id).all()
(C.CACHE / 'tracks_val').mkdir(parents=True, exist_ok=True)


def run(rid, path):
    f = C.CACHE / 'tracks_val' / f'{rid}.npz'
    if not f.exists():
        np.savez_compressed(f, **C.tracks(C.ROOT / path))
    tr = C.load_tracks(rid, 'tracks_val')
    bp = dict(np.load(C.BPACT_VAL / f'{rid}.npz'))
    notes = timed[timed.recording_id == rid]
    if notes.empty:
        return None
    ft = frame_tracks(tr, bp)

    def extra(n, lo, hi):                       # same extra sources as s3_candidates.run
        frames = np.arange(lo, hi + 1)
        before = np.array([C.med(tr['midi'], i - C.SIDE, i) for i in frames])
        after = np.array([C.med(tr['midi'], i, i + C.SIDE) for i in frames])
        change = np.nan_to_num(np.abs(after - before), nan=0.0)
        out = {}
        for f in {int(frames[k]) for k in range(len(frames))
                  if change[k] >= 0.3 and change[k] == change[max(0, k - 3):k + 4].max()}:
            out[f] = 'pitch_small'
        for f in peaks(ft['bpon'], frames, 0.3):
            out.setdefault(f, 'bp_onset')
        for f in peaks(ft['valley'], frames, 3.0):
            out.setdefault(f, 'valley')
        for f in peaks(ft['spec'], frames, 0.15):
            out.setdefault(f, 'timbre')
        return out
    args = (tr['midi'], tr['voiced'], tr['vprob'], tr['flux'], tr['db'])
    cands = pd.DataFrame([r for n in notes.itertuples() for r in C.candidates_for_note(n, *args, extra_frames=extra)])
    if cands.empty:
        return None
    return pd.concat([cands, K.per_recording(cands, notes, tr, bp)], axis=1)


if __name__ == '__main__':
    import librosa
    librosa.pyin(np.zeros(C.SR, dtype=np.float32), fmin=65, fmax=1100, sr=C.SR, hop_length=C.HOP)
    parts = Parallel(n_jobs=4, verbose=2)(delayed(run)(r.recording_id, r.audio_path) for r in man.itertuples())
    out = pd.concat([p for p in parts if p is not None], ignore_index=True)
    assert out[K.NEW_BASE].notna().all().all()
    out.to_csv(C.CACHE / 'clues_v2_val.csv', index=False)
    print('val candidates', len(out), 'notes with candidates', out.note_id.nunique(), 'of', len(timed))
    print(out.source.value_counts().to_dict())

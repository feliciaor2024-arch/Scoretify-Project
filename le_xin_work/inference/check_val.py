"""Check: the pipeline gives the SAME val tables as the chain v4 stage scripts, step by step.

Uses the val Basic Pitch notes (val_features.csv) and cached val activations (le_xin_work/bp_activations/val), so it
runs in the base environment without basic-pitch. The val answer key is only used for the final F1 printout.
Nothing is tuned here. data/test is never opened.
Usage: python check_val.py [n_jobs]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import features as F
from pipeline import Corrector, NOTE_COLUMNS

HERE = Path(__file__).resolve().parent
LX = HERE.parent
ROOT = LX.parent
DATA = ROOT / 'basic_pitch_runs' / 'correction_handover'
V4 = LX / 'chain_v4'
read = lambda p: pd.read_csv(p, dtype={'note_id': str, 'recording_id': str, 'sample_id': str})

man = read(DATA / 'recording_manifest.csv').query("split == 'val'").reset_index(drop=True)
assert len(man) == 160 and man.audio_path.str.startswith('data/val/').all()          # val only
raw = read(DATA / 'val_features.csv')


def tracks(r):
    act = dict(np.load(LX / 'bp_activations' / 'val' / f'{r.recording_id}.npz'))
    rec = F.Recording(r.recording_id, ROOT / r.audio_path, act, r.duration_seconds_if_validated)
    for key in ['flux', 'db', 'pyin', 'f0', 'mfcc']:
        getattr(rec, key)
    rec._c.pop('y')                                       # keep the worker's reply small
    return rec


if __name__ == '__main__':
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    import librosa
    librosa.pyin(np.zeros(F.SR, dtype=np.float32), fmin=65, fmax=1100, sr=F.SR, hop_length=F.HOP)   # compile once
    cache = HERE / 'cache' / 'val_tracks.pkl'                 # not in git; delete it to recompute
    if cache.exists():
        recs = pd.read_pickle(cache)
    else:
        recs = {r.rid: r for r in Parallel(n_jobs=jobs, verbose=2)(delayed(tracks)(r) for r in man.itertuples())}
        cache.parent.mkdir(exist_ok=True)
        pd.to_pickle(recs, cache)
    out = Corrector().run(raw, recs, order=list(man.recording_id))

    expected = [('timing', ROOT / 'notebooks/correction_handover/exports/le_xin_timing_val_rf_audio_pitch.csv'),
                ('split', V4 / 'val_stage1_split_v2.csv'), ('delete', V4 / 'val_stage2_delete_v2.csv'),
                ('start_snap', V4 / 'val_stage3_snap_v3.csv'), ('notes', V4 / 'val_stage4_end_snap.csv')]
    ok = True
    for name, path in expected:
        mine, saved = out[name][NOTE_COLUMNS].set_index('note_id'), read(path)[NOTE_COLUMNS].set_index('note_id')
        same_ids = set(mine.index) == set(saved.index)
        diff = (mine.loc[saved.index, ['start', 'end', 'pitch']] - saved[['start', 'end', 'pitch']]).abs() if same_ids else None
        worst = float(diff.to_numpy().max()) if same_ids else float('nan')
        n_off = int((diff > 1e-9).any(axis=1).sum()) if same_ids else -1
        good = same_ids and n_off == 0
        ok &= good
        print(f'{name:11s} notes {len(mine):5d} (saved {len(saved):5d})  same ids {same_ids}  notes that differ {n_off}  '
              f'max diff {worst:.2e}  {"OK" if good else "DIFFERENT"}')

    sys.path.insert(0, str(ROOT / 'notebooks'))
    sys.path.insert(0, str(ROOT / 'notebooks' / 'correction_handover'))
    import bp_experiment as bx
    import correction_utils as cu
    events, _ = bx.references(man.to_dict('records'))
    refs = pd.DataFrame([dict(note_id=f'{rid}:ref{i:05d}', recording_id=rid, start=e[0], end=e[1], pitch=e[2])
                         for rid, ev in events.items() for i, e in enumerate(ev)])
    for name in ['basic_pitch', 'timing', 'split', 'delete', 'start_snap', 'notes']:
        s = cu.evaluate_end_to_end(out[name][NOTE_COLUMNS], refs, man)['summary']
        print(f'val {name:11s} onset F1 {s["song_macro_f1"]:.4f}  offset F1 {s["song_macro_offset_f1"]:.4f}')
    print('ALL STEPS MATCH' if ok else 'SOME STEPS DIFFER')

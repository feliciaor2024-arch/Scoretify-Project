"""Step 4: compute the richer clues (clues.py) for a candidate table and cache them.
Usage: python s4_clues.py old|v2"""
import sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import common as C
import clues as K

which = sys.argv[1]
manifest, references, folds, raw, timed = C.load()
src = {'old': C.HERE / 'le_xin_cache' / 'split_candidates_train_v1.csv', 'v2': C.CACHE / 'cands_v2_train.csv'}[which]
cands = pd.read_csv(src, dtype={'note_id': str, 'recording_id': str, 'sample_id': str})

def run(rid, g):
    tr = C.load_tracks(rid)
    bp = dict(np.load(C.BPACT / f'{rid}.npz'))
    return K.per_recording(g, timed[timed.recording_id == rid], tr, bp)

if __name__ == '__main__':
    parts = Parallel(n_jobs=4, verbose=1)(delayed(run)(rid, g) for rid, g in cands.groupby('recording_id'))
    new = pd.concat(parts).loc[cands.index]
    out = pd.concat([cands, new], axis=1)
    assert out[K.NEW_BASE].notna().all().all(), out[K.NEW_BASE].isna().sum()[lambda s: s > 0]
    out.to_csv(C.CACHE / f'clues_{which}_train.csv', index=False)
    print(which, len(out))

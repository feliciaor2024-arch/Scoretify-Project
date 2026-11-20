"""Step 6: per-fold and per-song comparison of split versions (training songs, out-of-fold cuts).
Usage: python s6_compare.py v2:rf_rich v2:hgb_rich ..."""
import sys
import numpy as np, pandas as pd
import common as C

manifest, references, folds, raw, timed = C.load()
old_c = pd.read_csv(C.HERE / 'le_xin_cache' / 'split_candidates_train_v1.csv', dtype={'note_id': str})
tables = {'timing only': timed[C.cu.NOTE_COLUMNS],
          'old split (v1)': C.apply_splits(timed, old_c[np.load(C.CACHE / 'old_rf_cut.npy')])}
cl = {w: pd.read_csv(C.CACHE / f'clues_{w}_train.csv', dtype={'note_id': str}) for w in ['old', 'v2']}
for tag in sys.argv[1:]:
    w, name = tag.split(':')
    tables[tag] = C.apply_splits(timed, cl[w.rstrip('t')][np.load(C.CACHE / f'cut_{w}_{name}.npy')])
song = {}
for k, t in tables.items():
    out = C.cu.evaluate_end_to_end(t, references, manifest)
    song[k] = pd.Series({s['sample_id']: s['f1'] for s in out['songs']})
    print(f"{k:22s} onset {out['summary']['song_macro_f1']:.4f} offset {out['summary']['song_macro_offset_f1']:.4f} notes {len(t)}")
S = pd.DataFrame(song)
S['fold'] = S.index.map(folds)
print('\nonset F1 per fold (song macro):')
print(S.groupby('fold').mean().round(4).to_string())
for k in list(tables)[2:]:
    d = S[k] - S['old split (v1)']
    print(f'{k}: vs old split, songs better {int((d > 1e-12).sum())}, worse {int((d < -1e-12).sum())}, same {int((d.abs() <= 1e-12).sum())};'
          f" vs timing only better {int((S[k] > S['timing only'] + 1e-12).sum())}, worse {int((S[k] < S['timing only'] - 1e-12).sum())}")

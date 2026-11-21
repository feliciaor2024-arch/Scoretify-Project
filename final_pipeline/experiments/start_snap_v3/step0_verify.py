"""Step 0: check that common.py reproduces the notebook (option table, scorer, snap v2 scores)."""
import time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from common import *

t0 = time.time()
v2 = load_v2()
mine = build_options(chain, 0.10)
print(f'built options in {time.time() - t0:.0f} s: {len(mine)} rows vs notebook {len(v2)}')
key = ['note_id', 't']
a = v2.sort_values(key).reset_index(drop=True)
b = mine.sort_values(key).reset_index(drop=True)
assert (a.note_id.to_numpy() == b.note_id.to_numpy()).all() and np.allclose(a.t, b.t)
cols = CLUES_V2 + ['label', 'labelled_note']
diff = {c: float(np.nanmax(np.abs(a[c].astype(float) - b[c].astype(float)))) for c in cols}
bad = {c: d for c, d in diff.items() if d > 1e-6}
print('max abs difference per clue above 1e-6:', bad or 'none')

s0 = cu.evaluate_end_to_end(chain, references, manifest)['summary']
s1 = evaluate(chain)
print('scorer check (official vs memoised):', s0['song_macro_f1'], s1['song_macro_f1'], s0['song_macro_offset_f1'], s1['song_macro_offset_f1'])

make_rf = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                         class_weight='balanced_subsample', n_jobs=4, random_state=0)
def rf_fp(tr, te, clues=CLUES_V2):
    return make_rf().fit(tr[clues], tr.label).predict_proba(te[clues])[:, 1]
t0 = time.time()
_, _, st, m = nested(rf_fp, v2, 'rf_v2')
print(f'nested RF v2 in {time.time() - t0:.0f} s')
rows = [report('Final chain (no snap)', pd.Series(dtype=float)), report('snap v2 (RF, 45 clues) reproduced', st, m)]
print(pd.DataFrame(rows).to_string(index=False))

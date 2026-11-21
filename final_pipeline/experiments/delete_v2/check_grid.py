"""Follow-up: the GB runs picked 0.90, the top of the grid, so rerun them with a finer, wider grid (up to 0.975).
Also: songs better/worse than Felicia-only, and which clues the RF leans on. Writes results_grid.csv."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
import models as M
from common import *

M.GRID = np.r_[np.arange(.6, .976, .025), 1.01]
base = M.NOTE + M.AUDIO + M.BPACT
jobs = [('replace', 'y_official_pool', 'GB', base), ('more', 'y_official_kept', 'GB', base + ['felicia_prob'] + M.SNAP),
        ('replace', 'y_official_pool', 'RF', base + ['felicia_prob'])]
out = Parallel(n_jobs=3)(delayed(M.run)(*j) for j in jobs)
res = pd.DataFrame([o[0] for o in out])
res.to_csv(OUT / 'results_grid.csv', index=False)
print(res.to_string())

# songs better / worse than Felicia-only (onset F1 per song)
F = M.F
songs = lambda notes: {s: f1(c) for s, c in song_counts(notes).items()}


def song_counts(notes):
    g = {rid: n[['start', 'end', 'pitch']].to_numpy(float) for rid, n in notes.groupby('recording_id')}
    acc = {}
    for rid in MANIFEST.recording_id:
        c = rec_counts(rid, g.get(rid, np.zeros((0, 3))))
        acc.setdefault(MANIFEST.set_index('recording_id').sample_id[rid], np.zeros(4))
        acc[MANIFEST.set_index('recording_id').sample_id[rid]] += c
    return acc


fel = songs(F[F.felicia_kept])
for (s, l, m, f), (r, sc) in zip(jobs, out):
    kept = F.set_index('note_id').loc[sc.note_id[~sc.delete.to_numpy()]].reset_index()
    new = songs(kept)
    diff = np.array([new[k] - fel[k] for k in fel])
    print(f'{s} {l} {m}: songs better {(diff > 1e-9).sum()}, worse {(diff < -1e-9).sum()}, same {(np.abs(diff) <= 1e-9).sum()}')

# clue importance (RF on all pool notes, official label) - for the explanation only
X = F[base].to_numpy(float)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=.4, class_weight='balanced_subsample',
                            n_jobs=4, random_state=0).fit(np.where(np.isnan(X), -1, X), F.y_official_pool.astype(int))
imp = pd.Series(rf.feature_importances_, base).sort_values(ascending=False)
imp.to_csv(OUT / 'rf_importance.csv')
print(imp.head(15).round(3).to_string())

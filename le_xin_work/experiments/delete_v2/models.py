"""Delete-model prototypes on the corrected notes, out-of-fold by song, delete threshold chosen NESTED on onset F1.

Settings
  A 'replace': all 7,982 pre-delete notes; the model decides alone (Felicia's step replaced).
  B 'more'   : the 7,550 notes Felicia kept; the model deletes extra notes on top of her step.
For each outer fold k: inner out-of-fold scores on the 4 other folds -> pick the threshold with the best onset F1
on those training songs -> refit on the 4 folds -> score fold k -> delete notes with score >= threshold.
Writes results.csv, oof_scores.csv and per-config deletions next to this script."""
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from common import *

F = read(OUT / 'features_train.csv')
NOTE = ['duration', 'log_dur', 'pitch', 'amplitude', 'is_piece', 'n_pieces', 'prev_gap', 'next_gap', 'pitch_prev',
        'pitch_next', 'n_overlap', 'overlap_frac', 'overlapped_by_longer', 'rel_pos', 'first', 'last',
        'rec_notes_per_s', 'pitch_vs_rec', 'amp_vs_rec', 'dur_vs_rec']
AUDIO = [c for c in F if c.startswith(('f0_', 'a_'))]
BPACT = [c for c in F if c.startswith('bp_')]
SNAP = [c for c in F if c.startswith('sn_')]
GRID = np.r_[np.arange(.3, .91, .05), 1.01]                     # 1.01 = delete nothing

MODELS = {'RF': lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=.4,
                                               class_weight='balanced_subsample', n_jobs=1, random_state=0),
          'GB': lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=15,
                                                       min_samples_leaf=20, l2_regularization=1., random_state=0)}


def fit_predict(model, X, y, train, test):
    return MODELS[model]().fit(X[train], y[train]).predict_proba(X[test])[:, 1]


def best_threshold(notes, sc, rids):
    """Threshold with the best onset F1 on these recordings (ties -> delete fewer)."""
    f = [score(notes[sc < t], rids, offset=False)[0] for t in GRID]
    return GRID[int(np.flatnonzero(np.isclose(f, max(f), atol=1e-9))[-1])]


def run(setting, label, model, feats):
    d = F[F.felicia_kept] if setting == 'more' else F
    d = d.reset_index(drop=True)
    X = d[feats].to_numpy(float)
    X = np.where(np.isnan(X), -1, X)
    y = d[label].astype(int).to_numpy()
    fold = d.fold.to_numpy()
    oof, thr = np.zeros(len(d)), {}
    for k in range(5):
        outer = fold != k
        if model == 'rule':                                      # 1 - share of the note sung near its pitch
            sc = 1 - d.f0_near_pitch.to_numpy()
            inner = sc[outer]
            oof[~outer] = sc[~outer]
        else:
            inner = np.zeros(outer.sum())
            fo = fold[outer]
            for j in sorted(set(fo)):
                tr = np.flatnonzero(outer)[fo != j]
                te = np.flatnonzero(outer)[fo == j]
                inner[fo == j] = fit_predict(model, X, y, tr, te)
            oof[~outer] = fit_predict(model, X, y, np.flatnonzero(outer), np.flatnonzero(~outer))
        rids = d[outer].recording_id.unique()
        thr[k] = best_threshold(d[outer], inner, rids)
    delete = oof >= d.fold.map(thr).to_numpy()
    kept = d[~delete]
    on, off = score(kept)
    yo = d.y_official_kept if setting == 'more' else d.y_official_pool
    return dict(setting=setting, label=label, model=model, features=len(feats), notes_out=len(kept),
                deleted=int(delete.sum()), deleted_truly_false=int((delete & d.y_loose).sum()),
                deleted_unmatched=int((delete & (yo == 1)).sum()),
                onset_F1=round(on, 4), offset_F1=round(off, 4),
                thresholds='/'.join(f'{thr[k]:.2f}' for k in range(5))), \
        pd.DataFrame(dict(note_id=d.note_id, score=oof, delete=delete))


if __name__ == '__main__':
    base = NOTE + AUDIO + BPACT
    jobs = []
    for label in ['y_official', 'y_loose']:
        for model in ['RF', 'GB']:
            jobs += [('replace', label, model, base), ('replace', label, model, base + ['felicia_prob']),
                     ('more', label, model, base + ['felicia_prob'] + SNAP)]
    jobs += [('replace', 'y_official', 'rule', ['f0_near_pitch']), ('more', 'y_official', 'rule', ['f0_near_pitch'])]
    jobs = [(s, (l + ('_kept' if s == 'more' else '_pool')) if l == 'y_official' else l, m, f) for s, l, m, f in jobs]
    out = Parallel(n_jobs=4, verbose=5)(delayed(run)(*j) for j in jobs)
    res = pd.DataFrame([o[0] for o in out])
    res['felicia_prob_used'] = ['felicia_prob' in j[3] for j in jobs]
    res.to_csv(OUT / 'results.csv', index=False)
    for (s, l, m, f), (_, sc) in zip(jobs, out):
        sc.to_csv(OUT / f'oof_{s}_{l}_{m}{"_fp" if "felicia_prob" in f else ""}.csv', index=False)
    pd.set_option('display.width', 250)
    print(res.to_string())

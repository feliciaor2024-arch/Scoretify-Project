"""Pickers for the snap step. Each `fp_*(train_rows, test_rows)` returns a score per test row where a higher score
means 'more likely the right option'; `choose` then moves a note only if its best move beats 'stay' by the margin.
Rankers return per-note softmax probabilities, so the same margin grid makes sense for every picker."""
import sys
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from common import CLUES_V2
import lightgbm as lgb

JOBS = 4

def make_rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                  class_weight='balanced_subsample', n_jobs=JOBS, random_state=0)

def make_hgb():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40,
                                          l2_regularization=1.0, class_weight='balanced', random_state=0)

def softmax_per_note(score, note_id):
    s = pd.Series(score, index=np.arange(len(score)))
    g = s.groupby(np.asarray(note_id))
    e = np.exp(s - g.transform('max'))
    return (e / e.groupby(np.asarray(note_id)).transform('sum')).to_numpy()

def fp_rf(clues=CLUES_V2):
    return lambda tr, te: make_rf().fit(tr[clues], tr.label).predict_proba(te[clues])[:, 1]

def fp_hgb(clues=CLUES_V2):
    return lambda tr, te: make_hgb().fit(tr[clues], tr.label).predict_proba(te[clues])[:, 1]

def fp_rf_hgb(clues=CLUES_V2):
    """Average of the forest and boosting probabilities."""
    return lambda tr, te: (fp_rf(clues)(tr, te) + fp_hgb(clues)(tr, te)) / 2

def _groups(note_id):
    nid = np.asarray(note_id)
    change = np.r_[True, nid[1:] != nid[:-1]]
    assert len(set(nid)) == change.sum(), 'rows of a note must be next to each other'
    return np.diff(np.r_[np.flatnonzero(change), len(nid)])

def fp_lgb_rank(objective='lambdarank', clues=CLUES_V2, seed=0):
    """LightGBM ranker: compares a note's options with each other; output = per-note softmax of its scores."""
    def fp(tr, te):
        m = lgb.LGBMRanker(objective=objective, n_estimators=300, learning_rate=0.05, num_leaves=15,
                           min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           reg_lambda=1.0, n_jobs=JOBS, random_state=seed, verbose=-1)
        m.fit(tr[clues], tr.label, group=_groups(tr.note_id))
        return softmax_per_note(m.predict(te[clues]), te.note_id)
    return fp

AGG = ['attack', 'louder', 'pitch_jump', 'bp_onset_at', 'bp_note_rise', 'voiced_after']
def note_level(rows, clues=CLUES_V2):
    """One row per note: the 'stay' option's clues + the strongest / best-matching move option's values."""
    stay = rows[rows.is_stay == 1].set_index('note_id')[clues]
    mv = rows[rows.is_stay == 0].groupby('note_id')
    agg = pd.concat([mv[AGG].max().add_prefix('mvmax_'), mv[['match_100', 'gap_after', 'settle_abs']].min().add_prefix('mvmin_')], axis=1)
    out = stay.join(agg)
    return out.fillna({c: -1.0 for c in agg.columns})

def fp_two_stage(clues=CLUES_V2):
    """Stage A: should this note move at all? (note-level). Stage B: if so, which move option? (move rows only).
    Option score: stay = 1 - P(move); move option = P(move) * share of that option within the note's moves."""
    def fp(tr, te):
        A_tr, A_te = note_level(tr, clues), note_level(te, clues)
        stay_ok = tr[tr.is_stay == 1].set_index('note_id').label
        move_ok = tr[tr.is_stay == 0].groupby('note_id').label.max().reindex(stay_ok.index).fillna(0)
        yA = ((stay_ok == 0) & (move_ok == 1)).astype(int).reindex(A_tr.index)
        pA = pd.Series(make_hgb().fit(A_tr, yA).predict_proba(A_te)[:, 1], A_te.index)
        mv_tr = tr[(tr.is_stay == 0) & tr.note_id.map(move_ok).eq(1).to_numpy()]
        mv_te = te.is_stay.to_numpy() == 0
        qB = np.zeros(len(te))
        if mv_te.any():
            raw = make_rf().fit(mv_tr[clues], mv_tr.label).predict_proba(te.loc[mv_te, clues])[:, 1]
            s = pd.Series(raw, te.index[mv_te])
            qB[mv_te] = (s / s.groupby(te.note_id[mv_te]).transform('sum').clip(lower=1e-9)).to_numpy()
        pa = te.note_id.map(pA).to_numpy()
        return np.where(te.is_stay.to_numpy() == 1, 1 - pa, pa * qB)
    return fp

"""Step 2: end-snap picker, out-of-fold by song on the training songs (rules vs decision tree vs random forest).

Per note: move the end to the best MOVE option only if P(right) beats 'stay' by >= margin. The margin is chosen
per held-out fold on the other 4 folds only (nested), by OFFSET F1 (moving ends cannot change onset F1).
Writes results_train_oof.csv and the chosen ends (end_snap_train_oof_<version>.csv) into le_xin_work/experiments/end_snap/.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from common import OUT, SUFFIX, cu, read, load

manifest, references, folds, chain = load()
cands = read(OUT / f'end_candidates_train{SUFFIX}.csv')
cands['fold'] = cands.recording_id.map(manifest.set_index('recording_id').sample_id).map(folds)
assert cands.fold.notna().all() and set(cands.note_id) == set(chain.note_id)
NOT_CLUES = {'note_id', 'recording_id', 't', 'label', 'labelled_note', '_true_end', 'fold'}
CLUES = [c for c in cands.columns if c not in NOT_CLUES]
ALL = sorted(cands.fold.unique())
train_rows = cands.labelled_note.to_numpy()
y = cands.label.to_numpy()
rec_fold = manifest.set_index('recording_id').sample_id.map(folds)
MARGINS = [-0.30, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 9.0]
note_fold = cands.drop_duplicates('note_id').set_index('note_id').fold


def oof_prob(make_model, fold_set=ALL, clues=CLUES):
    p = np.full(len(cands), np.nan)
    for k in fold_set:
        test = (cands.fold == k).to_numpy()
        train = cands.fold.isin([f for f in fold_set if f != k]).to_numpy() & train_rows
        p[test] = make_model().fit(cands.loc[train, clues], y[train]).predict_proba(cands.loc[test, clues])[:, 1]
    return p


def choose(prob, margin):
    c = cands[['note_id', 't', 'is_stay']].assign(p=prob)
    stay_p = c[c.is_stay == 1].set_index('note_id').p
    moves = c[c.is_stay == 0].dropna(subset=['p'])
    best = moves.loc[moves.groupby('note_id').p.idxmax()].set_index('note_id')
    go = best.p - stay_p.reindex(best.index) >= margin
    return best.loc[go, 't']


def apply_ends(new_end):
    out = chain[cu.NOTE_COLUMNS].copy()
    e = out.note_id.map(new_end)
    out['end'] = np.where(e.notna(), e, out.end)
    out['end'] = np.maximum(out.end, out.start + 0.01)               # keep every note at least 10 ms long
    return out


def f1_on(fold_set, new_end):
    keep = manifest[manifest.recording_id.map(rec_fold).isin(fold_set)]
    t = apply_ends(new_end)
    t = t[t.recording_id.isin(keep.recording_id)]
    s = cu.evaluate_end_to_end(t, references[references.recording_id.isin(keep.recording_id)], keep)['summary']
    return s['song_macro_f1'], s['song_macro_offset_f1']


def nested(make_model, clues=CLUES):
    outer = oof_prob(make_model, ALL, clues)
    picked, ends = {}, []
    for k in ALL:
        others = [f for f in ALL if f != k]
        inner = oof_prob(make_model, others, clues)
        m = max(MARGINS, key=lambda g: f1_on(others, choose(inner, g))[1])
        picked[int(k)] = m
        ch = choose(outer, m)
        ends.append(ch[ch.index.map(note_fold) == k])
        print('  fold', k, 'margin', m, flush=True)
    return outer, pd.concat(ends), picked


right = cands[cands.label == 1].groupby('note_id').t.apply(np.array)
lab_ids = cands[cands.is_stay == 1].set_index('note_id').labelled_note
lab_ids = lab_ids[lab_ids].index


def labelled_right(new_end):
    t = apply_ends(new_end).set_index('note_id').end
    return np.mean([np.any(np.abs(right.get(n, np.array([])) - t.at[n]) <= 1e-6) for n in lab_ids])


if __name__ == '__main__':
    which = sys.argv[1:] or ['rules', 'dt', 'rf']
    versions = [('Snap v2 input (no end snap)', pd.Series(dtype=float), '-')]
    # Ceiling: pick a right option whenever one exists (prefer stay) -> the most an end snap could give
    st = cands[cands.is_stay == 1].set_index('note_id').label
    r = cands[(cands.is_stay == 0) & (cands.label == 1)]
    r = r[~r.note_id.map(st).astype(bool)]
    ceil = r.loc[r.assign(a=r.dt.abs()).groupby('note_id').a.idxmin()].set_index('note_id').t
    versions.append(('Ceiling: always pick a right option (answer key)', ceil, '-'))
    if 'rules' in which:
        mv = cands[cands.is_stay == 0]
        for kind in ['next_start', 'bp_off', 'voice_off']:
            o = mv[mv[f'is_{kind}'] == 1]
            versions.append((f'Rule: snap end to nearest {kind}', o.loc[o.abs_dt.groupby(o.note_id).idxmin()].set_index('note_id').t, '-'))
    if 'dt' in which:
        print('decision tree...', flush=True)
        _, e, m = nested(lambda: DecisionTreeClassifier(min_samples_leaf=40, class_weight='balanced', random_state=0))
        versions.append(('Decision tree picker', e, m))
        e.to_csv(OUT / f'end_snap_train_oof_dt{SUFFIX}.csv', header=['new_end'])
    if 'rf' in which:
        print('random forest...', flush=True)
        prob, e, m = nested(lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                                           class_weight='balanced_subsample', n_jobs=4, random_state=0))
        versions.append(('Random forest picker', e, m))
        e.to_csv(OUT / f'end_snap_train_oof_rf{SUFFIX}.csv', header=['new_end'])
        cands[['note_id', 't']].assign(p=prob).to_csv(OUT / f'end_snap_rf_oof_prob{SUFFIX}.csv', index=False)
        # recipe for a final model: margin = best all-train out-of-fold offset F1 (val not used)
        final_margin = max(MARGINS, key=lambda g: f1_on(ALL, choose(prob, g))[1])
        import json
        (OUT / f'end_snap_rf_margins{SUFFIX}.json').write_text(json.dumps({'per_fold': m, 'final_all_train': final_margin}))
        print('final margin (all-train OOF):', final_margin, flush=True)
    rows = []
    for name, ends, margins in versions:
        s = cu.evaluate_end_to_end(apply_ends(ends), references, manifest)['summary']
        assert s['recordings'] == len(manifest)
        rows.append(dict(version=name, ends_moved=len(ends), labelled_ends_right=f'{labelled_right(ends):.1%}',
                         onset_F1=round(s['song_macro_f1'], 4), offset_F1=round(s['song_macro_offset_f1'], 4),
                         margins=margins))
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f'results_train_oof_{"_".join(which)}{SUFFIX}.csv', index=False)
    print(res.to_string(index=False))

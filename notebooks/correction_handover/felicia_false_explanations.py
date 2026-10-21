"""Numbers behind felicia_false_explanations.md (EXPLORATORY, unapproved v2 labels).

Reuses the set-A model code of felicia_false_starter.ipynb (sections 3, 6 and 8):
- training: held-out decisions from the five fixed song folds (run_cv), v2 labels;
- validation: models refitted on all labelled training notes (fit_final), applied once.
Validation has no correction labels (the handover never labels validation), so on both splits
the shared evaluator match (50 ms onset, 50 cents, as onset_matched in the notebook) is also
reported: 'unmatched' = the note matches no reference note. Test is never opened.

Run from anywhere:  .venv/Scripts/python notebooks/correction_handover/felicia_false_explanations.py
Writes exports/felicia_false_explanations_numbers_v2_exploratory.json and prints the same numbers.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from librosa import midi_to_hz
from mir_eval.transcription import match_notes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import precision_recall_curve

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import correction_utils as cu
import bp_experiment as exp

DATA = HERE / '..' / '..' / 'basic_pitch_runs' / 'correction_handover'
SEED, TARGET_PRECISION = 42, 0.90
LOW, HIGH = 44.5, 71.5          # the decision tree's pitch splits
QUIET = 0.425                   # the decision tree's amplitude split
SHORT, FLOOR = 0.2, 0.14        # 'short'; within one frame of BP's 127.7 ms minimum note length
LATE = 0.8                      # last 20% of the clip
FEATURES = list(cu.NOTE_FEATURES)

# ---- data (same loading as the notebook) ----
notes = pd.read_csv(DATA / 'train_features.csv', dtype={'note_id': str, 'recording_id': str, 'sample_id': str})
v2 = pd.read_csv(DATA / 'proposed_v2' / 'train_targets_v2.csv', dtype={'note_id': str}).set_index('note_id')['false_note']
y = notes['note_id'].map(v2).dropna().astype(int)       # 1 = delete, 0 = retain (3,893 notes)
folds = pd.read_csv(DATA / 'folds.csv').set_index('sample_id')['fold']
note_fold = notes['sample_id'].map(folds)
train_refs = pd.read_csv(DATA / 'references_train.csv', dtype={'note_id': str, 'recording_id': str})
train_manifest = pd.read_csv(DATA / 'training_recording_manifest.csv', dtype={'recording_id': str, 'sample_id': str})

val_events, audit = exp.references([r for r in exp.inputs() if r['split'] == 'val'])
assert not audit['issues']
val_refs = pd.DataFrame([dict(recording_id=rid, start=s, end=e, pitch=p) for rid, ev in val_events.items() for s, e, p in ev])
val_refs.insert(0, 'note_id', [f'{rid}:ref{i:05d}' for i, rid in enumerate(val_refs['recording_id'])])
val_manifest = pd.read_csv(DATA / 'recording_manifest.csv', dtype={'recording_id': str, 'sample_id': str})
val_manifest = val_manifest[val_manifest['split'].eq('val')].reset_index(drop=True)
val_notes = pd.read_csv(DATA / 'val_features.csv', dtype={'note_id': str, 'recording_id': str, 'sample_id': str})


def onset_matched(table, references):
    refs = dict(tuple(references.groupby('recording_id')))
    hit = pd.Series(False, index=table.index)
    for rid, n in table.groupby('recording_id'):
        if rid in refs:
            r = refs[rid]
            pairs = match_notes(r[['start', 'end']].to_numpy(float), midi_to_hz(r['pitch'].to_numpy(float)),
                                n[['start', 'end']].to_numpy(float), midi_to_hz(n['pitch'].to_numpy(float)),
                                onset_tolerance=.05, pitch_tolerance=50., offset_ratio=None, strict=False)
            hit.loc[n.index[[e for _, e in pairs]]] = True
    return hit


def describe(frame, manifest, references):
    """Prediction-derived descriptors (analysis only, never model inputs) + evaluator match."""
    f = frame.copy()
    f['position'] = f['start'] / f['audio_duration']
    f['dev_from_median'] = f['pitch'] - f.groupby('recording_id')['pitch'].transform('median')
    f['vocal_range'] = f['recording_id'].map(manifest.set_index('recording_id')['vocal_range'])
    interval = pd.Series(np.nan, index=f.index)  # pitch minus the loudest note it overlaps in time
    for _, g in f.groupby('recording_id'):
        s, e, p, a = (g[c].to_numpy() for c in ['start', 'end', 'pitch', 'amplitude'])
        for k, i in enumerate(g.index):
            o = np.flatnonzero((s < e[k]) & (e > s[k]) & (np.arange(len(g)) != k))
            if len(o):
                interval[i] = p[k] - p[o[np.argmax(a[o])]]
    f['overlap_interval'] = interval
    f['matched'] = onset_matched(f, references)
    last_end = references.groupby('recording_id')['end'].max()
    f['after_last_ref'] = f['start'] > f['recording_id'].map(last_end).fillna(-np.inf)  # singing has finished
    return f


T, V = describe(notes, train_manifest, train_refs), describe(val_notes, val_manifest, val_refs)
T['v2'] = notes['note_id'].map(v2)

# ---- models (set A, identical to the notebook) ----
MODELS = {
    'decision_tree': (Pipeline([('clf', DecisionTreeClassifier(random_state=SEED))]),
                      {'clf__max_depth': [3, 5, 8, None], 'clf__min_samples_leaf': [5, 20, 50],
                       'clf__class_weight': [None, 'balanced']}),
    'random_forest': (Pipeline([('clf', RandomForestClassifier(n_estimators=300, max_features='sqrt', random_state=SEED))]),
                      {'clf__max_depth': [None, 10], 'clf__min_samples_leaf': [1, 5, 20],
                       'clf__class_weight': [None, 'balanced_subsample']}),
    'logistic_regression': (Pipeline([('scale', StandardScaler()), ('clf', LogisticRegression(max_iter=5000))]),
                            {'clf__C': [0.01, 0.1, 1, 10], 'clf__class_weight': [None, 'balanced']}),
    'gaussian_nb': (Pipeline([('scale', 'passthrough'), ('clf', GaussianNB())]),
                    {'scale': ['passthrough', PowerTransformer()], 'clf__var_smoothing': [1e-9, 1e-6, 1e-3]}),
}
X_all, X_lab, f_lab = notes[FEATURES].astype(float), notes.loc[y.index, FEATURES].astype(float), note_fold[y.index]


def pick_cutoff(y_true, prob):
    precision, _, cutoffs = precision_recall_curve(y_true, prob)
    ok = precision[:-1] >= TARGET_PRECISION
    return float(cutoffs[ok].min()) if ok.any() else np.inf


def search(name, X, labels, groups):
    pipe, grid = MODELS[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        s = GridSearchCV(pipe, grid, scoring='f1', cv=LeaveOneGroupOut(), n_jobs=-1).fit(X, labels, groups=groups)
        inner = cross_val_predict(s.best_estimator_, X, labels, groups=groups, cv=LeaveOneGroupOut(),
                                  method='predict_proba', n_jobs=-1)[:, 1]
    return s, pick_cutoff(labels, inner)


def run_cv(name):
    delete = pd.Series(False, index=notes.index)
    for k in sorted(folds.unique()):
        tr, held = f_lab.ne(k), note_fold.eq(k)
        s, cut = search(name, X_lab[tr], y[tr], f_lab[tr])
        delete[held] = s.predict_proba(X_all.loc[held])[:, 1] >= cut
    return delete


def score(table, references, manifest):
    r = cu.evaluate_end_to_end(table, references, manifest)
    return r['summary'], pd.DataFrame(r['songs']).set_index('sample_id')['f1']


# ---- helpers for the numbers ----
def pct(mask, within=None):
    within = pd.Series(True, index=mask.index) if within is None else within
    return round(100 * float(mask[within].mean()), 1) if within.any() else None


def band_numbers(f, band, labelled):
    out = dict(notes=int(band.sum()), unmatched_pct=pct(~f['matched'], band),
               median_pitch=float(f.loc[band, 'pitch'].median()), below_E2_pct=pct(f['pitch'] < 40, band),
               median_dev_from_recording_median=float(f.loc[band, 'dev_from_median'].median()),
               overlaps_another_note_pct=pct(f['overlap_interval'].notna(), band),
               vocal_range=f.loc[band, 'vocal_range'].value_counts().to_dict())
    if labelled:
        out.update(v2_false=int((f.loc[band, 'v2'] == 1).sum()), v2_real=int((f.loc[band, 'v2'] == 0).sum()),
                   v2_unlabelled=int(f.loc[band, 'v2'].isna().sum()))
        fb = band & f['v2'].eq(1)
        out.update(v2_false_median_dev=float(f.loc[fb, 'dev_from_median'].median()),
                   v2_false_overlap_pct=pct(f['overlap_interval'].notna(), fb),
                   v2_false_octave_overlaps=int(f.loc[fb, 'overlap_interval'].abs().eq(12).sum()),
                   v2_false_vocal_range=f.loc[fb, 'vocal_range'].value_counts().to_dict())
    return out


def signal_numbers(f, groups):
    """groups: name -> boolean mask. Each signal reported for every group."""
    sig = dict(amplitude_median=lambda m: round(float(f.loc[m, 'amplitude'].median()), 3),
               quiet_pct=lambda m: pct(f['amplitude'] <= QUIET, m),
               position_median=lambda m: round(float(f.loc[m, 'position'].median()), 3),
               last20_pct=lambda m: pct(f['position'] > LATE, m),
               clip_seconds_median=lambda m: round(float(f.loc[m, 'audio_duration'].median()), 2),
               duration_median=lambda m: round(float(f.loc[m, 'duration'].median()), 3),
               short_pct=lambda m: pct(f['duration'] < SHORT, m),
               at_floor_pct=lambda m: pct(f['duration'] < FLOOR, m),
               prev_overlap_pct=lambda m: pct(f['prev_gap'] < 0, m),
               any_overlap_pct=lambda m: pct(f['overlap_interval'].notna(), m),
               overlap_within_2_semitones_pct=lambda m: pct(f['overlap_interval'].abs().le(2),
                                                           m & f['overlap_interval'].notna()),
               overlap_octave_pct=lambda m: pct(f['overlap_interval'].abs().eq(12), m & f['overlap_interval'].notna()))
    return {name: {k: fn(m) for k, fn in sig.items()} | dict(notes=int(m.sum())) for name, m in groups.items()}


def unmatched_rate_by(f, conditions):
    return {name: dict(notes=int(c.sum()), unmatched_pct=pct(~f['matched'], c),
                       rest_unmatched_pct=pct(~f['matched'], ~c)) for name, c in conditions.items()}


def conditions(f):
    return dict(quiet=f['amplitude'] <= QUIET, last20=f['position'] > LATE, short=f['duration'] < SHORT,
                prev_overlap=f['prev_gap'] < 0, low=f['pitch'] <= LOW, high=f['pitch'] > HIGH)


def overlap_mix(f, m):
    """Among notes in m that overlap another note: interval to the loudest overlapped note."""
    iv = f.loc[m, 'overlap_interval'].dropna()
    a = iv.abs()
    return dict(overlapping=int(len(iv)), within_2_pct=pct(a.le(2)), octave_12_pct=pct(a.eq(12)),
                twelfth_19_pct=pct(a.eq(19)), two_octaves_24_pct=pct(a.eq(24)),
                harmonic_12_19_24_pct=pct(a.isin([12, 19, 24])), above_pct=pct(iv.gt(0))) if len(iv) else dict(overlapping=0)


def deletion_profile(f, d):
    return dict(deleted=int(d.sum()), deleted_bp_matched=int((d & f['matched']).sum()),
                deleted_after_last_reference=int((d & f['after_last_ref']).sum()),
                deleted_overlap=overlap_mix(f, d),
                deleted_low=int((d & f['pitch'].le(LOW)).sum()), deleted_high=int((d & f['pitch'].gt(HIGH)).sum()),
                deleted_mid=int((d & f['pitch'].between(LOW, HIGH, inclusive='neither')).sum()),
                median_pitch=float(f.loc[d, 'pitch'].median()) if d.any() else None,
                median_amplitude=round(float(f.loc[d, 'amplitude'].median()), 3) if d.any() else None,
                median_position=round(float(f.loc[d, 'position'].median()), 3) if d.any() else None,
                median_duration=round(float(f.loc[d, 'duration'].median()), 3) if d.any() else None,
                prev_overlap_pct=pct(f['prev_gap'] < 0, d) if d.any() else None)


# ---- signals ----
NUM = dict(definitions=dict(low_pitch=f'pitch <= {LOW}', high_pitch=f'pitch > {HIGH}', quiet=f'amplitude <= {QUIET}',
                            late='start / audio_duration > 0.8', short=f'duration < {SHORT} s',
                            at_floor=f'duration < {FLOOR} s', unmatched='no reference within 50 ms onset and 50 cents'))
NUM['train_bands'] = {'low': band_numbers(T, T['pitch'] <= LOW, True), 'high': band_numbers(T, T['pitch'] > HIGH, True),
                      'mid': band_numbers(T, T['pitch'].between(LOW, HIGH, inclusive='neither'), True)}
NUM['val_bands'] = {'low': band_numbers(V, V['pitch'] <= LOW, False), 'high': band_numbers(V, V['pitch'] > HIGH, False),
                    'mid': band_numbers(V, V['pitch'].between(LOW, HIGH, inclusive='neither'), False)}
NUM['train_signals'] = signal_numbers(T, {'v2_false': T['v2'].eq(1), 'v2_real': T['v2'].eq(0),
                                          'unmatched': ~T['matched'], 'matched': T['matched']})
NUM['val_signals'] = signal_numbers(V, {'unmatched': ~V['matched'], 'matched': V['matched']})
NUM['train_unmatched_rate'] = unmatched_rate_by(T, conditions(T))
NUM['val_unmatched_rate'] = unmatched_rate_by(V, conditions(V))
for split, f in [('train', T), ('val', V)]:
    high, late = f['pitch'] > HIGH, f['position'] > LATE
    NUM[f'{split}_extra'] = dict(
        high_overlap=overlap_mix(f, high), low_overlap=overlap_mix(f, f['pitch'] <= LOW),
        notes_in_recordings_without_references=int((~f['recording_id'].isin(set((train_refs if split == 'train' else val_refs)['recording_id']))).sum()),
        mid_overlap=overlap_mix(f, f['pitch'].between(LOW, HIGH, inclusive='neither')),
        unmatched_overlap=overlap_mix(f, ~f['matched']), matched_overlap=overlap_mix(f, f['matched']),
        high_share_of_range_pct={r: pct(high, f['vocal_range'].eq(r)) for r in sorted(f['vocal_range'].unique())},
        after_last_ref=dict(notes=int(f['after_last_ref'].sum()), unmatched_pct=pct(~f['matched'], f['after_last_ref']),
                            share_of_last20_pct=pct(f['after_last_ref'], late)),
        quiet_after_last_ref_pct=pct(f['after_last_ref'], f['amplitude'] <= QUIET))
    if split == 'train':
        false, real = f['v2'].eq(1), f['v2'].eq(0)
        NUM['train_extra'].update(
            v2_false_after_last_ref_pct=pct(f['after_last_ref'], false), v2_real_after_last_ref_pct=pct(f['after_last_ref'], real),
            v2_false_late_after_last_ref_pct=pct(f['after_last_ref'], false & late),
            v2_false_overlap=overlap_mix(f, false), v2_real_overlap=overlap_mix(f, real),
            quiet_v2=dict(false=int((false & f['amplitude'].le(QUIET)).sum()), real=int((real & f['amplitude'].le(QUIET)).sum()),
                          unlabelled=int((f['v2'].isna() & f['amplitude'].le(QUIET)).sum())),
            high_v2_by_range={r: dict(false=int((false & high & f['vocal_range'].eq(r)).sum()),
                                      real=int((real & high & f['vocal_range'].eq(r)).sum()),
                                      unlabelled=int((f['v2'].isna() & high & f['vocal_range'].eq(r)).sum()))
                              for r in sorted(f['vocal_range'].unique())})
NUM['min_duration'] = dict(train=round(float(T['duration'].min()), 4), val=round(float(V['duration'].min()), 4))

# ---- models ----
BP_T, BP_T_SONG = score(notes[cu.NOTE_COLUMNS], train_refs, train_manifest)
BP_V, BP_V_SONG = score(val_notes[cu.NOTE_COLUMNS], val_refs, val_manifest)
NUM['bp_unchanged'] = dict(train_f1=BP_T['song_macro_f1'], train_offset_f1=BP_T['song_macro_offset_f1'],
                           val_f1=BP_V['song_macro_f1'], val_offset_f1=BP_V['song_macro_offset_f1'])
NUM['models'] = {}
for name in MODELS:
    d_tr = run_cv(name)
    s, cut = search(name, X_lab, y, f_lab)
    model = s.best_estimator_
    d_val = pd.Series(model.predict_proba(val_notes[FEATURES].astype(float))[:, 1] >= cut, index=val_notes.index)
    task = cu.classification_metrics(y, d_tr[y.index].astype(int))
    st, st_song = score(notes.loc[~d_tr, cu.NOTE_COLUMNS], train_refs, train_manifest)
    sv, sv_song = score(val_notes.loc[~d_val, cu.NOTE_COLUMNS], val_refs, val_manifest)
    m = dict(params={k: repr(v) for k, v in s.best_params_.items()}, refit_cutoff=cut,
             train=dict(precision=task['precision'], recall=task['recall'], v2_false_deleted=int((d_tr & T['v2'].eq(1)).sum()),
                        v2_real_deleted=int((d_tr & T['v2'].eq(0)).sum()), unlabelled_deleted=int((d_tr & T['v2'].isna()).sum()),
                        f1=st['song_macro_f1'], delta_f1=st['song_macro_f1'] - BP_T['song_macro_f1'],
                        offset_f1=st['song_macro_offset_f1'], **deletion_profile(T, d_tr)),
             val=dict(f1=sv['song_macro_f1'], delta_f1=sv['song_macro_f1'] - BP_V['song_macro_f1'],
                      offset_f1=sv['song_macro_offset_f1'],
                      songs_improved=int((sv_song > BP_V_SONG + 1e-12).sum()), songs_worse=int((sv_song < BP_V_SONG - 1e-12).sum()),
                      **deletion_profile(V, d_val)))
    clf = model.named_steps['clf']
    if hasattr(clf, 'tree_'):
        t = clf.tree_
        leaf_t, leaf_v = clf.apply(X_all.to_numpy()), clf.apply(val_notes[FEATURES].to_numpy(float))
        lv = T.loc[y.index].assign(leaf=clf.apply(X_lab.to_numpy()))
        m['leaves'] = [dict(leaf=int(i), v2_false=int(lv.loc[lv.leaf.eq(i), 'v2'].eq(1).sum()),
                            v2_real=int(lv.loc[lv.leaf.eq(i), 'v2'].eq(0).sum()),
                            delete_prob=round(float(t.value[i, 0, 1] / t.value[i, 0].sum()), 3),
                            train_notes=int((leaf_t == i).sum()), val_notes=int((leaf_v == i).sum()),
                            val_unmatched_pct=pct(~V['matched'], pd.Series(leaf_v == i, index=V.index)))
                       for i in range(t.node_count) if t.children_left[i] == -1]
        m['importances'] = dict(zip(FEATURES, np.round(clf.feature_importances_, 3).tolist()))
    elif hasattr(clf, 'feature_importances_'):
        m['importances'] = dict(zip(FEATURES, np.round(clf.feature_importances_, 3).tolist()))
    elif hasattr(clf, 'coef_'):
        m['coef_per_sd'] = dict(zip(FEATURES, np.round(clf.coef_[0], 3).tolist()))
    else:
        m['retain_mean'] = dict(zip(FEATURES, np.round(clf.theta_[0], 3).tolist()))
        m['delete_mean'] = dict(zip(FEATURES, np.round(clf.theta_[1], 3).tolist()))
        m['retain_sd'] = dict(zip(FEATURES, np.round(np.sqrt(clf.var_[0]), 3).tolist()))
        m['delete_sd'] = dict(zip(FEATURES, np.round(np.sqrt(clf.var_[1]), 3).tolist()))
        m['class_prior'] = np.round(clf.class_prior_, 4).tolist()
    NUM['models'][name] = m

out = HERE / 'exports' / 'felicia_false_explanations_numbers_v2_exploratory.json'
out.write_text(json.dumps(NUM, indent=2, default=float), encoding='utf-8')
print(json.dumps(NUM, indent=2, default=float))

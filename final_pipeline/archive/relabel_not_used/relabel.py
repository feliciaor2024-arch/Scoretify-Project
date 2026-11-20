"""Relabel experiment: match Basic Pitch notes to reference notes AFTER a first timing correction.

Question: the v2 labels (3,697 of 7,623 notes) were made by matching the RAW, badly timed notes to the
answer key. If we first move the notes with the timing model and then match, do more notes get a
confident label, and does retraining the timing RF on those labels raise F1?

Leakage rules kept here:
- Features never change: the 31 note + audio + pitch features are computed from the RAW notes (cached).
  The corrected positions are used ONLY to decide which reference note a BP note belongs to.
- Label values are always reference start/end (shift measured from the RAW note, like v2).
- Strict nesting: for held-out fold k, the first-pass corrections of the 4 other folds come from an inner
  out-of-fold model that never saw fold k (nor the fold it predicts). Fold k's labels are never built or used
  when its model is trained. Gates are chosen on the 4 other folds only (Step 4b recipe).
Train songs only. No val, no test, no TextGrid.

Run from anywhere:  python final_pipeline/archive/relabel_not_used/relabel.py
"""
from pathlib import Path
import sys, json, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'notebooks' / 'correction_handover'
DATA = ROOT / 'basic_pitch_runs' / 'correction_handover'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import correction_utils as cu

# ---------- data (same loading as the timing notebook, cells 1/3/5a/5c) ----------
notes = pd.read_csv(DATA / 'train_features.csv', dtype={'note_id': str, 'recording_id': str, 'sample_id': str})
assert notes['split'].eq('train').all()
v2 = pd.read_csv(DATA / 'proposed_v2' / 'train_targets_v2.csv', dtype={'note_id': str})
v2 = notes[['note_id']].merge(v2, on='note_id', how='left', validate='one_to_one')
manifest = pd.read_csv(DATA / 'training_recording_manifest.csv')
refs = pd.read_csv(DATA / 'references_train.csv', dtype={'note_id': str, 'recording_id': str})
folds = pd.read_csv(DATA / 'folds.csv').set_index('sample_id')['fold']
fold_all = notes['sample_id'].map(folds).to_numpy()
assert not np.isnan(fold_all).any()
FOLDS = sorted(np.unique(fold_all))
rec_fold = manifest['sample_id'].map(folds)
audio_end = notes['recording_id'].map(manifest.set_index('recording_id')['duration_seconds_if_validated']).to_numpy()

cache = HERE / 'le_xin_cache'
audio = notes[['note_id']].merge(pd.read_csv(cache / 'audio_features_train_v1.csv', dtype={'note_id': str}),
                                 on='note_id', how='left', validate='one_to_one')
pitchf = notes[['note_id']].merge(pd.read_csv(cache / 'pitch_features_train_v1.csv', dtype={'note_id': str}),
                                  on='note_id', how='left', validate='one_to_one')
X = notes[cu.NOTE_FEATURES].astype(float)
XAP = pd.concat([X, audio.drop(columns='note_id'), pitchf.drop(columns='note_id')], axis=1)
assert XAP.shape == (len(notes), 31) and XAP.notna().all().all()
XAP = XAP.to_numpy()

v2_lab = v2['relationship'].eq('one_to_one').to_numpy()
v2_ts, v2_te = v2['reference_start'].to_numpy(float), v2['reference_end'].to_numpy(float)
v2_ref = np.where(v2_lab, notes['recording_id'] + ':ref' + v2['reference_index'].fillna(0).astype(int).map('{:05d}'.format), '')
eligible = v2['relationship'].isin(['one_to_one', 'ambiguous']).to_numpy()   # spurious/split candidates left as in v2
assert v2_lab.sum() == 3697

# ---------- matching rule (v2 rule: mutual-best overlap, checks 2-5, safety filter) ----------
def match(pos):
    """pos: DataFrame aligned with `notes` (start, end; pitch = BP pitch). Returns reference note_id or '' per note,
    using the v2 rule on these positions: mutual best overlap, overlap >= 60% of the shorter note, lengths within 3x,
    starts within 0.20 s, pitch within 12; safety filter: 2nd-largest overlap <= 30% of the note, pitch within 2."""
    out = np.full(len(notes), '', dtype=object)
    ref_by = dict(tuple(refs.groupby('recording_id')))
    for rid, idx in notes.groupby('recording_id').indices.items():
        ref = ref_by.get(rid)
        if ref is None:
            continue
        P = pos.iloc[idx][['start', 'end']].to_numpy(); pp = notes['pitch'].to_numpy()[idx]
        Q = ref[['start', 'end']].to_numpy(); qp = ref['pitch'].to_numpy(); qid = ref['note_id'].to_numpy()
        O = np.clip(np.minimum(P[:, None, 1], Q[None, :, 1]) - np.maximum(P[:, None, 0], Q[None, :, 0]), 0, None)
        best_ref, best_pred = O.argmax(1), O.argmax(0)
        for i in range(len(P)):
            j = best_ref[i]; o = O[i, j]
            if o <= 0 or best_pred[j] != i:
                continue
            dp, dr = P[i, 1] - P[i, 0], Q[j, 1] - Q[j, 0]; short = min(dp, dr)
            if o / short < .6 or max(dp, dr) / short > 3 or abs(P[i, 0] - Q[j, 0]) > .2 or abs(pp[i] - qp[j]) > 12:
                continue
            second = np.sort(O[i])[::-1][1] if O.shape[1] > 1 else 0.
            if second / dp > .30 or abs(pp[i] - qp[j]) > 2:
                continue
            out[idx[i]] = qid[j]
    return out

ref_t = refs.set_index('note_id')[['start', 'end']]

def labels_from(matched, scheme):
    """Return (ref_id per note or '', usable mask). scheme 'union' = v2 labels + new matches for v2-unlabelled
    eligible notes (dropped if that reference already has a v2 label); 'rematch' = only the new matching."""
    m = np.where(eligible, matched, '')
    if scheme == 'rematch':
        ids = m
    else:
        taken = set(v2_ref[v2_lab])
        new = (~v2_lab) & (m != '') & ~pd.Series(m).isin(taken).to_numpy()
        ids = np.where(v2_lab, v2_ref, np.where(new, m, ''))
    # a reference note may label only one BP note
    s = pd.Series(ids); dup = s.ne('') & s.duplicated(keep=False)
    ids = np.where(dup.to_numpy(), '', ids)
    return ids, ids != ''

def targets(ids, lab):
    t = ref_t.reindex(np.where(lab, ids, None))
    return t['start'].to_numpy(float), t['end'].to_numpy(float)

# ---------- timing model recipe (notebook Step 5c, n_jobs=4) ----------
make_rf = lambda: RandomForestRegressor(n_estimators=150, min_samples_leaf=20, max_features=0.6, n_jobs=4, random_state=0)
START_GATES = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
END_GATES = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 1.0]

def apply_shifts(shift, gate_start=0.0, gate_end=0.0):
    s = shift.copy()
    s[np.abs(s[:, 0]) < gate_start, 0] = 0
    s[np.abs(s[:, 1]) < gate_end, 1] = 0
    out = notes[cu.NOTE_COLUMNS].copy()
    out['start'] = np.maximum(0, notes['start'] + s[:, 0])
    out['end'] = np.minimum(notes['end'] + s[:, 1], audio_end)
    out['end'] = np.maximum(out['end'], out['start'] + 0.01)
    out['start'] = np.minimum(out['start'], out['end'] - 0.01)
    return out

def oof(ys, lab, fold_set):
    """Out-of-fold shifts within fold_set (train on the others in fold_set). ys = (target_start, target_end)."""
    y = [ys[0] - notes['start'].to_numpy(), ys[1] - notes['end'].to_numpy()]
    pred = np.zeros((len(notes), 2))
    for k in fold_set:
        test = fold_all == k
        train = np.isin(fold_all, [o for o in fold_set if o != k]) & lab
        for col in range(2):
            pred[test, col] = make_rf().fit(XAP[train], y[col][train]).predict(XAP[test])
    return pred

def f1_on(fold_set, shift, gs, ge):
    keep = manifest[rec_fold.isin(fold_set)]
    c = apply_shifts(shift, gs, ge)
    c = c[c['recording_id'].isin(keep['recording_id'])]
    r = refs[refs['recording_id'].isin(keep['recording_id'])]
    s = cu.evaluate_end_to_end(c, r, keep)['summary']
    return s['song_macro_f1'], s['song_macro_offset_f1']

def run(scheme, fixed=None):
    """Outer 5-fold by song with nested gate. scheme 'v2' = old labels; 'union'/'rematch' = labels built per outer
    fold from strictly nested first-pass corrections; fixed=(ids, lab) = one global label set (simple variant)."""
    shift, gated, picked, nlab = np.zeros((len(notes), 2)), np.zeros((len(notes), 2)), {}, {}
    for k in FOLDS:
        others = [j for j in FOLDS if j != k]
        in_others = np.isin(fold_all, others)
        if scheme == 'v2':
            ys, lab = (v2_ts, v2_te), v2_lab & in_others
        elif fixed is not None:
            ys, lab = targets(*fixed), fixed[1] & in_others
        else:
            first = oof((v2_ts, v2_te), v2_lab, others)            # never sees fold k
            ids, lab = labels_from(match(apply_shifts(first)), scheme)
            lab = lab & in_others                                   # fold k rows: no label, not used
            ys = targets(ids, lab)
        nlab[int(k)] = int(lab.sum())
        test = fold_all == k
        y = [ys[0] - notes['start'].to_numpy(), ys[1] - notes['end'].to_numpy()]
        for col in range(2):
            shift[test, col] = make_rf().fit(XAP[lab], y[col][lab]).predict(XAP[test])
        inner = oof(ys, lab, others)
        gs = max(START_GATES, key=lambda g: f1_on(others, inner, g, 0.0)[0])
        ge = max(END_GATES, key=lambda g: f1_on(others, inner, gs, g)[1])
        picked[int(k)] = (gs, ge)
        gated[test] = shift[test]
        gated[test & (np.abs(shift[:, 0]) < gs), 0] = 0
        gated[test & (np.abs(shift[:, 1]) < ge), 1] = 0
        print(f'  [{scheme}] fold {k}: {lab.sum()} training labels, gate {gs*1000:.0f}/{ge*1000:.0f} ms', flush=True)
    return shift, gated, picked, nlab

truth_v2 = np.column_stack([v2_ts, v2_te])[v2_lab]
def score(name, corrected, extra=None):
    task = cu.timing_metrics(truth_v2, corrected.loc[v2_lab, ['start', 'end']].to_numpy())
    e2e = cu.evaluate_end_to_end(corrected, refs, manifest)['summary']
    assert e2e['recordings'] == len(manifest)
    row = dict(model=name, onset_F1=round(e2e['song_macro_f1'], 4), offset_F1=round(e2e['song_macro_offset_f1'], 4),
               onset_mae_ms_v2labelled=round(task['onset_mae_seconds'] * 1000, 1),
               offset_mae_ms_v2labelled=round(task['offset_mae_seconds'] * 1000, 1))
    row.update(extra or {})
    print(row, flush=True)
    return row

if __name__ == '__main__':
    t0 = time.time()
    report = {}
    # 0. sanity: the matcher on RAW positions should give (almost) the v2 label set
    raw_ids, raw_lab = labels_from(match(notes), 'rematch')
    report['raw_rematch_labels'] = int(raw_lab.sum())
    report['raw_rematch_same_as_v2'] = int((raw_lab & v2_lab & (raw_ids == v2_ref)).sum())
    print('sanity raw rematch:', report, flush=True)

    # 1. counts + agreement when matching on the existing OOF timing export (gated) and on ungated OOF shifts
    export = pd.read_csv(HERE / 'exports' / 'le_xin_timing_rf_audio_pitch_v2_nestedgate.csv', dtype={'note_id': str, 'recording_id': str})
    assert export['note_id'].tolist() == notes['note_id'].tolist()
    oof_v2 = oof((v2_ts, v2_te), v2_lab, FOLDS)                    # ungated OOF (= notebook shift_rf_ap)
    sets = {}
    for src, pos in [('export_gated', export), ('oof_ungated', apply_shifts(oof_v2))]:
        m = match(pos)
        for scheme in ['rematch', 'union']:
            ids, lab = labels_from(m, scheme)
            sets[(src, scheme)] = (ids, lab)
            both = lab & v2_lab
            ts, te = targets(ids, lab)
            report[f'{src}_{scheme}'] = dict(
                labels=int(lab.sum()), new_vs_v2=int((lab & ~v2_lab).sum()), v2_lost=int((v2_lab & ~lab).sum()),
                both=int(both.sum()), same_reference_where_both=int((both & (ids == v2_ref)).sum()),
                new_label_start_shift_median_ms=round(float(np.median((ts - notes['start'].to_numpy())[lab & ~v2_lab])) * 1000, 1),
                new_label_abs_start_shift_gt50ms=round(float(np.mean(np.abs(ts - notes['start'].to_numpy())[lab & ~v2_lab] > .05)), 3),
                v2_label_abs_start_shift_gt50ms=round(float(np.mean(np.abs(v2_ts - notes['start'].to_numpy())[v2_lab] > .05)), 3))
            print(src, scheme, report[f'{src}_{scheme}'], flush=True)
    # save the label sets (export-based, for inspection only)
    for (src, scheme), (ids, lab) in sets.items():
        ts, te = targets(ids, lab)
        pd.DataFrame(dict(note_id=notes['note_id'], reference_note_id=ids, reference_start=ts, reference_end=te,
                          in_v2=v2_lab)).to_csv(OUT / f'labels_{src}_{scheme}.csv', index=False)

    # 2. retrain + score, all training recordings, out-of-fold by song, nested gate
    rows = [score('BP unchanged', notes[cu.NOTE_COLUMNS].copy())]
    runs = {}
    for name, scheme, fixed in [('v2 labels (old)', 'v2', None),
                                ('union labels, strict nested', 'union', None),
                                ('rematch-only labels, strict nested', 'rematch', None),
                                ('union labels from export (simple, weak leak)', 'fixed', sets[('export_gated', 'union')])]:
        shift, gated, picked, nlab = run(scheme, fixed)
        runs[name] = gated
        rows.append(score(name + ', ungated', apply_shifts(shift)))
        rows.append(score(name + ', nested gate', apply_shifts(gated),
                          dict(train_labels_per_fold=json.dumps(nlab), gates_ms=json.dumps({k: [round(a*1000), round(b*1000)] for k, (a, b) in picked.items()}))))
        pd.DataFrame(rows).to_csv(OUT / 'results.csv', index=False)
    np.save(OUT / 'gated_shifts.npy', np.stack([runs[n] for n in runs]))
    (OUT / 'report.json').write_text(json.dumps(report, indent=1))
    print(pd.DataFrame(rows).to_string())
    print(f'done in {time.time()-t0:.0f} s')

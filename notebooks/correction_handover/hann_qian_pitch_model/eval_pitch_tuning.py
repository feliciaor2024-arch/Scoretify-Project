"""Pitch correction: does adding singer-tuning features to pyin help?

Song-grouped 5-fold CV on train; confidence cutoff chosen by inner CV on the
training folds only. Trained on v2 labels; per-note scores on held-out
v2-labelled held-out notes (same rows for every setup); end-to-end via correction_utils.
Usage: python eval_pitch_tuning.py <GTSinger_100_unique_sample> <scratchpad>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT, S = Path(sys.argv[1]), Path(sys.argv[2])
DATA = ROOT / 'basic_pitch_runs' / 'correction_handover'
sys.path.insert(0, str(ROOT / 'notebooks' / 'correction_handover'))
import correction_utils as cu

n = pd.read_csv(DATA / 'train_features.csv', dtype={'note_id': str, 'recording_id': str, 'sample_id': str})
n = n.merge(pd.read_csv(S / 'train_pyin_features.csv', dtype={'note_id': str}), on='note_id', how='left', validate='one_to_one')
n = n.sort_values(['recording_id', 'start']).reset_index(drop=True)
g = n.groupby('recording_id')
n['prev_int'] = n.pitch - g.pitch.shift(1)
n['next_int'] = g.pitch.shift(-1) - n.pitch
n['rec_med'] = n.pitch - g.pitch.transform('median')
# Singer tuning: each recording's typical offset (voiced notes, duration-weighted median)
v = n[n.f0_has.eq(1)]
tune = v.groupby('recording_id').apply(lambda d: np.median(np.repeat(d.f0_cents_med.to_numpy(), np.maximum(1, (d.duration * 100).astype(int)))), include_groups=False)
n['rec_tune'] = n.recording_id.map(tune).fillna(0)
n['cents_detuned'] = np.where(n.f0_has.eq(1), n.f0_cents_med - n.rec_tune, 0)
n['prev_cents'] = g.f0_cents_med.shift(1) if False else n.groupby('recording_id').f0_cents_med.shift(1)
n['next_cents'] = n.groupby('recording_id').f0_cents_med.shift(-1)
n['toward_prev'] = -np.sign(n.prev_int) * n.cents_detuned  # >0: leaning back toward previous note
n = n.fillna(0)

BASE = cu.NOTE_FEATURES
CTX = BASE + ['prev_int', 'next_int', 'rec_med']
PYIN = ['f0_voiced_frac', 'f0_cents_med', 'f0_cents_iqr', 'f0_frac_up', 'f0_frac_down', 'f0_has']
TUNE = ['rec_tune', 'cents_detuned', 'prev_cents', 'next_cents', 'toward_prev']
SETUPS = {'+ context + pyin': CTX + PYIN, '+ context + pyin + tuning': CTX + PYIN + TUNE}

folds = pd.read_csv(DATA / 'folds.csv', dtype={'sample_id': str}).set_index('sample_id')['fold']
fold = n.sample_id.map(folds).to_numpy()
t2 = pd.read_csv(DATA / 'proposed_v2' / 'train_targets_v2.csv', dtype={'note_id': str}).set_index('note_id').loc[n.note_id]
y = (t2.pitch_delta.clip(-1, 1) + 1).to_numpy()  # 0,1,2 = -1,0,+1 ; NaN = unlabelled
use = np.isfinite(y)
ev = t2.pitch_delta.notna().to_numpy()
truth_all = t2.pitch_delta.to_numpy()
CUTS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]  # 1.01 = never change


def model():
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, random_state=0, n_jobs=-1)


def deltas(P, cut):
    cls = P.argmax(1) - 1
    return np.where((cls != 0) & (P.max(1) >= cut), cls, 0)


def net(P, rows, cut):
    d = deltas(P[rows], cut); t = truth_all[rows]
    return int(((d != 0) & (d == t)).sum()) - int(((d != 0) & (d != t)).sum())


manifest = pd.read_csv(DATA / 'training_recording_manifest.csv', dtype={'recording_id': str, 'sample_id': str})
refs = pd.read_csv(DATA / 'references_train.csv', dtype={'note_id': str, 'recording_id': str})
results = []
for name, feats in SETUPS.items():
    X = n[feats].to_numpy(float)
    fixed = broken = 0; f1s = []; cuts = []
    for k in range(5):
        tr, te = use & (fold != k), fold == k
        # inner CV on the 4 training folds to pick the cutoff
        Pin = np.zeros((len(n), 3))
        for j in set(range(5)) - {k}:
            a, b = use & (fold != k) & (fold != j), (fold == j)
            Pin[b] = model().fit(X[a], y[a].astype(int)).predict_proba(X[b])
        inner = ev & (fold != k)
        cut = max(CUTS, key=lambda c: (net(Pin, inner, c), c))
        cuts.append(cut)
        P = model().fit(X[tr], y[tr].astype(int)).predict_proba(X[te])
        d = deltas(P, cut)
        rows = ev[te]; t = truth_all[te][rows]; dd = d[rows]
        fixed += int(((dd != 0) & (dd == t)).sum()); broken += int(((dd != 0) & (dd != t)).sum())
        corrected = n.loc[te, cu.NOTE_COLUMNS].copy()
        corrected['pitch'] = corrected['pitch'] + d
        fm = manifest[manifest.sample_id.map(folds).eq(k)]
        f1s.append(cu.evaluate_end_to_end(corrected, refs[refs.recording_id.isin(fm.recording_id)], fm)['summary']['song_macro_f1'])
    acc = ((truth_all[ev] == 0).sum() + fixed - broken) / ev.sum()
    results.append(dict(setup=name, fixed=fixed, broken=broken, net=fixed - broken,
                        pitch_acc=round(acc, 4), song_macro_f1=round(np.mean(f1s), 4), cutoffs=cuts))

# no-change baseline
f1s = []
for k in range(5):
    fm = manifest[manifest.sample_id.map(folds).eq(k)]
    f1s.append(cu.evaluate_end_to_end(n.loc[fold == k, cu.NOTE_COLUMNS], refs[refs.recording_id.isin(fm.recording_id)], fm)['summary']['song_macro_f1'])
results.insert(0, dict(setup='no change', fixed=0, broken=0, net=0,
                       pitch_acc=round((truth_all[ev] == 0).mean(), 4), song_macro_f1=round(np.mean(f1s), 4), cutoffs=None))
pd.set_option('display.width', 200)
print(pd.DataFrame(results).to_string(index=False))
print('wrong-pitch notes available to fix:', int((truth_all[ev] != 0).sum()), 'of', int(ev.sum()))

# How well does the pyin offset alone point the right way on wrong notes?
lab = use & np.isfinite(y)
wrong = lab & (y != 1)
dirn = np.sign(n.f0_cents_med.to_numpy())
print('pyin sign matches true direction on wrong notes:',
      round(float((dirn[wrong] == (y[wrong] - 1)).mean()), 3), 'n =', int(wrong.sum()))
for lo, hi in [(0, 25), (25, 50), (50, 100), (100, 600)]:
    m = lab & (np.abs(n.f0_cents_med) >= lo) & (np.abs(n.f0_cents_med) < hi)
    print(f'|cents| {lo}-{hi}: notes {int(m.sum())}, share wrong {((y[m] != 1).mean() if m.any() else 0):.3f}')

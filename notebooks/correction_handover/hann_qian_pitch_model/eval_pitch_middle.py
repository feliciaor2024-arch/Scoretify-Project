"""Pitch correction: whole-note vs middle-of-note pyin features.

Builds per-note pyin features from the cached tracks (f0_tracks.npz) at several trims
(fraction cut from EACH end of the note; 0 = whole note). Notes with < MIN_FRAMES voiced
frames in the trimmed window fall back to the whole note.
Nested, song-grouped 5-fold CV on train with v2 labels only: for each held-out fold the trim
AND the confidence cutoff are chosen by inner CV on the other four folds.
Usage: python eval_pitch_middle.py <GTSinger_100_unique_sample>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from xgboost import XGBClassifier

ROOT = Path(sys.argv[1])
HERE = Path(__file__).resolve().parent
DATA = ROOT / 'basic_pitch_runs' / 'correction_handover'
sys.path.insert(0, str(ROOT / 'notebooks' / 'correction_handover'))
import correction_utils as cu

TRIMS = [0.0, 0.1, 0.2, 0.3]
MIN_FRAMES = 3
CUTS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]  # 1.01 = never change

n = pd.read_csv(DATA / 'train_features.csv', dtype={'note_id': str, 'recording_id': str, 'sample_id': str})
n = n.sort_values(['recording_id', 'start']).reset_index(drop=True)
tracks = np.load(HERE / 'f0_tracks.npz')


def pyin_feats(trim):
    rows = []
    for r in n.itertuples():
        t, f0 = tracks[f'{r.recording_id}_t'], tracks[f'{r.recording_id}_f0']
        whole = f0[(t >= r.start) & (t < r.end)]
        cut = trim * (r.end - r.start)
        w = f0[(t >= r.start + cut) & (t < r.end - cut)]
        if np.isfinite(w).sum() < MIN_FRAMES:
            w = whole
        v = w[np.isfinite(w)]
        if len(v) == 0:
            rows.append((0.0, 0.0, 0.0, 0.0, 0.0, 0)); continue
        c = 1200 * np.log2(v / librosa.midi_to_hz(r.pitch))
        c = (c + 600) % 1200 - 600  # fold octave errors
        rows.append((len(v) / max(len(w), 1), float(np.median(c)), float(np.subtract(*np.percentile(c, [75, 25]))),
                     float((c > 50).mean()), float((c < -50).mean()), 1))
    return pd.DataFrame(rows, columns=['f0_voiced_frac', 'f0_cents_med', 'f0_cents_iqr', 'f0_frac_up', 'f0_frac_down', 'f0_has'])


g = n.groupby('recording_id')
n['prev_int'] = (n.pitch - g.pitch.shift(1)).fillna(0)
n['next_int'] = (g.pitch.shift(-1) - n.pitch).fillna(0)
n['rec_med'] = n.pitch - g.pitch.transform('median')
CTX = cu.NOTE_FEATURES + ['prev_int', 'next_int', 'rec_med']
Xs = {trim: np.hstack([n[CTX].to_numpy(float), pyin_feats(trim).to_numpy(float)]) for trim in TRIMS}

folds = pd.read_csv(DATA / 'folds.csv', dtype={'sample_id': str}).set_index('sample_id')['fold']
fold = n.sample_id.map(folds).to_numpy()
t2 = pd.read_csv(DATA / 'proposed_v2' / 'train_targets_v2.csv', dtype={'note_id': str}).set_index('note_id').loc[n.note_id]
truth = t2.pitch_delta.to_numpy()
use = np.isfinite(truth)
y = np.clip(truth, -1, 1) + 1  # 0,1,2 = -1,0,+1


def model():
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, random_state=0, n_jobs=4)


def deltas(P, cut):
    cls = P.argmax(1) - 1
    return np.where((cls != 0) & (P.max(1) >= cut), cls, 0)


def score(d, rows):
    t = truth[rows]; d = d[rows]
    return int(((d != 0) & (d == t)).sum()), int(((d != 0) & (d != t)).sum())


def oof(X, train_mask, folds_used):
    P = np.zeros((len(n), 3))
    for j in folds_used:
        a, b = train_mask & (fold != j), fold == j
        P[b] = model().fit(X[a], y[a].astype(int)).predict_proba(X[b])
    return P


manifest = pd.read_csv(DATA / 'training_recording_manifest.csv', dtype={'recording_id': str, 'sample_id': str})
refs = pd.read_csv(DATA / 'references_train.csv', dtype={'note_id': str, 'recording_id': str})

# 1) Each trim on its own (cutoff chosen by inner CV) - for transparency
per_trim = {}
# 2) Nested: trim + cutoff chosen by inner CV
nested = dict(fixed=0, broken=0, f1=[], picks=[])
for k in range(5):
    tr, te = use & (fold != k), fold == k
    inner_rows = use & (fold != k)
    best = None
    for trim in TRIMS:
        Pin = oof(Xs[trim], use & (fold != k), set(range(5)) - {k})
        for cut in CUTS:
            fx, br = score(deltas(Pin, cut), inner_rows)
            key = (fx - br, cut)
            per_trim.setdefault((k, trim), (None, -1e9))
            if key > (per_trim[(k, trim)][1], per_trim[(k, trim)][0] or 0):
                per_trim[(k, trim)] = (cut, fx - br)
            if best is None or key > (best[2], best[1]):
                best = (trim, cut, fx - br)
    trim, cut, _ = best
    nested['picks'].append((trim, cut))
    P = model().fit(Xs[trim][tr], y[tr].astype(int)).predict_proba(Xs[trim][te])
    d = np.zeros(len(n)); d[te] = deltas(P, cut)
    fx, br = score(d, te & use)
    nested['fixed'] += fx; nested['broken'] += br
    corrected = n.loc[te, cu.NOTE_COLUMNS].copy(); corrected['pitch'] = corrected['pitch'] + d[te]
    fm = manifest[manifest.sample_id.map(folds).eq(k)]
    nested['f1'].append(cu.evaluate_end_to_end(corrected, refs[refs.recording_id.isin(fm.recording_id)], fm)['summary']['song_macro_f1'])

rows = []
for trim in TRIMS:
    fixed = broken = 0
    for k in range(5):
        tr, te = use & (fold != k), fold == k
        cut = per_trim[(k, trim)][0]
        P = model().fit(Xs[trim][tr], y[tr].astype(int)).predict_proba(Xs[trim][te])
        d = np.zeros(len(n)); d[te] = deltas(P, cut)
        fx, br = score(d, te & use); fixed += fx; broken += br
    rows.append(dict(setup=f'trim {int(trim*100)}% each end' if trim else 'whole note', fixed=fixed, broken=broken,
                     net=fixed - broken, pitch_acc=round(((truth[use] == 0).sum() + fixed - broken) / use.sum(), 4)))
rows.append(dict(setup='NESTED (trim+cutoff picked on train folds)', fixed=nested['fixed'], broken=nested['broken'],
                 net=nested['fixed'] - nested['broken'],
                 pitch_acc=round(((truth[use] == 0).sum() + nested['fixed'] - nested['broken']) / use.sum(), 4)))
pd.set_option('display.width', 200)
print(pd.DataFrame(rows).to_string(index=False))
print('nested picks per fold (trim, cutoff):', nested['picks'])
print('nested song_macro_f1:', round(float(np.mean(nested['f1'])), 4))
print('wrong-pitch notes:', int((truth[use] != 0).sum()), 'of', int(use.sum()), '| no-change pitch_acc', round(float((truth[use] == 0).mean()), 4))

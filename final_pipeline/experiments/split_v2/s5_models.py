"""Step 5: train split models on the richer clues, out-of-fold by song with a nested threshold, and score end-to-end.
Usage: python s5_models.py old|v2 name[,name...]   (names below; results appended to results.csv)"""
import sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import common as C
import clues as K
from exp import Experiment

which, names = sys.argv[1], sys.argv[2].split(',')
manifest, references, folds, raw, timed = C.load()
scorer = C.Scorer(manifest, references)
cands = pd.read_csv(C.CACHE / f'clues_{which}_train.csv', dtype={'note_id': str, 'recording_id': str, 'sample_id': str})
if 'label' not in cands:
    raise SystemExit('no labels')
cands = K.add_comparison(C.add_clues(cands))
cands['note_idx'] = pd.factorize(cands.note_id)[0]
TIMBRE = ['spec9', 'spec_near', 'hf_peak', 'hf_rise', 'zcr_peak', 'zcr_rise']
if 'spec9' in cands:
    for col in ['spec9', 'hf_rise']:
        cands[f'{col}_rank'] = cands.groupby('note_id')[col].rank(ascending=False, method='min')
        cands[f'{col}_vs_max'] = cands[col] - cands.groupby('note_id')[col].transform('max')
    TIMBRE += ['spec9_rank', 'spec9_vs_max', 'hf_rise_rank', 'hf_rise_vs_max']
truth = {k: v for rid, g in timed.groupby('recording_id')
         for k, v in C.true_cut_times(g, references[references.recording_id == rid]).items()}
needed = sum(map(len, truth.values()))
E = Experiment(cands, timed, manifest, folds, scorer)

RICH = C.OLD_FEATS + K.NEW_BASE + K.COMP_FEATS
NOTE_AGG = ['change100', 'settle_new', 'round_change100', 'spec_change', 'valley', 'bp_onset_new', 'bp_new_rise',
            'flux', 'pitch_change', 'piece_change']
NOTE_COLS = ['note_dur', 'note_pitch', 'cands_in_note', 'strong_in_note', 'note_stable_segs', 'note_seg_range',
             'note_pitch_spread', 'note_unvoiced', 'gap_before_note', 'gap_after_note']


def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                  class_weight='balanced_subsample', n_jobs=4, random_state=0)


def hgb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                          l2_regularization=1.0, class_weight='balanced', random_state=0)


class TwoStage:
    """Stage 1 (note level): does this note need a cut at all? Stage 2 (candidate level): is this candidate the cut?
    Score = P(note needs a cut) x P(candidate is a real cut). X's last column is the note index (grouping only)."""
    def __init__(self, make_cand, feats):
        self.make_cand, self.feats = make_cand, feats
        self.ia = [feats.index(c) for c in NOTE_AGG]
        self.ic = [feats.index(c) for c in NOTE_COLS]

    def note_table(self, X):
        df = pd.DataFrame(X[:, self.ia], columns=NOTE_AGG)
        df['g'] = X[:, -1]
        agg = df.groupby('g').agg(['max', 'mean'])
        agg.columns = [f'{a}_{b}' for a, b in agg.columns]
        top = pd.DataFrame(X[:, self.ic], columns=NOTE_COLS).assign(g=X[:, -1]).groupby('g').first()
        return pd.concat([agg, top], axis=1)

    def fit(self, X, y):
        nt = self.note_table(X)
        ny = pd.Series(y).groupby(X[:, -1]).max().loc[nt.index]
        self.note_model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                                 class_weight='balanced_subsample', n_jobs=4, random_state=0).fit(nt, ny)
        self.cand_model = self.make_cand().fit(X[:, :-1], y)
        return self

    def predict_proba(self, X):
        nt = self.note_table(X)
        pn = pd.Series(self.note_model.predict_proba(nt)[:, 1], nt.index).loc[X[:, -1]].to_numpy()
        p = self.cand_model.predict_proba(X[:, :-1])[:, 1] * pn
        return np.column_stack([1 - p, p])


class Avg:
    """Average of a random forest and a gradient-boosting model."""
    def fit(self, X, y):
        self.ms = [rf().fit(X, y), hgb().fit(X, y)]
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.ms], axis=0)


TH_WIDE = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def top1(p, t):
    """Only the most likely candidate of each note may be cut (at most one cut per note)."""
    best = pd.Series(p).groupby(cands.note_idx.to_numpy()).transform('max').to_numpy()
    return (p >= t) & (p == best)


RUNS = {
    'rf_old19':   (rf, C.OLD_FEATS, None, None),
    'rf_rich':    (rf, RICH, None, None),
    'hgb_old19':  (hgb, C.OLD_FEATS, None, None),
    'hgb_rich':   (hgb, RICH, None, None),
    'rf_rich_top1': (rf, RICH, top1, None),
    'rf_richt':   (rf, RICH + TIMBRE, None, None),
    'hgb_richt':  (hgb, RICH + TIMBRE, None, None),
    'avg_richt':  (Avg, RICH + TIMBRE, None, None),
    'avg_rich':   (Avg, RICH, None, None),
    'two_rf':     (lambda: TwoStage(rf, RICH), RICH + ['note_idx'], None, TH_WIDE),
    'two_hgb':    (lambda: TwoStage(hgb, RICH), RICH + ['note_idx'], None, TH_WIDE),
}

if __name__ == '__main__':
    for name in names:
        t0 = time.time()
        make, feats, decide, ths = RUNS[name]
        cut, chosen, prob = E.nested(make, cands[feats].to_numpy(float), decide, ths or TH_WIDE[4:] + [0.95])
        r = E.row(f'{which}:{name}', cut, needed)
        r.update(thresholds=str(chosen), candidates=len(cands), candidate_recall=round(E.y.sum() / needed, 3),
                 minutes=round((time.time() - t0) / 60, 1))
        print(r, flush=True)
        np.save(C.CACHE / f'prob_{which}_{name}.npy', prob)
        np.save(C.CACHE / f'cut_{which}_{name}.npy', cut)
        out = C.OUT / 'results.csv'
        pd.DataFrame([r]).to_csv(out, mode='a', header=not out.exists(), index=False)

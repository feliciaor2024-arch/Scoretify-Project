"""Out-of-fold (by song) training + nested threshold choice, same recipe as the split notebook's Step 2."""
import numpy as np
import pandas as pd
import common as C

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


class Experiment:
    def __init__(self, cands, timed, manifest, folds, scorer):
        self.cands, self.timed, self.man, self.score = cands.reset_index(drop=True), timed, manifest, scorer
        self.cfold = self.cands.sample_id.map(folds).to_numpy()
        self.rfold = manifest.sample_id.map(folds)
        self.ALL = sorted(np.unique(self.cfold))
        self.y = self.cands.label.to_numpy()

    def f1_on(self, fold_set, cut):
        recs = self.man[self.rfold.isin(fold_set)].recording_id
        keep = self.timed.recording_id.isin(recs)
        table = C.apply_splits(self.timed[keep], self.cands[cut & np.isin(self.cfold, fold_set)])
        return self.score(table, recs)

    def oof_prob(self, make_model, X, fold_set):
        p = np.zeros(len(self.cands))
        for k in fold_set:
            test, train = self.cfold == k, np.isin(self.cfold, [o for o in fold_set if o != k])
            p[test] = make_model().fit(X[train], self.y[train]).predict_proba(X[test])[:, 1]
        return p

    def nested(self, make_model, X, decide=None, thresholds=THRESHOLDS):
        """decide(prob, th) -> cut mask (default prob >= th). Threshold chosen on the other 4 folds only."""
        decide = decide or (lambda p, t: p >= t)
        outer, cut, chosen = self.oof_prob(make_model, X, self.ALL), np.zeros(len(self.cands), bool), {}
        for k in self.ALL:
            others = [o for o in self.ALL if o != k]
            inner = self.oof_prob(make_model, X, others)
            th = max(thresholds, key=lambda t: self.f1_on(others, decide(inner, t))[0])
            chosen[int(k)] = th
            cut |= (self.cfold == k) & decide(outer, th)
        return cut, chosen, outer

    def row(self, name, cut, needed):
        table = C.apply_splits(self.timed, self.cands[cut])
        on, off = self.score(table)
        tp = int((cut & (self.y == 1)).sum())
        return dict(model=name, cuts=int(cut.sum()), real_cuts=tp,
                    precision=round(tp / cut.sum(), 3) if cut.sum() else None,
                    found_of_needed=round(tp / needed, 3), onset_F1=round(on, 4), offset_F1=round(off, 4))

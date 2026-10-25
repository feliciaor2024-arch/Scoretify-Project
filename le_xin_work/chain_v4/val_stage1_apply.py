"""Chain v4 VALIDATION, stage 1: final split v2 model (RF + boosting average, v2 candidates, 74 richer clues) trained on
ALL training candidates, threshold from the training OOF scores, applied ONCE to the 20 val songs.
Val features: le_xin_work/experiments/split_v2/s8_val_features.py (no answer key). The val answer key (bp_experiment.references)
is used only for scoring at the end. Refuses to run a second time. Test is never opened; no TextGrid."""
import string, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'split_v2'))
import common as C
import clues as K
from exp import Experiment, THRESHOLDS
import bp_experiment as bx

OUT = Path(__file__).resolve().parent
TABLE, ACTS = OUT / 'val_stage1_split_v2.csv', OUT / 'val_stage1_split_v2_actions.csv'
assert not TABLE.exists() and not ACTS.exists(), 'val already run once; not running again'

RICH = C.OLD_FEATS + K.NEW_BASE + K.COMP_FEATS


class Avg:                                   # same as split_v2/s5_models.Avg
    def fit(self, X, y):
        self.ms = [RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                          class_weight='balanced_subsample', n_jobs=4, random_state=0).fit(X, y),
                   HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                                  l2_regularization=1.0, class_weight='balanced', random_state=0).fit(X, y)]
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.ms], axis=0)


# 1. Training side only: threshold = best all-train OOF onset F1 (same OOF probabilities as stage 1), then final fit
manifest, references, folds, raw, timed = C.load()
tr = K.add_comparison(C.add_clues(pd.read_csv(C.CACHE / 'clues_v2_train.csv',
                                              dtype={'note_id': str, 'recording_id': str, 'sample_id': str})))
E = Experiment(tr, timed, manifest, folds, C.Scorer(manifest, references))
oof = np.load(C.CACHE / 'prob_v2_avg_rich.npy')
scores = {t: E.f1_on(E.ALL, oof >= t)[0] for t in THRESHOLDS}
threshold = max(THRESHOLDS, key=scores.get)
print('training OOF onset F1 by threshold:', {t: round(s, 4) for t, s in scores.items()}, '-> threshold', threshold)
model = Avg().fit(tr[RICH].to_numpy(float), tr.label.to_numpy())

# 2. Val: features built without the answer key; apply once
all_man = pd.read_csv(C.DATA / 'recording_manifest.csv')
vman = all_man[all_man.split == 'val'].reset_index(drop=True)
assert len(vman) == 160 and vman.split.eq('val').all()
val_raw = pd.read_csv(C.DATA / 'val_features.csv', dtype={'note_id': str, 'recording_id': str, 'sample_id': str})
vtimed = pd.read_csv(C.HERE / 'exports' / 'le_xin_timing_val_rf_audio_pitch.csv', dtype={'note_id': str, 'recording_id': str})
vtimed = vtimed.merge(val_raw[['note_id', 'sample_id']], on='note_id', validate='one_to_one')
vc = K.add_comparison(C.add_clues(pd.read_csv(C.CACHE / 'clues_v2_val.csv',
                                              dtype={'note_id': str, 'recording_id': str, 'sample_id': str})))
assert vc.recording_id.isin(vman.recording_id).all()
vc['prob'] = model.predict_proba(vc[RICH].to_numpy(float))[:, 1]
chosen = vc[vc.prob >= threshold]

pieces = C.apply_splits(vtimed, chosen)                       # same pieces as train; id scheme <orig>_a, _b, ...
k = pieces.note_id.str.extract(r'^(.*?)(?:~(\d+))?$')
cut_ids = set(chosen.note_id)
orig, idx = k[0], k[1].fillna(0).astype(int)
pieces['note_id'] = [f'{o}_{string.ascii_lowercase[i]}' if o in cut_ids else o for o, i in zip(orig, idx)]
assert pieces.note_id.is_unique and idx.max() < 26

# 3. Score (val answer key used only here)
events, _ = bx.references(vman.to_dict('records'))
vrefs = pd.DataFrame([dict(note_id=f'{rid}:ref{i:05d}', recording_id=rid, start=e[0], end=e[1], pitch=e[2])
                      for rid, ev in events.items() for i, e in enumerate(ev)])
before = C.cu.evaluate_end_to_end(vtimed[C.cu.NOTE_COLUMNS], vrefs, vman)
after = C.cu.evaluate_end_to_end(pieces, vrefs, vman)
assert before['summary']['recordings'] == after['summary']['recordings'] == 160
b = {s['sample_id']: s['f1'] for s in before['songs']}
better = sum(s['f1'] > b[s['sample_id']] + 1e-12 for s in after['songs'])
worse = sum(s['f1'] < b[s['sample_id']] - 1e-12 for s in after['songs'])

pieces.to_csv(TABLE, index=False)
acts = chosen.sort_values(['note_id', 't'])[['note_id', 'recording_id', 't', 'prob']].rename(columns={'t': 'cut_time'})
acts['threshold'] = threshold
acts.to_csv(ACTS, index=False)
sb, sa = before['summary'], after['summary']
print(f"val candidates {len(vc)}, cuts {len(acts)} in {acts.note_id.nunique()} notes, notes {len(vtimed)} -> {len(pieces)}")
print(f"val input (timing): onset {sb['song_macro_f1']:.4f} offset {sb['song_macro_offset_f1']:.4f}")
print(f"val + split v2:     onset {sa['song_macro_f1']:.4f} offset {sa['song_macro_offset_f1']:.4f}")
print(f"songs better {better}, worse {worse}, same {len(after['songs']) - better - worse} (onset F1)")

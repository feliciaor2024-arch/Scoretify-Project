"""Step 2: rebuild the old split model (RF, 19 clues, nested threshold) from the cached v1 candidates."""
import time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
import common as C
from exp import Experiment

manifest, references, folds, raw, timed = C.load()
scorer = C.Scorer(manifest, references)
t0 = time.time()
print('timing only (fast scorer):', scorer(timed))
s = C.cu.evaluate_end_to_end(timed[C.cu.NOTE_COLUMNS], references, manifest)['summary']
print('timing only (cu):', s['song_macro_f1'], s['song_macro_offset_f1'], f'{time.time()-t0:.0f}s')
cands = C.add_clues(pd.read_csv(C.HERE / 'le_xin_cache' / 'split_candidates_train_v1.csv',
                                dtype={'note_id': str, 'recording_id': str, 'sample_id': str}))
truth = {k: v for rid, g in timed.groupby('recording_id')
         for k, v in C.true_cut_times(g, references[references.recording_id == rid]).items()}
needed = sum(map(len, truth.values()))
E = Experiment(cands, timed, manifest, folds, scorer)
rf = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                    class_weight='balanced_subsample', n_jobs=4, random_state=0)
cut, chosen, prob = E.nested(rf, cands[C.OLD_FEATS].to_numpy(float))
print('chosen', chosen, f'{time.time()-t0:.0f}s')
print(pd.DataFrame([E.row('old RF (19 clues)', cut, needed), E.row('ceiling', E.y == 1, needed)]))
table = C.apply_splits(timed, cands[cut])
s = C.cu.evaluate_end_to_end(table, references, manifest)['summary']
print('cu check:', round(s['song_macro_f1'], 4), round(s['song_macro_offset_f1'], 4))
np.save(C.CACHE / 'old_rf_oof_prob.npy', prob); np.save(C.CACHE / 'old_rf_cut.npy', cut)

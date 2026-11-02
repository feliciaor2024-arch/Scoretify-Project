"""Train the 7 final chain v4 models on ALL training songs and save them in models/ (with settings.json).

Same recipe, same training rows and same thresholds/margins as the scripts that made the chain v4 val tables:
  timing     notebooks/correction_handover/le_xin_timing_starter.ipynb Step 6 (RF + audio + pitch)
  split      chain_v4/val_stage1_apply.py
  delete     experiments/delete_v2/val_stage2.py
  start snap experiments/snap_v3/val_stage3.py
  end snap   experiments/end_snap/val_stage4.py
Thresholds, gates and margins were all chosen on training songs (out-of-fold); val and test play no part.
Needs the local training caches (not in git): le_xin_cache/*_train_v1.csv, split_v2/cache/clues_v2_train.csv,
delete_v2/features_chain_v4_stage1.csv, snap_v3/cache_v4/options_w150x.pkl, end_snap/end_candidates_train_v4.csv.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingClassifier
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
LX = HERE.parent
ROOT = LX.parent
HO = ROOT / 'notebooks' / 'correction_handover'
DATA = ROOT / 'basic_pitch_runs' / 'correction_handover'
EXP = LX / 'experiments'
OUT = HERE / 'models'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HO))
import correction_utils as cu
from pipeline import Avg

read = lambda p: pd.read_csv(p, dtype={'note_id': str, 'recording_id': str, 'sample_id': str})
settings = {}


def save(name, model):
    joblib.dump(model, OUT / f'{name}.joblib', compress=3)
    print(f'saved {name}: {(OUT / f"{name}.joblib").stat().st_size / 1e6:.1f} MB', flush=True)


# 1. Timing: 2 random forests (start shift, end shift) on 7 note + 10 loudness + 14 pitch clues
notes = read(DATA / 'train_features.csv')
targets = read(DATA / 'proposed_v2' / 'train_targets_v2.csv').rename(columns={
    'false_note': 'false_label', 'reference_start': 'target_start', 'reference_end': 'target_end', 'reference_pitch': 'target_pitch'})
assert notes.split.eq('train').all()
X, joined = cu.training_data(notes, targets, 'timing')
usable = joined[cu.TARGETS['timing']].notna().all(axis=1)
if 'ambiguous' in joined:
    usable &= ~joined['ambiguous'].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
audio = joined[['note_id']].merge(read(HO / 'le_xin_cache' / 'audio_features_train_v1.csv'), on='note_id', how='left', validate='one_to_one')
pitch = joined[['note_id']].merge(read(HO / 'le_xin_cache' / 'pitch_features_train_v1.csv'), on='note_id', how='left', validate='one_to_one')
clue_cols = [c for c in audio.columns if c != 'note_id'] + [c for c in pitch.columns if c != 'note_id']
XAP = pd.concat([X, audio.drop(columns='note_id'), pitch.drop(columns='note_id')], axis=1)
assert XAP.shape[1] == 31 and XAP.notna().all().all()
lab = usable.to_numpy()
print('timing: labelled training notes', int(lab.sum()))
for name, y in [('timing_start', joined.target_start - joined.start), ('timing_end', joined.target_end - joined.end)]:
    rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=20, max_features=0.6, n_jobs=-1, random_state=0)
    save(name, rf.fit(XAP[lab], y[lab]))
# gates = best all-train out-of-fold F1 (notebook Step 6 final_shift); printed there as 40 / 80 ms
settings['timing'] = dict(note_cols=list(X.columns), clue_cols=clue_cols, gate_start=0.04, gate_end=0.08)

# 2. Split: RF + gradient boosting, averaged; 74 clues; threshold 0.6 (best all-train OOF onset F1)
sys.path.insert(0, str(EXP / 'split_v2'))
import common as SC
import clues as SK
RICH = SC.OLD_FEATS + SK.NEW_BASE + SK.COMP_FEATS
tr = SK.add_comparison(SC.add_clues(read(SC.CACHE / 'clues_v2_train.csv')))
Xs, ys = tr[RICH].to_numpy(float), tr.label.to_numpy()
print('split: training candidates', len(tr))
save('split', Avg([
    RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.5, class_weight='balanced_subsample',
                           n_jobs=4, random_state=0).fit(Xs, ys),
    HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                   l2_regularization=1.0, class_weight='balanced', random_state=0).fit(Xs, ys)]))
settings['split'] = dict(clues=RICH, threshold=0.6)
for m in ['common', 'clues']:
    sys.modules.pop(m)
sys.path.remove(str(EXP / 'split_v2'))

# 3. Delete: gradient boosting on 49 clues; threshold 0.875 (best all-train OOF onset F1)
sys.path.insert(0, str(EXP / 'delete_v2'))
from stage2 import BASE, FEATS
tr = read(FEATS)
print('delete: training notes', len(tr))
save('delete', HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=20,
                                              l2_regularization=1., random_state=0)
     .fit(np.nan_to_num(tr[BASE].to_numpy(float), nan=-1), tr.y_official_pool.astype(int).to_numpy()))
settings['delete'] = dict(clues=list(BASE), threshold=0.875)
sys.modules.pop('common')
sys.path.remove(str(EXP / 'delete_v2'))

# 4. Start snap: LightGBM ranker on 47 clues; margins 0.10 (round 1) and 0.05 (round 2)
os.environ['SNAP_CACHE'] = 'cache_v4'
os.environ['SNAP_CHAIN'] = str(LX / 'chain_v4' / 'stage2_delete_v2_train_oof.csv')
sys.path.insert(0, str(EXP / 'snap_v3'))
import common as SN
from pickers import _groups
clues = SN.CLUES_V2 + SN.EXTRA_KIND_CLUES
tab = pd.read_pickle(SN.CACHE / 'options_w150x.pkl')
lab = tab[tab.labelled_note]
print('start snap: training options', len(lab), 'of', lab.note_id.nunique(), 'labelled notes')
rank = lgb.LGBMRanker(objective='lambdarank', n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=4,
                      random_state=0, verbose=-1)
save('snap_start', rank.fit(lab[clues], lab.label, group=_groups(lab.note_id)))
settings['snap_start'] = dict(clues=clues, margin_round1=0.10, margin_round2=0.05)

# 5. End snap: random forest on 45 clues; margin 0.0
train = read(EXP / 'end_snap' / 'end_candidates_train_v4.csv')
NOT_CLUES = {'note_id', 'recording_id', 't', 'label', 'labelled_note', '_true_end', 'fold'}
CLUES = [c for c in train.columns if c not in NOT_CLUES]
assert len(CLUES) == 45
lab = train.labelled_note.to_numpy()
print('end snap: training options', int(lab.sum()))
save('snap_end', RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                        class_weight='balanced_subsample', n_jobs=4, random_state=0)
     .fit(train.loc[lab, CLUES], train.label.to_numpy()[lab]))
margin = json.loads((EXP / 'end_snap' / 'end_snap_rf_margins_v4.json').read_text())['final_all_train']
settings['snap_end'] = dict(clues=CLUES, margin=margin)

settings['trained_on'] = 'training songs only (60 songs, 534 recordings); val and test never used for fitting or choosing'
(OUT / 'settings.json').write_text(json.dumps(settings, indent=1))
print('saved settings.json')

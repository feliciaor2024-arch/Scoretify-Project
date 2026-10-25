"""Chain v4 VALIDATION, stage 3: apply the final snap v3 model to the 20 val songs ONCE (no tuning on val).
Final ranker = LightGBM lambdarank (seed 0) trained on ALL labelled training options of the stage-3 run
(cache_v4/options_w150x.pkl, built on chain_v4/stage2_delete_v2_train_oof.csv). Margins = best onset F1 on all-train
out-of-fold scores (round 1 and round 2), as in stage3_README.md. Round 2 rebuilds options around the round-1 val
output and scores them with the same ranker (same as the out-of-fold training run). No answer key is used here.
Run:  SNAP_CHAIN=<stage2 train csv> SNAP_CACHE=cache_v4 python val_stage3.py"""
import os
import numpy as np, pandas as pd
assert os.environ.get('SNAP_CACHE') == 'cache_v4' and 'stage2_delete_v2_train_oof' in os.environ.get('SNAP_CHAIN', '')
import common
from common import *
from pickers import *

V4 = ROOT / 'le_xin_work' / 'chain_v4'
clues = CLUES_V2 + EXTRA_KIND_CLUES

# 1. final model + margins, from the training run only
tab = pd.read_pickle(CACHE / 'options_w150x.pkl')
outer1 = np.load(CACHE / 'lgbrank_w150x.npz')['outer']
r2 = pd.read_pickle(CACHE / 'lgbrank_w150x_round2.pkl')
fin_m1 = max(MARGINS, key=lambda g: f1_on(ALL, choose(outer1, g, tab))[0])
# round-1 out-of-fold output on train (the base of the round-2 options), rebuilt from the cached nested run
_, _, st1, _ = nested(fp_lgb_rank('lambdarank', clues, 0), tab, 'lgbrank_w150x')
base1_train = apply_starts(st1)
fin_m2 = max(MARGINS, key=lambda g: f1_on(ALL, choose(r2['outer'][1], g, r2['outer'][0]), base1_train)[0])
print('margins (all-train OOF): round 1', fin_m1, 'round 2', fin_m2)
assert (fin_m1, fin_m2) == (0.10, 0.05)                         # same as stage3_README.md
lab = tab[tab.labelled_note]
rank = lgb.LGBMRanker(objective='lambdarank', n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=JOBS,
                      random_state=0, verbose=-1)
from pickers import _groups
rank.fit(lab[clues], lab.label, group=_groups(lab.note_id))
print(f'final ranker trained on {len(lab)} options of {lab.note_id.nunique()} labelled training notes')

# 2. switch common.py to the val songs (audio curves, Basic Pitch val activations, no answer key)
all_man = pd.read_csv(DATA / 'recording_manifest.csv', dtype={'recording_id': str, 'sample_id': str})
val_man = all_man[all_man.split == 'val'].reset_index(drop=True)
assert len(val_man) == 160 and val_man.split.eq('val').all() and val_man.audio_path.str.startswith('data/val/').all()
val_chain = read(V4 / 'val_stage2_delete_v2.csv')
cu.validate_notes(val_chain, val_man)
common.manifest = val_man
common.REF = {}                                                 # no answer key: labels stay empty
common.BPACT = BPACT.parent / 'val'
common._ACT.clear()
vc = CACHE / 'curves_val.pkl'
if not vc.exists():
    pd.to_pickle({r.recording_id: common._curves(r) for r in val_man.itertuples()}, vc)
common._CURVES = pd.read_pickle(vc)

def snap_round(notes, margin):
    o = build_options(notes, 0.15, True)
    assert not o.labelled_note.any() and o[clues].notna().all().all()
    o['p'] = softmax_per_note(rank.predict(o[clues]), o.note_id)
    new = choose(o.p.to_numpy(), margin, o)
    return apply_starts(new, notes), o

val1, o1 = snap_round(val_chain, fin_m1)
val2, o2 = snap_round(val1, fin_m2)
assert (val2.note_id.to_numpy() == val_chain.note_id.to_numpy()).all() and (val2.start < val2.end).all()
assert (val2[['end', 'pitch']].to_numpy() == val_chain[['end', 'pitch']].to_numpy()).all()

acts = []
for rnd, before, after, o, m in [(1, val_chain, val1, o1, fin_m1), (2, val1, val2, o2, fin_m2)]:
    moved = before.start.to_numpy() != after.start.to_numpy()
    stay = o[o.is_stay == 1].set_index('note_id').p
    for nid, rid, old, new in zip(before.note_id[moved], before.recording_id[moved], before.start[moved], after.start[moved]):
        row = o[(o.note_id == nid) & (o.is_stay == 0)]
        row = row.iloc[np.argmin(np.abs(row.t.to_numpy() - new))]
        acts.append(dict(note_id=nid, recording_id=rid, round=rnd, old_start=old, new_start=new,
                         shift_ms=round((new - old) * 1000, 1), score=row.p, stay_score=stay[nid], margin=m))
acts = pd.DataFrame(acts)
print('starts moved: round 1', int((acts['round'] == 1).sum()), 'round 2', int((acts['round'] == 2).sum()),
      'total notes moved', int((val2.start.to_numpy() != val_chain.start.to_numpy()).sum()))

out, act_path = V4 / 'val_stage3_snap_v3.csv', V4 / 'val_stage3_snap_v3_actions.csv'
assert not out.exists() and not act_path.exists()               # run once
cu.export_corrections(val2[cu.NOTE_COLUMNS], val_man, out)
acts.to_csv(act_path, index=False)
print('wrote', out, act_path)

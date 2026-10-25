"""Chain v4 VALIDATION, stage 4: end snap on the 20 val songs, run ONCE (approved by Le Xin).

Final model = the stage-4 RF fitted on ALL labelled training options (end_candidates_train_v4.csv); margin = the one
with the best all-train out-of-fold offset F1 (end_snap_rf_margins_v4.json). Nothing is chosen on val.
Val candidates/clues are built exactly like training (audio, pyin, Basic Pitch val activations); no answer key.
Writes le_xin_work/chain_v4/val_stage4_end_snap.csv + val_stage4_end_snap_actions.csv (write once).
Score afterwards with le_xin_work/chain_v4/val_score.py.
"""
import json
import os
os.environ['END_SNAP_TAG'] = 'v4'
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import build_candidates as bc
from common import OUT, ROOT, DATA, cu, read

bc.BPACT = bc.BPACT.parent / 'val'       # cached Basic Pitch val activations
DEST = ROOT / 'le_xin_work' / 'chain_v4'
NOT_CLUES = {'note_id', 'recording_id', 't', 'label', 'labelled_note', '_true_end', 'fold'}

man = pd.read_csv(DATA / 'recording_manifest.csv', dtype={'recording_id': str, 'sample_id': str})
man = man[man.split == 'val'].reset_index(drop=True)
assert len(man) == 160 and man.split.eq('val').all()                      # test stays locked
val = read(DEST / 'val_stage3_snap_v3.csv')
cu.validate_notes(val, man)

# 1. Final model on all labelled training options
train = read(OUT / 'end_candidates_train_v4.csv')
CLUES = [c for c in train.columns if c not in NOT_CLUES]
assert len(CLUES) == 45
lab = train.labelled_note.to_numpy()
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                            class_weight='balanced_subsample', n_jobs=4, random_state=0)
rf.fit(train.loc[lab, CLUES], train.label.to_numpy()[lab])
margin = json.loads((OUT / 'end_snap_rf_margins_v4.json').read_text())['final_all_train']
print('margin (all-train OOF):', margin, flush=True)

# 2. Val candidates (same code as training, no labels)
VC = OUT / 'end_candidates_val_v4.csv'
if VC.exists():
    cands = read(VC)
else:
    rows = []
    for rec in man.itertuples():
        assert rec.audio_path.startswith('data/val/')
        rows += bc.rows_for_recording(rec, val[val.recording_id == rec.recording_id])
    cands = pd.DataFrame(rows)
    g = cands.groupby('note_id')
    cands['n_options'] = g.t.transform('size')
    cands['abs_dt'] = cands.dt.abs()
    stay = cands[cands.is_stay == 1].set_index('note_id')
    for col in ['drop', 'vshare_drop', 'bp_note_fall', 'pitch_jump', 'flux']:
        cands[f'{col}_rank'] = g[col].rank(ascending=False, method='min')
        cands[f'{col}_vs_stay'] = cands[col] - cands.note_id.map(stay[col])
    cands.to_csv(VC, index=False)
assert set(cands.note_id) == set(val.note_id) and cands[CLUES].notna().all().all()
cands['p'] = rf.predict_proba(cands[CLUES])[:, 1]

# 3. Choose: move to the best option if it beats 'stay' by >= margin
stay_p = cands[cands.is_stay == 1].set_index('note_id').p
moves = cands[cands.is_stay == 0]
best = moves.loc[moves.groupby('note_id').p.idxmax()].set_index('note_id')
best = best[best.p - stay_p.reindex(best.index) >= margin]

out = val[cu.NOTE_COLUMNS].copy()
e = out.note_id.map(best.t)
out['end'] = np.maximum(np.where(e.notna(), e, out.end), out.start + 0.01)
assert (out.end > out.start).all() and (out.start == val.start).all() and (out.pitch == val.pitch).all()

old = val.set_index('note_id').loc[best.index]
acts = pd.DataFrame({'note_id': best.index, 'recording_id': old.recording_id.to_numpy(), 'old_end': old.end.to_numpy(),
                     'new_end': out.set_index('note_id').end.loc[best.index].to_numpy(), 'prob': best.p.to_numpy(),
                     'stay_prob': stay_p.reindex(best.index).to_numpy(), 'margin': margin})
acts['shift_ms'] = ((acts.new_end - acts.old_end) * 1000).round(1)
acts['model'] = 'RandomForestClassifier end-snap picker (45 clues) fitted on all training options; margin from train OOF'
order = {n: i for i, n in enumerate(val.note_id)}
acts = acts.assign(_o=acts.note_id.map(order)).sort_values('_o').drop(columns='_o').reset_index(drop=True)

for table, name in [(out, 'val_stage4_end_snap.csv'), (acts, 'val_stage4_end_snap_actions.csv')]:
    path = DEST / name
    if path.exists():
        pd.testing.assert_frame_equal(read(path), table.reset_index(drop=True), check_dtype=False)
        print('Already saved, unchanged:', name)
    elif name == 'val_stage4_end_snap.csv':
        cu.export_corrections(table, man, path)
        print('Saved:', name)
    else:
        with path.open('x', encoding='utf-8', newline='') as h:
            table.to_csv(h, index=False)
        print('Saved:', name)
print(json.dumps(dict(notes=len(out), ends_moved=len(best), margin=margin,
                      shift_ms_median=float(acts.shift_ms.median()), later_share=round(float((acts.shift_ms > 0).mean()), 3))))

"""Chain v4, stage 4: write the end-snapped COMPLETE note table + action history from the out-of-fold RF picks.

Run after `END_SNAP_TAG=v4 python build_candidates.py` and `END_SNAP_TAG=v4 python end_snap_model.py rf`.
Writes final_pipeline/chain_v4/stage4_end_snap_train_oof.csv and stage4_end_snap_actions.csv (write once).
"""
import json
import os
os.environ['END_SNAP_TAG'] = 'v4'
import numpy as np
import pandas as pd
from common import OUT, ROOT, cu, read, load

DEST = ROOT / 'final_pipeline' / 'chain_v4'
manifest, references, folds, chain = load()
cands = read(OUT / 'end_candidates_train_v4.csv')
prob = read(OUT / 'end_snap_rf_oof_prob_v4.csv')
assert (prob.note_id.to_numpy() == cands.note_id.to_numpy()).all()
cands['p'] = prob.p.to_numpy()
new_end = pd.read_csv(OUT / 'end_snap_train_oof_rf_v4.csv', dtype={'note_id': str}).set_index('note_id').new_end
margins = json.loads((OUT / 'end_snap_rf_margins_v4.json').read_text())

out = chain[cu.NOTE_COLUMNS].copy()
e = out.note_id.map(new_end)
out['end'] = np.maximum(np.where(e.notna(), e, out.end), out.start + 0.01)
assert (out.end > out.start).all() and (out.start == chain.start).all() and (out.pitch == chain.pitch).all()

moved = chain.set_index('note_id').loc[new_end.index]
c = cands[cands.note_id.isin(new_end.index)]
c = c[(c.t - c.note_id.map(new_end)).abs() < 1e-6].drop_duplicates('note_id').set_index('note_id')
stay_p = cands[cands.is_stay == 1].set_index('note_id').p
fold = moved.recording_id.map(manifest.set_index('recording_id').sample_id).map(folds)
acts = pd.DataFrame({'note_id': new_end.index, 'recording_id': moved.recording_id.to_numpy(),
                     'old_end': moved.end.to_numpy(), 'new_end': out.set_index('note_id').end.loc[new_end.index].to_numpy(),
                     'prob': c.p.reindex(new_end.index).to_numpy(), 'stay_prob': stay_p.reindex(new_end.index).to_numpy(),
                     'fold': fold.to_numpy(), 'margin': fold.map(lambda k: margins['per_fold'][str(int(k))]).to_numpy()})
acts['shift_ms'] = ((acts.new_end - acts.old_end) * 1000).round(1)
acts['model'] = 'RandomForestClassifier end-snap picker (45 clues), nested margin; out-of-fold by song'
order = {n: i for i, n in enumerate(chain.note_id)}
acts = acts.assign(_o=acts.note_id.map(order)).sort_values('_o').drop(columns='_o').reset_index(drop=True)

for table, name in [(out, 'stage4_end_snap_train_oof.csv'), (acts, 'stage4_end_snap_actions.csv')]:
    path = DEST / name
    if path.exists():
        pd.testing.assert_frame_equal(read(path), table.reset_index(drop=True), check_dtype=False)
        print('Already saved, unchanged:', name)
    elif name.endswith('_oof.csv'):
        cu.export_corrections(table, manifest, path)
        print('Saved:', name)
    else:
        with path.open('x', encoding='utf-8', newline='') as h:
            table.to_csv(h, index=False)
        print('Saved:', name)

before = cu.evaluate_end_to_end(chain[cu.NOTE_COLUMNS], references, manifest)['summary']
after = cu.evaluate_end_to_end(read(DEST / 'stage4_end_snap_train_oof.csv'), references, manifest)
assert after['summary']['recordings'] == len(manifest)
b = {s['sample_id']: s['offset_f1'] for s in cu.evaluate_end_to_end(chain[cu.NOTE_COLUMNS], references, manifest)['songs']}
d = np.array([s['offset_f1'] - b[s['sample_id']] for s in after['songs']])
print(json.dumps(dict(notes=len(out), ends_moved=len(new_end),
                      before=(round(before['song_macro_f1'], 4), round(before['song_macro_offset_f1'], 4)),
                      after=(round(after['summary']['song_macro_f1'], 4), round(after['summary']['song_macro_offset_f1'], 4)),
                      songs_better=int((d > 1e-12).sum()), songs_worse=int((d < -1e-12).sum()), margins=margins)))

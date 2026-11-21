"""Chain v4, stage 3: snap v3 (LightGBM lambdarank, +-150 ms + 2 extra moment kinds, run twice, nested margins, seed 0)
on final_pipeline/chain_v4/stage2_delete_v2_train_oof.csv. Training songs only, out-of-fold by song.
Run:  SNAP_CHAIN=<stage2 csv> SNAP_CACHE=cache_v4 python stage3_v4.py"""
import os
import numpy as np, pandas as pd
assert os.environ.get('SNAP_CACHE') == 'cache_v4' and 'stage2_delete_v2' in os.environ.get('SNAP_CHAIN', '')
from common import *
from pickers import *
from step3_repeat import repeat

V4 = ROOT / 'final_pipeline' / 'chain_v4'
cu.validate_notes(chain, manifest)
print(f'input: {len(chain)} notes, {chain.recording_id.nunique()} recordings')
clues = CLUES_V2 + EXTRA_KIND_CLUES

def table(win, extra):
    f = CACHE / f'options_w{int(win * 1000)}{"x" if extra else ""}.pkl'
    if not f.exists():
        build_options(chain, win, extra).to_pickle(f)
    return pd.read_pickle(f)

def official(t):
    s = cu.evaluate_end_to_end(t[cu.NOTE_COLUMNS], references, manifest)['summary']
    assert s['recordings'] == len(manifest)
    return round(s['song_macro_f1'], 4), round(s['song_macro_offset_f1'], 4)

# snap v3
tab = table(0.15, True)
fp = fp_lgb_rank('lambdarank', clues, 0)
base1, final, m1, m2, starts2 = repeat(fp, tab, 'lgbrank_w150x', 0.15, True)
assert (final.note_id.to_numpy() == chain.note_id.to_numpy()).all() and (final.start < final.end).all()
assert (final[['end', 'pitch']].to_numpy() == chain[['end', 'pitch']].to_numpy()).all()

# action history: chosen option's score (per-note softmax) and how much it beat 'stay'
def scored(opts, p):
    o = opts[['note_id', 't', 'is_stay', 'fold']].assign(p=p)
    stay = o[o.is_stay == 1].set_index('note_id').p
    return o.assign(stay_p=o.note_id.map(stay))
outer1 = np.load(CACHE / 'lgbrank_w150x.npz')['outer']
r2 = pd.read_pickle(CACHE / 'lgbrank_w150x_round2.pkl')
acts = []
for rnd, before, after, (opts, p), margins in [(1, chain, base1, (tab, outer1), m1), (2, base1, final, r2['outer'], m2)]:
    moved = before.start.to_numpy() != after.start.to_numpy()
    sc = scored(opts, p)
    for nid, old, new in zip(before.note_id[moved], before.start[moved], after.start[moved]):
        row = sc[(sc.note_id == nid) & (sc.is_stay == 0)]
        row = row.iloc[np.argmin(np.abs(row.t.to_numpy() - new))]
        acts.append(dict(note_id=nid, recording_id=nid.split(':')[0], round=rnd, old_start=old, new_start=new,
                         shift_ms=round((new - old) * 1000, 1), score=row.p, stay_score=row.stay_p,
                         fold=int(row.fold), margin=margins[int(row.fold)]))
acts = pd.DataFrame(acts)
assert acts.recording_id.isin(manifest.recording_id).all()

# final-model recipe for val: margins chosen on ALL training songs' out-of-fold scores
fin_m1 = max(MARGINS, key=lambda g: f1_on(ALL, choose(outer1, g, tab))[0])
fin_m2 = max(MARGINS, key=lambda g: f1_on(ALL, choose(r2['outer'][1], g, r2['outer'][0]), base1)[0])

# plain snap v2 (forest, +-100 ms, 45 clues, one round) on the same input
t100 = table(0.10, False)
_, _, st_v2, m_v2 = nested(fp_rf(), t100, 'rf_w100')
v2 = apply_starts(st_v2)

res = pd.DataFrame([
    dict(version='input: stage 2 (timing -> split v2 -> delete v2)', starts_moved=0, onset_F1=official(chain)[0], offset_F1=official(chain)[1], margins=None),
    dict(version='snap v2 (forest, +-100 ms, 1 round)', starts_moved=len(st_v2), onset_F1=official(v2)[0], offset_F1=official(v2)[1], margins=m_v2),
    dict(version='snap v3 round 1 (ranker, +-150 ms + extra kinds)', starts_moved=int((acts['round'] == 1).sum()), onset_F1=official(base1)[0], offset_F1=official(base1)[1], margins=m1),
    dict(version='snap v3 round 1 + round 2 (final)', starts_moved=int((final.start.to_numpy() != chain.start.to_numpy()).sum()), onset_F1=official(final)[0], offset_F1=official(final)[1], margins=m2),
])
res['labelled_right'] = [round(100 * labelled_right(t), 1) for t in [chain, v2, base1, final]]
print(res.to_string(index=False))
print('round-2 moves:', int((acts['round'] == 2).sum()), '| final-model margins (all-train OOF): round 1', fin_m1, 'round 2', fin_m2)
res.to_csv(OUT / 'stage3_v4_results.csv', index=False)

out, act_path = V4 / 'stage3_snap_v3_train_oof.csv', V4 / 'stage3_snap_v3_actions.csv'
assert not out.exists() and not act_path.exists()
cu.export_corrections(final[cu.NOTE_COLUMNS], manifest, out)
acts.to_csv(act_path, index=False)
print('wrote', out, act_path)

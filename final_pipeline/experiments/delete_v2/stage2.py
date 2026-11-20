"""Chain v4, stage 2: best delete v2 recipe (GB, official label, 'replace', 49 base clues, nested threshold) on the
timing + split v2 pool. Training songs only, out-of-fold by song. Writes in final_pipeline/chain_v4/:
stage2_delete_v2_train_oof.csv (complete kept note table) and stage2_delete_v2_actions.csv; prints the scores.
Run features.py on the pool first (see __main__)."""
import subprocess
import sys
import numpy as np
import pandas as pd
from common import *
import correction_utils as cu
import models as M

CHAIN = OUT.parent / 'chain_v4'
POOL = CHAIN / 'stage1_split_v2_train_oof.csv'
FEATS = OUT / 'features_chain_v4_stage1.csv'
M.GRID = np.r_[np.arange(.3, .6, .05), np.arange(.6, .976, .025), 1.01]   # old grid's values + finer steps at the top
BASE = M.NOTE + M.AUDIO + M.BPACT

if __name__ == '__main__':
    if not FEATS.exists():
        subprocess.run([sys.executable, '-W', 'ignore', str(OUT / 'features.py'), str(POOL), str(FEATS)], check=True)
    M.F = read(FEATS)                                           # run() reads the module's F (sequential, no workers)
    pool_notes = read(POOL)
    assert len(M.F) == len(pool_notes) and set(M.F.note_id) == set(pool_notes.note_id)
    assert len(BASE) == 49
    r, sc = M.run('replace', 'y_official_pool', 'GB', BASE)
    thr = {k: float(M.GRID[np.abs(M.GRID - float(v)).argmin()])     # run() prints 2 decimals; snap back to the grid
           for k, v in enumerate(r['thresholds'].split('/'))}
    d = M.F.reset_index(drop=True)
    act = pd.DataFrame(dict(note_id=d.note_id, prob=sc.score.round(6), fold=d.fold, threshold=d.fold.map(thr).round(3),
                            deleted=sc.delete.astype(bool)))
    assert act.deleted.eq(act.prob >= act.threshold - 1e-9).all()
    kept = pool_notes[pool_notes.note_id.isin(act.note_id[~act.deleted])].reset_index(drop=True)

    # final-model threshold for val: best all-train onset F1 over the same grid, using the outer OOF probabilities
    f = [score(d[sc.score.to_numpy() < t], offset=False)[0] for t in M.GRID]
    final_thr = M.GRID[int(np.flatnonzero(np.isclose(f, max(f), atol=1e-9))[-1])]

    # baseline: Felicia's out-of-fold keep/delete decision (per original Basic Pitch note) on the same pool
    fel = read(EXPORTS / 'felicia_false_actions_train_oof_v2_exploratory.csv').set_index('note_id').action
    fel_kept = pool_notes[d.set_index('note_id').orig.loc[pool_notes.note_id].map(fel).eq('keep').to_numpy()]

    ev = lambda t: cu.evaluate_end_to_end(t[cu.NOTE_COLUMNS], REFS, MANIFEST)['summary']
    rows = []
    for name, t in [('stage 1 input (timing + split v2)', pool_notes), ('Felicia OOF decisions on same pool', fel_kept),
                    ('delete v2 GB (this stage)', kept)]:
        s = ev(t)
        assert s['recordings'] == 534
        rows.append(dict(version=name, notes=len(t), deleted=len(pool_notes) - len(t),
                         onset_F1=round(s['song_macro_f1'], 6), offset_F1=round(s['song_macro_offset_f1'], 6),
                         precision=round(s['song_macro_precision'], 4), recall=round(s['song_macro_recall'], 4)))
    res = pd.DataFrame(rows)
    print(res.to_string())
    print('fold thresholds', thr, '| final-model threshold (all-train OOF)', round(float(final_thr), 3))
    print('deleted truly-false', int((act.deleted & d.y_loose).sum()), 'of', int(act.deleted.sum()))

    for path, t in [(CHAIN / 'stage2_delete_v2_train_oof.csv', kept), (CHAIN / 'stage2_delete_v2_actions.csv', act)]:
        if path.exists():                                       # write once; a rerun checks it still matches
            old = read(path)
            assert old.shape == t.shape and (old.note_id.to_numpy() == t.note_id.to_numpy()).all(), path
        else:
            (cu.export_corrections(t, MANIFEST, path) if 'actions' not in path.name else t.to_csv(path, index=False))
    res.to_csv(OUT / 'stage2_results.csv', index=False)

"""Chain v4 VALIDATION, stage 2 (run once): final delete v2 model on the 20 val songs.
Final model = GB (stage 2 recipe) fit on ALL 8,493 training notes of the stage-1 pool; threshold = best all-train
onset F1 over the stage-2 grid using the training out-of-fold probabilities (nothing is chosen on val).
Writes final_pipeline/chain_v4/val_stage2_delete_v2.csv (+ _actions.csv). Score with chain_v4/val_score.py."""
import subprocess
import sys
import numpy as np
import pandas as pd
from common import *
import correction_utils as cu
import models as M
from stage2 import CHAIN, FEATS, BASE

VAL_POOL = CHAIN / 'val_stage1_split_v2.csv'
VAL_FEATS = OUT / 'features_chain_v4_val_stage1.csv'

if __name__ == '__main__':
    # 1. threshold from the training out-of-fold probabilities only
    tr = read(FEATS)
    oof = read(CHAIN / 'stage2_delete_v2_actions.csv').set_index('note_id').prob
    p_tr = tr.note_id.map(oof).to_numpy()
    f = [score(tr[p_tr < t], offset=False)[0] for t in M.GRID]
    thr = float(M.GRID[int(np.flatnonzero(np.isclose(f, max(f), atol=1e-9))[-1])])
    print(f'threshold from training OOF: {thr:.3f} (train OOF onset F1 at it {max(f):.4f})', flush=True)

    # 2. final model on all training notes
    X = np.nan_to_num(tr[BASE].to_numpy(float), nan=-1)
    model = M.MODELS['GB']().fit(X, tr.y_official_pool.astype(int).to_numpy())

    # 3. val clues (no answer key) and the single application
    if not VAL_FEATS.exists():
        subprocess.run([sys.executable, '-W', 'ignore', str(OUT / 'features.py'), str(VAL_POOL), str(VAL_FEATS), 'val'],
                       check=True)
    va = read(VAL_FEATS)
    pool_val = read(VAL_POOL)
    assert len(va) == len(pool_val) == 2587 and set(va.note_id) == set(pool_val.note_id) and va.y_official_pool.isna().all()
    prob = model.predict_proba(np.nan_to_num(va[BASE].to_numpy(float), nan=-1))[:, 1]
    act = pd.DataFrame(dict(note_id=va.note_id, recording_id=va.recording_id, prob=prob.round(6),
                            threshold=round(thr, 3), deleted=prob >= thr))
    kept = pool_val[pool_val.note_id.isin(act.note_id[~act.deleted])].reset_index(drop=True)
    man = read(DATA / 'recording_manifest.csv').query("split == 'val'").reset_index(drop=True)
    out, out_act = CHAIN / 'val_stage2_delete_v2.csv', CHAIN / 'val_stage2_delete_v2_actions.csv'
    cu.export_corrections(kept, man, out)                        # write once: refuses to overwrite
    act.to_csv(out_act, index=False)
    print(f'val: {len(pool_val)} notes in, {int(act.deleted.sum())} deleted, {len(kept)} kept;'
          f' share deleted {act.deleted.mean():.3f} (train OOF {oof.ge(0).mean() and (read(CHAIN / "stage2_delete_v2_actions.csv").deleted.mean()):.3f})')
    print('mean prob val', prob.mean().round(3), '| train OOF', oof.mean().round(3))

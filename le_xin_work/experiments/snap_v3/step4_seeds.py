"""Step 4: is the ranker's gain stable? Re-run lambdarank on +-150 ms + extra kinds with other random seeds,
and count songs better / worse than snap v2 (song onset F1)."""
import pandas as pd
from common import *
from common import _REF_ARR
from pickers import *

tab = pd.read_pickle(CACHE / 'options_w150x.pkl')
clues = CLUES_V2 + EXTRA_KIND_CLUES
_, _, st_v2, _ = nested(fp_rf(), load_v2(), 'rf_v2')
def songs(starts):
    t = apply_starts(starts)
    man = manifest
    rows = [{**r, **core.score_notes(_REF_ARR.get(r['recording_id'], EMPTY),
             t[t.recording_id == r['recording_id']][['start', 'end', 'pitch']].to_numpy())} for r in man.to_dict('records')]
    return pd.DataFrame(core.aggregate(rows)[0]).set_index('sample_id').f1
base = songs(st_v2)
rows = []
for seed in [0, 1, 2]:
    name = 'lgbrank_w150x' if seed == 0 else f'lgbrank_w150x_s{seed}'
    _, _, st, m = nested(fp_lgb_rank('lambdarank', clues, seed), tab, name)
    r = report(f'lambdarank +-150 ms + extra kinds, seed {seed}', st, m)
    s = songs(st)
    r.update(songs_better_than_v2=int((s > base + 1e-12).sum()), songs_worse=int((s < base - 1e-12).sum()), songs=len(s))
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(OUT / 'results_step4_seeds.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))

"""Step 1: better pickers on the SAME options and clues as snap v2 (45 clues, +-100 ms)."""
import time, json
import pandas as pd
from common import *
from pickers import *

v2 = load_v2()
runs = [('snap v2: random forest (reproduced)', 'rf_v2', fp_rf()),
        ('gradient boosting (HistGradientBoosting)', 'hgb_v2', fp_hgb()),
        ('forest + boosting average', 'rfhgb_v2', fp_rf_hgb()),
        ('ranker: LightGBM lambdarank (per-note softmax)', 'lgbrank_v2', fp_lgb_rank('lambdarank')),
        ('ranker: LightGBM xendcg (listwise softmax)', 'lgbxe_v2', fp_lgb_rank('rank_xendcg')),
        ('two-stage: move at all? (boosting) then which? (forest)', 'two_v2', fp_two_stage())]
rows = [report('Final chain (no snap)', pd.Series(dtype=float))]
for title, name, fp in runs:
    t0 = time.time()
    _, _, st, m = nested(fp, v2, name)
    rows.append(report(title, st, m))
    print(f'{name}: {time.time() - t0:.0f} s', rows[-1], flush=True)
res = pd.DataFrame(rows)
res.to_csv(OUT / 'results_step1_pickers.csv', index=False)
print(res.to_string(index=False))

"""Step 2: wider search. Rebuild the options with a +-150 ms window and/or two extra moment kinds
(Basic Pitch onset peak at the note's pitch; the moment the sung pitch settles on the note), then retrain."""
import sys, time
import pandas as pd
from common import *
from pickers import *

def table(win, extra):
    f = CACHE / f'options_w{int(win * 1000)}{"x" if extra else ""}.pkl'
    if f.exists():
        return pd.read_pickle(f)
    t = build_options(chain, win, extra)
    t.to_pickle(f)
    return t

def ceiling(t):
    lab = t[t.labelled_note]
    has = lab.groupby('note_id').label.max()
    return dict(options=len(t), per_note=round(len(t) / t.note_id.nunique(), 2), ceiling=round(100 * has.mean(), 1),
                stay_right=round(100 * lab[lab.is_stay == 1].label.mean(), 1))

pickers = {'rf': fp_rf, 'hgb': fp_hgb, 'rfhgb': fp_rf_hgb, 'lgbrank': lambda c: fp_lgb_rank('lambdarank', c)}
use = sys.argv[1:] or ['rf']
rows = [report('Final chain (no snap)', pd.Series(dtype=float))]
for win, extra in [(0.10, False), (0.10, True), (0.15, False), (0.15, True)]:
    t = table(win, extra)
    tag = f'+-{int(win * 1000)} ms' + (' + extra kinds' if extra else '')
    print(tag, ceiling(t), flush=True)
    clues = CLUES_V2 + (EXTRA_KIND_CLUES if extra else [])
    for p in use:
        t0 = time.time()
        _, _, st, m = nested(pickers[p](clues), t, f'{p}_w{int(win * 1000)}{"x" if extra else ""}')
        rows.append(report(f'{p}, {tag}', st, m))
        print(f'{time.time() - t0:.0f} s', rows[-1], flush=True)
res = pd.DataFrame(rows)
res.to_csv(OUT / f'results_step2_wider_{"_".join(use)}.csv', index=False)
print(res.to_string(index=False))

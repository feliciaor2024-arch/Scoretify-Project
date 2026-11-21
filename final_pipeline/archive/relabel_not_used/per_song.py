"""Per-song paired comparison of the relabel runs vs the old v2-label model (nested gate shifts from relabel.py)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import relabel as R

names = ['v2 labels (old)', 'union, strict', 'rematch-only, strict', 'union from export (simple)']
shifts = np.load(R.OUT / 'gated_shifts.npy')
per = {}
for n, s in zip(names, shifts):
    out = R.cu.evaluate_end_to_end(R.apply_shifts(s), R.refs, R.manifest)
    per[n] = pd.DataFrame(out['songs']).set_index('sample_id')[['f1', 'offset_f1']]
rng = np.random.default_rng(0)
rows = []
for n in names[1:]:
    for col in ['f1', 'offset_f1']:
        d = (per[n][col] - per[names[0]][col]).to_numpy()
        boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        rows.append(dict(model=n, metric=col, mean_gain=round(d.mean(), 4), songs_better=int((d > 1e-9).sum()),
                         songs_worse=int((d < -1e-9).sum()), songs=len(d),
                         ci95=f'[{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}]'))
t = pd.DataFrame(rows)
t.to_csv(R.OUT / 'per_song_comparison.csv', index=False)
print(t.to_string())

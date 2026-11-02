"""Chain v4 stage 1: export split v2 (RF + boosting average, v2 candidates, 74 richer clues, nested threshold) on the
TRAINING songs, out-of-fold by song, as a complete note table. Reads the OOF cuts/probabilities saved by
le_xin_work/experiments/split_v2/s5_models.py (run 'v2 avg_rich'); no val, no test, no TextGrid."""
import ast, string, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'split_v2'))
import common as C

OUT = Path(__file__).resolve().parent
manifest, references, folds, raw, timed = C.load()
cands = pd.read_csv(C.CACHE / 'clues_v2_train.csv', dtype={'note_id': str, 'recording_id': str, 'sample_id': str})
cut = np.load(C.CACHE / 'cut_v2_avg_rich.npy')
prob = np.load(C.CACHE / 'prob_v2_avg_rich.npy')
assert len(cut) == len(prob) == len(cands)
res = pd.read_csv(C.OUT / 'results.csv')
th = ast.literal_eval(res.loc[res.model == 'v2:avg_rich', 'thresholds'].iloc[-1])
cands['fold'] = cands.sample_id.map(folds)
cands['prob'] = prob
assert (cut == (prob >= cands.fold.map(th).to_numpy())).all()     # cuts = OOF prob >= that fold's nested threshold

# Same pieces as common.apply_splits; only the id scheme differs: every piece of a cut note gets <orig>_a, _b, ...
chosen = cands[cut]
pieces = C.apply_splits(timed, chosen)
k = pieces.note_id.str.extract(r'^(.*?)(?:~(\d+))?$')
cut_ids = set(chosen.note_id)
orig, idx = k[0], k[1].fillna(0).astype(int)
pieces['note_id'] = [f'{o}_{string.ascii_lowercase[i]}' if o in cut_ids else o for o, i in zip(orig, idx)]
assert pieces.note_id.is_unique and idx.max() < 26
assert pieces.recording_id.isin(manifest.recording_id).all() and set(timed.recording_id) <= set(pieces.recording_id)
pieces.to_csv(OUT / 'stage1_split_v2_train_oof.csv', index=False)

acts = chosen.sort_values(['note_id', 't'])[['note_id', 'recording_id', 't', 'prob', 'fold']].rename(columns={'t': 'cut_time'})
acts['threshold'] = acts.fold.map(th)
acts.to_csv(OUT / 'stage1_split_v2_actions.csv', index=False)

s = C.cu.evaluate_end_to_end(pieces, references, manifest)['summary']
assert s['recordings'] == len(manifest)
print(f"notes {len(pieces)} (from {len(timed)}), cuts {len(acts)} in {acts.note_id.nunique()} notes, thresholds {th}")
print(f"cu.evaluate_end_to_end: onset F1 {s['song_macro_f1']:.4f}, offset F1 {s['song_macro_offset_f1']:.4f}, recordings {s['recordings']}")

"""Why do notes stay unlabelled after correction? First failing check per note (matching on the existing gated OOF export)."""
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import relabel as R
pos = pd.read_csv(R.HERE/'exports'/'le_xin_timing_rf_audio_pitch_v2_nestedgate.csv', dtype={'note_id':str,'recording_id':str})
lab = pd.read_csv(R.OUT/'labels_export_gated_union.csv')['reference_note_id'].notna().to_numpy()
reason = np.full(len(R.notes), 'labelled', dtype=object)
ref_by = dict(tuple(R.refs.groupby('recording_id')))
for rid, idx in R.notes.groupby('recording_id').indices.items():
    ref = ref_by[rid]; P = pos.iloc[idx][['start','end']].to_numpy(); pp = R.notes['pitch'].to_numpy()[idx]
    Q = ref[['start','end']].to_numpy(); qp = ref['pitch'].to_numpy()
    O = np.clip(np.minimum(P[:,None,1],Q[None,:,1]) - np.maximum(P[:,None,0],Q[None,:,0]), 0, None)
    br, bp = O.argmax(1), O.argmax(0)
    for i in range(len(P)):
        g = idx[i]
        if lab[g]: continue
        if not R.eligible[g]: reason[g] = 'v2 spurious/split candidate (not re-matched)'; continue
        j = br[i]; o = O[i,j]; dp, dr = P[i,1]-P[i,0], Q[j,1]-Q[j,0]; short = min(dp, dr)
        second = np.sort(O[i])[::-1][1] if O.shape[1] > 1 else 0.
        if o <= 0: r = 'touches no reference note (false note)'
        elif bp[j] != i: r = 'its reference is covered more by another BP note (fragment / duplicate)'
        elif second/dp > .30: r = 'covers a 2nd reference note >30% (needs SPLIT)'
        elif max(dp,dr)/short > 3: r = 'length differs >3x'
        elif o/short < .6: r = 'overlap <60%'
        elif abs(P[i,0]-Q[j,0]) > .2: r = 'start still >200 ms off'
        elif abs(pp[i]-qp[j]) > 2: r = 'pitch >2 semitones off'
        else: r = 'reference already labelled to another note'
        reason[g] = r
s = pd.Series(reason).value_counts()
print(s.to_string()); s.to_csv(R.OUT/'why_unlabelled.csv', header=['notes'])

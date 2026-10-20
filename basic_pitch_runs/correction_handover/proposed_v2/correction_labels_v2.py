"""Provisional TRAIN labels, version 2 (mutual-best overlap + safety filter).

Changes ONLY the 'Unique overlap' check of correction_labels.py (v1).
v1: a prediction is matched only if it overlaps exactly one reference and that
    reference overlaps exactly one prediction (any overlap, even 1 ms, counts).
v2: a prediction is matched to the reference it overlaps MOST, provided that
    reference also overlaps this prediction MOST (mutual best).
Checks 2-5 are unchanged: overlap >= 60% of shorter note, duration ratio <= 3,
start difference <= 0.20 s, pitch difference <= 12 semitones.
Safety filter (replaces manual listening) for NEW v2-only matches:
    drop if the second-largest reference overlap > 30% of the prediction,
    or if |pitch difference| > 2 semitones.
All v1 one_to_one labels are kept unchanged. Only v1 'ambiguous' notes can be
upgraded; spurious/split candidates are left as in v1.

Usage (from the handover root):
    python basic_pitch_runs/correction_handover/proposed_v2/correction_labels_v2.py
Reads train_features.csv, references_train.csv, train_targets.csv (v1) from
basic_pitch_runs/correction_handover/ and writes train_targets_v2.csv next to this file.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
SECOND_OVERLAP_MAX = 0.30
PITCH_DIFF_MAX_NEW = 2


def build(features, references, targets_v1):
    targets = targets_v1.copy()
    targets['match_rule'] = np.where(targets.relationship == 'one_to_one', 'v1_original', '')
    refs = dict(tuple(references.groupby('recording_id')))
    feats = features.set_index('note_id')
    stats = dict(mutual_best=0, kept_new=0, dropped_second_overlap=0, dropped_pitch_jump=0)
    rows = targets.set_index('note_id')
    for rid, p in features.sort_values(['recording_id', 'start', 'end', 'pitch']).groupby('recording_id'):
        ref = refs.get(rid)
        if ref is None or not len(p):
            continue
        P = p[['start', 'end']].to_numpy(); Q = ref[['start', 'end']].to_numpy()
        O = np.clip(np.minimum(P[:, None, 1], Q[None, :, 1]) - np.maximum(P[:, None, 0], Q[None, :, 0]), 0, None)
        best_ref = O.argmax(1); best_pred = O.argmax(0)
        for i in range(len(P)):
            j = best_ref[i]; o = O[i, j]
            if o <= 0 or best_pred[j] != i:
                continue
            x = p.iloc[i]; y = ref.iloc[j]
            dp, dr = x.end - x.start, y.end - y.start; short = min(dp, dr)
            if o / short < .6 or max(dp, dr) / short > 3 or abs(x.start - y.start) > .2 or abs(x.pitch - y.pitch) > 12:
                continue
            stats['mutual_best'] += 1
            nid = x.note_id
            if rows.at[nid, 'relationship'] != 'ambiguous':
                continue  # keep v1 one_to_one / spurious / split exactly as they were
            second = np.sort(O[i])[::-1][1] if O.shape[1] > 1 else 0.
            if second / dp > SECOND_OVERLAP_MAX:
                stats['dropped_second_overlap'] += 1; continue
            if abs(y.pitch - x.pitch) > PITCH_DIFF_MAX_NEW:
                stats['dropped_pitch_jump'] += 1; continue
            sd, ed, pd_ = y.start - x.start, y.end - x.end, y.pitch - x.pitch
            rows.loc[nid, ['relationship', 'reference_index', 'false_note', 'pitch_delta', 'start_delta', 'end_delta',
                           'status', 'reference_start', 'reference_end', 'reference_pitch',
                           'wrong_start', 'wrong_end', 'wrong_pitch', 'match_rule']] = [
                'one_to_one', int(y.note_id.rsplit('ref', 1)[1]), 0, pd_, sd, ed,
                'provisional_auto_v2', y.start, y.end, y.pitch,
                int(abs(sd) > .05), int(abs(ed) > max(.05, .2 * dr)), int(pd_ != 0), 'v2_mutual_best']
            stats['kept_new'] += 1
    out = rows.reset_index()[list(targets_v1.columns) + ['match_rule']]
    return out, stats


if __name__ == '__main__':
    f = pd.read_csv(DATA / 'train_features.csv', dtype={'note_id': str, 'recording_id': str})
    r = pd.read_csv(DATA / 'references_train.csv', dtype={'note_id': str, 'recording_id': str})
    t = pd.read_csv(DATA / 'train_targets.csv', dtype={'note_id': str})
    out, stats = build(f, r, t)
    out.to_csv(HERE / 'train_targets_v2.csv', index=False)
    print(stats)
    print(out.relationship.value_counts().to_dict())
    print(out.match_rule.replace('', 'none').value_counts().to_dict())

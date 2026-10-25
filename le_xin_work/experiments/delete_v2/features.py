"""Build one row per note of the pre-delete pool (timing + split + snap starts): labels from the answer key,
and clues from the notes, the pyin pitch track, Basic Pitch's own activations, loudness and the snap step.
Writes features_train.csv next to this script. Training songs only.
Optional: python features.py <pool.csv> <out.csv> builds the same table for another complete training note table
(no Felicia/snap step: felicia_kept is all True and the snap clues stay empty).
python features.py <pool.csv> <out.csv> val  builds the clues for a VAL note table: no answer key is read,
the label columns stay empty, and the val caches (Basic Pitch activations, loudness) are used."""
import sys
import numpy as np
import pandas as pd
from common import *

SPLIT = sys.argv[3] if len(sys.argv) > 3 else 'train'
assert SPLIT in ('train', 'val')                                  # test is never built
bp = read(DATA / f'{SPLIT}_features.csv').set_index('note_id')
if len(sys.argv) > 2:
    p = read(sys.argv[1])
    man = MANIFEST if SPLIT == 'train' else read(DATA / 'recording_manifest.csv').query("split == 'val'")
    assert man.split.eq(SPLIT).all() and set(p.recording_id) <= set(man.recording_id)
    p['felicia_kept'] = True
    p['sample_id'] = p.recording_id.map(man.set_index('recording_id').sample_id)
    p['fold'] = p.sample_id.map(FOLDS) if SPLIT == 'train' else -1
    OUT_FILE = Path(sys.argv[2])
else:
    p, OUT_FILE = pool(), OUT / 'features_train.csv'
# original Basic Pitch id: split v1 pieces are '<id>~1', split v2 pieces '<id>_a', '<id>_b', ...
p['orig'] = original_id(p.note_id)
v2 = ~p.orig.isin(bp.index)
p.loc[v2, 'orig'] = p.loc[v2, 'orig'].str.replace(r'_[a-z]$', '', regex=True)
assert p.orig.isin(bp.index).all()
p = p.sort_values(['recording_id', 'start', 'pitch']).reset_index(drop=True)


# ---------- labels (answer key; never a model input) ----------
def labels(notes):
    """official_false: matched to no reference note by the official one-to-one rule (onset 50 ms, pitch 50 cents).
    loose_false: overlaps in time no reference note within 50 cents (nothing sung at that pitch then)."""
    off, loose = pd.Series(True, index=notes.index), pd.Series(True, index=notes.index)
    for rid, g in notes.groupby('recording_id'):
        ref = REFS[REFS.recording_id == rid][['start', 'end', 'pitch']].to_numpy(float)
        est = g[['start', 'end', 'pitch']].to_numpy(float)
        if len(ref):
            m = match_notes(ref[:, :2], midi_to_hz(ref[:, 2]), est[:, :2], midi_to_hz(est[:, 2]),
                            onset_tolerance=.05, pitch_tolerance=50., offset_ratio=None, strict=False)
            off.loc[g.index[[j for _, j in m]]] = False
            ov = (est[:, None, 0] < ref[None, :, 1]) & (est[:, None, 1] > ref[None, :, 0])
            same = np.abs(est[:, None, 2] - ref[None, :, 2]) <= .5
            loose.loc[g.index[(ov & same).any(1)]] = False
    return off, loose


k = p.felicia_kept
p['y_official_kept'] = np.nan
if SPLIT == 'train':                                               # val: no answer key here
    p['y_official_pool'], p['y_loose'] = labels(p)
    p.loc[k, 'y_official_kept'] = labels(p[k])[0].astype(float)
else:
    p['y_official_pool'], p['y_loose'] = np.nan, np.nan   # same rule, scored among Felicia's kept notes only

# ---------- note clues ----------
p['duration'] = p.end - p.start
p['log_dur'] = np.log(p.duration)
p['amplitude'] = p.orig.map(bp.amplitude)
p['is_piece'] = (p.note_id != p.orig).astype(int)
p['n_pieces'] = p.groupby('orig').note_id.transform('size')
p['felicia_prob'] = p.orig.map(read(EXPORTS / 'felicia_false_actions_train_oof_v2_exploratory.csv')
                               .set_index('note_id').prob_extra)          # her out-of-fold score, same folds
rows = []
for rid, g in p.groupby('recording_id', sort=False):
    s, e, pt = g.start.to_numpy(), g.end.to_numpy(), g.pitch.to_numpy()
    n = len(g)
    ov = np.minimum(e[:, None], e[None, :]) - np.maximum(s[:, None], s[None, :])
    np.fill_diagonal(ov, 0)
    ovp = np.clip(ov, 0, None)
    dur = e - s
    longer = (dur[None, :] > dur[:, None]) & (ovp > 0)
    rows.append(pd.DataFrame(dict(
        prev_gap=np.r_[s[0], s[1:] - e[:-1]], next_gap=np.r_[s[1:] - e[:-1], 5.],
        pitch_prev=np.r_[0, pt[1:] - pt[:-1]], pitch_next=np.r_[pt[1:] - pt[:-1], 0],
        n_overlap=(ovp > 0.01).sum(1), overlap_frac=(ovp.max(1) if n > 1 else np.zeros(n)) / dur,
        overlapped_by_longer=longer.any(1).astype(int),
        rel_pos=(s - s.min()) / max(e.max() - s.min(), 1e-3), first=np.r_[1, np.zeros(n - 1)], last=np.r_[np.zeros(n - 1), 1],
        rec_notes_per_s=n / max(e.max() - s.min(), 1e-3), pitch_vs_rec=pt - np.median(pt),
        amp_vs_rec=g.amplitude.to_numpy() - g.amplitude.median(), dur_vs_rec=np.log(dur / np.median(dur))),
        index=g.index))
p = p.join(pd.concat(rows))

# ---------- pyin pitch track over the note (Hann Qian's cache) ----------
f0 = dict(np.load(HO / 'hann_qian_pitch_model' / 'f0_tracks.npz'))


def track_clues(t, midi, s, e, pitch):
    out = {}
    inn = (t >= s) & (t < e)
    m = midi[inn]
    v = ~np.isnan(m)
    d = m[v] - pitch
    out['f0_frames'] = inn.sum()
    out['f0_voiced'] = v.mean() if len(m) else 0.
    out['f0_on_pitch'] = (np.abs(d) <= .5).sum() / max(len(m), 1)
    out['f0_near_pitch'] = (np.abs(d) <= 1.).sum() / max(len(m), 1)
    out['f0_octave'] = ((np.abs(np.abs(d) - 12) <= 1)).sum() / max(len(m), 1)
    out['f0_med_absdiff'] = np.median(np.abs(d)) if len(d) else 24.
    out['f0_med_diff'] = np.clip(np.median(d), -24, 24) if len(d) else 0.
    out['f0_std'] = np.std(m[v]) if v.sum() > 1 else 0.
    for name, lo, hi in [('before', s - .1, s), ('after', e, e + .1)]:
        w = midi[(t >= lo) & (t < hi)]
        out[f'f0_voiced_{name}'] = (~np.isnan(w)).mean() if len(w) else 0.
        out[f'f0_on_pitch_{name}'] = (np.abs(w - pitch) <= .5).mean() if len(w) else 0.
    return out


# ---------- Basic Pitch's own activations (same ONNX model as the frozen predictions) ----------
def bp_times(n, sr=22050, hop=256, annot=172, n_samples=22050 * 2 - 256):
    """basic_pitch.note_creation.model_frames_to_time, copied so basic-pitch need not be installed."""
    off = (hop / sr) * (annot - n_samples / hop) + .0018
    return np.arange(n) * hop / sr - off * np.floor(np.arange(n) / annot)


def act_clues(T, note, onset, s, e, pitch):
    b = int(round(pitch)) - 21
    inn = (T >= s) & (T < e)
    if not inn.any():
        inn[np.argmin(np.abs(T - (s + e) / 2))] = True
    a = note[inn]
    best = a.max(1)
    i0 = int(np.argmin(np.abs(T - s)))
    nb = a[:, max(b - 1, 0):b + 2].max(1)
    return dict(bp_note_mean=a[:, b].mean(), bp_note_max=a[:, b].max(), bp_note_min=a[:, b].min(),
                bp_near_mean=nb.mean(), bp_best_mean=best.mean(), bp_share=(a[:, b] / (best + 1e-6)).mean(),
                bp_is_argmax=(np.abs(a.argmax(1) - b) <= 1).mean(),
                bp_onset_start=onset[max(i0 - 2, 0):i0 + 3, b].max(),
                bp_onset_any_start=onset[max(i0 - 2, 0):i0 + 3].max(),
                bp_oct_mean=max(a[:, b - 12].mean() if b >= 12 else 0, a[:, b + 12].mean() if b + 12 < 88 else 0))


rows = []
for rid, g in p.groupby('recording_id', sort=False):
    t, hz = f0[f'{rid}_t'], f0[f'{rid}_f0']
    midi = 69 + 12 * np.log2(hz / 440.)
    with np.load((BPACT if SPLIT == 'train' else BPACT.parent / 'val') / f'{rid}.npz') as z:
        note, onset = z['note'], z['onset']                     # read once per recording (npz is lazy)
    T = bp_times(len(note))
    for i, r in g.iterrows():
        rows.append(dict(idx=i, **track_clues(t, midi, r.start, r.end, r.pitch),
                         **act_clues(T, note, onset, r.start, r.end, r.pitch)))
p = p.join(pd.DataFrame(rows).set_index('idx'))
p['bp_note_vs_rec'] = p.bp_note_mean - p.groupby('recording_id').bp_note_mean.transform('median')

# ---------- loudness around the boundaries (Le Xin's timing cache, per original Basic Pitch note) ----------
aud = read(CACHE / f'audio_features_{SPLIT}_v1.csv').set_index('note_id')
for c in ['db_at_start', 'db_at_end', 's_flux', 's_rise', 'e_drop']:
    p['a_' + c] = p.orig.map(aud[c])
p['a_db_vs_rec'] = p.a_db_at_start - p.groupby('recording_id').a_db_at_start.transform('median')

# ---------- snap step (only exists for the notes Felicia kept) ----------
cand = read(CACHE / f'snap_candidates_{SPLIT}_v2.csv')   # unused for chain v4 (no snap clues)
agg = cand.groupby('note_id').agg(sn_options=('n_options', 'first'), sn_attack_max=('attack', 'max'),
                                  sn_louder_max=('louder', 'max'), sn_jump_max=('pitch_jump', 'max'),
                                  sn_any_settle=('has_settle', 'max'), sn_bp_onset_max=('bp_onset_at', 'max'))
stay = cand[cand.is_stay == 1].set_index('note_id')[['match_100', 'steady_100', 'settle_abs', 'voiced_before',
                                                    'voiced_after', 'bp_note_rise']].add_prefix('sn_stay_')
act = read(EXPORTS / f'le_xin_snap_{"train_oof" if SPLIT == "train" else "val"}_v2_actions.csv').set_index('note_id')
p = p.join(agg, on='note_id').join(stay, on='note_id')
p['sn_moved'] = p.note_id.isin(act.index).astype(int)
p['sn_abs_shift'] = p.note_id.map(act.shift_ms).abs().fillna(0)

p.to_csv(OUT_FILE, index=False)
print(len(p), 'notes;', p.shape[1], 'columns')
if SPLIT == 'val':
    sys.exit()
print('false rate (official, pool):', p.y_official_pool.mean().round(3), ' loose:', p.y_loose.mean().round(3))
if (~k).any(): print('Felicia-deleted notes: official false', p[~k].y_official_pool.mean().round(3), ' loose', p[~k].y_loose.mean().round(3))
print('Felicia-kept: official false', p[k].y_official_kept.mean().round(3), ' loose', p[k].y_loose.mean().round(3))

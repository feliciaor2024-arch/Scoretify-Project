"""Step 1: list the real moments near every note's END, describe each with clues, label the right ones.

Mirror of the start snap (notebooks/correction_handover/le_xin_snap_model.ipynb), for note ends.
Options = 'stay' + moments within +-150 ms of the current end:
  voice_off   : pyin says singing -> silence
  voiced_drop : share of voiced frames falls by >= 0.5 from the 4 frames before to the 4 frames after
  quieter     : loudness falls >= 1 dB in one frame (local peak of the drop)
  pitch_leave : the sung pitch leaves the note's pitch (within 0.5 semitone -> not)
  pitch_jump  : the sung pitch jumps >= 0.5 semitone between frames
  next_start  : the next predicted note's start (exact time)
  bp_off      : Basic Pitch's 'this pitch is sounding' signal at the note pitch (+-1 bin) falls below 0.3
Clues come from the audio, Basic Pitch activations and the predicted notes only.
Labels (answer key, never a clue): true note = reference note with the same pitch (< 0.5 st) whose start is the
nearest within 50 ms of this note's start (= what mir_eval would match). An option is RIGHT if it lies within
max(50 ms, 20% of the true note's duration) of the true end (mir_eval's offset rule).
Output: final_pipeline/experiments/end_snap/end_candidates_train.csv
"""
import numpy as np
import pandas as pd
import librosa
from common import ROOT, HERE, OUT, BPACT, SUFFIX, load

SR, HOP = 22050, 256
FRAME = HOP / SR
WIN = 0.15
FAR = 12.0
SIDE = 5                      # ~58 ms
BP_THR, MIDI_OFFSET = 0.3, 21
F0 = np.load(HERE / 'hann_qian_pitch_model' / 'f0_tracks.npz')
KINDS = ['voice_off', 'voiced_drop', 'quieter', 'pitch_leave', 'pitch_jump', 'next_start', 'bp_off']


def bp_times(n):
    """basic_pitch.note_creation.model_frames_to_time (window-overlap correction included)."""
    t = np.arange(n) * HOP / SR
    off = (HOP / SR) * (172 - (SR * 2 - HOP) / HOP) + 0.0018
    return t - off * np.floor(np.arange(n) / 172)


def peaks(v, height):
    return np.flatnonzero((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] >= height)) + 1


def curves(rec):
    rid = rec.recording_id
    y, _ = librosa.load(ROOT / rec.audio_path, sr=SR, mono=True)
    flux = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    flux = flux / (np.percentile(flux, 99) + 1e-9)
    db = librosa.amplitude_to_db(librosa.feature.rms(y=y, hop_length=HOP)[0], ref=np.max)
    f0 = F0[f'{rid}_f0']
    n = min(len(flux), len(db), len(f0))
    db, flux = db[:n], flux[:n]
    midi = librosa.hz_to_midi(f0[:n])
    voiced = ~np.isnan(midi)
    jump = np.abs(np.diff(np.nan_to_num(midi), prepend=0.0))
    jump[~voiced] = 0; jump[1:][~voiced[:-1]] = 0
    vf = voiced.astype(float)
    k = np.ones(4)
    before = np.convolve(np.r_[np.zeros(4), vf], k, 'valid')[:n] / 4          # frames i-4..i-1
    after = np.convolve(np.r_[vf, np.zeros(4)], k, 'valid')[:n] / 4           # frames i..i+3
    drop = -np.diff(db, prepend=db[0])
    act = np.load(BPACT / f'{rid}.npz')
    bt = bp_times(len(act['note']))
    t = np.arange(n) * FRAME
    idx = np.clip(np.searchsorted(bt, t), 0, len(bt) - 1)
    idx = np.where((idx > 0) & (np.abs(bt[np.maximum(idx - 1, 0)] - t) < np.abs(bt[idx] - t)), idx - 1, idx)
    return dict(t=t, flux=flux, db=db, drop=drop, midi=midi, voiced=voiced, jump=jump, vshare_drop=before - after,
                bp_note=act['note'][idx], bp_onset=act['onset'][idx].max(axis=1))


def share(c, key, a, b):
    m = (c['t'] >= a) & (c['t'] < b)
    return float(c[key][m].mean()) if m.any() else 0.0


def sung_gap(c, a, b, pitch):
    seg = c['midi'][(c['t'] >= a) & (c['t'] < b)]
    seg = seg[~np.isnan(seg)]
    return min(abs(np.median(seg) - pitch), FAR) if len(seg) else FAR


def rows_for_recording(rec, notes):
    c = curves(rec)
    t, n_fr = c['t'], len(c['t'])
    glob = {'voice_off': np.flatnonzero(~c['voiced'][1:] & c['voiced'][:-1]) + 1,
            'voiced_drop': peaks(c['vshare_drop'], 0.5),
            'quieter': peaks(c['drop'], 1.0),
            'pitch_jump': np.flatnonzero(c['jump'] >= 0.5)}
    notes = notes.sort_values('start').reset_index(drop=True)
    nxt_start = notes.start.shift(-1).to_numpy()
    nxt_end = notes.end.shift(-1).to_numpy()
    nxt_pitch = notes.pitch.shift(-1).to_numpy()
    prv_end = notes.end.shift(1).to_numpy()
    rows = []
    for j, n in enumerate(notes.itertuples()):
        b = int(round(n.pitch)) - MIDI_OFFSET
        bins = list(range(max(b - 1, 0), min(b + 2, c['bp_note'].shape[1])))
        act = c['bp_note'][:, bins].max(axis=1)
        on = np.abs(c['midi'] - n.pitch) < 0.5
        kinds = dict(glob)
        kinds['pitch_leave'] = np.flatnonzero(on[:-1] & ~on[1:]) + 1
        kinds['bp_off'] = np.flatnonzero((act[:-1] >= BP_THR) & (act[1:] < BP_THR)) + 1
        here = min(int(round(n.end / FRAME)), n_fr - 1)
        ev = []                                                          # (time, frame, kind)
        for k, fr in kinds.items():
            for f in fr[np.abs(t[fr] - n.end) <= WIN]:
                ev.append((t[f], f, k))
        ns = nxt_start[j]
        if np.isfinite(ns) and abs(ns - n.end) <= WIN:
            ev.append((ns, min(int(round(ns / FRAME)), n_fr - 1), 'next_start'))
        # group: within 2 frames of the current end = 'stay'; other moments closer than 2 frames are one moment
        stay_kinds = {k for tt, f, k in ev if abs(f - here) <= 2}
        rest = sorted((e for e in ev if abs(e[1] - here) > 2), key=lambda e: e[0])
        groups = []
        for e in rest:
            if groups and e[1] - groups[-1][-1][1] <= 2:
                groups[-1].append(e)
            else:
                groups.append([e])
        options = [(n.end, here, stay_kinds, True)]
        for g in groups:
            ks = {e[2] for e in g}
            tt = next((e[0] for e in g if e[2] == 'next_start'), g[0][0])   # exact next start wins
            if tt <= n.start + 0.02:
                continue                                                  # end must stay after the start
            options.append((tt, g[0][1], ks, False))
        dur = n.end - n.start
        note_db = np.median(c['db'][(t >= n.start) & (t < n.end)]) if dur > FRAME else c['db'][here]
        for tt, i, ks, stay in options:
            lo, hi = max(i - 1, 0), min(i + 2, n_fr)
            ab, aa = (float(act[max(i - SIDE, 0):i].mean()) if i > 0 else 0.0), float(act[i:i + SIDE].mean())
            rows.append(dict(
                note_id=n.note_id, recording_id=n.recording_id, t=tt, dt=tt - n.end, is_stay=float(stay),
                **{f'is_{k}': float(k in ks) for k in KINDS},
                drop=c['drop'][lo:hi].max(), flux=c['flux'][lo:hi].max(), pitch_jump=c['jump'][lo:hi].max(),
                vshare_drop=c['vshare_drop'][lo:hi].max(),
                voiced_before=share(c, 'voiced', tt - 0.05, tt), voiced_after=share(c, 'voiced', tt, tt + 0.05),
                gap_before=sung_gap(c, tt - 0.05, tt, n.pitch), gap_after=sung_gap(c, tt, tt + 0.05, n.pitch),
                gap_after_next=(sung_gap(c, tt, tt + 0.05, nxt_pitch[j]) if np.isfinite(nxt_pitch[j]) else FAR),
                db_before=share(c, 'db', tt - 0.05, tt) - note_db, db_after=share(c, 'db', tt, tt + 0.05) - note_db,
                bp_note_before=ab, bp_note_after=aa, bp_note_fall=ab - aa,
                bp_onset_any=float(c['bp_onset'][max(i - 2, 0):i + 3].max()),
                new_dur=tt - n.start, note_dur=dur, note_pitch=n.pitch, is_piece=float('~' in n.note_id),
                next_gap=(ns - n.end) if np.isfinite(ns) else 5.0,
                to_next_start=min(ns - tt, 5.0) if np.isfinite(ns) else 5.0,
                to_next_end=min(nxt_end[j] - tt, 5.0) if np.isfinite(ns) else 5.0,
                next_pitch_diff=abs(nxt_pitch[j] - n.pitch) if np.isfinite(ns) else FAR,
                prev_gap=(n.start - prv_end[j]) if np.isfinite(prv_end[j]) else 5.0,
            ))
    return rows


def add_labels(cands, notes, references):
    ref = {rid: g for rid, g in references.groupby('recording_id')}
    tgt = {}
    for n in notes.itertuples():
        g = ref.get(n.recording_id)
        if g is None:
            continue
        g = g[(np.abs(g.pitch - n.pitch) < 0.5) & (np.abs(g.start - n.start) <= 0.05)]
        if len(g):
            r = g.iloc[np.argmin(np.abs(g.start - n.start))]
            tgt[n.note_id] = (r.end, max(0.05, 0.2 * (r.end - r.start)))
    out = cands.copy()
    te = out.note_id.map({k: v[0] for k, v in tgt.items()})
    tol = out.note_id.map({k: v[1] for k, v in tgt.items()})
    out['label'] = ((out.t - te).abs() <= tol).astype(int)
    out['labelled_note'] = te.notna()
    out['_true_end'] = te                                                # kept only for analysis, never a clue
    return out


if __name__ == '__main__':
    import time
    manifest, references, folds, chain = load()
    t0, rows = time.time(), []
    for k, rec in enumerate(manifest.itertuples()):
        assert rec.audio_path.startswith('data/train/')
        rows += rows_for_recording(rec, chain[chain.recording_id == rec.recording_id])
        if k % 100 == 0:
            print(k, f'{time.time() - t0:.0f}s', flush=True)
    cands = pd.DataFrame(rows)
    g = cands.groupby('note_id')
    cands['n_options'] = g.t.transform('size')
    cands['abs_dt'] = cands.dt.abs()
    stay = cands[cands.is_stay == 1].set_index('note_id')
    for col in ['drop', 'vshare_drop', 'bp_note_fall', 'pitch_jump', 'flux']:
        cands[f'{col}_rank'] = g[col].rank(ascending=False, method='min')
        cands[f'{col}_vs_stay'] = cands[col] - cands.note_id.map(stay[col])
    cands = add_labels(cands, chain, references)
    assert set(cands.note_id) == set(chain.note_id)
    cands.to_csv(OUT / f'end_candidates_train{SUFFIX}.csv', index=False)
    print('saved', len(cands), 'options for', cands.note_id.nunique(), 'notes', f'{time.time() - t0:.0f}s')

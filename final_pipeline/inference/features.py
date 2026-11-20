"""Clues for every stage of chain v4, for ANY recording (no answer key, no TextGrid, no training files needed).

Each function is copied from the script that made the chain v4 tables, with the file loading swapped for a
`Recording` object that computes the audio tracks once:
  timing  : notebooks/correction_handover/le_xin_timing_starter.ipynb (Steps 5a, 5c)
  split   : experiments/split_v2/common.py, clues.py, s3_candidates.py, s8_val_features.py
  delete  : experiments/delete_v2/features.py
  snap    : experiments/start_snap_v3/common.py (build_options and helpers)
  end snap: experiments/end_snap/build_candidates.py, val_stage4.py
check_val.py proves the copies give the same val tables as the original scripts.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import librosa

SR, HOP = 22050, 256
FRAME = HOP / SR
FAR = 12.0
MIDI_OFFSET = 21                       # Basic Pitch bin 0 = MIDI 21


def bp_times(n):
    """basic_pitch.note_creation.model_frames_to_time (window-overlap correction included)."""
    t = np.arange(n) * HOP / SR
    off = (HOP / SR) * (172 - (SR * 2 - HOP) / HOP) + 0.0018
    return t - off * np.floor(np.arange(n) / 172)


class Recording:
    """One recording: its audio, Basic Pitch activations and every track the stages need (computed once)."""

    def __init__(self, recording_id, audio_path, activations, duration):
        self.rid, self.path, self.duration = recording_id, Path(audio_path), float(duration)
        self.act = {k: np.asarray(activations[k]) for k in ('onset', 'note', 'contour')}
        self._c = {}

    def _get(self, key, make):
        if key not in self._c:
            self._c[key] = make()
        return self._c[key]

    @property
    def y(self):
        return self._get('y', lambda: librosa.load(self.path, sr=SR, mono=True)[0])

    @property
    def flux(self):                    # onset strength, scaled per recording
        def make():
            f = librosa.onset.onset_strength(y=self.y, sr=SR, hop_length=HOP)
            return f / (np.percentile(f, 99) + 1e-9)
        return self._get('flux', make)

    @property
    def db(self):
        return self._get('db', lambda: librosa.amplitude_to_db(librosa.feature.rms(y=self.y, hop_length=HOP)[0], ref=np.max))

    @property
    def pyin(self):                    # Le Xin's pyin (65-1100 Hz): timing pitch clues and split tracks
        return self._get('pyin', lambda: librosa.pyin(self.y, fmin=65, fmax=1100, sr=SR, hop_length=HOP))

    @property
    def f0(self):                      # Hann Qian's pyin (C2-C6): delete, snap and end-snap clues
        def make():
            f0, voiced, _ = librosa.pyin(self.y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'),
                                         sr=SR, hop_length=HOP)
            return librosa.times_like(f0, sr=SR, hop_length=HOP), np.where(voiced, f0, np.nan)
        return self._get('f0', make)

    @property
    def mfcc(self):
        return self._get('mfcc', lambda: librosa.feature.mfcc(y=self.y, sr=SR, hop_length=HOP, n_mfcc=13))


# =============================================================== stage 1: timing (31 clues per Basic Pitch note)
WIN = 0.10


def _near(times, values, t, pick):
    w = np.flatnonzero(np.abs(times - t) <= WIN)
    if len(w) == 0:
        return 0.0, 0.0
    i = w[pick(values[w])]
    return times[i] - t, values[i]


def _transition_dt(times, voiced, t, to_voiced):
    switch = np.flatnonzero(voiced[1:] != voiced[:-1]) + 1
    switch = switch[voiced[switch] == to_voiced]
    dts = times[switch] - t
    dts = dts[np.abs(dts) <= WIN]
    return (dts[np.argmin(np.abs(dts))], 1.0) if len(dts) else (0.0, 0.0)


def _pitch_gap(times, midi, a, b, note_pitch):
    seg = midi[(times >= a) & (times < b)]
    seg = seg[~np.isnan(seg)]
    return min(abs(np.median(seg) - note_pitch), FAR) if len(seg) else FAR


def timing_clues(rec, notes):
    """Loudness (10) + pitch (14) clues near each note's start and end. notes: one recording's raw Basic Pitch notes."""
    flux, db = rec.flux, rec.db
    rise = np.diff(db, prepend=db[0])
    times = librosa.frames_to_time(np.arange(len(flux)), sr=SR, hop_length=HOP)
    f0, voiced, vprob = rec.pyin
    midi = np.where(voiced, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)
    jump = np.abs(np.gradient(np.nan_to_num(midi, nan=0.0)))
    jump[np.isnan(midi)] = 0.0
    ptimes = librosa.frames_to_time(np.arange(len(midi)), sr=SR, hop_length=HOP)
    voiced = voiced.astype(bool)
    rows = []
    for n in notes.itertuples():
        s_flux_dt, s_flux = _near(times, flux, n.start, np.argmax)
        s_rise_dt, s_rise = _near(times, rise, n.start, np.argmax)
        e_drop_dt, e_drop = _near(times, rise, n.end, np.argmin)
        e_flux_dt, e_flux = _near(times, flux, n.end, np.argmax)
        frame = lambda t: min(int(round(t * SR / HOP)), len(db) - 1)
        pframe = lambda t: min(int(round(t * SR / HOP)), len(ptimes) - 1)
        von_dt, von_found = _transition_dt(ptimes, voiced, n.start, True)
        voff_dt, voff_found = _transition_dt(ptimes, voiced, n.end, False)
        sj_dt, sj = _near(ptimes, jump, n.start, np.argmax)
        ej_dt, ej = _near(ptimes, jump, n.end, np.argmax)
        rows.append(dict(note_id=n.note_id,
                         s_flux_dt=s_flux_dt, s_flux=s_flux, s_rise_dt=s_rise_dt, s_rise=s_rise,
                         e_drop_dt=e_drop_dt, e_drop=e_drop, e_flux_dt=e_flux_dt, e_flux=e_flux,
                         db_at_start=db[frame(n.start)], db_at_end=db[frame(n.end)],
                         p_von_dt=von_dt, p_von_found=von_found, p_voff_dt=voff_dt, p_voff_found=voff_found,
                         p_sjump_dt=sj_dt, p_sjump=sj, p_ejump_dt=ej_dt, p_ejump=ej,
                         p_vprob_start=vprob[pframe(n.start)], p_vprob_end=vprob[pframe(n.end)],
                         p_gap_before_start=_pitch_gap(ptimes, midi, n.start - 0.05, n.start, n.pitch),
                         p_gap_after_start=_pitch_gap(ptimes, midi, n.start, n.start + 0.05, n.pitch),
                         p_gap_before_end=_pitch_gap(ptimes, midi, n.end - 0.05, n.end, n.pitch),
                         p_gap_after_end=_pitch_gap(ptimes, midi, n.end, n.end + 0.05, n.pitch)))
    return pd.DataFrame(rows)


# =============================================================== stage 2: split (74 clues per cut candidate)
SIDE, EDGE = 5, 0.06


def split_tracks(rec):
    f0, voiced, vprob = rec.pyin
    midi = np.where(voiced, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)
    flux, db, mfcc = rec.flux, rec.db, rec.mfcc
    n = min(len(midi), len(flux), len(db), mfcc.shape[1])
    return dict(midi=midi[:n], voiced=voiced[:n].astype(bool), vprob=vprob[:n], flux=flux[:n], db=db[:n],
                mfcc=mfcc[:, :n].astype(np.float32))


def _med(midi, a, b):
    seg = midi[max(a, 0):max(b, 0)]
    seg = seg[~np.isnan(seg)]
    return np.median(seg) if len(seg) >= 2 else np.nan


def _candidates_for_note(n, midi, voiced, vprob, flux, db, extra_frames):
    lo, hi = int(np.ceil((n.start + EDGE) / FRAME)), int((n.end - EDGE) / FRAME)
    hi = min(hi, len(midi) - 1)
    if hi <= lo:
        return []
    frames = np.arange(lo, hi + 1)
    before = np.array([_med(midi, i - SIDE, i) for i in frames])
    after = np.array([_med(midi, i, i + SIDE) for i in frames])
    change = np.nan_to_num(np.abs(after - before), nan=0.0)

    def peaks(score, floor):
        return {f for k, f in enumerate(frames)
                if score[k] >= floor and score[k] == score[max(0, k - 3):k + 4].max()}
    picked = peaks(change, 0.5) | peaks(flux[frames], 0.25) | {f for f in frames if voiced[f] and not voiced[f - 1]}
    source = {f: 'old' for f in picked}
    for f, s in extra_frames(n, lo, hi).items():
        if all(abs(f - g) > 3 for g in picked) and f not in source:
            source[f] = s
    strength = {f: change[f - lo] + flux[f] for f in source}
    kept = []
    for f in sorted([f for f in source if source[f] == 'old'], key=lambda f: -strength[f]):
        if all(abs(f - g) > 3 for g in kept):
            kept.append(f)
    for f in sorted([f for f in source if source[f] != 'old'], key=lambda f: -strength[f]):
        if all(abs(f - g) > 3 for g in kept):
            kept.append(f)
    rows = []
    note_lo, note_hi = int(n.start / FRAME), min(int(n.end / FRAME), len(midi))
    note_db = np.median(db[note_lo:note_hi]) if note_hi > note_lo else db[lo]
    for f in sorted(kept):
        k, t = f - lo, f * FRAME
        b, a = before[k], after[k]
        rows.append(dict(
            note_id=n.note_id, recording_id=n.recording_id, t=t,
            since_start=t - n.start, to_end=n.end - t, rel_pos=(t - n.start) / (n.end - n.start),
            note_dur=n.end - n.start, note_pitch=n.pitch,
            pitch_change=change[k], pitch_change_signed=0.0 if np.isnan(a - b) else a - b,
            before_vs_note=FAR if np.isnan(b) else min(abs(b - n.pitch), FAR),
            after_vs_note=FAR if np.isnan(a) else min(abs(a - n.pitch), FAR),
            flux=flux[max(f - 2, 0):f + 3].max(),
            rise=np.diff(db[max(f - 3, 0):f + 4]).max() if f + 4 - max(f - 3, 0) > 1 else 0.0,
            dip=note_db - db[max(f - 3, 0):f + 4].min(),
            silent_near=float((~voiced[max(f - 3, 0):f + 4]).any()),
            vprob=vprob[f],
            piece_pitch_before=_med(midi, max(int(t / FRAME) - 13, note_lo), f),
            piece_pitch_after=_med(midi, f, min(f + 13, note_hi)),
            source=source[f]))
    return rows


def _frame_tracks(tr, bp):
    midi, db, mfcc = tr['midi'], tr['db'], tr['mfcc'][1:]
    n = len(midi)
    cs = np.concatenate([np.zeros((mfcc.shape[0], 1)), np.cumsum(mfcc, axis=1)], axis=1)
    spec = np.zeros(n)
    for f in range(5, n - 5):
        mb, ma = (cs[:, f] - cs[:, f - 5]) / 5, (cs[:, f + 5] - cs[:, f]) / 5
        spec[f] = 1 - mb @ ma / (np.linalg.norm(mb) * np.linalg.norm(ma) + 1e-9)
    valley = np.zeros(n)
    for f in range(9, n - 9):
        valley[f] = min(db[f - 9:f - 2].max(), db[f + 2:f + 9].max()) - db[f - 3:f + 4].min()
    on = bp['onset'].max(axis=1)
    bpon = np.interp(np.arange(n) * FRAME, bp_times(len(on)), on)
    return dict(spec=spec, valley=valley, bpon=bpon)


def _frame_peaks(score, frames, floor):
    s = score[frames]
    return {int(f) for k, f in enumerate(frames) if s[k] >= floor and s[k] == s[max(0, k - 3):k + 4].max()}


def _settle(midi, f, nmax=13, run=4):
    for j in range(f, min(f + nmax, len(midi) - run)):
        seg = midi[j:j + run]
        if not np.isnan(seg).any() and np.ptp(seg) < 0.5:
            return j, np.median(midi[j:j + run + 4][~np.isnan(midi[j:j + run + 4])])
    return None, np.nan


def _stable_segments(midi, lo, hi, run=6):
    r = np.round(midi[lo:hi])
    segs, k = [], 0
    while k < len(r):
        if np.isnan(r[k]):
            k += 1
            continue
        j = k
        while j < len(r) and r[j] == r[k]:
            j += 1
        if j - k >= run and (not segs or segs[-1] != r[k]):
            segs.append(r[k])
        k = j
    return len(segs), (np.ptp(segs) if segs else 0.0)


def _nanmed(x):
    x = x[~np.isnan(x)]
    return np.median(x) if len(x) >= 2 else np.nan


def _split_rich(cands, notes, tr, bp):
    """experiments/split_v2/clues.per_recording."""
    midi, db, mfcc, voiced = tr['midi'], tr['db'], tr['mfcc'], tr['voiced']
    n = len(midi)
    on, nt = bp['onset'], bp['note']
    bt = bp_times(len(on))
    ns = notes.set_index('note_id')
    order = notes.sort_values('start')
    prev_end = dict(zip(order.note_id, order.end.shift(1)))
    next_start = dict(zip(order.note_id, order.start.shift(-1)))
    note_info, rows = {}, []
    for c in cands.itertuples():
        f = min(max(int(round(c.t / FRAME)), 1), n - 1)
        b9, a9 = midi[max(f - 9, 0):f], midi[f:f + 9]
        pb, pa = _nanmed(b9), _nanmed(a9)
        sj, sval = _settle(midi, f)
        note = ns.loc[c.note_id]
        lo_n, hi_n = int(note.start / FRAME), min(int(note.end / FRAME), n)
        if c.note_id not in note_info:
            nseg, nrange = _stable_segments(midi, lo_n, hi_n)
            inside = midi[lo_n:hi_n]
            inside = inside[~np.isnan(inside)]
            spread = np.percentile(inside, 90) - np.percentile(inside, 10) if len(inside) > 4 else 0.0
            note_info[c.note_id] = (nseg, nrange, spread, float(np.mean(~voiced[lo_n:hi_n])) if hi_n > lo_n else 0.0)
        nseg, nrange, spread, unvoiced_share = note_info[c.note_id]
        mb, ma = mfcc[1:, max(f - 5, 0):f].mean(axis=1), mfcc[1:, f:f + 5].mean(axis=1)
        spec = 1 - np.dot(mb, ma) / (np.linalg.norm(mb) * np.linalg.norm(ma) + 1e-9)
        left, right = db[max(f - 9, 0):max(f - 2, 1)], db[f + 2:f + 9]
        centre = db[max(f - 3, 0):f + 4]
        valley = (min(left.max(), right.max()) - centre.min()) if len(left) and len(right) else 0.0
        rise50 = db[f:f + 4].mean() - db[max(f - 4, 0):f].mean()
        i = int(np.argmin(np.abs(bt - c.t)))
        lo, hi = max(i - 2, 0), min(i + 3, len(on))

        def bins(p):
            b = int(round(p)) - MIDI_OFFSET
            return list(range(max(b - 1, 0), min(b + 2, on.shape[1]))) if 0 <= b + 1 and b - 1 < on.shape[1] else []
        old_b = bins(c.note_pitch)
        new_b = bins(pa if not np.isnan(pa) else c.note_pitch)

        def act(arr, bb, a, z):
            return float(arr[max(a, 0):max(z, 0), bb].max(axis=1).mean()) if bb and z > max(a, 0) else 0.0
        rows.append(dict(
            pb100=pb, pa100=pa,
            change100=FAR if np.isnan(pa - pb) else min(abs(pa - pb), FAR),
            round_change100=FAR if np.isnan(pa - pb) else min(abs(round(pa) - round(pb)), FAR),
            std_b100=min(np.nanstd(b9), FAR) if np.sum(~np.isnan(b9)) > 1 else FAR,
            std_a100=min(np.nanstd(a9), FAR) if np.sum(~np.isnan(a9)) > 1 else FAR,
            voiced_b100=float(np.mean(~np.isnan(b9))) if len(b9) else 0.0,
            voiced_a100=float(np.mean(~np.isnan(a9))) if len(a9) else 0.0,
            settle_dt=(sj - f) * FRAME if sj is not None else 0.2,
            has_settle=float(sj is not None),
            settle_new=FAR if (np.isnan(sval) or np.isnan(pb)) else min(abs(sval - pb), FAR),
            settle_vs_note=FAR if np.isnan(sval) else min(abs(sval - c.note_pitch), FAR),
            after100_vs_note=FAR if np.isnan(pa) else min(abs(pa - c.note_pitch), FAR),
            before100_vs_note=FAR if np.isnan(pb) else min(abs(pb - c.note_pitch), FAR),
            spec_change=spec, valley=valley, rise50=rise50,
            bp_onset_any=float(on[lo:hi].max()),
            bp_onset_old=float(on[lo:hi, old_b].max()) if old_b else 0.0,
            bp_onset_new=float(on[lo:hi, new_b].max()) if new_b else 0.0,
            bp_new_rise=act(nt, new_b, i, i + 5) - act(nt, new_b, i - 5, i),
            bp_old_fall=act(nt, old_b, i - 5, i) - act(nt, old_b, i, i + 5),
            note_stable_segs=nseg, note_seg_range=nrange, note_pitch_spread=spread, note_unvoiced=unvoiced_share,
            gap_before_note=(note.start - prev_end[c.note_id]) if not pd.isna(prev_end[c.note_id]) else 5.0,
            gap_after_note=(next_start[c.note_id] - note.end) if not pd.isna(next_start[c.note_id]) else 5.0))
    return pd.DataFrame(rows, index=cands.index)


def split_candidates(rec, notes):
    """Cut candidates inside each note of one recording, with their per-recording clues (s8_val_features.run)."""
    tr = split_tracks(rec)
    ft = _frame_tracks(tr, rec.act)

    def extra(n, lo, hi):
        frames = np.arange(lo, hi + 1)
        before = np.array([_med(tr['midi'], i - SIDE, i) for i in frames])
        after = np.array([_med(tr['midi'], i, i + SIDE) for i in frames])
        change = np.nan_to_num(np.abs(after - before), nan=0.0)
        out = {}
        for f in {int(frames[k]) for k in range(len(frames))
                  if change[k] >= 0.3 and change[k] == change[max(0, k - 3):k + 4].max()}:
            out[f] = 'pitch_small'
        for f in _frame_peaks(ft['bpon'], frames, 0.3):
            out.setdefault(f, 'bp_onset')
        for f in _frame_peaks(ft['valley'], frames, 3.0):
            out.setdefault(f, 'valley')
        for f in _frame_peaks(ft['spec'], frames, 0.15):
            out.setdefault(f, 'timbre')
        return out
    args = (tr['midi'], tr['voiced'], tr['vprob'], tr['flux'], tr['db'])
    cands = pd.DataFrame([r for n in notes.itertuples() for r in _candidates_for_note(n, *args, extra)])
    if cands.empty:
        return None
    return pd.concat([cands, _split_rich(cands, notes, tr, rec.act)], axis=1)


SPLIT_COMPARED = ['change100', 'settle_new', 'spec_change', 'valley', 'bp_onset_new', 'bp_new_rise', 'flux', 'pitch_change']


def split_note_clues(df):
    """split_v2 common.add_clues + clues.add_comparison (compares the candidates of one note)."""
    df = df.copy()
    df['cands_in_note'] = df.groupby('note_id').t.transform('size')
    df['strength_rank'] = df.assign(s=df.pitch_change + df.flux).groupby('note_id').s.rank(ascending=False)
    pb, pa = df.piece_pitch_before, df.piece_pitch_after
    df['piece_change'] = (pa.round() - pb.round()).abs().fillna(FAR)
    df['piece_after_vs_note'] = (pa - df.note_pitch).abs().fillna(FAR).clip(upper=FAR)
    df['piece_before_vs_note'] = (pb - df.note_pitch).abs().fillna(FAR).clip(upper=FAR)
    g = df.groupby('note_id')
    for col in SPLIT_COMPARED:
        df[f'{col}_rank'] = g[col].rank(ascending=False, method='min')
        df[f'{col}_vs_max'] = df[col] - g[col].transform('max')
    df['to_prev_cand'] = (df.t - g.t.shift(1)).fillna(df.since_start)
    df['to_next_cand'] = (g.t.shift(-1) - df.t).fillna(df.to_end)
    df['strong_in_note'] = g.change100.transform(lambda s: float((s >= 0.75).sum()))
    return df


# =============================================================== stage 3: delete (49 clues per note)
def _track_clues(t, midi, s, e, pitch):
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


def _act_clues(T, note, onset, s, e, pitch):
    b = int(round(pitch)) - MIDI_OFFSET
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


def delete_clues(pool, raw, timing_audio, recs):
    """pool: notes after split; raw: the Basic Pitch notes (note_id -> amplitude); timing_audio: timing_clues() of the
    raw notes (loudness per original note); recs: recording_id -> Recording. Returns pool rows sorted like features.py."""
    bp = raw.set_index('note_id')
    p = pool.copy()
    p['orig'] = p.note_id.str.split('~').str[0]
    v2 = ~p.orig.isin(bp.index)
    p.loc[v2, 'orig'] = p.loc[v2, 'orig'].str.replace(r'_[a-z]$', '', regex=True)
    assert p.orig.isin(bp.index).all()
    p = p.sort_values(['recording_id', 'start', 'pitch']).reset_index(drop=True)
    p['duration'] = p.end - p.start
    p['log_dur'] = np.log(p.duration)
    p['amplitude'] = p.orig.map(bp.amplitude)
    p['is_piece'] = (p.note_id != p.orig).astype(int)
    p['n_pieces'] = p.groupby('orig').note_id.transform('size')
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
    rows = []
    for rid, g in p.groupby('recording_id', sort=False):
        t, hz = recs[rid].f0
        midi = 69 + 12 * np.log2(hz / 440.)
        note, onset = recs[rid].act['note'], recs[rid].act['onset']
        T = bp_times(len(note))
        for i, r in g.iterrows():
            rows.append(dict(idx=i, **_track_clues(t, midi, r.start, r.end, r.pitch),
                             **_act_clues(T, note, onset, r.start, r.end, r.pitch)))
    p = p.join(pd.DataFrame(rows).set_index('idx'))
    p['bp_note_vs_rec'] = p.bp_note_mean - p.groupby('recording_id').bp_note_mean.transform('median')
    aud = timing_audio.set_index('note_id')
    for c in ['db_at_start', 'db_at_end', 's_flux', 's_rise', 'e_drop']:
        p['a_' + c] = p.orig.map(aud[c])
    p['a_db_vs_rec'] = p.a_db_at_start - p.groupby('recording_id').a_db_at_start.transform('median')
    return p


# =============================================================== stage 4: start snap (options + 47 clues)
SNAP_COMPARED = ['attack', 'louder', 'pitch_jump', 'bp_onset_at', 'bp_note_rise']


def _snap_curves(rec):
    flux, db = rec.flux, rec.db
    rise = np.diff(db, prepend=db[0])
    f0 = rec.f0[1]
    n = min(len(flux), len(rise), len(f0))
    midi = librosa.hz_to_midi(f0[:n])
    voiced = ~np.isnan(midi)
    jump = np.abs(np.diff(np.nan_to_num(midi), prepend=0.0))
    jump[~voiced] = 0; jump[1:][~voiced[:-1]] = 0
    return dict(t=librosa.frames_to_time(np.arange(n), sr=SR, hop_length=HOP),
                flux=flux[:n], rise=rise[:n], midi=midi, voiced=voiced, jump=jump)


def _peaks(v, height):
    return np.flatnonzero((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] >= height)) + 1


def _voiced_share(c, a, b):
    m = (c['t'] >= a) & (c['t'] < b)
    return float(c['voiced'][m].mean()) if m.any() else 0.0


def _sung_gap(c, a, b, pitch):
    seg = c['midi'][(c['t'] >= a) & (c['t'] < b)]
    seg = seg[~np.isnan(seg)]
    return min(abs(np.median(seg) - pitch), FAR) if len(seg) else FAR


def _settle_time(c, s0, p, win):
    on = np.abs(c['midi'] - p) < 0.5
    for j in np.flatnonzero((c['t'] >= s0 - win) & (c['t'] <= s0 + win)):
        if on[j:j + 4].size == 4 and on[j:j + 4].all():
            return c['t'][j]
    return np.nan


def snap_options(notes, recs, win=0.15):
    """snap_v3 common.build_options(notes, win, extra_kinds=True), without labels."""
    curves = {rid: recs[rid]._get('snap_curves', lambda r=recs[rid]: _snap_curves(r)) for rid in notes.recording_id.unique()}
    notes = notes.sort_values(['recording_id', 'start']).copy()
    notes['prev_gap'] = (notes.start - notes.groupby('recording_id').end.shift(1)).fillna(0.0)
    rows = []
    for rid, nts in notes.groupby('recording_id', sort=False):
        c = curves[rid]
        onset = recs[rid].act['onset']
        otimes = librosa.frames_to_time(np.arange(len(onset)), sr=SR, hop_length=HOP)
        otimes = otimes - ((HOP / SR) * (172 - (SR * 2 - HOP) / HOP) + 0.0018) * np.floor(np.arange(len(onset)) / 172)
        kinds = {'attack': _peaks(c['flux'], 0.1), 'louder': _peaks(c['rise'], 1.0),
                 'voice_on': np.flatnonzero(c['voiced'][1:] & ~c['voiced'][:-1]) + 1,
                 'pitch_jump': np.flatnonzero(c['jump'] >= 0.5)}
        all_frames = np.unique(np.concatenate(list(kinds.values())))
        near_kind = {k: set(np.concatenate([f - 1, f, f + 1])) for k, f in kinds.items()}
        for n in nts.itertuples():
            here = min(int(round(n.start / FRAME)), len(c['t']) - 1)
            b = int(round(n.pitch)) - MIDI_OFFSET
            v = onset[:, max(b - 1, 0):min(b + 2, onset.shape[1])].max(axis=1)
            bpf = np.round(otimes[_peaks(v, 0.3)] / FRAME).astype(int)
            st = _settle_time(c, n.start, n.pitch, win)
            sf = np.array([int(round(st / FRAME))]) if not np.isnan(st) else np.array([], int)
            bpf, sf = bpf[(bpf >= 0) & (bpf < len(c['t']))], sf[(sf >= 0) & (sf < len(c['t']))]
            extra = {'bp_onset': set(np.concatenate([bpf - 1, bpf, bpf + 1])),
                     'settle': set(np.concatenate([sf - 1, sf, sf + 1]))}
            cand = np.unique(np.concatenate([all_frames, bpf, sf]))
            frames = cand[(np.abs(c['t'][cand] - n.start) <= win) & (np.abs(cand - here) > 2)]
            merged = [f for i, f in enumerate(frames) if i == 0 or f - frames[i - 1] > 2]
            options = [(n.start, here, True)] + [(c['t'][f], f, False) for f in merged]
            for t, i, stay in options:
                lo, hi = max(i - 1, 0), min(i + 2, len(c['t']))
                near = set(range(i - 2, i + 3)) if stay else {i}
                r = dict(note_id=n.note_id, recording_id=rid, t=t, dt=t - n.start, is_stay=float(stay),
                         **{f'is_{k}': float(bool(near & near_kind[k])) for k in kinds})
                r.update({f'is_{k}': float(bool(near & s)) for k, s in extra.items()})
                r.update(attack=c['flux'][lo:hi].max(), louder=c['rise'][lo:hi].max(), pitch_jump=c['jump'][lo:hi].max(),
                         voiced_before=_voiced_share(c, t - 0.05, t), voiced_after=_voiced_share(c, t, t + 0.05),
                         gap_before=_sung_gap(c, t - 0.05, t, n.pitch), gap_after=_sung_gap(c, t, t + 0.05, n.pitch),
                         note_dur=n.end - n.start, note_pitch=n.pitch, is_piece=float('~' in n.note_id),
                         prev_gap=n.prev_gap)
                rows.append(r)
    o = pd.DataFrame(rows)
    g = o.groupby('note_id', sort=False)
    o['n_options'] = g.t.transform('size')
    o['attack_rank'] = g.attack.rank(ascending=False, method='min')
    o['abs_dt'] = o.dt.abs()
    o = _snap_bp_clues(o, recs)
    return _snap_new_clues(o, notes, recs)


def _snap_bp_clues(c, recs, W=2, SIDE_=5):
    vals = np.zeros((len(c), 4))
    for rid, idx in c.groupby('recording_id', sort=False).indices.items():
        onset, note = recs[rid].act['onset'], recs[rid].act['note']
        times = librosa.frames_to_time(np.arange(len(onset)), sr=SR, hop_length=HOP)
        times = times - ((HOP / SR) * (172 - (SR * 2 - HOP) / HOP) + 0.0018) * np.floor(np.arange(len(onset)) / 172)
        for r in idx:
            t, p = c.t.iat[r], c.note_pitch.iat[r]
            i = int(np.argmin(np.abs(times - t)))
            b = int(round(p)) - MIDI_OFFSET
            nb = list(range(max(b - 1, 0), min(b + 2, onset.shape[1])))
            lo, hi = max(i - W, 0), min(i + W + 1, len(onset))
            vals[r] = [float(onset[lo:hi, nb].max()), float(onset[lo:hi].max()),
                       float(note[max(i - SIDE_, 0):i, nb].max(axis=1).mean()) if i > 0 else 0.0,
                       float(note[i:i + SIDE_, nb].max(axis=1).mean())]
    for j, col in enumerate(['bp_onset_at', 'bp_onset_any', 'bp_note_before', 'bp_note_after']):
        c[col] = vals[:, j]
    return c


def _snap_new_clues(options, notes, recs):
    c = options.copy()
    c['bp_note_rise'] = c.bp_note_after - c.bp_note_before
    n = notes.sort_values(['recording_id', 'start']).copy()
    g = n.groupby('recording_id')
    n['prev_start'], n['prev_end'], n['next_start'] = g.start.shift(1), g.end.shift(1), g.start.shift(-1)
    n = n.set_index('note_id')
    c['to_prev_start'] = (c.t - c.note_id.map(n.prev_start)).fillna(5.0).clip(upper=5.0)
    c['to_prev_end'] = (c.t - c.note_id.map(n.prev_end)).fillna(5.0).clip(-5.0, 5.0)
    c['to_next_start'] = (c.note_id.map(n.next_start) - c.t).fillna(5.0).clip(upper=5.0)
    settle_dt = np.full(len(c), np.nan); match_100 = np.full(len(c), FAR); steady_100 = np.full(len(c), FAR)
    for rid, idx in c.groupby('recording_id', sort=False).indices.items():
        tt, hz = recs[rid].f0
        midi = librosa.hz_to_midi(hz)
        settled = {}
        for i in idx:
            nid, t, p = c.note_id.iat[i], c.t.iat[i], c.note_pitch.iat[i]
            if nid not in settled:
                s0 = n.at[nid, 'start']
                on = np.abs(midi - p) < 0.5
                hit = [j for j in np.flatnonzero((tt >= s0 - 0.15) & (tt <= s0 + 0.15)) if on[j:j + 4].size == 4 and on[j:j + 4].all()]
                settled[nid] = tt[hit[0]] if hit else np.nan
            settle_dt[i] = t - settled[nid]
            seg = midi[(tt >= t) & (tt < t + 0.10)]
            seg = seg[~np.isnan(seg)]
            if len(seg):
                match_100[i] = min(abs(np.median(seg) - p), FAR)
            if len(seg) > 1:
                steady_100[i] = min(np.std(seg), FAR)
    settle_dt = pd.Series(settle_dt, c.index)
    c['has_settle'] = settle_dt.notna().astype(float)
    c['settle_abs'] = settle_dt.abs().fillna(1.0)
    c['settle_dt'] = settle_dt.fillna(1.0)
    c['match_100'], c['steady_100'] = match_100, steady_100
    g = c.groupby('note_id', sort=False)
    stay = c[c.is_stay == 1].set_index('note_id')
    for col in SNAP_COMPARED:
        c[f'{col}_vs_stay'] = c[col] - c.note_id.map(stay[col])
        c[f'{col}_is_max'] = (c[col] == g[col].transform('max')).astype(float)
    c['match_rank'] = g.match_100.rank(method='min')
    c['settle_rank'] = g.settle_abs.rank(method='min')
    return c


# =============================================================== stage 5: end snap (options + 45 clues)
END_KINDS = ['voice_off', 'voiced_drop', 'quieter', 'pitch_leave', 'pitch_jump', 'next_start', 'bp_off']
END_WIN, BP_THR = 0.15, 0.3


def _end_curves(rec):
    flux, db = rec.flux, rec.db
    f0 = rec.f0[1]
    n = min(len(flux), len(db), len(f0))
    db, flux = db[:n], flux[:n]
    midi = librosa.hz_to_midi(f0[:n])
    voiced = ~np.isnan(midi)
    jump = np.abs(np.diff(np.nan_to_num(midi), prepend=0.0))
    jump[~voiced] = 0; jump[1:][~voiced[:-1]] = 0
    vf = voiced.astype(float)
    k = np.ones(4)
    before = np.convolve(np.r_[np.zeros(4), vf], k, 'valid')[:n] / 4
    after = np.convolve(np.r_[vf, np.zeros(4)], k, 'valid')[:n] / 4
    drop = -np.diff(db, prepend=db[0])
    act = rec.act
    bt = bp_times(len(act['note']))
    t = np.arange(n) * FRAME
    idx = np.clip(np.searchsorted(bt, t), 0, len(bt) - 1)
    idx = np.where((idx > 0) & (np.abs(bt[np.maximum(idx - 1, 0)] - t) < np.abs(bt[idx] - t)), idx - 1, idx)
    return dict(t=t, flux=flux, db=db, drop=drop, midi=midi, voiced=voiced, jump=jump, vshare_drop=before - after,
                bp_note=act['note'][idx], bp_onset=act['onset'][idx].max(axis=1))


def _share(c, key, a, b):
    m = (c['t'] >= a) & (c['t'] < b)
    return float(c[key][m].mean()) if m.any() else 0.0


def _end_rows(rec, notes):
    """end_snap build_candidates.rows_for_recording."""
    c = _end_curves(rec)
    t, n_fr = c['t'], len(c['t'])
    glob = {'voice_off': np.flatnonzero(~c['voiced'][1:] & c['voiced'][:-1]) + 1,
            'voiced_drop': _peaks(c['vshare_drop'], 0.5),
            'quieter': _peaks(c['drop'], 1.0),
            'pitch_jump': np.flatnonzero(c['jump'] >= 0.5)}
    notes = notes.sort_values('start').reset_index(drop=True)
    nxt_start, nxt_end = notes.start.shift(-1).to_numpy(), notes.end.shift(-1).to_numpy()
    nxt_pitch, prv_end = notes.pitch.shift(-1).to_numpy(), notes.end.shift(1).to_numpy()
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
        ev = []
        for k, fr in kinds.items():
            for f in fr[np.abs(t[fr] - n.end) <= END_WIN]:
                ev.append((t[f], f, k))
        ns = nxt_start[j]
        if np.isfinite(ns) and abs(ns - n.end) <= END_WIN:
            ev.append((ns, min(int(round(ns / FRAME)), n_fr - 1), 'next_start'))
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
            tt = next((e[0] for e in g if e[2] == 'next_start'), g[0][0])
            if tt <= n.start + 0.02:
                continue
            options.append((tt, g[0][1], ks, False))
        dur = n.end - n.start
        note_db = np.median(c['db'][(t >= n.start) & (t < n.end)]) if dur > FRAME else c['db'][here]
        for tt, i, ks, stay in options:
            lo, hi = max(i - 1, 0), min(i + 2, n_fr)
            ab, aa = (float(act[max(i - 5, 0):i].mean()) if i > 0 else 0.0), float(act[i:i + 5].mean())
            rows.append(dict(
                note_id=n.note_id, recording_id=n.recording_id, t=tt, dt=tt - n.end, is_stay=float(stay),
                **{f'is_{k}': float(k in ks) for k in END_KINDS},
                drop=c['drop'][lo:hi].max(), flux=c['flux'][lo:hi].max(), pitch_jump=c['jump'][lo:hi].max(),
                vshare_drop=c['vshare_drop'][lo:hi].max(),
                voiced_before=_share(c, 'voiced', tt - 0.05, tt), voiced_after=_share(c, 'voiced', tt, tt + 0.05),
                gap_before=_sung_gap(c, tt - 0.05, tt, n.pitch), gap_after=_sung_gap(c, tt, tt + 0.05, n.pitch),
                gap_after_next=(_sung_gap(c, tt, tt + 0.05, nxt_pitch[j]) if np.isfinite(nxt_pitch[j]) else FAR),
                db_before=_share(c, 'db', tt - 0.05, tt) - note_db, db_after=_share(c, 'db', tt, tt + 0.05) - note_db,
                bp_note_before=ab, bp_note_after=aa, bp_note_fall=ab - aa,
                bp_onset_any=float(c['bp_onset'][max(i - 2, 0):i + 3].max()),
                new_dur=tt - n.start, note_dur=dur, note_pitch=n.pitch, is_piece=float('~' in n.note_id),
                next_gap=(ns - n.end) if np.isfinite(ns) else 5.0,
                to_next_start=min(ns - tt, 5.0) if np.isfinite(ns) else 5.0,
                to_next_end=min(nxt_end[j] - tt, 5.0) if np.isfinite(ns) else 5.0,
                next_pitch_diff=abs(nxt_pitch[j] - n.pitch) if np.isfinite(ns) else FAR,
                prev_gap=(n.start - prv_end[j]) if np.isfinite(prv_end[j]) else 5.0))
    return rows


def end_options(notes, recs, order):
    """End-snap options + clues for every note (val_stage4.py). order: recording ids in processing order."""
    rows = []
    for rid in order:
        rows += _end_rows(recs[rid], notes[notes.recording_id == rid])
    o = pd.DataFrame(rows)
    g = o.groupby('note_id')
    o['n_options'] = g.t.transform('size')
    o['abs_dt'] = o.dt.abs()
    stay = o[o.is_stay == 1].set_index('note_id')
    for col in ['drop', 'vshare_drop', 'bp_note_fall', 'pitch_jump', 'flux']:
        o[f'{col}_rank'] = g[col].rank(ascending=False, method='min')
        o[f'{col}_vs_stay'] = o[col] - o.note_id.map(stay[col])
    return o

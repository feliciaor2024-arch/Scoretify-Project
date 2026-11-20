"""Richer per-candidate clues for the split model (audio + Basic Pitch activations + predicted notes only).

Ideas borrowed from the start-snap model (Step 4): pitch settle, steadiness over 100 ms, Basic Pitch note activation
rise, comparison with the other candidates of the same note. Never reads the answer key.
"""
import numpy as np
import pandas as pd
import common as C

MIDI_OFFSET = 21


def bp_times(n):
    """basic_pitch.note_creation.model_frames_to_time (basic-pitch is not installed in base python)."""
    t = np.arange(n) * 256 / 22050
    off = 256 / 22050 * (172 - (22050 * 2 - 256) / 256) + 0.0018
    return t - off * np.floor(np.arange(n) / 172)


def _med(x):
    x = x[~np.isnan(x)]
    return np.median(x) if len(x) >= 2 else np.nan


def _settle(midi, f, nmax=13, run=4):
    """First frame at/after f (within nmax frames) where pitch stays within 0.5 st of its run median for `run` frames."""
    for j in range(f, min(f + nmax, len(midi) - run)):
        seg = midi[j:j + run]
        if not np.isnan(seg).any() and np.ptp(seg) < 0.5:
            return j, np.median(midi[j:j + run + 4][~np.isnan(midi[j:j + run + 4])])
    return None, np.nan


def _stable_segments(midi, lo, hi, run=6):
    """Number of distinct rounded pitches held steady for >= run frames (~70 ms) between frames lo..hi."""
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


def per_recording(cands, notes, tr, bp):
    """cands: rows of one recording (note_id, t, note_pitch ...); notes: that recording's note table."""
    midi, db, mfcc, voiced, flux = tr['midi'], tr['db'], tr['mfcc'], tr['voiced'], tr['flux']
    n = len(midi)
    on, nt = bp['onset'], bp['note']
    bt = bp_times(len(on))
    ns = notes.set_index('note_id')
    order = notes.sort_values('start')
    prev_end = dict(zip(order.note_id, order.end.shift(1)))
    next_start = dict(zip(order.note_id, order.start.shift(-1)))
    note_info = {}
    rows = []
    for c in cands.itertuples():
        f = int(round(c.t / C.FRAME))
        f = min(max(f, 1), n - 1)
        b9, a9 = midi[max(f - 9, 0):f], midi[f:f + 9]
        pb, pa = _med(b9), _med(a9)
        sj, sval = _settle(midi, f)
        note = ns.loc[c.note_id]
        lo_n, hi_n = int(note.start / C.FRAME), min(int(note.end / C.FRAME), n)
        if c.note_id not in note_info:
            nseg, nrange = _stable_segments(midi, lo_n, hi_n)
            inside = midi[lo_n:hi_n]
            inside = inside[~np.isnan(inside)]
            spread = np.percentile(inside, 90) - np.percentile(inside, 10) if len(inside) > 4 else 0.0
            note_info[c.note_id] = (nseg, nrange, spread, float(np.mean(~voiced[lo_n:hi_n])) if hi_n > lo_n else 0.0)
        nseg, nrange, spread, unvoiced_share = note_info[c.note_id]
        # spectral (timbre) change: MFCC 1..12 mean over 5 frames before vs after, cosine distance
        mb, ma = mfcc[1:, max(f - 5, 0):f].mean(axis=1), mfcc[1:, f:f + 5].mean(axis=1)
        spec = 1 - np.dot(mb, ma) / (np.linalg.norm(mb) * np.linalg.norm(ma) + 1e-9)
        # loudness valley: how far loudness drops between the louder stretches on each side
        left, right = db[max(f - 9, 0):max(f - 2, 1)], db[f + 2:f + 9]
        centre = db[max(f - 3, 0):f + 4]
        valley = (min(left.max(), right.max()) - centre.min()) if len(left) and len(right) else 0.0
        rise50 = db[f:f + 4].mean() - db[max(f - 4, 0):f].mean()
        # Basic Pitch activations at the old (note) pitch and at the NEW pitch sung after the cut
        i = int(np.argmin(np.abs(bt - c.t)))
        lo, hi = max(i - 2, 0), min(i + 3, len(on))
        def bins(p):
            b = int(round(p)) - MIDI_OFFSET
            return list(range(max(b - 1, 0), min(b + 2, on.shape[1]))) if 0 <= b + 1 and b - 1 < on.shape[1] else []
        old_b = bins(c.note_pitch)
        new_p = pa if not np.isnan(pa) else c.note_pitch
        new_b = bins(new_p)
        def act(arr, bb, a, z):
            return float(arr[max(a, 0):max(z, 0), bb].max(axis=1).mean()) if bb and z > max(a, 0) else 0.0
        rows.append(dict(
            pb100=pb, pa100=pa,
            change100=FAR if np.isnan(pa - pb) else min(abs(pa - pb), C.FAR),
            round_change100=FAR if np.isnan(pa - pb) else min(abs(round(pa) - round(pb)), C.FAR),
            std_b100=min(np.nanstd(b9), C.FAR) if np.sum(~np.isnan(b9)) > 1 else C.FAR,
            std_a100=min(np.nanstd(a9), C.FAR) if np.sum(~np.isnan(a9)) > 1 else C.FAR,
            voiced_b100=float(np.mean(~np.isnan(b9))) if len(b9) else 0.0,
            voiced_a100=float(np.mean(~np.isnan(a9))) if len(a9) else 0.0,
            settle_dt=(sj - f) * C.FRAME if sj is not None else 0.2,
            has_settle=float(sj is not None),
            settle_new=C.FAR if (np.isnan(sval) or np.isnan(pb)) else min(abs(sval - pb), C.FAR),
            settle_vs_note=C.FAR if np.isnan(sval) else min(abs(sval - c.note_pitch), C.FAR),
            after100_vs_note=C.FAR if np.isnan(pa) else min(abs(pa - c.note_pitch), C.FAR),
            before100_vs_note=C.FAR if np.isnan(pb) else min(abs(pb - c.note_pitch), C.FAR),
            spec_change=spec,
            valley=valley, rise50=rise50,
            bp_onset_any=float(on[lo:hi].max()),
            bp_onset_old=float(on[lo:hi, old_b].max()) if old_b else 0.0,
            bp_onset_new=float(on[lo:hi, new_b].max()) if new_b else 0.0,
            bp_new_rise=act(nt, new_b, i, i + 5) - act(nt, new_b, i - 5, i),
            bp_old_fall=act(nt, old_b, i - 5, i) - act(nt, old_b, i, i + 5),
            note_stable_segs=nseg, note_seg_range=nrange, note_pitch_spread=spread, note_unvoiced=unvoiced_share,
            gap_before_note=(note.start - prev_end[c.note_id]) if not pd.isna(prev_end[c.note_id]) else 5.0,
            gap_after_note=(next_start[c.note_id] - note.end) if not pd.isna(next_start[c.note_id]) else 5.0,
        ))
    return pd.DataFrame(rows, index=cands.index)


FAR = C.FAR
NEW_BASE = ['change100', 'round_change100', 'std_b100', 'std_a100', 'voiced_b100', 'voiced_a100', 'settle_dt',
            'has_settle', 'settle_new', 'settle_vs_note', 'after100_vs_note', 'before100_vs_note', 'spec_change',
            'valley', 'rise50', 'bp_onset_any', 'bp_onset_old', 'bp_onset_new', 'bp_new_rise', 'bp_old_fall',
            'note_stable_segs', 'note_seg_range', 'note_pitch_spread', 'note_unvoiced', 'gap_before_note',
            'gap_after_note']
COMPARED = ['change100', 'settle_new', 'spec_change', 'valley', 'bp_onset_new', 'bp_new_rise', 'flux', 'pitch_change']


def add_comparison(df):
    """Compare each candidate with the other candidates of the same note."""
    df = df.copy()
    g = df.groupby('note_id')
    for col in COMPARED:
        df[f'{col}_rank'] = g[col].rank(ascending=False, method='min')
        df[f'{col}_vs_max'] = df[col] - g[col].transform('max')
    t = df.t.to_numpy()
    prev_t = g.t.shift(1)
    next_t = g.t.shift(-1)
    df['to_prev_cand'] = (df.t - prev_t).fillna(df.since_start)
    df['to_next_cand'] = (next_t - df.t).fillna(df.to_end)
    # segment pitch: piece between this candidate and its neighbours (use 100 ms medians as proxy)
    df['strong_in_note'] = g.change100.transform(lambda s: float((s >= 0.75).sum()))
    return df


COMP_FEATS = [f'{c}_{s}' for c in COMPARED for s in ['rank', 'vs_max']] + ['to_prev_cand', 'to_next_cand',
                                                                          'strong_in_note']

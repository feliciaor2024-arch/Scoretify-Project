"""Shared helpers for the split v2 experiments (training songs only; test is never opened).

Copies the split notebook's candidate finder, labels and apply_splits so the old model can be rebuilt exactly,
and adds a fast end-to-end scorer that gives the same numbers as cu.evaluate_end_to_end (it caches each
recording's match counts, so only recordings whose notes changed are re-scored).
"""
from pathlib import Path
import sys
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'notebooks' / 'correction_handover'
OUT = Path(__file__).resolve().parent
CACHE = OUT / 'cache'
DATA = ROOT / 'basic_pitch_runs' / 'correction_handover'
BPACT = Path(r'C:\Users\derp2\AppData\Local\Temp\claude\C--Users-derp2-SMU-Y2SEM1-DMBA-GTSinger-Correction-Group-Handover'
             r'\6efc453f-37d1-4798-af6f-53bef5f326a6\scratchpad\bpact_train')
BPACT_VAL = BPACT.parent / 'bpact_val'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'notebooks'))
import correction_utils as cu
import bp_tuning_core as core

SR, HOP = 22050, 256
FRAME = HOP / SR
SIDE = 5
EDGE = 0.06
FAR = 12.0


def load():
    manifest = pd.read_csv(DATA / 'training_recording_manifest.csv')
    references = pd.read_csv(DATA / 'references_train.csv', dtype={'note_id': str, 'recording_id': str})
    folds = pd.read_csv(DATA / 'folds.csv').set_index('sample_id')['fold']
    raw = pd.read_csv(DATA / 'train_features.csv', dtype={'note_id': str, 'recording_id': str, 'sample_id': str})
    assert raw['split'].eq('train').all() and manifest['split'].eq('train').all()      # test stays locked
    assert manifest.audio_path.str.startswith('data/train/').all()
    timed = pd.read_csv(HERE / 'exports' / 'le_xin_timing_rf_audio_pitch_v2_nestedgate.csv',
                        dtype={'note_id': str, 'recording_id': str})
    timed = timed.merge(raw[['note_id', 'sample_id']], on='note_id', validate='one_to_one')
    return manifest, references, folds, raw, timed


# ---------- audio tracks (same settings as the split notebook) ----------
def tracks(path):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    f0, voiced, vprob = librosa.pyin(y, fmin=65, fmax=1100, sr=SR, hop_length=HOP)
    midi = np.where(voiced, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)
    flux = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    flux = flux / (np.percentile(flux, 99) + 1e-9)
    db = librosa.amplitude_to_db(librosa.feature.rms(y=y, hop_length=HOP)[0], ref=np.max)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=13)
    n = min(len(midi), len(flux), len(db), mfcc.shape[1])
    return dict(midi=midi[:n], voiced=voiced[:n].astype(bool), vprob=vprob[:n], flux=flux[:n], db=db[:n],
                mfcc=mfcc[:, :n].astype(np.float32))


def load_tracks(rid, folder='tracks'):
    a = np.load(CACHE / folder / f'{rid}.npz')
    return {k: a[k] for k in a}


# ---------- split notebook's candidate finder (copied unchanged) ----------
def med(midi, a, b):
    seg = midi[max(a, 0):max(b, 0)]
    seg = seg[~np.isnan(seg)]
    return np.median(seg) if len(seg) >= 2 else np.nan


def candidates_for_note(n, midi, voiced, vprob, flux, db, extra_frames=None):
    """Split notebook's finder. extra_frames(n, lo, hi) -> set of extra frames to add (v2 recall sources)."""
    lo, hi = int(np.ceil((n.start + EDGE) / FRAME)), int((n.end - EDGE) / FRAME)
    hi = min(hi, len(midi) - 1)
    if hi <= lo:
        return []
    frames = np.arange(lo, hi + 1)
    before = np.array([med(midi, i - SIDE, i) for i in frames])
    after = np.array([med(midi, i, i + SIDE) for i in frames])
    change = np.nan_to_num(np.abs(after - before), nan=0.0)

    def peaks(score, floor):
        return {f for k, f in enumerate(frames)
                if score[k] >= floor and score[k] == score[max(0, k - 3):k + 4].max()}
    picked = peaks(change, 0.5) | peaks(flux[frames], 0.25) | {f for f in frames if voiced[f] and not voiced[f - 1]}
    source = {f: 'old' for f in picked}
    if extra_frames is not None:
        for f, s in extra_frames(n, lo, hi).items():
            if all(abs(f - g) > 3 for g in picked) and f not in source:
                source[f] = s
        # extras never displace an old candidate; they are clustered among themselves below
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
            note_id=n.note_id, recording_id=n.recording_id, sample_id=n.sample_id, t=t,
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
            piece_pitch_before=med(midi, max(int(t / FRAME) - 13, note_lo), f),
            piece_pitch_after=med(midi, f, min(f + 13, note_hi)),
            source=source[f]))
    return rows


def true_cut_times(rec_notes, rec_refs):
    starts = rec_notes['start'].to_numpy()
    out = {}
    for n in rec_notes.itertuples():
        inside = rec_refs[(rec_refs.start > n.start + 0.05) & (rec_refs.start < n.end - 0.05)].start.to_numpy()
        inside = [b for b in inside if not (np.abs(starts - b) <= 0.08).any()]
        if inside:
            out[n.note_id] = inside
    return out


def label(cands, truth):
    """Notebook labelling: each true cut marks its nearest candidate (within 50 ms) in that note."""
    cands = cands.copy()
    cands['label'] = 0
    for nid, times in truth.items():
        c = cands[cands.note_id == nid]
        for b in times:
            if len(c) and np.abs(c.t - b).min() <= 0.05:
                cands.loc[(c.t - b).abs().idxmin(), 'label'] = 1
    return cands


def add_clues(df):
    df = df.copy()
    df['cands_in_note'] = df.groupby('note_id').t.transform('size')
    df['strength_rank'] = df.assign(s=df.pitch_change + df.flux).groupby('note_id').s.rank(ascending=False)
    pb, pa = df.piece_pitch_before, df.piece_pitch_after
    df['piece_change'] = (pa.round() - pb.round()).abs().fillna(FAR)
    df['piece_after_vs_note'] = (pa - df.note_pitch).abs().fillna(FAR).clip(upper=FAR)
    df['piece_before_vs_note'] = (pb - df.note_pitch).abs().fillna(FAR).clip(upper=FAR)
    return df


OLD_FEATS = ['since_start', 'to_end', 'rel_pos', 'note_dur', 'note_pitch', 'pitch_change', 'pitch_change_signed',
             'before_vs_note', 'after_vs_note', 'flux', 'rise', 'dip', 'silent_near', 'vprob', 'cands_in_note',
             'strength_rank', 'piece_change', 'piece_after_vs_note', 'piece_before_vs_note']


def apply_splits(note_table, cuts):
    out, by_note = [], {nid: g.sort_values('t') for nid, g in cuts.groupby('note_id')}
    for n in note_table.itertuples():
        c = by_note.get(n.note_id)
        if c is None or c.empty:
            out.append(dict(note_id=n.note_id, recording_id=n.recording_id, start=n.start, end=n.end, pitch=n.pitch))
            continue
        edges = [n.start, *c.t.tolist(), n.end]
        pitches = [c.piece_pitch_before.iloc[0], *c.piece_pitch_after.tolist()]
        for k in range(len(edges) - 1):
            p = pitches[k]
            out.append(dict(note_id=n.note_id if k == 0 else f'{n.note_id}~{k}', recording_id=n.recording_id,
                            start=edges[k], end=edges[k + 1],
                            pitch=float(np.clip(round(p), 0, 127)) if not np.isnan(p) else n.pitch))
    return pd.DataFrame(out)[cu.NOTE_COLUMNS]


# ---------- fast scorer (same numbers as cu.evaluate_end_to_end) ----------
class Scorer:
    def __init__(self, manifest, references):
        self.man = cu.validate_manifest(manifest)
        self.ref = {rid: g[['start', 'end', 'pitch']].to_numpy(float)
                    for rid, g in references.groupby('recording_id')}
        self.memo = {}

    def rec_counts(self, rid, est):
        key = (rid, hashlib.md5(np.ascontiguousarray(est).tobytes()).hexdigest())
        if key not in self.memo:
            self.memo[key] = core.score_notes(self.ref.get(rid, np.zeros((0, 3))), est)
        return self.memo[key]

    def __call__(self, table, recording_ids=None):
        man = self.man if recording_ids is None else self.man[self.man.recording_id.isin(recording_ids)]
        est = {rid: g[['start', 'end', 'pitch']].to_numpy(float) for rid, g in table.groupby('recording_id')}
        rows = [{**r, **self.rec_counts(r['recording_id'], est.get(r['recording_id'], np.zeros((0, 3))))}
                for r in man.to_dict('records')]
        s = core.aggregate(rows)[1]
        return s['song_macro_f1'], s['song_macro_offset_f1']

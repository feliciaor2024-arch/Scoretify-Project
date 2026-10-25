"""Shared paths, the note pool and a fast copy of the official scorer (training songs only; test is never read)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HO = ROOT / 'notebooks' / 'correction_handover'
EXPORTS, CACHE = HO / 'exports', HO / 'le_xin_cache'
DATA = ROOT / 'basic_pitch_runs' / 'correction_handover'
OUT = Path(__file__).resolve().parent
BPACT = Path(r'C:\Users\derp2\AppData\Local\Temp\claude\C--Users-derp2-SMU-Y2SEM1-DMBA-GTSinger-Correction-Group-Handover'
             r'\6efc453f-37d1-4798-af6f-53bef5f326a6\scratchpad\bpact_train')
sys.path[:0] = [str(HO), str(ROOT / 'notebooks')]

read = lambda p: pd.read_csv(p, dtype={'note_id': str, 'recording_id': str, 'sample_id': str})
original_id = lambda ids: ids.str.split('~').str[0]
MANIFEST = read(DATA / 'training_recording_manifest.csv')
assert MANIFEST.split.eq('train').all()                       # train only: no val, no test here
REFS = read(DATA / 'references_train.csv')                    # answer key: labels and scoring only
FOLDS = read(DATA / 'folds.csv').set_index('sample_id').fold


def pool():
    """Every note of the chain just before Felicia's delete step (timing + split), with the snapped start
    for the notes Felicia kept (snap only ran on those). felicia_kept marks her decision."""
    split = read(EXPORTS / 'le_xin_split_train_oof_v1.csv')
    snap = read(EXPORTS / 'le_xin_snap_train_oof_v2.csv').set_index('note_id')
    p = split.copy()
    p['felicia_kept'] = p.note_id.isin(snap.index)
    k = p.felicia_kept
    p.loc[k, 'start'] = p.loc[k, 'note_id'].map(snap.start).to_numpy()
    p.loc[k, 'end'] = p.loc[k, 'note_id'].map(snap.end).to_numpy()
    p['sample_id'] = p.recording_id.map(MANIFEST.set_index('recording_id').sample_id)
    p['fold'] = p.sample_id.map(FOLDS)
    return p


# ---- fast scorer: same matching as bp_tuning_core.score_notes, pooled per song, then song-macro mean ----
from mir_eval.transcription import match_notes
from librosa import midi_to_hz
import bp_tuning_core as core

_REF = {rid: g[['start', 'end', 'pitch']].to_numpy(float) for rid, g in REFS.groupby('recording_id')}
_SONG = MANIFEST.set_index('recording_id').sample_id
_memo = {}


def rec_counts(rid, est, offset=True):
    """(ref, est, tp_onset, tp_offset) for one recording; est = array start,end,pitch.
    offset=False skips the offset matching (tp_offset = 0) to save time while tuning on onset F1."""
    key = (rid, est.tobytes(), offset)
    if key in _memo:
        return _memo[key]
    ref = core._events(_REF.get(rid, np.zeros((0, 3))))
    e = core._events(est, predicted=True) if len(est) else np.zeros((0, 3))
    tp = [0, 0]
    if len(ref) and len(e):
        for i, ratio in enumerate([None, .2][:1 + offset]):
            tp[i] = len(match_notes(ref[:, :2], midi_to_hz(ref[:, 2]), e[:, :2], midi_to_hz(e[:, 2]),
                                    onset_tolerance=.05, pitch_tolerance=50., offset_ratio=ratio,
                                    offset_min_tolerance=.05, strict=False))
    _memo[key] = out = (len(ref), len(e), tp[0], tp[1])
    return out


def f1(c):
    return 2 * c[2] / (c[0] + c[1]) if c[0] + c[1] else 0.


def score(notes, rids=None, offset=True):
    """Song-macro onset / offset F1 over the given recordings (default: all 534 training recordings)."""
    rids = MANIFEST.recording_id.tolist() if rids is None else list(rids)
    g = {rid: n[['start', 'end', 'pitch']].to_numpy(float) for rid, n in notes.groupby('recording_id')}
    songs = {}
    for rid in rids:
        c = rec_counts(rid, g.get(rid, np.zeros((0, 3))), offset)
        s = songs.setdefault(_SONG[rid], np.zeros(4))
        s += c
    on = np.mean([f1(s) for s in songs.values()])
    off = np.mean([f1([s[0], s[1], s[3]]) for s in songs.values()])
    return on, off

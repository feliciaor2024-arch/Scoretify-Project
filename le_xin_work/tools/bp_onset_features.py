"""Basic Pitch onset/note probabilities around each split candidate (audio-derived; no reference data).

Basic Pitch writes a note only when its onset probability passes 0.99 (the tuned setting). Inside a long
note, a weaker onset peak (say 0.6) can still mark a real new note. These clues measure that.

Usage (needs basic-pitch + onnxruntime, see requirements-lock.txt):
  python bp_onset_features.py <repo_root> <candidates.csv> <split> <activation_cache_dir> <out.csv>
Activations are computed with the project's own ONNX model (notebooks/bp_tuning_core.make_model) and cached
per recording; decoding them with the tuned config reproduces the frozen predictions exactly.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

repo, cand_path, split, act_dir, out_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), sys.argv[5]
assert split in ('train', 'val')                                   # test stays locked
sys.path.insert(0, str(repo / 'notebooks'))
import bp_tuning_core as core
from basic_pitch.inference import run_inference
from basic_pitch.note_creation import model_frames_to_time

MIDI_OFFSET = 21          # Basic Pitch note bin 0 = MIDI 21 (A0)
W = 2                     # +-2 frames (~23 ms) around the candidate
man = pd.read_csv(repo / 'basic_pitch_runs/correction_handover/recording_manifest.csv')
man = man[man.split == split].set_index('recording_id')
cands = pd.read_csv(cand_path, dtype={'note_id': str, 'recording_id': str, 'sample_id': str})
model = None
rows = []
for rid, g in cands.groupby('recording_id'):
    rec = man.loc[rid]
    assert rec.split == split and rec.audio_path.startswith(f'data/{split}/')
    f = act_dir / f'{rid}.npz'
    if f.exists():
        arr = dict(np.load(f))
    else:
        model = model or core.make_model()[0]
        arr = run_inference(repo / rec.audio_path, model)
        act_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, onset=arr['onset'].astype(np.float32), note=arr['note'].astype(np.float32),
                            contour=arr['contour'].astype(np.float32))
    onset, note = arr['onset'], arr['note']
    times = model_frames_to_time(len(onset))
    n_bins = onset.shape[1]

    def bins(p):
        b = int(round(p)) - MIDI_OFFSET
        return list(range(max(b - 1, 0), min(b + 2, n_bins)))       # +-1 semitone

    for c in g.itertuples():
        i = int(np.argmin(np.abs(times - c.t)))
        lo, hi = max(i - W, 0), min(i + W + 1, len(onset))
        after_p = c.piece_pitch_after if not np.isnan(c.piece_pitch_after) else c.note_pitch
        before_p = c.piece_pitch_before if not np.isnan(c.piece_pitch_before) else c.note_pitch
        nb, ab, bb = bins(c.note_pitch), bins(after_p), bins(before_p)
        rows.append(dict(
            note_id=c.note_id, t=c.t,
            bp_onset_any=float(onset[lo:hi].max()),                    # any new-note signal here
            bp_onset_note_pitch=float(onset[lo:hi, nb].max()),          # re-attack of the same pitch
            bp_onset_after_pitch=float(onset[lo:hi, ab].max()),         # attack of the pitch sung next
            bp_note_after=float(note[i:i + 10, ab].mean()),             # is the next pitch active after?
            bp_note_before=float(note[max(i - 10, 0):i + 1, bb].mean()),  # was the earlier pitch active before?
            bp_note_dip=float(note[max(i - 10, 0):i, nb].mean() - note[lo:hi, nb].min())))  # does the note sag here?
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f'{split}: {len(rows)} candidates, {cands.recording_id.nunique()} recordings -> {out_path}')

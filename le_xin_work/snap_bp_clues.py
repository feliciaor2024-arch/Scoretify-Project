"""Basic Pitch activation clues for each snap option (audio-derived; no reference data).

For every option (a possible new start time t for a note), measure Basic Pitch's own evidence at the note's pitch:
does its onset signal fire at t, and does its "this pitch is sounding" signal switch on at t?

Usage (needs basic-pitch + onnxruntime, see requirements-lock.txt):
  python snap_bp_clues.py <repo_root> <snap_candidates.csv> <split> <activation_cache_dir> <out.csv>
Activations are computed with the project's own ONNX model (notebooks/bp_tuning_core.make_model) and cached per
recording (same cache as bp_onset_features.py). Output rows are in the same order as the candidate file.
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
W = 2                     # +-2 frames (~23 ms) around the option for the onset signal
SIDE = 5                  # ~58 ms before / after the option for the note signal
man = pd.read_csv(repo / 'basic_pitch_runs/correction_handover/recording_manifest.csv')
man = man[man.split == split].set_index('recording_id')
cands = pd.read_csv(cand_path, dtype={'note_id': str, 'recording_id': str, 'sample_id': str})
model = None
out = pd.DataFrame(index=cands.index, columns=['bp_onset_at', 'bp_onset_any', 'bp_note_before', 'bp_note_after'], dtype=float)
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
    for c in g.itertuples():
        i = int(np.argmin(np.abs(times - c.t)))
        b = int(round(c.note_pitch)) - MIDI_OFFSET
        nb = list(range(max(b - 1, 0), min(b + 2, onset.shape[1])))  # the note's pitch +-1 semitone
        lo, hi = max(i - W, 0), min(i + W + 1, len(onset))
        out.loc[c.Index] = [float(onset[lo:hi, nb].max()),             # onset signal at this pitch, here
                            float(onset[lo:hi].max()),                 # onset signal at any pitch, here
                            float(note[max(i - SIDE, 0):i, nb].max(axis=1).mean()) if i > 0 else 0.0,  # pitch sounding before?
                            float(note[i:i + SIDE, nb].max(axis=1).mean())]                            # pitch sounding after?
out.insert(0, 'note_id', cands.note_id)
out.insert(1, 't', cands.t)
out.to_csv(out_path, index=False)
print(f'{split}: {len(out)} options, {cands.recording_id.nunique()} recordings -> {out_path}')

"""Score chain v4 tables on the 20 VAL songs (answer key used for scoring only). Usage: python val_score.py [table.csv ...]"""
import sys
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'notebooks'), str(REPO / 'notebooks' / 'correction_handover')]
import correction_utils as cu
import bp_experiment as bx

HERE = Path(__file__).parent
man = pd.read_csv(REPO / 'basic_pitch_runs/correction_handover/recording_manifest.csv', dtype={'recording_id': str, 'sample_id': str})
man = man[man.split == 'val'].reset_index(drop=True)
assert len(man) == 160 and man.split.eq('val').all()                       # test stays locked
events, _ = bx.references(man.to_dict('records'))
refs = pd.DataFrame([dict(note_id=f'{rid}:ref{i:05d}', recording_id=rid, start=e[0], end=e[1], pitch=e[2])
                     for rid, ev in events.items() for i, e in enumerate(ev)])
rd = lambda p: pd.read_csv(p, dtype={'note_id': str, 'recording_id': str})
STAGES = [('timing (val input)', REPO / 'notebooks/correction_handover/exports/le_xin_timing_val_rf_audio_pitch.csv'),
          ('+ split v2', HERE / 'val_stage1_split_v2.csv'), ('+ delete v2', HERE / 'val_stage2_delete_v2.csv'),
          ('+ snap v3', HERE / 'val_stage3_snap_v3.csv'), ('+ end snap', HERE / 'val_stage4_end_snap.csv')]
prev = None
for name, path in STAGES:
    if not path.exists():
        continue
    t = rd(path)[cu.NOTE_COLUMNS]
    assert t.note_id.is_unique and t.recording_id.isin(man.recording_id).all() and (t.end > t.start).all()
    r = cu.evaluate_end_to_end(t, refs, man)
    s, songs = r['summary'], {x['sample_id']: x['f1'] for x in r['songs']}
    vs = '' if prev is None else f"  songs better/worse vs previous: {sum(songs[k] > prev[k] + 1e-12 for k in songs)}/{sum(songs[k] < prev[k] - 1e-12 for k in songs)}"
    print(f"{name:20s} notes {len(t):5d}  onset F1 {s['song_macro_f1']:.4f}  offset F1 {s['song_macro_offset_f1']:.4f}{vs}", flush=True)
    prev = songs

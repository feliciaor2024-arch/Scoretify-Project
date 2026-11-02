"""Score the one test run (made by run_test.py) per stage, plus the sheet. Base Python environment.

correction_utils refuses test on purpose, so this uses the same scoring core (bp_tuning_core.score_notes /
aggregate) and the same reference parser rule as bp_experiment.references. `python score_test.py --check-val`
first proves this route gives exactly cu.evaluate_end_to_end's numbers on val.
"""
import sys
import json
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LX = HERE.parent
ROOT = LX.parent
OUT = LX / 'test_run'
sys.path[:0] = [str(ROOT / 'notebooks'), str(ROOT / 'notebooks' / 'correction_handover')]
spec = importlib.util.spec_from_file_location('bp_core', ROOT / 'notebooks' / 'bp_tuning_core.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
read = lambda p: pd.read_csv(p, dtype={'note_id': str, 'recording_id': str, 'sample_id': str})


def references(man):
    """bp_experiment.references without its train/val lock: as-annotated note_start/end, pitch > 0 only."""
    rows = []
    for r in man.itertuples():
        for item in json.loads((ROOT / r.json_path).read_text(encoding='utf-8')):
            notes, starts, ends = item.get('note', []), item.get('note_start', []), item.get('note_end', [])
            if not len(notes) == len(starts) == len(ends):
                continue
            for p, s, e in zip(notes, starts, ends):
                if p == 0:
                    continue
                if all(isinstance(v, (int, float)) and np.isfinite(v) for v in [p, s, e]) and \
                        0 < p <= 127 and int(p) == p and 0 <= s < e <= float(r.duration_seconds_if_validated) + 0.001:
                    rows.append((r.recording_id, float(s), float(e), int(p)))
    return pd.DataFrame(rows, columns=['recording_id', 'start', 'end', 'pitch'])


def score(notes, refs, man):
    """Same as cu.evaluate_end_to_end: every recording counts (also ones with no notes), song-macro F1."""
    recs = []
    for record in man.to_dict('records'):
        rid = record['recording_id']
        recs.append({**record, **core.score_notes(refs.loc[refs.recording_id.eq(rid), ['start', 'end', 'pitch']].to_numpy(),
                                                  notes.loc[notes.recording_id.eq(rid), ['start', 'end', 'pitch']].to_numpy())})
    songs, summary = core.aggregate(recs)
    return summary, pd.DataFrame(songs)


def check_val():
    import correction_utils as cu
    man = read(ROOT / 'basic_pitch_runs/correction_handover/recording_manifest.csv').query("split == 'val'")
    refs = references(man)
    cu_refs = refs.assign(note_id=[f'r{i}' for i in range(len(refs))])
    for path in [ROOT / 'basic_pitch_runs/correction_handover/val_features.csv', LX / 'chain_v4/val_stage4_end_snap.csv']:
        notes = read(path)
        a = score(notes, refs, man)[0]
        b = cu.evaluate_end_to_end(notes[cu.NOTE_COLUMNS], cu_refs[cu.NOTE_COLUMNS], man)['summary']
        same = all(a[k] == b[k] for k in ['song_macro_f1', 'song_macro_offset_f1'])
        print(f'{path.name:28s} this {a["song_macro_f1"]:.4f}/{a["song_macro_offset_f1"]:.4f}  '
              f'cu {b["song_macro_f1"]:.4f}/{b["song_macro_offset_f1"]:.4f}  {"SAME" if same else "DIFFERENT"}')


if __name__ == '__main__':
    if '--check-val' in sys.argv:
        check_val()
        sys.exit()
    man = read(OUT / 'test_manifest.csv')
    refs = references(man)
    dur = man.set_index('recording_id').duration_seconds_if_validated.astype(float)
    rows, per_song = [], {}
    for name in ['basic_pitch', 'timing', 'split', 'delete', 'start_snap', 'notes', 'sheet_notes']:
        notes = read(OUT / f'test_{name}.csv')
        notes['end'] = notes.end.clip(upper=notes.recording_id.map(dur))      # sheet rounding can pass the end
        s, songs = score(notes, refs, man)
        per_song[name] = songs.set_index('sample_id').f1
        rows.append(dict(stage=name, notes=len(notes), onset_f1=round(s['song_macro_f1'], 4),
                         offset_f1=round(s['song_macro_offset_f1'], 4)))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'test_scores.csv', index=False)
    print(f'test: {len(man)} recordings, {man.sample_id.nunique()} songs, {len(refs)} reference notes')
    print(table.to_string(index=False))
    gain = per_song['notes'] - per_song['basic_pitch']
    print(f'songs improved by the pipeline (onset F1, step 5 vs Basic Pitch): {(gain > 0).sum()}/{len(gain)}')
    pd.DataFrame(per_song).to_csv(OUT / 'test_scores_by_song.csv')

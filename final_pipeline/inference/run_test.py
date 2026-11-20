"""THE ONE TEST RUN: the finished chain v4 pipeline on the 20 locked test songs (164 recordings), audio -> notes -> sheet.

Run once, after group approval, with nothing changed afterwards. Needs the Basic Pitch environment (Python 3.13,
requirements.txt). It refuses to run again if final_pipeline/test_run/ already has results.
Nothing is tuned or chosen here: models and settings are the ones in models/. Scoring is a separate step
(score_test.py), so this script never reads the test answer key.
Usage: python run_test.py [n_jobs]
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pipeline import Corrector, bp_model, bp_activations, bp_notes, write_sheet, NOTE_COLUMNS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE.parent / 'test_run'
STAGES = ['basic_pitch', 'timing', 'split', 'delete', 'start_snap', 'notes']


def test_manifest():
    """Test rows of selected_recordings.csv, with paths in data/test (same rule as the train/val manifest)."""
    s = pd.read_csv(ROOT / 'selected_recordings.csv', dtype=str).query("split == 'test'").reset_index(drop=True)
    for col in ['audio_path', 'json_path', 'musicxml_path', 'textgrid_path']:
        s[col] = 'data/test/' + s[col].str.removeprefix('source_files/')
    assert len(s) == 164 and s.sample_id.nunique() == 20 and s.validation_status.eq('passed').all()
    return s


def run_chunk(rows):
    model, corrector = bp_model(), Corrector()
    done = []
    for r in rows:
        audio = ROOT / r['audio_path']
        t0 = time.time()
        acts = bp_activations(audio, model)
        raw = bp_notes(acts, r['recording_id'])
        out = corrector.correct(audio, r['recording_id'], bp=(acts, raw))
        sheet = write_sheet(out['notes'], audio, OUT / 'sheets', r['recording_id']) if len(out['notes']) else None
        done.append(dict(out, sheet=sheet, seconds=time.time() - t0, rid=r['recording_id']))
    return done


if __name__ == '__main__':
    if (OUT / 'test_notes.csv').exists():
        sys.exit(f'{OUT} already has the test run. It is run once only; do not rerun.')
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    (OUT / 'sheets').mkdir(parents=True, exist_ok=True)
    man = test_manifest()
    man.to_csv(OUT / 'test_manifest.csv', index=False)
    rows = man.to_dict('records')
    t0 = time.time()
    results = [x for part in Parallel(n_jobs=jobs, verbose=5)(delayed(run_chunk)(list(c))
                                                               for c in np.array_split(np.array(rows, dtype=object), jobs))
               for x in part]
    for name in STAGES:
        pd.concat([x[name][NOTE_COLUMNS] for x in results if name in x]).to_csv(OUT / f'test_{name}.csv', index=False)
    pd.concat([x['sheet']['notes'] for x in results if x['sheet']]).to_csv(OUT / 'test_sheet_notes.csv', index=False)
    pd.DataFrame([dict(recording_id=x['rid'], seconds=round(x['seconds'], 2), bp_notes=len(x['basic_pitch']),
                       notes=len(x['notes']), bpm=x['sheet']['bpm'] if x['sheet'] else None) for x in results]
                 ).to_csv(OUT / 'test_run_log.csv', index=False)
    print(f'{len(results)} recordings in {time.time() - t0:.0f} s -> {OUT}')

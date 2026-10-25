"""Shared paths/loaders for the end-snap experiment (training songs only; test and val are not opened)."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'notebooks' / 'correction_handover'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import correction_utils as cu  # noqa: E402

DATA, EXPORTS = ROOT / 'basic_pitch_runs' / 'correction_handover', HERE / 'exports'
BPACT = ROOT / 'le_xin_work' / 'bp_activations' / 'train'
import os  # noqa: E402
TAG = os.environ.get('END_SNAP_TAG', '')                              # '' = first run (snap v2 input); 'v4' = chain v4
INPUTS = {'': EXPORTS / 'le_xin_snap_train_oof_v2.csv',
          'v4': ROOT / 'le_xin_work' / 'chain_v4' / 'stage3_snap_v3_train_oof.csv'}   # timing -> split v2 -> delete v2 -> snap v3
SUFFIX = f'_{TAG}' if TAG else ''
read = lambda path: pd.read_csv(path, dtype={'note_id': str, 'recording_id': str, 'sample_id': str})


def load():
    manifest = pd.read_csv(DATA / 'training_recording_manifest.csv', dtype={'recording_id': str, 'sample_id': str})
    assert manifest['split'].eq('train').all()                       # test (and val) stay closed
    references = read(DATA / 'references_train.csv')                  # answer key: labels and scoring only
    folds = pd.read_csv(DATA / 'folds.csv').set_index('sample_id')['fold']
    chain = read(INPUTS[TAG])
    cu.validate_notes(chain, manifest)
    return manifest, references, folds, chain

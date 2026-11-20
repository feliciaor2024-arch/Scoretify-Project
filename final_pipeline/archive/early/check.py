import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools")); import shim
import pandas as pd
R=str(__import__('pathlib').Path(__file__).resolve().parents[2]).replace('\\', '/') + '/'; D=R+'basic_pitch_runs/correction_handover/'
sys.path.insert(0,R+'notebooks/correction_handover'); import correction_utils as cu
f=pd.read_csv(D+'train_features.csv',dtype={'note_id':str,'recording_id':str})
man=pd.read_csv(D+'training_recording_manifest.csv'); refs=pd.read_csv(D+'references_train.csv',dtype={'note_id':str,'recording_id':str})
print(cu.evaluate_end_to_end(f[cu.NOTE_COLUMNS],refs,man)['summary'])

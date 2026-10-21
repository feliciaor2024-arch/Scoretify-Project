import sys; sys.path.insert(0,"."); import shim
import pandas as pd
R='/mnt/user-data/uploads/GTSinger_Correction_Group_Handover/GTSinger_100_unique_sample/'; D=R+'basic_pitch_runs/correction_handover/'
sys.path.insert(0,R+'notebooks/correction_handover'); import correction_utils as cu
f=pd.read_csv(D+'train_features.csv',dtype={'note_id':str,'recording_id':str})
man=pd.read_csv(D+'training_recording_manifest.csv'); refs=pd.read_csv(D+'references_train.csv',dtype={'note_id':str,'recording_id':str})
print(cu.evaluate_end_to_end(f[cu.NOTE_COLUMNS],refs,man)['summary'])

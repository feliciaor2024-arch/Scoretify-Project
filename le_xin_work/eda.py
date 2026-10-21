import sys, numpy as np, pandas as pd
R='/mnt/user-data/uploads/GTSinger_Correction_Group_Handover/GTSinger_100_unique_sample/'
D=R+'basic_pitch_runs/correction_handover/'
sys.path.insert(0,R+'notebooks/correction_handover')
import correction_utils as cu
f=pd.read_csv(D+'train_features.csv',dtype={'note_id':str,'recording_id':str,'sample_id':str})
t=pd.read_csv(D+'proposed_v2/train_targets_v2.csv',dtype={'note_id':str})
rm=pd.read_csv(D+'recording_manifest.csv',dtype={'recording_id':str})
j=f.merge(t,on='note_id').merge(rm[['recording_id','language','singer','vocal_range','technique_context']],on='recording_id')
o=j[j.relationship=='one_to_one'].copy()
print(len(o),'matched notes')
for c in ['start_delta','end_delta']:
    d=o[c]*1000
    print(c,'ms: mean %.1f median %.1f std %.1f  p5 %.0f p25 %.0f p75 %.0f p95 %.0f'%(d.mean(),d.median(),d.std(),*np.percentile(d,[5,25,75,95])))
    print('  >0 (BP too early):',(d>0).mean().round(3),' |err|>50ms:',(d.abs()>50).mean().round(3),' |err|>100ms:',(d.abs()>100).mean().round(3))
print('wrong_start',o.wrong_start.mean().round(3),'wrong_end',o.wrong_end.mean().round(3))
o['dbin']=pd.cut(o.duration,[0,.2,.4,.8,5])
print(o.groupby('dbin',observed=True).agg(n=('note_id','size'),s_med=('start_delta',lambda x:x.median()*1000),s_abs=('start_delta',lambda x:x.abs().mean()*1000),e_med=('end_delta',lambda x:x.median()*1000),e_abs=('end_delta',lambda x:x.abs().mean()*1000)).round(1))
for g in ['language','technique_context','vocal_range']:
    print(o.groupby(g).agg(n=('note_id','size'),s_med=('start_delta',lambda x:x.median()*1000),s_abs=('start_delta',lambda x:x.abs().mean()*1000),e_med=('end_delta',lambda x:x.median()*1000),e_abs=('end_delta',lambda x:x.abs().mean()*1000),ws=('wrong_start','mean'),we=('wrong_end','mean')).round(2))
print(o[['start_delta','end_delta','duration','amplitude','prev_gap','next_gap','pitch','start']].corr().round(2).loc[['start_delta','end_delta']])
# match_rule split
print(o.groupby('match_rule').agg(n=('note_id','size'),s_abs=('start_delta',lambda x:x.abs().mean()*1000),e_abs=('end_delta',lambda x:x.abs().mean()*1000)).round(1))
# on all notes how many are 'ambiguous' & their share
print(j.relationship.value_counts())
folds=pd.read_csv(D+'folds.csv'); print(folds.fold.value_counts().sort_index().to_dict())

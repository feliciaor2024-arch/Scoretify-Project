import sys; sys.path.insert(0,"."); import shim
import sys, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
R='/mnt/user-data/uploads/GTSinger_Correction_Group_Handover/GTSinger_100_unique_sample/'
D=R+'basic_pitch_runs/correction_handover/'
sys.path.insert(0,R+'notebooks/correction_handover')
import correction_utils as cu
f=pd.read_csv(D+'train_features.csv',dtype={'note_id':str,'recording_id':str,'sample_id':str})
t=pd.read_csv(D+'proposed_v2/train_targets_v2.csv',dtype={'note_id':str})
man=pd.read_csv(D+'training_recording_manifest.csv'); refs=pd.read_csv(D+'references_train.csv',dtype={'note_id':str,'recording_id':str})
folds=pd.read_csv(D+'folds.csv').set_index('sample_id')['fold']
f=f.sort_values(['recording_id','start']).reset_index(drop=True)
# prediction-derived features only (allowlist + boundary flags + log dur, per DATA_DICTIONARY suggestion)
f['first']=(f.groupby('recording_id').cumcount()==0).astype(float)
f['last']=(f.groupby('recording_id').cumcount(ascending=False)==0).astype(float)
f['logdur']=np.log(f.duration)
FEATS=cu.NOTE_FEATURES+['first','last','logdur']
j=f.merge(t[['note_id','relationship','start_delta','end_delta','wrong_start','wrong_end']],on='note_id',how='left')
lab=(j.relationship=='one_to_one').to_numpy()
fold=j.sample_id.map(folds).to_numpy()
X=j[FEATS].to_numpy(float); ys=j.start_delta.to_numpy(); ye=j.end_delta.to_numpy()

def make(kind,p):
    if kind=='dt': return DecisionTreeRegressor(min_samples_leaf=p,random_state=0)
    return RandomForestRegressor(n_estimators=150,min_samples_leaf=p,max_features=.6,n_jobs=2,random_state=0)
def fit_pred(kind,tr,te,grid):
    """Fit on labelled notes in tr; choose min_samples_leaf by inner song-grouped CV (inside training folds only)."""
    best=None
    g=j.sample_id.to_numpy()[tr]
    for p in grid:
        err=0
        for a,b in GroupKFold(3).split(X[tr],groups=g):
            for y in (ys,ye):
                m=make(kind,p).fit(X[tr][a],y[tr][a]); err+=np.abs(m.predict(X[tr][b])-y[tr][b]).mean()
        if best is None or err<best[0]: best=(err,p)
    out=[]
    for y in (ys,ye):
        out.append(make(kind,best[1]).fit(X[tr],y[tr]).predict(X[te]))
    return np.c_[out[0],out[1]],best[1]

def oof(kind,grid=None):
    ds=np.zeros((len(j),2)); chosen={}
    for k in sorted(set(fold)):
        te=np.where(fold==k)[0]; tr=np.where((fold!=k)&lab)[0]
        if kind=='none': continue
        if kind=='shift': ds[te]=[np.median(ys[tr]),np.median(ye[tr])]
        else: ds[te],chosen[int(k)]=fit_pred(kind,tr,te,grid)
    return ds,chosen

def apply(ds,gate_s=0.,gate_e=0.,scale=1.):
    n=j[cu.NOTE_COLUMNS].copy()
    ds=ds.copy()
    ds[np.abs(ds[:,0])<gate_s,0]=0; ds[np.abs(ds[:,1])<gate_e,1]=0
    n['start']=np.maximum(0,n.start+scale*ds[:,0]); n['end']=n.end+scale*ds[:,1]
    dur=j.audio_duration.to_numpy(); n['end']=np.minimum(n.end,dur); n['end']=np.maximum(n.end,n.start+0.01)
    n['start']=np.minimum(n.start,n.end-0.01)
    return n
def score(n):
    r=cu.evaluate_end_to_end(n,refs,man)['summary']
    return {k:round(r[k],4) for k in ['song_macro_f1','song_macro_offset_f1','song_macro_precision','song_macro_recall','micro_f1']}
def task(n):
    m=lab; truth=j.loc[m,['reference_start','reference_end']].to_numpy() if 'reference_start' in j else None
    return None

tt=t.set_index('note_id'); rs=tt.loc[j.note_id[lab],'reference_start'].to_numpy(); re_=tt.loc[j.note_id[lab],'reference_end'].to_numpy()
def tmet(n):
    m=cu.timing_metrics(np.c_[rs,re_],n.loc[lab,['start','end']].to_numpy())
    return {k:round(m[k],4) for k in ['onset_mae_seconds','offset_mae_seconds','onset_median_absolute_error_ms','onset_within_50ms','offset_within_tolerance']}
rows=[]
def rec(name,n,extra=None):
    r=dict(model=name,**tmet(n),**score(n)); 
    if extra: r['note']=str(extra)
    rows.append(r); print(r,flush=True)
rec('BP unchanged',j[cu.NOTE_COLUMNS])
ds,_=oof('shift'); rec('Median shift (rule: constant)',apply(ds))
res={}
for kind,grid in [('dt',[5,10,20,40]),('rf',[3,5,10,20])]:
    ds,ch=oof(kind,grid); res[kind]=ds; rec(kind.upper()+' regressor',apply(ds),ch)
# conservative gating/shrinkage on RF
for gs,ge,sc in [(.02,.05,1.),(.0,.0,.75),(.02,.05,.75),(.03,.08,1.)]:
    rec(f'RF gated s>{gs} e>{ge} x{sc}',apply(res['rf'],gs,ge,sc))
pd.DataFrame(rows).to_csv('stageA_results.csv',index=False)
np.save('rf_ds.npy',res['rf']); np.save('dt_ds.npy',res['dt'])

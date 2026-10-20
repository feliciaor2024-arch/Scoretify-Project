"""Build provisional correction handover; never label test or validation targets."""
from pathlib import Path
import json,csv,collections,hashlib
import numpy as np
import pandas as pd
from correction_labels import correspond
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'basic_pitch_runs/correction_handover'

def tables_for_record(row,pred,ref=None):
    if row['split'] not in ('train','val'): raise PermissionError('Test locked')
    if row['split']=='val' and ref is not None: raise PermissionError('Validation labels not exported')
    pred=sorted(pred,key=lambda p:(p[0],p[1],p[2]))
    if any(not np.isfinite(p[:4]).all() or not 0<=p[0]<p[1] or not 0<p[2]<=127 or int(p[2])!=p[2] for p in pred): raise ValueError('Malformed prediction')
    base=dict(recording_id=row['recording_id'],sample_id=row['sample_id'],split=row['split'])
    features=[];targets=[];pairs=[];pair_targets=[]
    labels=correspond(pred,ref) if ref is not None else None
    for i,p in enumerate(pred):
        nid=f"{row['recording_id']}:n{i:05d}"
        features.append(dict(base,note_id=nid,start=p[0],end=p[1],pitch=p[2],amplitude=p[3],duration=p[1]-p[0],prev_gap=p[0]-pred[i-1][1] if i else 0.,next_gap=pred[i+1][0]-p[1] if i+1<len(pred) else 0.,audio_duration=float(row['duration_seconds_if_validated'])))
        if labels:
            t=dict(note_id=nid,**labels['notes'][i]); j=t['reference_index']
            if t['relationship']=='one_to_one':
                r=ref[j]
                t.update(reference_start=r[0],reference_end=r[1],reference_pitch=r[2],wrong_start=int(abs(t['start_delta'])>.05),wrong_end=int(abs(t['end_delta'])>max(.05,.2*(r[1]-r[0]))),wrong_pitch=int(t['pitch_delta']!=0))
            targets.append(t)
    # Candidate generation uses predictions only: immediate same-pitch neighbours.
    for i in range(len(pred)-1):
        a,b=pred[i:i+2]; gap=b[0]-a[1]
        if a[2]!=b[2] or not -.02<=gap<=.12: continue
        pid=f"{row['recording_id']}:p{i:05d}"
        pairs.append(dict(base,pair_id=pid,note_id_a=features[i]['note_id'],note_id_b=features[i+1]['note_id'],gap=gap,pitch_delta=b[2]-a[2],duration_sum=a[1]-a[0]+b[1]-b[0]))
        if labels:
            lab=None
            if any(i in g and i+1 in g for g in labels['merge_groups']): lab=1
            elif all(labels['notes'][k]['relationship']=='one_to_one' for k in [i,i+1]) and labels['notes'][i]['reference_index']!=labels['notes'][i+1]['reference_index']: lab=0
            pair_targets.append(dict(pair_id=pid,merge=lab,status='provisional_auto' if lab is not None else 'unresolved'))
    return features,targets,pairs,pair_targets,labels


def build_artifacts(out,rows,predictions,references):
    out.mkdir(parents=True,exist_ok=True)
    if any(r['split'] not in ('train','val') for r in rows): raise PermissionError('Test locked')
    assert set(predictions)=={r['recording_id'] for r in rows}, 'Missing/extra prediction recording'
    assert set(references)=={r['recording_id'] for r in rows if r['split']=='train'}, 'Training references only'
    all_features={'train':[],'val':[]};targets=[];pairs=[];pair_targets=[];diagnostics=[]
    for row in rows:
        rid=row['recording_id']
        f,t,p,pt,d=tables_for_record(row,predictions[rid],references.get(rid))
        all_features[row['split']]+=f
        if row['split']=='train':
            targets+=t; pairs+=p;pair_targets+=pt
            diagnostics.append(dict(recording_id=rid,sample_id=row['sample_id'],predictions=len(f),references=len(references[rid]),merge_groups=len(d['merge_groups']),missing_reference_candidates=len(d['missing_reference_indices']),**dict(collections.Counter(n['relationship'] for n in d['notes']))))
    featcols=['note_id','recording_id','sample_id','split','start','end','pitch','amplitude','duration','prev_gap','next_gap','audio_duration']
    targetcols=['note_id','relationship','reference_index','false_note','pitch_delta','start_delta','end_delta','status','reference_start','reference_end','reference_pitch','wrong_start','wrong_end','wrong_pitch']
    paircols=['pair_id','note_id_a','note_id_b','recording_id','sample_id','split','gap','pitch_delta','duration_sum']
    for split,fs in all_features.items(): pd.DataFrame(fs,columns=featcols).to_csv(out/f'{split}_features.csv',index=False)
    pd.DataFrame(targets,columns=targetcols).to_csv(out/'train_targets.csv',index=False)
    pd.DataFrame(pairs,columns=paircols).to_csv(out/'merge_candidates.csv',index=False)
    pd.DataFrame(pair_targets,columns=['pair_id','merge','status']).to_csv(out/'merge_targets.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'recording_manifest.csv',index=False)
    pd.DataFrame([r for r in rows if r['split']=='train']).to_csv(out/'training_recording_manifest.csv',index=False)
    reference_rows=[dict(note_id=f'{rid}:ref{i:05d}',recording_id=rid,start=event[0],end=event[1],pitch=event[2]) for rid,events in references.items() for i,event in enumerate(events)]
    pd.DataFrame(reference_rows,columns=['note_id','recording_id','start','end','pitch']).to_csv(out/'references_train.csv',index=False)
    pd.DataFrame(diagnostics).fillna(0).to_csv(out/'training_diagnostics.csv',index=False)
    summary=dict(train_notes=len(all_features['train']),val_notes=len(all_features['val']),training_relationships=dict(collections.Counter(t['relationship'] for t in targets)),training_label_status=dict(collections.Counter(t['status'] for t in targets)),merge_candidates=len(pairs),merge_labels=dict(collections.Counter(str(t['merge']) for t in pair_targets)),false_labels=dict(collections.Counter(str(t.get('false_note')) for t in targets)),test_included=False,group_review_approved=False,labels='PROVISIONAL_HEURISTIC_NOT_HUMAN_GROUND_TRUTH')
    (out/'label_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


"""Frozen sample experiment utilities; only train/validation reference parsing permitted."""
from pathlib import Path
import csv,json,hashlib,sys,time,copy
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'basic_pitch_runs/tuning_001'

def dump(name,obj):
    path=OUT/name; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')

def csvread(name):
    with (ROOT/name).open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def frozen_check():
    baseline=json.loads((ROOT/'package_integrity.json').read_text(encoding='utf-8'))
    changes=[rel for rel,digest in baseline.items() if not (ROOT/rel).is_file() or sha(ROOT/rel)!=digest]
    if changes: raise RuntimeError('Frozen package inputs changed: '+str(changes[:10]))
    return dict(passed=True,protected_files=len(baseline),changed_paths=[])


def inputs():
    s=csvread('selected_100.csv'); r=csvread('selected_recordings.csv'); f=csvread('selected_files.csv')
    assert len(s)==100 and len({x['canonical_song_id'] for x in s})==100
    assert Counter(x['split'] for x in s)==Counter(train=60,val=20,test=20)
    audio={x['recording_id']:x for x in f if x['role']=='audio'}
    rows=[dict(x,audio_sha256=audio[x['recording_id']]['local_sha256']) for x in r if x['split'] in ('train','val')]
    exports={(x['recording_id'],x['role']):x['export_path'] for x in f}
    for x in rows:
        # Original manifests retain source download provenance; use packaged exports.
        for role in ['audio','json','musicxml','textgrid']:
            x[role+'_path']=exports[(x['recording_id'],role)]
        p=(ROOT/x['audio_path']).resolve()
        assert p.is_relative_to(ROOT) and p.is_file()
    return rows

def references(rows):
    refs={}; issues=[]; counters=Counter(); lengths=[]; same_pitch_boundaries=0
    for row in rows:
        if row['split'] not in ('train','val'): raise PermissionError('Test references locked')
        duration=float(row['duration_seconds_if_validated']); events=[]
        obj=json.loads((ROOT/row['json_path']).read_text(encoding='utf-8'))
        for wi,item in enumerate(obj):
            notes=item.get('note',[]); starts=item.get('note_start',[]); ends=item.get('note_end',[])
            if not (len(notes)==len(starts)==len(ends)):
                issues.append(dict(recording_id=row['recording_id'],word_index=wi,reason='note/timing array length mismatch')); continue
            for ni,(pitch,start,end) in enumerate(zip(notes,starts,ends)):
                counters['annotation_notes']+=1
                if pitch==0:
                    counters['rest_or_breath_events']+=1; continue
                if not all(isinstance(v,(int,float)) and np.isfinite(v) for v in [pitch,start,end]) or not (0<pitch<=127 and int(pitch)==pitch and 0<=start<end<=duration+0.001):
                    issues.append(dict(recording_id=row['recording_id'],word_index=wi,note_index=ni,pitch=pitch,start=start,end=end,duration=duration,reason='invalid pitch or timing bounds')); continue
                events.append([float(start),float(end),int(pitch)])
                lengths.append(float(end-start))
                if len(item.get('note_dur',[]))==len(notes) and abs(float(item['note_dur'][ni])-(end-start))>0.001: counters['notated_duration_disagrees_gt_1ms']+=1
        events.sort(key=lambda x:(x[0],x[1],x[2]))
        for a,b in zip(events,events[1:]):
            if a[2]==b[2] and abs(a[1]-b[0])<0.001: same_pitch_boundaries+=1
        refs[row['recording_id']]=events
    audit=dict(recordings=len(rows),pitched_notes=len(lengths),counters=dict(counters),same_pitch_adjacent_boundaries=same_pitch_boundaries,issues=issues,parser_policy='as-annotated note_start/end seconds, pitch>0 only; preserve repeated notes/ties as separate annotation events; no score-duration reconstruction, no per-recording alignment optimization, no source modification',shorter_than_default_fraction=float(np.mean(np.array(lengths)<0.1277)) if lengths else 0,median_note_seconds=float(np.median(lengths)) if lengths else None)
    return refs,audit

def config(name,onset=.5,frame=.3,length=127.7,melodia=True):
    return dict(name=name,onset_threshold=onset,frame_threshold=frame,minimum_note_length=length,melodia_trick=melodia)

def run_cache(rows,model,provenance):
    import bp_tuning_core as core
    records=[]; start=time.perf_counter()
    for i,r in enumerate(rows,1):
        metadata=core.cache_recording(ROOT,r,OUT/'activation_cache',model,dict(provenance,audio_sha256=r['audio_sha256']))
        records.append(metadata)
        with (OUT/'inference_log.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(dict(split=r['split'],**metadata))+'\n')
        if i%25==0 or i==len(rows): print('Activation cache:',i,'/',len(rows),'elapsed',round(time.perf_counter()-start,1),'s',flush=True)
    dump('cache_manifest_'+rows[0]['split']+'.json',records)
    return {r['recording_id']:r for r in records}

def evaluate(rows,refs,cache,configs,phase):
    import bp_tuning_core as core
    results={}; all_leaders=[]
    for c in configs:
        folder=OUT/phase/c['name']; folder.mkdir(parents=True,exist_ok=True)
        conf={k:v for k,v in c.items() if k!='name'}
        dump(str(folder.relative_to(OUT)/'config.json'),c)
        pd.DataFrame(rows).to_csv(folder/'input_manifest.csv',index=False)
        metrics=[]; predictions={}; start=time.perf_counter()
        for i,r in enumerate(rows,1):
            with np.load(cache[r['recording_id']]['npz_path'],allow_pickle=False) as saved: arrays={k:saved[k] for k in saved.files}
            try:
                midi,notes=core.decode(arrays,conf)
                counts=core.score_notes(refs[r['recording_id']],notes)
            except Exception as exc:
                dump(str(folder.relative_to(OUT)/'failure.json'),dict(recording_id=r['recording_id'],error=str(exc)))
                raise
            metrics.append(dict(recording_id=r['recording_id'],sample_id=r['sample_id'],language=r['language'],singer=r['singer'],vocal_range=r['vocal_range'],technique_context=r['technique_context'],subgroup=r['subgroup'],**counts))
            predictions[r['recording_id']]=[[float(n[0]),float(n[1]),int(n[2]),float(n[3])] for n in notes]
            if i%150==0: print(phase,c['name'],i,'/',len(rows),flush=True)
        song_rows,summary=core.aggregate(metrics)
        summary=dict(summary,config=c['name'],elapsed_seconds=time.perf_counter()-start,attempted=len(rows),successful=len(metrics),failed=0)
        pd.DataFrame(metrics).to_csv(folder/'recording_metrics.csv',index=False)
        pd.DataFrame(song_rows).to_csv(folder/'song_metrics.csv',index=False)
        dump(str(folder.relative_to(OUT)/'predicted_notes.json'),predictions)
        dump(str(folder.relative_to(OUT)/'summary.json'),summary)
        (folder/'report.md').write_text('# '+c['name']+' — '+phase+'\n\nActual reference-based metrics, not output counts.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n',encoding='utf-8')
        results[c['name']]=dict(summary=summary,recording_metrics=metrics,song_metrics=song_rows,predictions=predictions)
        all_leaders.append(summary)
        print(phase,c['name'],json.dumps(summary),flush=True)
    pd.DataFrame(all_leaders).to_csv(OUT/(phase+'_leaderboard.csv'),index=False)
    return results

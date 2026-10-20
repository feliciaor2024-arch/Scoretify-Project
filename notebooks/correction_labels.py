"""Conservative provisional TRAIN labels. Correspondence is not strict F1 matching."""

def correspond(pred,ref,split='train'):
    if split!='train': raise PermissionError('Label development is TRAIN only')
    edges={i:[j for j,r in enumerate(ref) if min(p[1],r[1])>max(p[0],r[0])] for i,p in enumerate(pred)}
    inverse={j:[i for i,js in edges.items() if j in js] for j in range(len(ref))}
    notes=[dict(relationship='ambiguous',reference_index=None,false_note=None,status='unresolved') for p in pred]
    groups=[]
    for j,ids in inverse.items():
        r=ref[j]
        if len(ids)>1 and all(edges[i]==[j] and pred[i][2]==r[2] for i in ids):
            ordered=sorted(ids,key=lambda i:pred[i][0])
            covered=0.; cursor=r[0]
            for i in ordered:
                start=max(r[0],pred[i][0],cursor); end=min(r[1],pred[i][1])
                covered+=max(0.,end-start); cursor=max(cursor,end)
            compatible=covered/(r[1]-r[0])>=.6 and abs(pred[ordered[0]][0]-r[0])<=.2 and abs(pred[ordered[-1]][1]-r[1])<=.2
            if compatible and all(-.02<=pred[b][0]-pred[a][1]<=.06 for a,b in zip(ordered,ordered[1:])):
                groups.append(ordered)
                for i in ids: notes[i]=dict(relationship='fragment',reference_index=j,false_note=None,status='provisional_auto')
        elif len(ids)==1 and edges[ids[0]]==[j]:
            i=ids[0]; p=pred[i]
            overlap=min(p[1],r[1])-max(p[0],r[0])
            duration_ratio=max(p[1]-p[0],r[1]-r[0])/min(p[1]-p[0],r[1]-r[0])
            if duration_ratio>3 or overlap/min(p[1]-p[0],r[1]-r[0])<.6 or abs(p[0]-r[0])>.2 or abs(p[2]-r[2])>12:
                continue
            notes[i]=dict(relationship='one_to_one',reference_index=j,false_note=0,pitch_delta=r[2]-p[2],start_delta=r[0]-p[0],end_delta=r[1]-p[1],status='provisional_auto')
    for i,js in edges.items():
        if len(js)>1 and all(inverse[j]==[i] and ref[j][2]==pred[i][2] and (min(pred[i][1],ref[j][1])-max(pred[i][0],ref[j][0]))/(ref[j][1]-ref[j][0])>=.6 for j in js):
            notes[i]=dict(relationship='split_candidate',reference_index=None,false_note=None,status='unresolved')
        if not js and all(max(0,max(pred[i][0],r[0])-min(pred[i][1],r[1]))>.20 for r in ref):
            notes[i]=dict(relationship='spurious_candidate',reference_index=None,false_note=1,status='provisional_auto')
    return dict(notes=notes,merge_groups=groups,missing_reference_indices=[j for j,ids in inverse.items() if not ids])

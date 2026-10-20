"""Correction handover contracts. Targets/references never enter feature selection."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

NOTE_COLUMNS = ['note_id', 'recording_id', 'start', 'end', 'pitch']
NOTE_FEATURES = ['start', 'end', 'pitch', 'amplitude', 'duration', 'prev_gap', 'next_gap']
PAIR_FEATURES = ['gap', 'pitch_delta', 'duration_sum']

def validate_manifest(manifest):
    required = {'recording_id', 'sample_id', 'split'}
    if not required.issubset(manifest.columns):
        raise ValueError(f'Manifest requires {required}')
    result = manifest.copy(deep=True)
    if result[list(required)].isna().any().any() or result.recording_id.duplicated().any():
        raise ValueError('Missing or duplicate manifest IDs')
    if not result.split.isin(['train', 'val']).all():
        raise ValueError('Only train/val are allowed; test is forbidden')
    if (result.groupby('sample_id').split.nunique() > 1).any():
        raise ValueError('A song cannot occur in both train and val')
    return result

def validate_notes(notes, manifest):
    manifest = validate_manifest(manifest)
    if not set(NOTE_COLUMNS).issubset(notes.columns):
        raise ValueError(f'Notes require {NOTE_COLUMNS}')
    out = notes[NOTE_COLUMNS].copy(deep=True)
    if out[['note_id','recording_id']].isna().any().any() or out.note_id.duplicated().any():
        raise ValueError('Missing or duplicate note IDs')
    if any(out[c].astype(str).str.strip().eq('').any() for c in ['note_id','recording_id']):
        raise ValueError('Empty IDs')
    if not out.recording_id.isin(manifest.recording_id).all():
        raise ValueError('Unknown recording ID')
    if 'split' in notes and not notes['split'].eq(notes.recording_id.map(manifest.set_index('recording_id').split)).all():
        raise ValueError('Split disagrees with manifest')
    try:
        out[['start','end','pitch']] = out[['start','end','pitch']].astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError('Times and MIDI pitch must be numeric') from exc
    if not np.isfinite(out[['start','end','pitch']].to_numpy()).all():
        raise ValueError('Times/pitch must be finite')
    if (out.start < 0).any() or (out.end <= out.start).any() or not out.pitch.between(0,127).all():
        raise ValueError('Require start >= 0, end > start, MIDI pitch in [0,127]')
    duration_col=next((c for c in ['audio_duration','duration_seconds_if_validated'] if c in manifest),None)
    if duration_col:
        durations=manifest.set_index('recording_id')[duration_col].astype(float)
        if not np.isfinite(durations).all() or (durations<=0).any(): raise ValueError('Invalid audio durations')
        if (out.end>out.recording_id.map(durations)+.001).any(): raise ValueError('Event exceeds audio duration (1ms reference rounding allowance)')
    return out


def export_corrections(notes, manifest, path):
    """Export the COMPLETE replacement note set; omissions delete notes.

    Merge: retain an original ID for the merged event and omit absorbed IDs.
    Empty output is permitted. Existing files are never overwritten.
    """
    out = validate_notes(notes, manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as handle:
        out.to_csv(handle, index=False)
    return path


def _paired(truth, predicted, columns=None):
    a, b = np.asarray(truth, dtype=float), np.asarray(predicted, dtype=float)
    if a.shape != b.shape or a.ndim != (2 if columns else 1):
        raise ValueError('Target and prediction shapes must agree')
    if columns and a.shape[1] != columns:
        raise ValueError('Wrong number of columns')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Metrics require finite values')
    return a, b


def classification_metrics(truth, predicted):
    """Binary labels: 1 = merge pair / remove false note; 0 = no action."""
    a,b = _paired(truth,predicted)
    if not np.isin(a,[0,1]).all() or not np.isin(b,[0,1]).all():
        raise ValueError('Binary labels must be 0/1')
    tp,fp,fn,tn = [int(np.sum(mask)) for mask in [(a==1)&(b==1),(a==0)&(b==1),(a==1)&(b==0),(a==0)&(b==0)]]
    return dict(n=len(a),tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,accuracy=(tp+tn)/len(a) if len(a) else None)


def timing_metrics(truth, predicted):
    """Nx2 absolute start/end seconds for task-labelled notes only."""
    a,b = _paired(truth,predicted,2)
    if any((v[:,0]<0).any() or (v[:,1]<=v[:,0]).any() for v in (a,b)):
        raise ValueError('Invalid timing intervals')
    out = dict(n=len(a))
    for i,name in enumerate(['onset','offset']):
        d = b[:,i]-a[:,i]
        out.update({name+'_mae_seconds':float(np.mean(abs(d))) if len(d) else None,name+'_rmse_seconds':float(np.sqrt(np.mean(d*d))) if len(d) else None,name+'_median_absolute_error_ms':float(np.median(abs(d))*1000) if len(d) else None})
    out['onset_within_50ms']=float(np.mean(abs(b[:,0]-a[:,0])<=.05)) if len(a) else None
    out['offset_within_tolerance']=float(np.mean(abs(b[:,1]-a[:,1])<=np.maximum(.05,.2*(a[:,1]-a[:,0])))) if len(a) else None
    return out


def pitch_metrics(truth, predicted):
    """Absolute MIDI pitch; fractional MIDI allowed, 1 semitone = 100 cents."""
    a,b = _paired(truth,predicted)
    if any(((v<0)|(v>127)).any() for v in (a,b)):
        raise ValueError('MIDI pitch out of range')
    d = (b-a)*100
    return dict(n=len(a),mae_cents=float(np.mean(abs(d))) if len(a) else None,mae_semitones=float(np.mean(abs(b-a))) if len(a) else None,octave_error_rate=float(np.mean(np.isclose(abs(b-a),12))) if len(a) else None,rmse_cents=float(np.sqrt(np.mean(d*d))) if len(a) else None,within_50_cents=float(np.mean(abs(d)<=50)) if len(a) else None)


TARGETS = {'merge':['merge_label'],'false':['false_label'],'timing':['target_start','target_end'],'pitch':['target_pitch']}

def training_data(features, targets, task):
    """Return allowlisted X and separately joined labels. Null labels are excluded later.

    All feature rows must have target rows, including ambiguous/null-label rows.
    Never use joined labels as model input. Pair features use pair_id.
    """
    if task not in TARGETS:
        raise ValueError('Unknown task')
    key = 'pair_id' if task == 'merge' else 'note_id'
    allowed = PAIR_FEATURES if task == 'merge' else NOTE_FEATURES
    for frame in (features,targets):
        if key not in frame or frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError('Missing or duplicate training keys')
        if 'split' in frame and not frame.split.eq('train').all():
            raise ValueError('Starter training tables must contain train only')
    if 'split' not in features or not features.split.eq('train').all():
        raise ValueError('Features require explicit train split')
    if set(features[key]) != set(targets[key]):
        raise ValueError('Target row coverage must be exactly 100%; retain ambiguous rows with null labels')
    if not set(allowed).issubset(features) or not set(TARGETS[task]).issubset(targets):
        raise ValueError('Missing allowlisted features or task targets')
    x = features[allowed].copy(deep=True).astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Features must be finite; explicitly impute boundary gaps upstream')
    label_columns = TARGETS[task] + (['ambiguous'] if 'ambiguous' in targets else [])
    overlap = set(label_columns).intersection(features.columns)
    if overlap:
        raise ValueError('Target columns must be separate from features')
    joined = features.merge(targets[[key]+label_columns],on=key,how='left',validate='one_to_one',sort=False)
    return x, joined


def evaluate_end_to_end(corrected, references, manifest):
    """Score complete train/val recordings, including zero-estimate recordings.

    Caller supplies the COMPLETE reference table and explicit recording manifest.
    No task-label/ambiguity argument exists: do not prefilter references or manifest.
    References use the same five columns but IDs are reference-local, not matched IDs.
    """
    manifest = validate_manifest(manifest)
    est, ref = validate_notes(corrected,manifest), validate_notes(references,manifest)
    spec = importlib.util.spec_from_file_location('_handover_bp_core',Path(__file__).resolve().parents[1]/'bp_tuning_core.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rows = []
    for record in manifest.to_dict('records'):
        rid = record['recording_id']
        metrics = core.score_notes(ref.loc[ref.recording_id.eq(rid),['start','end','pitch']].to_numpy(),est.loc[est.recording_id.eq(rid),['start','end','pitch']].to_numpy())
        rows.append({**record,**metrics})
    songs,summary = core.aggregate(rows)
    by_split = {split:core.aggregate([r for r in rows if r['split']==split])[1] for split in manifest.split.unique()}
    return dict(recordings=rows,songs=songs,summary=summary,by_split=by_split)

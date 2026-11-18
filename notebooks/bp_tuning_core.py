"""Reusable train/validation-only Basic Pitch tuning primitives."""
from pathlib import Path
import hashlib
import json
import os
import re
import tempfile
import time
import zipfile
import numpy as np


def sha(path):
    """Streaming SHA256 of a file."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_arrays(arrays):
    frames = None
    for key, bins in [('note', 88), ('onset', 88), ('contour', 264)]:
        value = np.asarray(arrays[key])
        if value.ndim != 2 or value.shape[1] != bins or value.shape[0] == 0:
            raise ValueError(f'Invalid {key} activation shape')
        if value.dtype.kind not in 'fi' or not np.isfinite(value).all():
            raise ValueError(f'Invalid {key} activation values')
        if (value < 0).any() or (value > 1).any():
            raise ValueError(f'{key} activations must lie in [0, 1]')
        if frames is not None and frames != value.shape[0]:
            raise ValueError('Activation time dimensions disagree')
        frames = value.shape[0]
    return {key: list(arrays[key].shape) for key in ('note', 'onset', 'contour')}


def _atomic_write(path, writer):
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def cache_recording(root: Path, record: dict, cache_dir: Path, model, provenance: dict) -> dict:
    """Cache raw activations; duration_seconds is original inference wall time.

    A hit preserves the original runtime, rather than reporting cache-read time.
    Caller supplies audio_sha256, modelsha and versions provenance. No logs are written.
    """
    if record.get('split') not in ('train', 'val'):
        raise ValueError('Only train/val recordings are allowed')
    rid = str(record['recording_id'])
    if not re.fullmatch(r'[A-Za-z0-9_-]+', rid):
        raise ValueError('Unsafe recording_id')
    for key in ('audio_sha256', 'modelsha', 'versions'):
        if key not in provenance:
            raise ValueError(f'Missing provenance: {key}')
    canonical = json.loads(json.dumps(provenance, sort_keys=True, allow_nan=False))
    if canonical != provenance:
        raise ValueError('Provenance must use JSON-native values')
    cache_dir = Path(cache_dir).resolve()
    path = cache_dir / (rid + '.npz')
    sidecar = path.with_suffix('.json')
    if path.is_file() and sidecar.is_file():
        try:
            metadata = json.loads(sidecar.read_text(encoding='utf-8'))
            if metadata['provenance'] == provenance and metadata['npzsha'] == sha(path):
                with np.load(path, allow_pickle=False) as arrays:
                    shapes = _validate_arrays(arrays)
                runtime = float(metadata['duration_seconds'])
                if not np.isfinite(runtime) or runtime < 0:
                    raise ValueError('Invalid cached runtime')
                return dict(recording_id=rid, npz_path=str(path), duration_seconds=runtime,
                            cache_hit=True, shapes=shapes)
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
            pass  # Invalid entries are regenerated, never decoded.
    from basic_pitch.inference import run_inference
    audio_path = Path(root) / record['audio_path']
    if sha(audio_path) != provenance['audio_sha256']:
        raise ValueError('Audio checksum disagrees with provenance')
    start = time.perf_counter()
    arrays = run_inference(audio_path, model)
    runtime = time.perf_counter() - start
    shapes = _validate_arrays(arrays)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, lambda handle: np.savez_compressed(handle, **{key: arrays[key] for key in shapes}))
    metadata = dict(provenance=canonical, npzsha=sha(path), duration_seconds=runtime, shapes=shapes)
    _atomic_write(sidecar, lambda handle: handle.write(json.dumps(metadata, sort_keys=True, allow_nan=False).encode('utf-8')))
    return dict(recording_id=rid, npz_path=str(path), duration_seconds=runtime, cache_hit=False, shapes=shapes)


def make_model():
    """Load bundled ONNX model, replacing its session with bounded CPU ORT."""
    import onnxruntime as ort
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import Model
    path = build_icassp_2022_model_path(FilenameSuffix.onnx).resolve()
    model = Model(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    model.model = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    model.model_type = Model.MODEL_TYPES.ONNX
    return model, path, sha(path)


def decode(arrays, config):
    """Decode copied activations; minimum_note_length is in milliseconds."""
    from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
    from basic_pitch.note_creation import model_output_to_notes
    _validate_arrays(arrays)
    ms = float(config.get('minimum_note_length', 127.7))
    if not np.isfinite(ms) or ms < 0:
        raise ValueError('minimum_note_length must be finite and nonnegative')
    return model_output_to_notes(
        {key: np.array(arrays[key], copy=True) for key in ('note', 'onset', 'contour')},
        onset_thresh=config.get('onset_threshold', .5),
        frame_thresh=config.get('frame_threshold', .3),
        min_note_len=int(np.round(ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP))),
        melodia_trick=config.get('melodia_trick', True),
        min_freq=None, max_freq=None, infer_onsets=True, include_pitch_bends=True,
        multiple_pitch_bends=False, midi_tempo=120)


def _events(events, predicted=False):
    if predicted:
        try:
            events = [event[:3] for event in events]
        except (TypeError, IndexError) as exc:
            raise ValueError('Expected note event tuples') from exc
    values = np.asarray(events, dtype=float)
    if values.shape == (0,):
        values = values.reshape(0, 3)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError('Expected Nx3 onset, offset, MIDI events')
    if not np.isfinite(values).all():
        raise ValueError('Events must be finite')
    if (values[:, 0] < 0).any() or (values[:, 1] <= values[:, 0]).any():
        raise ValueError('Events require nonnegative onset and offset > onset')
    if ((values[:, 2] < 0) | (values[:, 2] > 127)).any():
        raise ValueError('MIDI pitch must be between 0 and 127')
    return values


def _rates(counts):
    result = dict(counts)
    for tp_key, prefix in [('tp_onset', ''), ('tp_offset', 'offset_')]:
        tp, nr, ne = counts[tp_key], counts['ref_count'], counts['est_count']
        result[prefix + 'precision'] = tp / ne if ne else 0.
        result[prefix + 'recall'] = tp / nr if nr else 0.
        result[prefix + 'f1'] = 2 * tp / (nr + ne) if nr + ne else 0.
    return result


def score_notes(reference_events, predicted):
    """One-to-one mir_eval note matching, with and without offsets."""
    from librosa import midi_to_hz
    from mir_eval.transcription import match_notes
    ref, est = _events(reference_events), _events(predicted, predicted=True)
    counts = dict(ref_count=len(ref), est_count=len(est), tp_onset=0, tp_offset=0)
    if len(ref) and len(est):
        for key, ratio in [('tp_onset', None), ('tp_offset', .2)]:
            counts[key] = len(match_notes(
                ref[:, :2], midi_to_hz(ref[:, 2]), est[:, :2], midi_to_hz(est[:, 2]),
                onset_tolerance=.05, pitch_tolerance=50., offset_ratio=ratio,
                offset_min_tolerance=.05, strict=False))
    return _rates(counts)


def aggregate(record_metrics):
    """Pool excerpt counts within sample_id before macro-averaging songs."""
    keys = ('ref_count', 'est_count', 'tp_onset', 'tp_offset')
    groups = {}
    for record in record_metrics:
        sid = record['sample_id']
        if sid not in groups:
            groups[sid] = dict(sample_id=sid, **{key: record.get(key) for key in ('language', 'singer')},
                               vocal_range=record.get('vocal_range', record.get('range')),
                               **dict.fromkeys(keys, 0), recordings=0)
        song = groups[sid]
        for key in keys:
            value = record[key]
            if int(value) != value or value < 0:
                raise ValueError('Counts must be nonnegative integers')
            song[key] += int(value)
        if any(record[key] > min(record['ref_count'], record['est_count']) for key in ('tp_onset', 'tp_offset')):
            raise ValueError('True positives exceed event counts')
        song['recordings'] += 1
    rows = [_rates(groups[sid]) for sid in sorted(groups)]
    pooled = _rates({key: sum(row[key] for row in rows) for key in keys})
    summary = {'song_macro_' + key: float(np.mean([row[key] for row in rows])) if rows else 0.
               for key in ('f1', 'offset_f1', 'precision', 'recall')}
    summary.update(micro_f1=pooled['f1'], songs=len(rows), recordings=sum(row['recordings'] for row in rows))
    return rows, summary

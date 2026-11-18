"""Read-only frozen dataset/handover verification; no fitting, inference or scoring.

Run with Python 3.10+ from any directory. Test annotations are hashed, never parsed.
"""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HANDOVER = ROOT / 'basic_pitch_runs' / 'correction_handover'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def packaged_path(relative):
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), f'Path outside package: {relative}')
    require(path.is_file(), f'Missing packaged file: {relative}')
    return path


def verify():
    lock = read_json(ROOT / 'package_integrity.json')
    for relative, digest in lock.items():
        require(sha(packaged_path(relative)) == digest, f'Frozen hash mismatch: {relative}')

    songs = read_csv(ROOT / 'selected_100.csv')
    recordings = read_csv(ROOT / 'selected_recordings.csv')
    files = read_csv(ROOT / 'selected_files.csv')
    split_rows = read_csv(ROOT / 'split_manifest.csv')
    by_song = {row['sample_id']: row for row in songs}
    require(len(songs) == len(by_song) == 100, 'Song identity count')
    require(len({row['canonical_song_id'] for row in songs}) == 100, 'Canonical song identity')
    require(Counter(row['split'] for row in songs) == Counter(train=60, val=20, test=20), 'Song splits')
    require(len(split_rows) == 100, 'Split manifest count')
    for row in split_rows:
        source = by_song[row['sample_id']]
        require(row['split'] == source['split'], 'Split manifest membership')
        require(row['canonical_song_id'] == source['canonical_song_id'], 'Split canonical identity')
    require(len(recordings) == 858 and len({row['recording_id'] for row in recordings}) == 858,
            'Recording identities')
    require(Counter(row['split'] for row in recordings) == Counter(train=534, val=160, test=164),
            'Recording splits')
    require(len(files) == 3432 and len({row['export_path'] for row in files}) == 3432,
            'Unique file exports')
    exports = {(row['recording_id'], row['role']): row for row in files}
    require(len(exports) == 3432, 'Recording/role identity')
    for row in files:
        path = packaged_path(row['export_path'])
        require(path.stat().st_size == int(row['actual_size']), f'File size: {path}')
        require(sha(path) == row['local_sha256'], f'Selected file hash: {path}')
    counts = Counter(row['sample_id'] for row in recordings)
    for row in songs:
        require(counts[row['sample_id']] == int(row['recording_count']), 'Song excerpt count')
    for row in recordings:
        song = by_song[row['sample_id']]
        for key in ('canonical_song_id', 'split', 'language', 'singer', 'vocal_range'):
            require(row[key] == song[key], f'Recording metadata: {key}')
        for role in ('audio', 'json', 'musicxml', 'textgrid'):
            entry = exports[(row['recording_id'], role)]
            require(entry['sample_id'] == row['sample_id'], 'Export song membership')
            require(entry['export_path'].startswith('data/' + row['split'] + '/'), 'Export split path')

    input_version = read_json(HANDOVER / 'input_version.json')
    note_counts = {}
    for split, expected_recordings, expected_notes in [('train', 534, 7623), ('val', 160, 2329)]:
        filename = f'selected_predictions_{split}.json'
        require(sha(HANDOVER / filename) == input_version['prediction_sha256'][filename],
                f'Prediction byte version: {split}')
        predictions = read_json(HANDOVER / filename)
        expected_ids = {row['recording_id'] for row in recordings if row['split'] == split}
        require(set(predictions) == expected_ids and len(predictions) == expected_recordings,
                f'Prediction recording set: {split}')
        features = read_csv(HANDOVER / f'{split}_features.csv')
        by_note = {row['note_id']: row for row in features}
        require(len(by_note) == len(features) == expected_notes, f'Feature note identities: {split}')
        seen = set()
        for recording_id, events in predictions.items():
            for index, event in enumerate(sorted(events, key=lambda e: (e[0], e[1], e[2]))):
                note_id = f'{recording_id}:n{index:05d}'
                feature = by_note[note_id]
                require(feature['recording_id'] == recording_id and feature['split'] == split,
                        'Feature recording/split')
                require([float(feature[key]) for key in ('start', 'end', 'pitch', 'amplitude')]
                        == [float(value) for value in event[:4]], f'Prediction/feature values: {note_id}')
                seen.add(note_id)
        require(seen == set(by_note), f'Feature coverage: {split}')
        note_counts[split] = len(features)

    version = read_json(HANDOVER / 'label_version.json')
    require(version['prediction_version'] == input_version, 'Label prediction version')
    require(sha(ROOT / 'notebooks/correction_labels.py') == version['algorithm_sha256'],
            'Original label algorithm version')
    require(sha(ROOT / 'notebooks/build_correction_handover.py') == version['exporter_sha256'],
            'Original exporter version')
    for filename, digest in version['table_sha256'].items():
        require(sha(HANDOVER / filename) == digest, f'Packaged v1 table hash: {filename}')
    manifest = read_csv(HANDOVER / 'training_recording_manifest.csv')
    train_ids = {row['recording_id'] for row in recordings if row['split'] == 'train'}
    require(len(manifest) == 534 and {row['recording_id'] for row in manifest} == train_ids,
            'Complete training evaluation manifest')
    for row in manifest:
        for role in ('audio', 'json', 'musicxml', 'textgrid'):
            require(row[role + '_path'] == exports[(row['recording_id'], role)]['export_path'],
                    'Packaged manifest path')
    # Reconstruct source-relative CSV bytes in memory only, with original CRLF.
    source = io.StringIO(newline='')
    writer = csv.DictWriter(source, fieldnames=list(manifest[0]), lineterminator='\r\n')
    writer.writeheader()
    provenance = version['manifest_provenance']
    for row in manifest:
        original = dict(row)
        for column in provenance['path_columns']:
            value = original[column]
            require(value.startswith(provenance['packaged_path_prefix']), 'Manifest path prefix')
            original[column] = provenance['source_path_prefix'] + value[len(provenance['packaged_path_prefix']):]
        writer.writerow(original)
    require(hashlib.sha256(source.getvalue().encode()).hexdigest()
            == provenance['source_training_recording_manifest_sha256'], 'Source-relative manifest provenance')

    targets = read_csv(HANDOVER / 'train_targets.csv')
    train_notes = {row['note_id'] for row in read_csv(HANDOVER / 'train_features.csv')}
    require(len(targets) == len(train_notes) and {row['note_id'] for row in targets} == train_notes,
            'v1 target coverage')
    v2 = read_json(HANDOVER / 'proposed_v2/label_version_v2.json')
    require(not v2['approved'] and not input_version['labels_approved'], 'Provisional approval scope')
    for filename, digest in v2['sha256'].items():
        path = HANDOVER / 'proposed_v2' / filename
        require(sha(path) == digest, f'Proposed v2 version: {filename}')
    for filename in ('merge_candidates.csv', 'merge_targets.csv'):
        require(not read_csv(HANDOVER / filename), 'Shared v1 pairs must remain empty')
    folds = read_csv(HANDOVER / 'folds.csv')
    require(len(folds) == 60 and {row['sample_id'] for row in folds}
            == {row['sample_id'] for row in songs if row['split'] == 'train'}, 'Training fold coverage')

    derivative_name = 'no_correction_song_metrics_metadata_v2'
    metadata_version = read_json(HANDOVER / (derivative_name + '.json'))
    for relative, digest in metadata_version['source_sha256'].items():
        require(sha(ROOT / relative) == digest, f'Subgroup source hash: {relative}')
    require(sha(HANDOVER / (derivative_name + '.csv')) == metadata_version['artifact_sha256'],
            'Subgroup derivative hash')
    original = read_csv(HANDOVER / 'no_correction_song_metrics.csv')
    derivative = read_csv(HANDOVER / (derivative_name + '.csv'))
    require(len(original) == len(derivative) == metadata_version['rows'] == 60, 'Song metric rows')
    for old, new in zip(original, derivative):
        require(old['sample_id'] == new['sample_id'], 'Song metric row order')
        for key, value in old.items():
            if key not in ('language', 'singer', 'range'):
                require(value == new[key], f'Saved metric string changed: {key}')
        for key in ('language', 'singer', 'vocal_range'):
            require(new[key] == by_song[new['sample_id']][key] and bool(new[key]), 'Song subgroup metadata')
    return dict(passed=True, frozen_package_files=len(lock), selected_files=len(files),
                songs=len(songs), recordings=len(recordings), prediction_notes=note_counts,
                metadata_rows=len(derivative), source_manifest_provenance=True,
                fitting=False, inference=False, scoring=False, test_annotations_parsed=False)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))

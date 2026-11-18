# Scientific deliverables and reproducibility

## Read the stages in order

| Stage | Entry point | What is established / how to use it |
|---|---|---|
| Frozen dataset | `selected_100.csv`, `split_manifest.csv`, `selected_recordings.csv`, `selected_files.csv` | 100 songs; 60/20/20 train/validation/test songs; 534/160/164 excerpts; 858 excerpts, 3,432 files, about 2.515 audio hours. Keep selections and splits fixed. |
| Sample verification and EDA | `GTSinger_100_EDA_and_BasicPitch.ipynb` | Read saved outputs. Historical preparation only: no BP installation/inference or test evaluation. Its unrun configurations are not the later benchmark. |
| BP decoder selection | `notebooks/GTSinger_100_BasicPitch_Tuning.ipynb` | Read saved outputs; retain the accidental-rerun guard. Thirteen training configurations produced three frozen validation finalists. Validation selected onset 0.99; frame 0.30, minimum note length 127.7 ms and Melodia enabled remained at defaults. Neural weights were not trained. |
| Initial correction handover | [Historical handover](README%20copy.md), `notebooks/correction_handover/Shared_Correction_Label_Audit.ipynb`, `notebooks/correction_handover/DATA_DICTIONARY.md` | Frozen train/validation predictions, initial features, automatic targets and review queues. Shared-review execution writes derived proposals; it is not a read-only Run All. Group approval remains false in the supplied stage records. |
| Later fitted correction experiments | Four `*_starter.ipynb` files in `notebooks/correction_handover/` | Despite retained filenames, these contain fitted models and exports. Felicia, Le Xin and Hann Qian use proposed-v2 note targets; Queenie creates notebook-local v2 pairs, separate from the zero-row shared v1 pair tables. Read saved outputs rather than starting a new fit. |
| Later split, snap and combined chain | `le_xin_split_model.ipynb`, `le_xin_snap_model.ipynb`, `combined_correction_system.ipynb` in the same folder | Later experiments, with chain-v4 status dated 2026-10-27 in the combined notebook. Initial whitelist/candidate counts do not describe every later model. |
| Application / deployed-model environment | [RUN_GUIDE.md](RUN_GUIDE.md), `backend/README.md`, `le_xin_work/inference/README.md` | Separate app and Python-3.13 inference environment; not a substitute for the original BP experiment environment. |

There are nine notebooks under `notebooks/` and one root EDA notebook. Initial handover instructions describe an earlier development stage; their no-model/no-test statements are not current repository-wide status claims.

## Scientific interpretation

The split is **song-grouped, not singer-disjoint**. All held-out singers also occur in training (20 train, 19 validation, 19 test singers). Do not claim unseen-singer generalization. Language, singer and folder-derived range/context describe this selected sample; folder range is not measured acoustic pitch and technique context is not an event-level technique label.

BP validation song-macro onset+pitch F1 was 0.18547149856270645 by default, 0.27173987311241976 at onset 0.95 and 0.27264306924988774 at onset 0.99. Selected offset-aware F1 was 0.16024855433363022. Training chose the shortlist; validation chose the decoder. The small 0.95/0.99 difference does not prove a uniquely optimal threshold. Later use of the same validation songs remains development evidence, not a fresh independent test.

The initial seven note features and three pair features are a **manual initial whitelist**, not experimentally optimal selection or the final feature set of all later experiments. Reference annotations create targets and evaluate predictions; they must not become inference inputs. Labels are provisional automatic correspondences; fitted-model outputs do not establish manual review.

The four `notebooks/correction_handover/exports/*_unchanged.csv` files are intentional per-person aliases of the same no-change baseline, not four independent experiments.

## Read-only verification and loading

From the repository root, using Python 3.10 or newer (standard library only):

```sh
python notebooks/verify_scientific_inputs.py
```

This checks literal SHA256 locks, selected-file sizes, manifest identities and packaged paths, prediction/feature agreement, initial label provenance and the subgroup-ready derivative. It performs no fitting, inference, sampling or transcription scoring and writes no repository files. Test files are hashed only for frozen-package integrity; test annotations are not parsed. A mismatch is an error, not an invitation to regenerate a lock.

For an environment containing NumPy and pandas, the existing path remapper is also usable without BP inference:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('notebooks').resolve()))
import bp_experiment as exp
print(exp.frozen_check())
rows = exp.inputs()  # train/validation only; resolves packaged export paths
assert len(rows) == 694
assert all((exp.ROOT / r['audio_path']).is_file() for r in rows)
```

`selected_recordings.csv` retains original `source_files/` paths as provenance. Actual files live at `selected_files.csv` **export_path**; handover manifests use the packaged `data/train/` and `data/val/` layout. Do not create a duplicate source tree to satisfy provenance paths.

### Frozen byte contract and artifact versions

`package_integrity.json` and `selected_files.csv` pin original file bytes, not normalized JSON values. Native line endings can differ between files. `.gitattributes` marks immutable datasets/manifests/handover inputs `-text` so Git does not change them with `core.autocrlf` or platform defaults. Preserve those exact bytes; do not run newline formatters over them or replace frozen hashes with hashes of a preferred text representation. Version a genuinely different scientific artifact separately.

`input_version.json` pins the selected train/validation prediction bytes. `label_version.json` identifies packaged v1 tables as `provisional_v1_packaged_paths_r2`: the original label-generation algorithm/exporter hashes remain identified, while `manifest_provenance` distinguishes the source-relative manifest from the packaged path layout. Only the four path columns differ in scope; song identities, row order, labels, prediction IDs and non-path metadata are unchanged. The source-relative manifest digest remains explicit, independently of the packaged manifest digest.

For stratified training-baseline reporting, use `basic_pitch_runs/correction_handover/no_correction_song_metrics_metadata_v2.csv` and its companion provenance JSON. This is a one-to-one metadata join by `sample_id` to `selected_100.csv`, retaining the original numeric strings and row order. `vocal_range` is the consistent folder-range key. The original `no_correction_song_metrics.csv` remains the historical numeric artifact with blank metadata. No predictions, references or scores were regenerated for the metadata derivative. Future shared-notebook scoring uses a separate `no_correction_song_metrics_metadata_rescored.csv`, not this join-only artifact.

## Test chronology and freeze boundary

1. **EDA/preparation:** no BP inference or test evaluation. Test files appear only in package checks and descriptive counts.
2. **BP decoder selection and initial handover:** training search, validation shortlist selection and train/validation export only. The BP notebook and `baseline_config.json`, `input_version.json`, `group_review_gate.json` retain stage-specific no-test flags. These records are historical, not global flags for all later repository activity.
3. **Later correction pipeline:** `le_xin_work/inference/README.md` records a completed test run on **2026-10-27**, after the chain-v4 models/settings were fixed. Saved `le_xin_work/test_run/test_manifest.csv` and `test_run_log.csv` contain 164 recordings; `test_scores.csv` contains seven stage rows. These are existing saved artifacts, not a new evaluation. The documented freeze boundary is the fixed chain-v4 models/settings before that run; repository records alone do not independently establish the authorization history.

Do not use test results for further selection, rerun test to reconcile documentation, or attribute the later test run to BP decoder tuning. Any further experiment needs an explicit protocol and separate authorization.

## Environments and reproduction limits

- Saved notebook outputs and the standard-library verification command need no model installation.
- `requirements-lock.txt` records the original notebook dependency set and declares `xgboost==3.1.1` for the evolved pitch notebook. This is a dependency declaration, not evidence of which XGBoost version produced its historical outputs. The complete environment has not been newly installed/resolved or re-executed here; do not infer current-platform or hosted-Colab compatibility. Read-only loading was checked separately from a full environment rebuild.
- Shared review needs NumPy, pandas, plotting/audio-display dependencies and, for scoring, librosa/mir_eval. It can use frozen predictions without Basic Pitch inference. Run it only when regeneration of its proposal/review files is intended; save curated targets in a distinct versioned file.
- Full historical EDA is not runnable unchanged from this tree: `validation_report.json` and original source copies are omitted. Full BP retuning also requires omitted activation caches, report trees and helper snapshots (`phase_a_workflow.py`, `test_bp_tuning_core.py`, `execute_bp_notebook.py`). Removing the notebook guard does not supply those artifacts.
- Original sampling code, seed and selection universe are not bundled. The manifests verify the chosen sample, not seed-based reconstruction of the original selection algorithm. The EDA's larger 12,230-file preservation record describes its original workspace, not this 3,439-entry package lock.
- The app/inference requirements and saved correction models are a separate stage. Neither local loading checks nor a matching model checksum establishes numerical equivalence to the original BP run on another platform.

A full resampling, retuning or model-training reproduction requires a separately specified workspace and experiment protocol. Keep original results readable and distinguish artifact verification from numerical re-execution.

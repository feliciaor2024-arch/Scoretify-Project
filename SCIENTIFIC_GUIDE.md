# Scientific deliverables and reproducibility

## Read the stages in order

| Stage | Entry point | What is established / how to use it |
|---|---|---|
| Frozen dataset | [Songs](selected_100.csv), [split](split_manifest.csv), [recordings](selected_recordings.csv), [files](selected_files.csv) | 100 songs; 60/20/20 train/validation/test songs; 534/160/164 excerpts; 858 excerpts, 3,432 files, about 2.515 audio hours. Keep selections and splits fixed. |
| Sample verification and EDA | [EDA notebook](eda/GTSinger_100_EDA_and_BasicPitch.ipynb), saved cells 16/18 | Composition tables/charts and BP input preparation, not the decoder benchmark. No BP installation/inference or test evaluation. Read saved outputs. |
| BP decoder selection | [BP tuning notebook](notebooks/GTSinger_100_BasicPitch_Tuning.ipynb), saved cells 25 (consolidated evidence), 29/31 (validation/selection), 35 (export) | Thirteen training configurations produced three frozen validation finalists. Validation selected onset 0.99; frame 0.30, minimum note length 127.7 ms and Melodia enabled remained at defaults. Neural weights were not trained. Keep the accidental-rerun guard. |
| Initial correction handover | [Historical instructions](notebooks/correction_handover/README.md), [shared label audit](notebooks/correction_handover/Shared_Correction_Label_Audit.ipynb), [data dictionary](notebooks/correction_handover/DATA_DICTIONARY.md) | Frozen train/validation predictions, initial features, automatic targets and review queues. Shared review writes proposals; it is not read-only Run All. Group approval remains false in the supplied stage records. |
| Later fitted correction experiments | [Felicia: delete](notebooks/correction_handover/felicia_false_starter.ipynb), [Le Xin: timing](notebooks/correction_handover/le_xin_timing_starter.ipynb), [Hann Qian: pitch](notebooks/correction_handover/hann_qian_pitch_starter.ipynb), [Queenie: merge](notebooks/correction_handover/queenie_merge_starter.ipynb) | Retained starter filenames contain fitted experiments and exports. The first three use proposed-v2 note targets; Queenie constructs notebook-local v2 pairs, separate from zero-row shared v1 tables. Read saved outputs rather than fitting again. |
| Later split, snap and combined chain | [Split notebook](notebooks/correction_handover/le_xin_split_model.ipynb), [snap notebook](notebooks/correction_handover/le_xin_snap_model.ipynb), [combined chain](notebooks/correction_handover/combined_correction_system.ipynb), [model comparisons](final_pipeline/experiments/) | Later correction experiments; combined notebook Step 5 records chain-v4 status on 27 October 2026. Initial whitelist/candidate counts do not describe all later models. |
| Final pipeline and saved test | [Overview](final_pipeline/README.md), [step-by-step pipeline](final_pipeline/pipeline.md), [runtime settings](final_pipeline/inference/models/settings.json), [test scores](final_pipeline/test_run/test_scores.csv) | Five correction steps; seven estimators in six joblib containers. Saved runtime split uses 64 features; historical stage descriptions of “74” are not its current input list. Test was completed later than the initial handover. |
| Report figures | [Figure script](report/make_figures.py), [figures](report/figures/) | Redraws Figs. 1–6 of the report from saved repository files only (sample manifests, `data/` annotations/audio, `final_pipeline/test_run/`). Nothing is retrained or re-run; Fig. 4–6 numbers are re-scored with the scorer in [score_test.py](final_pipeline/inference/score_test.py). |
| Application / model environment | [Run guide](RUN_GUIDE.md), [backend API](backend/README.md), [inference usage](final_pipeline/inference/README.md) | Separate app/Python-3.13 inference environment; not a substitute for the original BP experiment environment. |

There are nine notebooks under `notebooks/` and one EDA notebook under `eda/`. Initial handover instructions describe an earlier stage; their no-model/no-test statements are not current repository-wide claims. [notebooks/correction_handover/](notebooks/correction_handover/) holds notebooks and experiment exports; [basic_pitch_runs/correction_handover/](basic_pitch_runs/correction_handover/) holds frozen shared prediction/feature/target inputs and baselines.

## Report-to-evidence map

The table follows the current report’s Sections 5.1–5.4 and Tables 9/11/12/13. Every figure listed below is drawn by [report/make_figures.py](report/make_figures.py) into [report/figures/](report/figures/). Cell numbers are 1-based, including markdown cells. Read saved outputs; these pointers do not authorize reruns.

| Report evidence | Submitted source | Interpretation / limits |
|---|---|---|
| Sample composition / Fig. 1 | [EDA](eda/GTSinger_100_EDA_and_BasicPitch.ipynb), cells 16/18; [selected_100.csv](selected_100.csv), `language` × `split`, `vocal_range` | Saved composition tables and embedded charts. Report image: `report/figures/fig1_dataset.png`. |
| BP decoder selection | [BP notebook](notebooks/GTSinger_100_BasicPitch_Tuning.ipynb), cells 25/29/31/35 | Training shortlist, validation selection and frozen exports; decoder tuning, not neural-weight training. |
| Table 9: initial four models, validation | [Delete validation CSV](notebooks/correction_handover/exports/felicia_false_results_val_v2_exploratory.csv); [timing notebook](notebooks/correction_handover/le_xin_timing_starter.ipynb), Steps 3/6; [pitch notebook](notebooks/correction_handover/hann_qian_pitch_starter.ipynb), Pitch model; [merge validation CSV](notebooks/correction_handover/exports/queenie_merge_v2_VALIDATION_results_20261020_0107.csv) | These are initial-model results, not the final chain. Timing’s early 0.3115 is distinct from the later audio/pitch-feature timing result 0.3755. |
| Table 11: timing, split, delete model comparisons (training OOF) | [Timing notebook](notebooks/correction_handover/le_xin_timing_starter.ipynb), Steps 3/5b; [split results](final_pipeline/experiments/split_v2/results.csv), `v2:hgb_rich`, `v2:rf_rich`, `v2:avg_rich`; [delete results](final_pipeline/experiments/delete_v2/results.csv), `replace / y_official_pool`, 49 features, RF vs GB | Compare within a task on the same input pool/features. Some comparisons precede final gates or later chain versions, so absolute scores differ from final stage scores. |
| Table 12: start/end snap comparisons (training OOF) | [Start-snap results](final_pipeline/experiments/start_snap_v3/results.csv), `2 wider`, ±150 ms + extra kinds, forest vs ranker; [end-snap results](final_pipeline/experiments/end_snap/results_train_oof_rules_dt_rf.csv), decision tree vs random forest | The start comparison is one round; final inference snaps twice. End-only changes leave onset F1 unchanged. |
| Table 13: final chain training OOF and test; Fig. 4 numerical evidence | [Combined notebook](notebooks/correction_handover/combined_correction_system.ipynb), Step 5; [chain stage recipes/tables](final_pipeline/chain_v4/); [test scores](final_pipeline/test_run/test_scores.csv), [per-song scores](final_pipeline/test_run/test_scores_by_song.csv) | Test run recorded on 27 October 2026. Final-note F1 is separate from sheet-readback F1; no scores regenerated here. |
| Fig. 2: training pitch/spectrogram illustration | [Recording manifest](basic_pitch_runs/correction_handover/recording_manifest.csv), `5fe72323ae3e5a6c18dd`: GTS021, English/EN-Alto-1/Breathy/someday or oneday/Breathy_Group, clip `0002`; WAV/JSON under `data/train/` | Report image `report/figures/fig2_pitch_vs_notes.png`: pYIN (22,050 Hz, hop 256) over the annotated notes, near-silent frames (< −35 dB) hidden. Not an embedded chart in the submitted EDA notebook. |
| Fig. 3: final pipeline diagram | [Step-by-step pipeline](final_pipeline/pipeline.md) | Diagram only, no data. Report image: `report/figures/fig3_pipeline.png`. |
| Fig. 4: test scores by stage | [Test scores](final_pipeline/test_run/test_scores.csv), `onset_f1` / `offset_f1`, seven stages | Saved numerical evidence. Report image: `report/figures/fig4_test_by_step.png`. |
| Fig. 5: illustrative before/after recording | [Test manifest](final_pipeline/test_run/test_manifest.csv), `a17354f2bfe3c15d97c0`: GTS018, English/EN-Tenor-1/Breathy/Someone Like You/Control_Group, clip `0005`; [BP notes](final_pipeline/test_run/test_basic_pitch.csv), [corrected notes](final_pipeline/test_run/test_notes.csv), JSON under `data/test/` | Saved event geometry: 14 BP notes, 12 corrected notes and 15 reference notes, including the deleted pitch-53 note near 0.4 s. Report image: `report/figures/fig5_example_recording.png`. |
| Fig. 6: per-song test gains | [Per-song scores](final_pipeline/test_run/test_scores_by_song.csv), `basic_pitch` / `notes`, sorted by `notes` | Saved numerical evidence. Report image: `report/figures/fig6_test_by_song.png`. |

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

### Redrawing the report figures

```sh
python report/make_figures.py        # all six figures
python report/make_figures.py 4 6    # only Figs. 4 and 6
```

Install its packages first with `python -m pip install -r report/requirements.txt` (the `backend/requirements.txt` packages plus `matplotlib`; Fig. 2 needs `librosa`, Fig. 5 needs `librosa` and `mir_eval`). It reads only saved files and overwrites only `report/figures/fig<N>_*.png`; it fits nothing, runs no inference and does not regenerate `final_pipeline/test_run/`. Drawing Figs. 5–6 from saved test tables is not a new test evaluation.

### Frozen byte contract and artifact versions

`package_integrity.json` and `selected_files.csv` pin original file bytes, not normalized JSON values. Native line endings can differ between files. `.gitattributes` marks immutable datasets/manifests/handover inputs `-text` so Git does not change them with `core.autocrlf` or platform defaults. Preserve those exact bytes; do not run newline formatters over them or replace frozen hashes with hashes of a preferred text representation. Version a genuinely different scientific artifact separately.

`input_version.json` pins the selected train/validation prediction bytes. `label_version.json` identifies packaged v1 tables as `provisional_v1_packaged_paths_r2`: the original label-generation algorithm/exporter hashes remain identified, while `manifest_provenance` distinguishes the source-relative manifest from the packaged path layout. Only the four path columns differ in scope; song identities, row order, labels, prediction IDs and non-path metadata are unchanged. The source-relative manifest digest remains explicit, independently of the packaged manifest digest.

For stratified training-baseline reporting, use `basic_pitch_runs/correction_handover/no_correction_song_metrics_metadata_v2.csv` and its companion provenance JSON. This is a one-to-one metadata join by `sample_id` to `selected_100.csv`, retaining the original numeric strings and row order. `vocal_range` is the consistent folder-range key. The original `no_correction_song_metrics.csv` remains the historical numeric artifact with blank metadata. No predictions, references or scores were regenerated for the metadata derivative. Future shared-notebook scoring uses a separate `no_correction_song_metrics_metadata_rescored.csv`, not this join-only artifact.

## Test chronology and freeze boundary

1. **EDA/preparation:** no BP inference or test evaluation. Test files appear only in package checks and descriptive counts.
2. **BP decoder selection and initial handover:** training search, validation shortlist selection and train/validation export only. The BP notebook and `baseline_config.json`, `input_version.json`, `group_review_gate.json` retain stage-specific no-test flags. These records are historical, not global flags for all later repository activity.
3. **Later correction pipeline:** [inference README](final_pipeline/inference/README.md) records a completed test run on **2026-10-27**, after the chain-v4 models/settings were fixed. Saved [test manifest](final_pipeline/test_run/test_manifest.csv) and [run log](final_pipeline/test_run/test_run_log.csv) contain 164 recordings; [test scores](final_pipeline/test_run/test_scores.csv) contains seven stage rows. These are existing saved artifacts, not a new evaluation. The documented freeze boundary is the fixed chain-v4 models/settings before that run; repository records alone do not independently establish the authorization history.

Do not use test results for further selection, rerun test to reconcile documentation, or attribute the later test run to BP decoder tuning. Any further experiment needs an explicit protocol and separate authorization.

## Environments and reproduction limits
- Follow [Run guide](RUN_GUIDE.md) to run the current application. Historical notebooks used separate environments; the application environment does not reproduce every earlier experiment.
- The repository includes frozen manifests, predictions and saved notebook results. Some original sampling code, EDA inputs and full Basic Pitch activation caches are not included, so full resampling or retuning requires additional files.
- Run python [notebooks/verify_scientific_inputs.py](notebooks/verify_scientific_inputs.py) to check the supplied data and prediction files without running models. Read saved notebook outputs before attempting execution, and do not rerun the final test as part of setup.
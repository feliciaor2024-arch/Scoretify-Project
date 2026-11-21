# Initial correction handover — historical stage snapshot

**Scope:** after Basic Pitch decoder selection and before correction-model fitting. Present-tense instructions below describe that initial stage, not the later experiments now saved in the linked notebooks. Read [SCIENTIFIC_GUIDE.md](../../SCIENTIFIC_GUIDE.md) for current navigation and test chronology. Notebook filenames were retained as their contents evolved.

## The issue we are solving
Basic Pitch has already converted the singing audio into predicted notes, but the notes still contain mistakes. Our job is to build traditional-ML correction components that OUTPUT CORRECTED NOTES: merge fragments, remove genuinely extra notes, fix timing, and fix pitch.

**Basic Pitch is finished for this handover. Correction models are NOT trained yet. The shared training labels are NOT approved yet.** Do not mistake the starter notebooks or their unchanged-note exports for finished models.

**Label version in use:** the Felicia, Le Xin and Hann Qian starters now load the **proposed v2 labels** (`basic_pitch_runs/correction_handover/proposed_v2/train_targets_v2.csv`, mutual-best overlap rule). v2 is automatic, NOT manually verified and NOT approved. The original v1 `train_targets.csv` is unchanged. See [proposed v2 label notes](../../basic_pitch_runs/correction_handover/proposed_v2/README_v2.md). Counts below are v2 with v1 in brackets.

The fixed input is original **tuning_001: onset 0.99, frame 0.30, minimum note length 127.7 ms, Melodia on**. Everyone uses the same predictions and note IDs. Do not resample songs, change splits, or substitute predictions.

## Seven notebooks at the initial handover stage
Literal project paths and shell commands below are relative to the repository root, unless a table explicitly names files within the frozen handover. Clickable links resolve from this document. Open in Jupyter or VS Code; read saved outputs first.

| Order | Notebook | Purpose |
|---|---|---|
| Background (first) | [EDA notebook](../../eda/GTSinger_100_EDA_and_BasicPitch.ipynb) | Original executed Phase A: frozen-sample verification (100 songs, 60/20/20 split, 858 recordings, 3,432 files, 2.51 h audio, 9 languages, 20 singers, 4 vocal ranges), descriptive EDA tables/charts, and Basic Pitch input preparation. No BP inference. Do not rerun (see below). |
| Background | [BP tuning notebook](../GTSinger_100_BasicPitch_Tuning.ipynb) | Original executed BP experiment: what was tried and why 0.99 was selected. Do not rerun. |
| 1: everyone | [Shared_Correction_Label_Audit.ipynb](Shared_Correction_Label_Audit.ipynb) | Inspect automatic label proposals, listen to examples, compare predicted/reference notes, and agree on shared labels. |
| 2: Queenie | [queenie_merge_starter.ipynb](queenie_merge_starter.ipynb) | Future merge/keep model. Currently blocked by zero candidate pairs. |
| 2: Felicia | [felicia_false_starter.ipynb](felicia_false_starter.ipynb) | Future extra-note deletion model; retain notes that can be repaired. |
| 2: Le Xin | [le_xin_timing_starter.ipynb](le_xin_timing_starter.ipynb) | Future model correcting note start/end times. |
| 2: Hann Qian | [hann_qian_pitch_starter.ipynb](hann_qian_pitch_starter.ipynb) | Pitch v1 (XGBoost, up/keep/down one semitone). Left out of the final pipeline: no gain. Scripts, results and how to run: [hann_qian_pitch_model/README.md](hann_qian_pitch_model/README.md). Its pyin tracks (`f0_tracks.npz`) are still used by the final pipeline. |

The person notebooks load data, inspect proposed targets, calculate a no-change baseline and demonstrate export format. They contain no fitted correction models. Training them is the next development phase AFTER shared label review.

## Initial-stage blockers — before correction fitting
1. **Automatic labels are proposals, not human ground truth.** Of 7,623 training predictions, 3,893 have provisional labels and 3,730 remain unresolved under v2 (v1: 1,142 and 6,481). The v2 rule adds 2,751 matches automatically and has not been listened to. An unmatched prediction might need a timing/pitch repair rather than deletion. Blank means unknown, NEVER zero. One note can need both timing and pitch correction.
2. **Queenie has zero merge candidates.** The current rule considers neighbouring same-pitch notes with a gap from -0.02 to 0.12 seconds. Zero candidates under this rule does not prove there are no merge errors. Agree on whether/how to broaden the rule and version the resulting candidates/labels before supervised merge training.
3. **No combined correction system exists yet.** Evaluate each component before combining; agree how deletion, merging, timing and pitch decisions interact. Missing-note insertion and splitting a long prediction are outside the four initial tasks.
4. **Final test is locked.** The dataset includes test files for completeness, NOT permission to inspect test answers for development or run test inference/evaluation. Separate approval is required. Validation was already used for BP selection and is not fresh independent evidence.

## What to do together first
1. Extract the ENTIRE ZIP and preserve its folder structure. Do not distribute isolated notebook files.
2. Read the shared notebook's saved outputs. Review the displayed audio and note overlays together.
3. Record decisions, reviewer and reasons in `basic_pitch_runs/correction_handover/review_queue.csv`; use the merge review queue when a valid candidate set exists.
4. Agree on the correspondence policy, usable training examples and handling of unresolved cases. You need a reviewed usable set, not invented answers for every note.
5. Save actual reviewed targets in a NEW file such as `train_targets_reviewed_v1.csv`, with a version and reviewers. The shared notebook rebuilds `train_targets.csv` as provisional proposals; NEVER store human-curated targets there. A decision queue alone is not yet a machine-ready corrected target table.
6. After real agreement, record the approved label version and change each starter to load that reviewed file. The starters currently point at the proposed v2 file for exploration only; that does not mean v2 is approved. The approval flag currently remains false; running notebooks does not constitute approval. Suggested check before choosing: train on v1 and on v2 and compare on the same held-out folds.
7. Develop each model using the fixed song-grouped `folds.csv`. Fit preprocessing and tune only inside training folds. Never put reference/target columns into model inputs.

## Folder map — everything else supports those notebooks
- `data/`: complete frozen 100-song sample, organized into train/val/test. Includes audio and original JSON, MusicXML and TextGrid annotations. No duplicate source download folder or reserves.
- [eda/GTSinger_100_EDA_and_BasicPitch.ipynb](../../eda/GTSinger_100_EDA_and_BasicPitch.ipynb): the Phase A EDA notebook; its discovery checks the working folder and its parent for `selected_100.csv`.
- `notebooks/`: the six initial handover/BP notebooks plus required Python helpers at this stage. The later repository contains nine notebooks under `notebooks/`, plus the EDA notebook under `eda/`. Helpers are dependencies, not extra tasks to read.
- `basic_pitch_runs/correction_handover/`: shared prediction inputs, training proposals, review queues, folds and baseline outputs. No separate tuning-candidate run folders.
- `selected_100.csv`, `selected_recordings.csv`, `selected_files.csv`, `split_manifest.csv`: fixed sample/recording/file/split manifests. Do not edit. Use `export_path` / recording paths to locate packaged data; source `local_path` entries are original download provenance and are intentionally not duplicated.
- `hf_source_manifest.json`, `SAMPLE_FROZEN_DO_NOT_RESAMPLE.txt`, `package_integrity.json`: source provenance, freeze notice and byte-integrity checks.
- `requirements-lock.txt`: Python dependencies.

### Shared tables in plain English
| File | Meaning |
|---|---|
| `train_features.csv` | What the model may observe for each predicted training note. |
| `train_targets.csv` | Automatically proposed training answers (v1); review required. |
| `proposed_v2/train_targets_v2.csv` | Proposed v2 answers (mutual-best overlap, more matches) with a `match_rule` column; loaded by the starters; not approved. |
| `val_features.csv` | Validation inputs, without correction targets. |
| `merge_candidates.csv`, `merge_targets.csv` | Queenie's proposed note pairs and labels; currently empty. |
| `selected_predictions_train.json`, `selected_predictions_val.json` | Frozen 0.99 prediction events used by all tasks. |
| `references_train.csv` | Complete reference notes for evaluation, NOT model inputs. |
| `training_recording_manifest.csv` | All training recordings for scoring, including empty predictions. |
| `folds.csv` | Fixed training folds: excerpts of the same song stay together. |
| `input_version.json`, `label_version.json`, `baseline_config.json` | Identify predictions, original provisional-label snapshot and final BP settings. Human-reviewed labels need their own version. `proposed_v2/label_version_v2.json` identifies the v2 proposal. |

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for detailed columns. Some historic audit-only artifacts named there are intentionally omitted from this minimal package; they are not required to use the notebooks.

## Rules, current example counts and evaluation metrics

These are **automatic rules for proposing TRAINING labels**, not rules learned by a correction model. All current labels are provisional and require group review. A reference note is a note from the dataset's original annotation. Only Queenie works with pairs of predicted notes; the others work with individual notes and proposed reference matches.

### Each person's rules, available examples and task baseline

| Person / task | Current examples available | Current automatic rule | Task-specific metrics | No-change comparison |
|---|---|---|---|---|
| **Queenie: merge / keep separate** | **0 candidate pairs; 0 labelled pairs. Currently blocked.** | Consider immediately neighbouring predictions in the same recording, with identical pitch and gap **-0.02 to 0.12 s**. Propose merge only for accepted fragments of the same reference note (details below). Propose keep when each has an accepted one-to-one match to a different reference note. Otherwise unknown. | Merge precision, recall and F1; confusion matrix. Higher precision/recall/F1 is better. Currently unavailable because there are no pairs. | Never merge any pair. |
| **Felicia: remove extra notes** | **3,893 provisionally labelled notes: 196 extra/delete, 3,697 retain** (v1: 1,142 = 196 + 946). Another **3,730** (v1: 6,481) have no usable false-note label. | Propose delete only when the prediction overlaps no reference and is **more than 0.20 s away from every reference**. Propose retain for an accepted one-to-one match. Otherwise unknown; unmatched does NOT automatically mean extra. | Extra-note detection precision, recall and F1; confusion matrix. Higher is better. Precision matters because false positives delete real notes. | Keep every note; make no deletions. |
| **Le Xin: correct timing** | **3,697 prediction-reference matches** (v1: 946). Among these, **1,876** (v1: 562) have wrong-start flags and **1,033** (v1: 265) have wrong-end flags; these groups can overlap. | Use accepted one-to-one matches. Target = reference start/end; adjustment = reference minus predicted time. Wrong-start flag: error **>0.05 s**. Wrong-end flag: error **>max(0.05 s, 20% of reference duration)**. | Start and end MAE/RMSE (lower is better); median absolute error; fraction of starts within 50 ms and ends within tolerance (higher is better). | Leave BP start/end times unchanged on the same labelled examples. |
| **Hann Qian: correct pitch** | **3,697 prediction-reference matches: 292 wrong-pitch and 3,405 already-correct examples** (v1: 946 = 76 + 870). | Use accepted one-to-one matches. Target = reference MIDI pitch; adjustment = reference minus predicted pitch. Wrong-pitch flag: integer MIDI pitches differ. | Pitch MAE/RMSE in cents (and MAE in semitones), fraction within 50 cents, octave-error rate. Lower errors and higher within-tolerance fraction are better. | Leave BP pitches unchanged on the same labelled examples. |

Le Xin and Hann Qian use the **same 3,697 matched notes** (v1: 946), not separate sets. Already-correct examples are important: the models must learn when not to change a note. There are 7,623 training predictions overall; label coverage is not true error prevalence. These counts will need updating if the group creates a new reviewed label version.

### Shared rule: accepted one-to-one prediction/reference match

All conditions must pass. This correspondence rule determines eligible examples for Felicia, Le Xin and Hann Qian; it is deliberately different from strict transcription-F1 matching so repairable errors can receive targets.

| Check | Current requirement |
|---|---|
| Unique overlap (v1) | The prediction overlaps exactly one reference, and that reference overlaps exactly one prediction. Overlap must have positive duration. |
| Mutual-best overlap (proposed v2, replaces the row above) | The prediction is matched to the reference it overlaps most, if that reference also overlaps it most. New v2-only matches are dropped if a second reference covers > 30% of the prediction or pitch differs by > 2 semitones. All v1 matches are kept. |
| Sufficient overlap | Overlap covers at least **60% of the shorter note**. |
| Duration compatibility | The longer note is no more than **3 times** the shorter note's duration. |
| Start compatibility | Start times differ by no more than **0.20 s**. |
| Pitch compatibility | Pitches differ by no more than **12 semitones**. This permits incorrect-pitch notes to receive repair targets. |
| If a check fails | Leave the correspondence unresolved, rather than invent an answer. |

### Queenie's additional fragment-group checks

Passing the initial candidate-pair rule is NOT enough to label a pair merge. The fragment group must satisfy all of these:

| Check | Current requirement |
|---|---|
| Same reference | Multiple predictions overlap the same reference note; each prediction overlaps only that reference. |
| Pitch | Every fragment has the same pitch as the reference. |
| Coverage | Together, fragments cover at least **60% of reference duration**, without double-counting overlaps. |
| Outer boundaries | First fragment's start and last fragment's end are each within **0.20 s** of the reference boundaries. |
| Internal gaps | Consecutive fragments have gaps **-0.02 to 0.06 s**. This is stricter than the initial candidate filter. Negative gap means overlap. |

The labeller also leaves potential one-prediction-to-several-reference split cases unresolved: a split proposal requires more than one overlapping reference, each overlapping only that prediction, matching its pitch, and each covered by at least 60%. Splitting is outside the four initial correction tasks. Other difficult correspondences remain ambiguous. Blank labels mean **unknown**, never zero, keep or correct. Reference-derived information creates/reviews targets and scores outputs; it must never be supplied as model input features.

### Two levels of evaluation — not one combined F1

| Evaluation level | Question answered | Metrics |
|---|---|---|
| **Task-specific** | Is this person's model good at its particular correction? | The classification or timing/pitch metrics above, on reviewed examples in held-out song groups. |
| **Full transcription** | After applying this correction, is the resulting note list better? | The same **song-macro onset+pitch note F1** used for BP selection, plus **offset-aware note F1**, on complete corrected outputs and references. |

Felicia's extra-note classification F1 and Queenie's merge classification F1 are **not the same metric** as transcription note F1. Do not average the four people's task metrics into a combined score. Score the actual corrected transcription instead.

| BP transcription metric | Requirements for a matched note | Why report it? |
|---|---|---|
| **Onset+pitch note F1 (primary)** | One-to-one note matching: start within **50 ms**, pitch within **50 cents**. End time is not checked. | Shared measure of transcription quality, consistent with BP selection. |
| **Offset-aware note F1** | Same start/pitch checks, plus end within **max(50 ms, 20% of reference duration)**. | Essential for timing work: fixing only note endings may not change primary F1 at all. |

Within each song, matching counts are aggregated across its excerpts before calculating precision/recall/F1; then song scores are averaged. **Song-macro means averaging across songs**, not combining people's models. Higher F1 is better. MAE is average absolute error; RMSE penalizes large errors more strongly. One semitone is 100 cents. Classification precision asks whether proposed actions are correct; recall asks how many true action cases were found. Accuracy alone can hide class imbalance, especially with 3,405 already-correct versus 292 incorrect pitch examples (v1: 870 versus 76).

### Unchanged 0.99 BP baseline scores

| Dataset | Onset+pitch song-macro note F1 | Offset-aware song-macro note F1 |
|---|---:|---:|
| All training songs | **0.2628** | **0.1539** |
| Validation songs (original BP tuning result) | **0.2726** | **0.1602** |

These are dataset-specific reference scores, NOT universal targets for every fold. Compare on **the same songs, same references and same scoring rules**. Do not compare a correction model's training-fold score directly with the BP validation score 0.2726.

### Fair comparison procedure

1. Approve/version usable shared labels before fitting. Keep uncertain labels unknown.
2. Use the fixed song-grouped training folds. Train on the other songs and predict corrections for the held-out songs; fit scaling, feature selection and hyperparameter selection without using the held-out fold. Use inner training folds for model selection when reporting outer-fold estimates.
3. Measure task metrics against the no-change task baseline on the same held-out reviewed examples.
4. Apply each component to the full held-out prediction set and measure full-transcription metrics against unchanged BP predictions on those exact songs. Do not drop difficult/unlabelled notes, missing reference events or empty-output recordings from transcription scoring.
5. Compare **BP unchanged**, **BP + Queenie only**, **BP + Felicia only**, **BP + Le Xin only**, **BP + Hann Qian only**. Queenie's comparison waits for a usable reviewed candidate set and a fitted model.
6. Later, agree on conflict handling and compare **BP + combined corrections**. No component order or combined gain is proven yet.
7. Report both task and transcription results. A good task score does not guarantee better overall note F1. Separate improvements on originally wrong examples from damage to originally correct examples where useful.

Training songs were already inspected during BP development, so correction-fold results remain development estimates, not completely untouched evidence. Validation was used for BP selection; further use is continued development. At the initial handover stage, the final test remained separately locked. No correction-model improvement is claimed in this handover.

## What each person should return
- An executed notebook with model choices, input features, label version and results.
- A saved model or reproducible training code.
- A COMPLETE corrected note table with `note_id, recording_id, start, end, pitch` (seconds and MIDI pitch), not merely error flags or a list of edits.
- A separate action history: deleted notes, absorbed merge IDs, timing/pitch changes.
- Task metrics and full-transcription metrics against the no-change baseline, using held-out song groups. F1 higher is better; timing/pitch errors lower are better. Training scores alone are not evidence of generalization.

A corrected CSV REPLACES the original note list: omitted notes count as deleted. Keep unchanged notes too. A merge retains one source ID and logs the absorbed IDs separately. Score all recordings and all reference notes, not only the easy labelled subset.

## Running vs reading
Saved outputs can be read without installing dependencies. The original shared/person notebook environment used Python 3.10; `requirements-lock.txt` records that environment and declares the later pitch notebook’s XGBoost dependency. Installation of the complete historical environment is not verified for current runtimes. Running later person notebooks fits models; read their saved outputs unless a new experiment is explicitly intended. For read-only input verification, use the standard-library command in `SCIENTIFIC_GUIDE.md`. The historical environment setup, from the repository root (not this README’s folder), was:

```sh
python -m venv .venv
# Windows:
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Choose that environment as the notebook kernel in your notebook editor. Run notebooks from their own folder or this extracted root. This package contains the dataset needed for shared review; no Basic Pitch inference is needed to use the supplied predictions. Selected audio is also embedded in the saved shared notebook.

The EDA notebook ([EDA notebook](../../eda/GTSinger_100_EDA_and_BasicPitch.ipynb)) is read-only history: read its saved outputs. It predates the BP experiment, so its statements that Basic Pitch is not installed, its Python 3.14 environment and its three prepared run configs (onset 0.5/0.6/0.4) are superseded by the BP notebook and the fixed tuning_001 input. Do not rerun it in this package: it needs `validation_report.json` and the original source `local_path` copies, which are intentionally omitted, and it would write `eda_outputs/`, `reports/`, `basic_pitch_runs/run_00*` folders and `notebooks/basic_pitch_train_val.py`. Its `RUN_BASIC_PITCH` and `RUN_FINAL_TEST` flags stay False; test data appears only in integrity checks and descriptive counts.

The BP notebook has an explicit accidental-rerun guard. Original experiment code and outputs are retained. Its historical 0.95 comparisons are scientific history, NOT an alternative input package. All standalone 0.95 run folders and the retired experiment are excluded. Removing those comparisons from the original notebook would misrepresent how 0.99 was selected.

Full historical retuning is deliberately NOT a supported one-click workflow in this compact handover: caches, historical output trees, builder/test scripts and duplicate provenance reports are omitted. Request a separately authorized reproduction workspace if required. See the scientific guide for the later fitted experiments and recorded test run.

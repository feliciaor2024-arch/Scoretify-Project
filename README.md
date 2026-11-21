# Scoretify — singing to sheet music

Scoretify turns a solo singing recording into sheet music. It uses Basic Pitch to predict notes, applies five correction steps, and turns the corrected notes into a score. Users can view and play the result, then download PDF, MusicXML or MIDI files.

## Start here

- **Run the application:** follow the [Run Guide](RUN_GUIDE.md) for requirements, installation, startup and troubleshooting.
- **Review the research:** follow the [Scientific Guide](SCIENTIFIC_GUIDE.md) for the dataset, experiments, saved results and limits on reproducing the work.

## How the project developed

1. **Dataset sampling and EDA:** select and freeze the songs, check the files, and explore the data. EDA means exploratory data analysis.
2. **Basic Pitch tuning:** compare settings that turn the pretrained model's signals into notes. The neural network itself was not retrained.
3. **Correction-model handover:** prepare the predicted notes, initial input features and reference targets for the correction experiments.
4. **Correction experiments:** test ways to fix errors in the predicted notes.
5. **Final pipeline:** combine five correction steps: timing, splitting, deletion, start adjustment and end adjustment.
6. **Web application:** connect the final pipeline to recording, upload, score display and downloads.

This is the development sequence. The app does not repeat sampling, tuning or model training when a user uploads audio.

## Where to find the work

| Part | Files and explanation |
|---|---|
| Selected dataset | [Selected songs](selected_100.csv) and [train/validation/test split](split_manifest.csv) |
| Dataset checks and EDA | [EDA notebook](eda/GTSinger_100_EDA_and_BasicPitch.ipynb), with saved tables and charts |
| Basic Pitch tuning | [Tuning notebook](notebooks/GTSinger_100_BasicPitch_Tuning.ipynb), with saved comparisons and selected settings |
| Initial correction handover | [Handover README](notebooks/correction_handover/README.md) and [data dictionary](notebooks/correction_handover/DATA_DICTIONARY.md) |
| Correction experiments | [Experiment reading guide](SCIENTIFIC_GUIDE.md#read-the-stages-in-order) and [later model comparisons](final_pipeline/experiments/) |
| Final pipeline | [Overview and results](final_pipeline/README.md) and [step-by-step explanation](final_pipeline/pipeline.md) |
| Web application | [Frontend details and tests](frontend/README.md) and [backend API](backend/README.md) |

Read the saved notebook outputs first. Some historical notebooks need files or environments that are not included here; see the Scientific Guide before trying to rerun them.

### Why there are two correction-handover folders

After BP tuning, the selected predictions became the starting point for correction-model development:

- [basic_pitch_runs/correction_handover/](basic_pitch_runs/correction_handover/) holds the frozen predictions, initial features, reference targets and version records supplied to that work. Do not replace these inputs with corrected notes.
- [notebooks/correction_handover/](notebooks/correction_handover/) holds the review notebooks, correction experiments, helpers and their outputs. Later models extended or replaced some initial features.

The [original proposal](docs/archive/proposal-overview.md) is kept as historical context, not as a description of the final system.

## Dataset and results

The frozen GTSinger sample contains **100 songs and 858 audio excerpts**, split into **60 training, 20 validation and 20 test songs**. Songs do not overlap between splits, but singers can appear in more than one split.

The table uses **pitch + note-start F1**: a score from 0 to 1 that balances how many predicted notes are correct and how many reference notes were found. Higher is better. Scores are calculated per song and averaged equally across songs.

| Comparison | Data used | Before | After |
|---|---|---:|---:|
| Default BP → tuned BP | 20 validation songs | 0.1855 | 0.2726 |
| Tuned BP → final corrected notes | 20 test songs | 0.2397 | 0.4148 |

These rows describe different comparisons on different splits. Validation was used during development; it is not an independent final test. The test results come from the saved final run on 27 October 2026.

These are note-prediction scores, before rounding note times to create sheet music. See the [saved pipeline results](final_pipeline/README.md) for the full results, including scores after sheet-music conversion.

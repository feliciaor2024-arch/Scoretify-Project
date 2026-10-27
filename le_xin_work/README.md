# le_xin_work: Le Xin's correction work (GTSinger, IS424 G2)

Start here. For a plain-language explanation of the whole part (what was built, what was and wasn't used, why not 0.8),
read `notebooks/correction_handover/le_xin_correction_explained.md` (summary) or `le_xin_explainer.md` (step by step). For how each step and model of the final pipeline works, read `pipeline.md`. The **final system is `chain_v4/`**; everything in `experiments/` is how each piece of it was built.
The notebooks in `notebooks/correction_handover/` (timing, split, snap, combined) are the earlier, step-by-step versions.

## Folders

| Folder | What it is |
|---|---|
| `chain_v4/` | **Final correction chain.** One complete note table per stage, for training songs (out-of-fold) and the 20 val songs, plus an action history per stage, a README per stage and `val_score.py`. |
| `experiments/split_v2/` | Better "cut one note into two" picker (stage 1). |
| `experiments/delete_v2/` | Better "delete false notes" model (stage 2). Replaces Felicia's delete step, so the group must agree before using it. |
| `experiments/snap_v3/` | Better "move each start to a nearby audio moment" picker (stage 3). |
| `experiments/end_snap/` | Same idea for note ends (stage 4). |
| `experiments/relabel/` | Tried rebuilding training labels from corrected notes. No gain, not used. |
| `tools/` | Helpers: `shim.py` (stands in for librosa + mir_eval), `snap_bp_clues.py`, `bp_onset_features.py` (Basic Pitch clues). |
| `early/` | First scripts (`eda.py`, `models.py`, `check.py`, `stageA_results.csv`). Kept for the record. |
| `bp_activations/` | Basic Pitch's saved outputs per recording (`train/`, `val/`, about 850 MB). **Not in git**; rebuild with `tools/snap_bp_clues.py` if missing. |

Each experiment folder has its own README/RESULTS.md, its scripts, a `logs/` folder and a `cache/` folder (not in git).

## The chain, in order

Basic Pitch notes → timing fix (notebook) → **split v2** → **delete v2** → **snap v3** (starts) → **end snap** (ends)

Each stage is trained only on training songs, out-of-fold by song (`folds.csv`), using the output of the stage before it.
Thresholds and margins are chosen on training songs only. Val was run once, with nothing tuned on it.

| Stage | Train onset F1 | Train offset F1 | Val onset F1 | Val offset F1 |
|---|---|---|---|---|
| Basic Pitch unchanged | 0.2628 | 0.1539 | 0.2726 | 0.1602 |
| + timing | 0.3541 | 0.2176 | 0.3755 | 0.2360 |
| + split v2 | 0.3622 | 0.2342 | 0.3844 | 0.2584 |
| + delete v2 | 0.3810 | 0.2475 | 0.4050 | 0.2730 |
| + snap v3 | 0.4213 | 0.2772 | 0.4563 | 0.3159 |
| **+ end snap (final)** | **0.4213** | **0.2844** | **0.4563** | **0.3213** |

Score any val table again with `python le_xin_work/chain_v4/val_score.py`.

## Rules followed
- `data/test` is locked: never opened or scored.
- No reference/target columns as model inputs; the answer key is used only for labels and scoring.
- TextGrid is not used (it is effectively the answer key).
- Every output is a complete note table `note_id, recording_id, start, end, pitch`; source tables are never overwritten.

## Rerunning
Python 3.14 with numpy, pandas, scikit-learn, scipy, librosa, mir_eval and lightgbm 4.7.0.
Rebuilding `bp_activations/` also needs basic-pitch + onnxruntime. Each stage's README in `chain_v4/` gives the exact
script and the final-model recipe.

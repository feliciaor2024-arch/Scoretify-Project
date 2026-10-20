# Proposed label version 2 (more labelled notes)

**Status: proposed, automatic, NOT manually verified, NOT yet approved by the group.**
Original files (`train_targets.csv`, `correction_labels.py`) are untouched.

## What changed
Only the "Unique overlap" check of the shared matching rule (README table "Shared rule", first row;
`notebooks/correction_labels.py` line 21).

- **v1:** a predicted note is matched only if it touches exactly one real note and that real note touches only it. Any overlap, even 1 ms, breaks the match.
- **v2:** a predicted note is matched to the real note it overlaps **most**, if that real note also overlaps it **most** (mutual best).
- Checks 2-5 unchanged: overlap >= 60%, lengths within 3x, starts within 0.20 s, pitch within 12 semitones.
- Safety filter for newly added matches (in place of manual listening): dropped if a second real note covers > 30% of the predicted note (256 dropped) or the pitch differs by > 2 semitones (40 dropped).
- All 946 v1 labels are kept exactly. Spurious (196) and split (98) candidates are left as in v1.

## Result
| | v1 | v2 |
|---|---:|---:|
| Matched (one_to_one) notes | 946 | 3,697 |
| Wrong pitch | 76 | 292 |
| Wrong start | 562 | 1,876 |
| Wrong end | 265 | 1,033 |
| Still ambiguous | 6,383 | 3,632 |

Column `match_rule` shows `v1_original`, `v2_mutual_best`, or blank.

## How to use
`train_targets_v2.csv` has the same columns as `train_targets.csv` (plus `match_rule`) and the same 7,623 note IDs.
In your starter notebook, change the targets filename from `train_targets.csv` to `proposed_v2/train_targets_v2.csv`.

Suggested check: train on v1 and on v2, compare on the same held-out folds.

Report wording: "Labels were generated automatically with a mutual-best-overlap rule and a safety filter; they were not manually verified."

To regenerate: `python basic_pitch_runs/correction_handover/proposed_v2/correction_labels_v2.py` from the handover root.

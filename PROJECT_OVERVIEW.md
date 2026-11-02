# MusicCare: Singing-to-Sheet AI System (project overview)

Source: the project proposal PDF (`dma_project_.pdf`). This file summarises it for teammates and Claude Code. For the Le Xin timing task, see `CLAUDE.md` in the same folder.

## One-line summary
Turn a sung melody into standard, quantised sheet music (MusicXML/MIDI, with tempo, rhythm, rests and bars), then let the singer refine it through natural-language commands.

## Problem
- Writing down a sung melody needs years of ear-training or a trained transcriber. Most people who can sing an idea cannot notate it.
- Existing pitch tools (CREPE, Basic Pitch, CREPE Notes) detect pitch and note events but stop there. They do not produce readable sheet music, do not handle the messiness of human singing (vibrato, slides, off-pitch notes, uneven timing), and do not let a non-technical user edit the result.
- Gap we fill: an end-to-end system from raw vocal recording to editable sheet music, with an LLM-based editing layer.
- Motivation: a 2024 study of the top 100 US Spotify artists found 87% are self-taught and 92% have no music degree, so many creators depend on their voice to carry an idea.

## Planned pipeline
1. **Pitch detection (deep learning):** Basic Pitch as the primary baseline, CREPE for comparison.
2. **Note formation and post-processing (ML/rules):** group pitch into notes, fix errors, estimate tempo, quantise durations, add rests and bar divisions.
3. **Sheet-music generation:** export MIDI or MusicXML (music21), render with OpenSheetMusicDisplay.
4. **LLM editing layer:** translate requests such as "make the final note longer" into validated structured edits. The LLM works on the note data, not on raw audio.
5. **Human-singing validation:** singers of different ranges, skill levels, microphones and rooms.

## Datasets named in the proposal
| Dataset | What it is | Status |
|---|---|---|
| MIR-ST500 | 500 pop songs with vocal note labels (pitch, onset, offset) and YouTube links. Downloaded with `get_youtube.py` from github.com/york135/singing_transcription_ICASSP2021 | Proposed |
| Annotated-VocalSet | 10.1 h of monophonic recordings from 20 professional singers, all 5 vowels and extended techniques. zenodo.org/records/1442513 | Proposed |
| Children's Song Dataset (CSD) | 100 children's songs (Korean/English), aligned with MIDI and lyrics | Proposed |
| RWC-MDB-P | 100 popular songs with MIDI melody annotations | Dropped |
| Cmedia | 200 YouTube pop songs (mostly Chinese) | Dropped |
| jaCappella | 50 Japanese a cappella songs with stems and scores | Proposed |

**Note for the team:** the handover uses **GTSinger** (100-song sample, split by singer and song), not the datasets above. The proposal is the original plan; the handover is the concrete data we are working on now.

## Tools
- Basic Pitch (primary), CREPE (comparison)
- pandas, numpy, matplotlib, seaborn, librosa: data mining (pitch range, note durations, tempo, rests, pitch jumps, errors by range, differences between datasets)
- librosa: onset detection, beat tracking, tempo
- music21: MusicXML
- OpenSheetMusicDisplay: render MusicXML in the browser (JavaScript/TypeScript)

## Schedule (from the proposal)
| Week | Task |
|---|---|
| 4 | Topic finalised (MusicCare) |
| 5 | Introduction, problem, literature review, datasets, tools |
| 6 | Tools section; preliminary baseline; evaluation strategy; slides due 27 Sep 23:59 |
| 7 | Proposal presentation, 29 Sep |
| 8 | Full EDA; vocal separation for polyphonic songs; baseline on full evaluation subset |
| 9 | Post-processing: tempo, quantisation, rests, bars |
| 10 | Sheet-music generation (MIDI/MusicXML) |
| 11 | LLM editing layer (kept small because of the midterm); user testing and human-singing validation |
| 12 | Final presentation (slides due two days before slot) |
| 13 | (no task listed) |
| 14 | Final evaluation, report, code cleanup. Submit by 11:59 pm, 24 Nov |

## Where the correction work fits
The handover (GTSinger correction) sits between steps 1 and 2: it takes Basic Pitch's notes and fixes them before quantisation.
- Merge fragments (Queenie), remove false notes (Felicia), fix start/end timing (Le Xin), fix pitch (Hann Qian)
- Output: a corrected note table `note_id, recording_id, start, end, pitch`
- Status: labels are automatic and not yet group-approved. Timing (Le Xin): random forest on note + audio-only (loudness, pitch) features, trained on the provisional v2 labels (out-of-fold by song). Validation songs: onset F1 0.2726 → 0.3755, offset F1 0.1602 → 0.2360, all 20 songs improved. No TextGrid input. Results must be rerun once reviewed labels exist; test is still locked.

## Open questions
- Which dataset is the main evaluation set: GTSinger (current handover) or the proposal's list?
- Do we still plan to use CREPE, given Basic Pitch is the chosen baseline?
- The timeline has no task in week 13.

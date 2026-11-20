# Data-Mining-Project

**Scientific submission:** start with [SCIENTIFIC_GUIDE.md](SCIENTIFIC_GUIDE.md) for the frozen dataset, EDA, Basic Pitch decoder selection, correction handover, reproducibility limits and test chronology.

**New to the project? Start with [RUN_GUIDE.md](RUN_GUIDE.md)**: step-by-step setup and running the live demo.

## Run the Scoretify frontend

The frontend is in `frontend/`; the back end that runs the transcription model is in `backend/`. For the live demo, start the back end first (setup and run steps: `backend/README.md`), then the frontend below. Without the back end, `Generate score` reports that the server is not reachable; to show the old fixed sample instead, set `VITE_SCORE_PROVIDER=mock` before `npm run dev`.

### Prerequisites

- Node.js 22.12+ and npm (tested with Node.js 26.8.1).
- Google Chrome is recommended for this project's microphone demo and is required by the current browser-test configuration.

### First-time setup

Open a terminal in the cloned `Data-Mining-Project` repository, then run:

```bash
cd frontend
npm ci
npm run dev
```

Open the address printed by Vite, normally `http://127.0.0.1:5173`, in your browser. If that port is occupied, use the alternative address printed in the terminal.

Keep the terminal running while you use the app. Press **Ctrl+C** to stop it.

### Start again later

From the repository root:

```bash
cd frontend
npm run dev
```

Run `npm ci` again when the dependency lockfile changes or dependencies are missing.

### Try the demo

1. Select **Try sample**, or record/upload a clip up to **30 seconds** (uploads up to **20 MB**).
2. Select **Generate score**. The clip is sent to the local back end, which transcribes and corrects the notes (about 3-8 s for a 10 s clip).
3. Play the result as transcribed notes alone or mixed with your voice, follow the cursor on the score, zoom, and download the score as PDF, MusicXML or MIDI.

Allow microphone access when prompted if you want to record. Recording needs localhost or HTTPS and a supported browser. Audio is sent only to the back end you run locally.

### Checks and production build

Run these from `frontend/`:

```bash
npm test          # Unit and component tests
npm run test:e2e  # Browser tests using installed Chrome and a simulated microphone
npm run build    # TypeScript check and production files in dist/
```

See `frontend/README.md` for the file map, mock-data boundaries, and testing details, and `backend/README.md` for the API.

## Correction pipeline (v2, the actual run)

Basic Pitch (tuned) writes down the notes, then 5 correction steps (7 trained models) fix them, then a sheet is made:
timing → split → delete → start snap → end snap → sheet. This is `final_pipeline/`, it is what the demo runs, and it
raised test onset F1 from 0.240 to 0.415 (offset 0.134 to 0.281; all 20 test songs improve).

The 4 first (v1) models (timing, merge, pitch, delete) are in `notebooks/correction_handover/`. We kept timing
(improved), replaced delete, added split, start snap and end snap, and removed merge and pitch (no gain). The
step-by-step table is in `final_pipeline/README.md`.

Run it on one recording (Python 3.13, `final_pipeline/inference/requirements.txt`):

```bash
python final_pipeline/inference/pipeline.py song.wav --out results
```

This writes the corrected notes (CSV), a MIDI file and a MusicXML score. Re-check the reported val numbers with
`python final_pipeline/chain_v4/val_score.py`. More: `final_pipeline/README.md`, `final_pipeline/inference/README.md`.

## Project overview

MusicCare: Singing-to-Sheet AI System: An AI-driven system is designed to convert vocal performances into structured music sheets. It leverages data mining, machine learning (ML), deep learning (DL), and large language models (LLMs) to transcribe human singing into standard musical notation, assisting musicians, composers, and hobbyists in capturing their melodies easily.

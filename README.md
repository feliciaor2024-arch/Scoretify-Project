# Data-Mining-Project

## Run the Scoretify frontend

The frontend is in `frontend/`. It currently uses **mocked transcription results**; no backend, Python environment, model setup, or API key is required. Recording and upload previews work locally, but every demo generation displays the same labelled sample score—not a transcription of your audio.

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
2. Select **Generate demo score** to see the simulated processing stages.
3. Play the sample score to see note highlighting; try zoom and the PDF, MusicXML, and MIDI downloads.

Allow microphone access when prompted if you want to record. Recording needs localhost or HTTPS and a supported browser. Audio stays in the browser; no audio is uploaded to a backend. Downloads contain the fixed sample, not a transcription of your recording.

### Checks and production build

Run these from `frontend/`:

```bash
npm test          # Unit and component tests
npm run test:e2e  # Browser tests using installed Chrome and a simulated microphone
npm run build    # TypeScript check and production files in dist/
```

See `frontend/README.md` for the file map, mock-data boundaries, and testing details.

## Project overview

MusicCare: Singing-to-Sheet AI System: An AI-driven system is designed to convert vocal performances into structured music sheets. It leverages data mining, machine learning (ML), deep learning (DL), and large language models (LLMs) to transcribe human singing into standard musical notation, assisting musicians, composers, and hobbyists in capturing their melodies easily.

# Scoretify · Studio preview

An offline-friendly React + TypeScript + Vite music workspace. Warm ivory, ink/violet typography, coral recording controls and a real Verovio-engraved sample score. With the back end running (`../backend/README.md`), **Generate score** transcribes the input with the chain v4 model and shows the result. No noise suppression is applied.

## Run

Requires Node 22.12+ (verified with Node 26.8.1) and npm.

```sh
# From the Data-Mining-Project repository root:
cd frontend
npm ci
npm run dev
```

Open the localhost address printed by Vite (default `http://127.0.0.1:5173`). Microphone recording needs localhost or HTTPS, permission, and MediaRecorder support. Browser-supported audio formats vary; WAV or MP3 is a good starting point. Files are decoded locally with Web Audio and rejected when empty, not audio, undecodable, over 20 MB or over 30 seconds.

```sh
npm test                 # Vitest unit / component tests
npm run build            # TypeScript check + production bundle in dist/
npm run test:e2e         # Playwright against installed Google Chrome
npm run format           # Prettier
```

The browser suite launches its **own isolated Chrome profile** and a temporary server on port 5181. It does not use personal Chrome cookies. It uses a fake microphone / automatic fake permission, not hardware or a human singer. Screenshots go to `test-results/`. If Chrome is not installed, install it or change `channel` in `playwright.config.ts` and install the corresponding Playwright browser.

## File map

- `src/App.tsx` — workspace shell, help dialog, cancellable generation lifecycle and provenance.
- `src/components/AudioInput.tsx` — real browser recording, actual input-level meter, local audio validation/preview, drag/drop and sample selection. Tracks, intervals, stale permission requests and object URLs are cleaned up.
- `src/components/ScoreViewer.tsx` — genuine supplied SVG, zoom, sample audio playback, JSON-ID note highlighting, scrubber and real sample downloads.
- `src/lib/audio.ts` — size/type/decode/duration validation.
- `src/components/ResultViewer.tsx` — a transcribed score: MusicXML drawn by OpenSheetMusicDisplay (loaded on demand), a cursor that follows playback, a switch between the transcribed notes alone and mixed with the voice, and MusicXML/MIDI downloads.
- `src/lib/demo.ts` — typed `ScoreProvider` seam, result types and the abortable mock adapter.
- `src/lib/api.ts` — the live provider (`POST /api/transcribe`) and the default-provider choice.
- `src/lib/wav.ts` — decodes any browser-playable clip and re-encodes it as mono 16-bit WAV for the back end.
- `src/styles.css` — responsive Luminous recital visual system and reduced-motion support.
- `src/**/*.test.*` — lifecycle, validation, provenance and note-ID tests.
- `e2e/studio.spec.ts` — real Chrome sample workflow, local file decoding, download availability and fake-device recording checks.
- `public/demo/*` — **read-only synthetic fixtures** supplied separately: SVG, JSON note timings, WAV, PDF, MusicXML and MIDI.

## Mock boundary

“Generate demo score” waits through explicitly **simulated** stages. It always reveals the same fixed **Evening melody** fixture, regardless of the recording or uploaded input. Tempo/metre are informational, fixed at 120 BPM / 4/4. Synthetic sine-wave sample playback matches the eight note IDs and times in `sample.json`; original input playback is separate. There is no lyric recognition/alignment. The initial canvas is labelled **Demo preview**, not an input result.

Only trusted, bundled `/demo/sample.svg` is inlined. It is cropped in memory, preserving SVG definitions/references and removing duplicate/debug page furniture. User files never become SVG or HTML. Sample downloads are genuine static files, not downloads of the user's audio or a generated transcription.

The mock is used under Vitest and when `VITE_SCORE_PROVIDER=mock` (the Playwright suite sets this, so it never needs the back end). Otherwise the app uses `apiScoreProvider`, which converts the clip to WAV in the browser and sends it to `/api/transcribe`; `npm run dev` proxies `/api` to `http://127.0.0.1:8000`, and `VITE_API_URL` can point elsewhere. Labels switch from "Mock demo" to "Live model" accordingly.

## Manual acceptance still needed

Human hardware microphone permission/denial and audible quality, Safari/mobile hardware codecs, projector legibility in the intended room, and assistive-technology review. Fake-device success does not establish real singing transcription. No transcription is implemented.

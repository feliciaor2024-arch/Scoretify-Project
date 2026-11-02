# Scoretify · Studio preview

An offline-friendly React + TypeScript + Vite music workspace. Warm ivory, ink/violet typography, coral recording controls and a real Verovio-engraved sample score. **Frontend only. No backend, model, API, database, transcription or noise suppression is configured.**

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
- `src/lib/demo.ts` — typed `ScoreProvider` seam with an abortable mock adapter.
- `src/styles.css` — responsive Luminous recital visual system and reduced-motion support.
- `src/**/*.test.*` — lifecycle, validation, provenance and note-ID tests.
- `e2e/studio.spec.ts` — real Chrome sample workflow, local file decoding, download availability and fake-device recording checks.
- `public/demo/*` — **read-only synthetic fixtures** supplied separately: SVG, JSON note timings, WAV, PDF, MusicXML and MIDI.

## Mock boundary

“Generate demo score” waits through explicitly **simulated** stages. It always reveals the same fixed **Evening melody** fixture, regardless of the recording or uploaded input. Tempo/metre are informational, fixed at 120 BPM / 4/4. Synthetic sine-wave sample playback matches the eight note IDs and times in `sample.json`; original input playback is separate. There is no lyric recognition/alignment. The initial canvas is labelled **Demo preview**, not an input result.

Only trusted, bundled `/demo/sample.svg` is inlined. It is cropped in memory, preserving SVG definitions/references and removing duplicate/debug page furniture. User files never become SVG or HTML. Sample downloads are genuine static files, not downloads of the user's audio or a generated transcription.

The `ScoreProvider` interface is an eventual integration seam only: no live-provider implementation or API configuration exists. The app requests local static fixture files; it makes no external requests and sends no input audio anywhere.

## Manual acceptance still needed

Human hardware microphone permission/denial and audible quality, Safari/mobile hardware codecs, projector legibility in the intended room, and assistive-technology review. Fake-device success does not establish real singing transcription. No transcription is implemented.

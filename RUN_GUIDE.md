# Run guide: Scoretify live demo

This guide takes you from a fresh computer to the working demo: sing or upload a short melody, and the app turns it
into sheet music using our trained model. No prior knowledge of the project is needed. It takes about 20-30 minutes
the first time, mostly downloads.

```
 Browser (frontend/)                            Your computer (backend/)
 ┌─────────────────────────────┐   audio clip   ┌──────────────────────────────────────┐
 │ record or upload a clip     │ ─────────────► │ Basic Pitch finds the notes          │
 │ see the score, play it,     │                │ 5 correction steps fix the notes     │
 │ download PDF/MusicXML/MIDI  │ ◄───────────── │ the result is written as sheet music │
 └─────────────────────────────┘  score + audio └──────────────────────────────────────┘
   http://127.0.0.1:5173                           http://127.0.0.1:8000
```

You will run **two programs at the same time**, each in its own terminal window: the back end (Python, runs the
model) and the front end (the web page).

The steps below are for **Windows**, in **PowerShell**. On macOS or Linux, see [the note at the end](#macos-and-linux).

---

## 1. Install the tools (once per computer)

| Tool | Version | Where to get it | Check it worked (in a new PowerShell window) |
|---|---|---|---|
| Git | any recent | https://git-scm.com/downloads | `git --version` |
| Python | **3.13** exactly | https://www.python.org/downloads/ (pick a 3.13.x release) | `py -3.13 --version` |
| Node.js | 22.12 or newer (LTS) | https://nodejs.org | `node --version` |
| Google Chrome | any recent | https://www.google.com/chrome | (recommended browser for the demo) |

- **Python must be 3.13.** The saved models only load with the exact package versions we tested, and those were
  tested on 3.13. Another Python version you already have is fine; `py -3.13` picks the right one.
- In the Python installer, keep **"py launcher"** ticked.
- Open a **new** PowerShell window after installing, so it finds the new tools.

## 2. Get the code (once)

Pick a folder with a **short path**, such as `C:\Projects` or your `Downloads` folder, and avoid deep or synced
folders like OneDrive. Windows refuses file paths longer than 260 characters, and some Python package files sit
about 150 characters deep inside the project, so the project folder's own path must stay under roughly 100
characters.

Open PowerShell in that folder and run:

```powershell
git clone https://github.com/feliciaor2024-arch/Data-Mining-Project.git
cd Data-Mining-Project
```

The download is large (about 850 MB), because it includes the audio data set. That is expected.

**Every command from here on is run from this `Data-Mining-Project` folder** unless a step says otherwise.

## 3. Set up the back end (once)

```powershell
py -3.13 -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install -r backend/requirements.txt
venv\Scripts\python -m pip install --no-deps basic-pitch==0.4.0
```

What this does: it creates a private Python environment in a folder called `venv` (so nothing clashes with other
Python projects), then installs the packages the model needs, plus the web server.

- The installs take a few minutes.
- **The last line must have `--no-deps`.** Without it, pip tries to install TensorFlow, which fails on Python 3.13
  (and is not needed: we use Basic Pitch's lighter ONNX model).
- At the end, pip may print a red **"dependency conflicts"** message about `tensorflow` and `resampy`. **This is
  expected and safe to ignore.**

Check that it worked:

```powershell
venv\Scripts\python -c "import basic_pitch, sklearn, lightgbm, fastapi; print('back end OK')"
```

It should print `back end OK` (warnings about TensorFlow, CoreML or TFLite before it are normal).

## 4. Set up the front end (once)

```powershell
cd frontend
npm ci
cd ..
```

This installs the web page's packages into `frontend/node_modules`. Warnings starting with `npm warn` (for example
about install scripts or deprecated packages) are normal.

---

## 5. Start the demo (every time)

### Terminal 1: the back end

```powershell
venv\Scripts\python -m uvicorn app:app --app-dir backend --port 8000
```

Wait for `Application startup complete.` Leave this window open: closing it stops the server.

The server then warms up for about 20 seconds in the background (the first transcription after starting is the
slowest, so it does one in advance). To check it is ready, open http://127.0.0.1:8000/api/health in your browser. It
should show:

```json
{"models_loaded":true,"warmed_up":true}
```

The very first start on a new computer can take a few minutes longer, while the audio libraries prepare themselves.

### Terminal 2: the front end

Open a **second** PowerShell window in the `Data-Mining-Project` folder and run:

```powershell
cd frontend
npm run dev
```

It prints an address, normally **http://127.0.0.1:5173**. Open it in Chrome. The badge at the top of the page
should say **Live model**.

## 6. Try it

1. **Upload a clip:** click **Upload**, then **Choose audio file**, and pick this clip from the project's data set
   (11 seconds of singing):
   `data\train\English\EN-Alto-1\Breathy\someday or oneday\Breathy_Group\0000.wav`
2. Optionally type a **Song name** (for example *Someday*). It becomes the title on the score and in the
   downloads, and you can change it later without generating again. Leave it empty for "Your melody".
3. Click **Generate score**. After a few seconds you should see **"Score ready · 10 notes"** and the sheet music on
   the right.
4. **Listen:** press play. Use the switch above the play button to hear **Transcribed notes** only, or **Notes + your
   voice**. The highlighted cursor on the score follows the music.
5. **Download:** **PDF** (printable score), **MusicXML** (opens in MuseScore and other notation apps) or **MIDI**.
6. **Record yourself:** click **Record**, then **Start recording**, and allow the microphone when Chrome asks. Sing
   or hum a simple tune for 5-10 seconds, click **Stop recording**, then **Generate score**.

Limits: clips up to **30 seconds** and **20 MB**, **one voice**, no background music. The tempo is estimated from
your audio (it can come out double or half for slow songs), and the time signature is always 4/4. Lyrics are not
transcribed.

## 7. Stop, and run again later

- To stop: press **Ctrl+C** in each terminal window (or close them).
- To run again later, only **step 5** is needed: start the back end in one terminal and the front end in another.
- If a teammate pushes updates: run `git pull`, then repeat steps 3 and 4 only if `backend/requirements.txt` or
  `frontend/package-lock.json` changed.

---

## Troubleshooting

| What you see | What to do |
|---|---|
| `py` is not recognised, or `py -3.13` says no such Python | Install Python 3.13 (step 1) with the py launcher, then open a new PowerShell window. |
| pip: *Could not install packages due to an OSError: No such file or directory* (mentions **Long Path support**) | The project folder's path is too long (see step 2). Delete the `venv` folder, move the project to a shorter path such as `C:\Projects\Data-Mining-Project`, and repeat step 3. Or ask an administrator to [enable long paths](https://pip.pypa.io/warnings/enable-long-paths). |
| pip: *No matching distribution found for tensorflow* | You installed basic-pitch without `--no-deps`, or used `final_pipeline/inference/requirements.txt`. Run the step 3 commands exactly as shown. |
| pip warns about tensorflow / resampy conflicts | Expected; ignore it (see step 3). |
| *ModuleNotFoundError: No module named 'pkg_resources'* | Wrong resampy version. Run `venv\Scripts\python -m pip install resampy==0.4.3`. |
| Back end: *address already in use* / port 8000 | Another program uses port 8000, possibly an earlier back end that is still running. Close it, or find it with `netstat -ano \| findstr :8000`. The front end expects port 8000. |
| Page says **"The transcription server is not reachable"** | The back end is not running. Start it (step 5, terminal 1) and check http://127.0.0.1:8000/api/health. |
| The page says **Mock demo** instead of **Live model** | The variable `VITE_SCORE_PROVIDER` is set to `mock` in that terminal. Open a new terminal and run `npm run dev` again. |
| First **Generate score** is slow | The back end is still warming up. Wait until `/api/health` shows `"warmed_up":true`. |
| **"No sung notes were found"** | Sing louder or closer to the microphone, with a clear melody and no background music. |
| Microphone does not work | Use the `http://127.0.0.1:5173` address (the browser only allows microphones on localhost or HTTPS), and allow microphone access in Chrome's address bar. |
| Vite says port 5173 is in use | It picks another port automatically. Open the address it prints. |
| `npm ci` fails | Check `node --version` is 22.12 or newer. Run `npm ci` from inside the `frontend` folder. |

When the back end hits an unexpected error, the page says *"Check the back-end log"*. The details (a Python
traceback) are in the terminal 1 window.

## Optional: run the automated checks

From the `frontend` folder (these do not need the back end):

```powershell
npm test            # unit and component tests
npm run build       # type check and production build
npm run test:e2e    # browser tests in Chrome, with a simulated microphone
npm run format      # tidy the code with Prettier
```

- The browser tests start their own Chrome profile and a temporary server on port 5181, so they do not touch
  your own Chrome or a running `npm run dev`. If Chrome is not installed, install it or change `channel` in
  `frontend/playwright.config.ts`.
- Both test suites use the **mock** (see below), so they never need the back end. They check the app's behaviour,
  not transcription quality or a real microphone.

To show the old fixed sample score without the back end (for example on a laptop without Python), start the front
end with `$env:VITE_SCORE_PROVIDER="mock"; npm run dev`. In this mock mode, **Generate demo score** always shows the
same built-in "Evening melody" sample, whatever you record or upload, and the badge says **Mock demo**.

## macOS and Linux

Not tested yet, but the steps should be the same with these changes:

- Use `python3.13` instead of `py -3.13`.
- Use `venv/bin/python` instead of `venv\Scripts\python`.
- Use forward slashes in paths, for example `data/train/English/...`.

## Where things are

| Folder / file | What it is | More detail |
|---|---|---|
| `frontend/` | The web page (React + TypeScript + Vite) | [Front-end code](#front-end-code) below |
| `backend/` | The web server that runs the model (FastAPI) | `backend/README.md` (API, how a request flows) |
| `final_pipeline/inference/` | The transcription model: Basic Pitch + 5 correction steps + sheet writer | `final_pipeline/inference/README.md` |
| `final_pipeline/inference/models/` | The 7 trained models | |
| `data/` | The GTSinger audio clips (train / val / test) | |
| `venv/`, `backend/runs/` | Created on your computer (Python environment, generated scores). Not in git. | |

### Front-end code

All in `frontend/src/`:

| File | What it does |
|---|---|
| `App.tsx` | The page: input panel, song name, Generate button, help dialog, live or mock mode |
| `components/AudioInput.tsx` | Recording (with the level meter), file upload and drag-and-drop, the sample clip |
| `components/ResultViewer.tsx` | A transcribed score: drawing, playback cursor, the playback switch, zoom, downloads |
| `components/ScoreViewer.tsx` | The fixed sample score shown before anything is generated |
| `lib/api.ts` | Sends the clip to the back end (`POST /api/transcribe`) and picks live or mock mode |
| `lib/wav.ts` | Converts any browser-playable clip to WAV before upload |
| `lib/pdf.ts` | Makes the PDF in the browser (titles in non-Latin scripts, such as Chinese, are drawn as an image) |
| `lib/title.ts` | The song name: default title, download file names, title inside the MusicXML |
| `lib/audio.ts`, `lib/demo.ts` | Upload checks (type, size, length); the mock provider and result types |
| `*.test.ts(x)`, `../e2e/` | Unit tests (Vitest) and browser tests (Playwright) |

`frontend/public/demo/` holds the built-in sample (audio, score and downloads). `npm run dev` passes `/api`
requests to the back end on port 8000; `VITE_API_URL` can point elsewhere.

import { useEffect, useRef, useState } from "react";
import {
  ArrowUpRight,
  Music2,
  RotateCcw,
  X,
  Check,
  LoaderCircle,
} from "lucide-react";
import type { ScoreProvider, ScoreResult } from "./lib/demo";
import { defaultProvider } from "./lib/api";
import { AudioInput, type AudioClip } from "./components/AudioInput";
import { ScoreViewer } from "./components/ScoreViewer";
import { ResultViewer } from "./components/ResultViewer";
import "./styles.css";
export default function App({
  provider = defaultProvider,
}: {
  provider?: ScoreProvider;
}) {
  const [clip, setClip] = useState<AudioClip | null>(null),
    [stage, setStage] = useState(""),
    [result, setResult] = useState<ScoreResult | null>(null),
    [error, setError] = useState(""),
    [resetKey, setResetKey] = useState(0),
    [help, setHelp] = useState(false);
  const live = !!provider.live,
    transcription = result?.kind === "transcription" ? result : null;
  const job = useRef<AbortController | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => () => job.current?.abort(), []);
  useEffect(() => {
    if (help) dialog.current?.showModal();
    else dialog.current?.close();
  }, [help]);
  const cancel = () => {
    job.current?.abort();
    job.current = null;
    setStage("");
  };
  const reset = () => {
    cancel();
    setClip(null);
    setResult(null);
    setError("");
    setResetKey((k) => k + 1);
  };
  const generate = async () => {
    if (!clip || job.current) return;
    const controller = new AbortController();
    job.current = controller;
    setResult(null);
    setError("");
    try {
      const value = await provider.generate({
        input: clip,
        signal: controller.signal,
        onStage: (s) => {
          if (job.current === controller) setStage(s);
        },
      });
      if (job.current === controller) {
        setResult(value);
        setStage("");
      }
    } catch (e) {
      if (!controller.signal.aborted)
        setError(
          e instanceof Error
            ? e.message
            : live
              ? "The score could not be generated. Try again."
              : "The demo could not be prepared. Try again.",
        );
    } finally {
      if (job.current === controller) {
        job.current = null;
        setStage("");
      }
    }
  };
  return (
    <>
      <header className="site-header">
        <a className="brand" href="#studio" aria-label="Scoretify studio">
          <span className="brand-mark">
            <Music2 size={25} />
          </span>
          Scoretify<span className="brand-dot">.</span>
        </a>
        <nav aria-label="Main navigation">
          <a className="active" href="#studio">
            Studio
          </a>
          <button onClick={() => setHelp(true)}>
            How it works <ArrowUpRight size={15} />
          </button>
        </nav>
        <span className="mock-badge">
          <span />
          {live ? "Live model" : "Mock demo"}
        </span>
      </header>
      <main id="studio">
        <section className="intro">
          <div>
            <p className="eyebrow">A LITTLE MELODY. A NEW BEGINNING.</p>
            <h1>
              Your melody, <em>on paper.</em>
            </h1>
            <p>
              A space for the tune in your head. Record a little, or bring your
              own audio.
            </p>
          </div>
          <div className="intro-aside">
            <span>THE STUDIO / 01</span>
            <p>
              One voice.
              <br />A world of possibility.
            </p>
          </div>
        </section>
        <div className="workspace">
          <aside className="input-panel">
            <div className="section-title">
              <span className="step">01</span>
              <h2>Start with a sound</h2>
              <button
                className="icon-button"
                aria-label="Reset studio"
                onClick={reset}
              >
                <RotateCcw size={17} />
              </button>
            </div>
            <p className="panel-lead">
              A solo hum or a sung melody.
              <br /> Keep it simple. Make it yours.
            </p>
            <AudioInput
              key={resetKey}
              disabled={!!stage}
              onChange={(value) => {
                cancel();
                setResult(null);
                setClip(value);
                setError("");
              }}
            />
            {live ? (
              <div className="fixture-settings">
                <span>
                  Score settings <span className="small-label">AUTO</span>
                </span>
                <div>
                  <strong>
                    ♩ = {transcription ? transcription.bpm : "auto"}{" "}
                    <small>BPM</small>
                  </strong>
                  <strong>
                    4/4 <small>METRE</small>
                  </strong>
                </div>
                <p>
                  The tempo is estimated from your audio. The metre is always
                  4/4.
                </p>
              </div>
            ) : (
              <div className="fixture-settings">
                <span>
                  Sample score settings{" "}
                  <span className="small-label">FIXED</span>
                </span>
                <div>
                  <strong>
                    ♩ = 120 <small>BPM</small>
                  </strong>
                  <strong>
                    4/4 <small>METRE</small>
                  </strong>
                </div>
                <p>
                  Informational only. Your input does not change this fixture.
                </p>
              </div>
            )}
            <button
              className="primary generate"
              onClick={generate}
              disabled={!clip || !!stage}
            >
              {stage ? (
                <LoaderCircle size={18} className="spin" />
              ) : (
                <Music2 size={19} />
              )}
              {live ? "Generate score" : "Generate demo score"}{" "}
              <ArrowUpRight size={18} />
            </button>
            {stage && (
              <div className="generation" role="status">
                <span className="eyebrow">
                  {live ? "TRANSCRIBING" : "SIMULATED WORKFLOW"}
                </span>
                <p>{stage}</p>
                <button className="text-button" onClick={cancel}>
                  {live ? "Cancel" : "Cancel preview"}
                </button>
              </div>
            )}
            {result && (
              <p className="success" role="status">
                <Check size={16} />
                {transcription
                  ? `Score ready · ${transcription.notes.length} notes`
                  : "Sample score ready"}
              </p>
            )}
            {error && (
              <p role="alert" className="error">
                {error}
              </p>
            )}
            <p className="privacy">
              {live
                ? "Audio is sent only to the transcription server you run."
                : "Stays on this device. No audio is uploaded."}
            </p>
          </aside>
          <section className="score-panel" aria-label="Score workspace">
            <div className="score-top">
              <div>
                <span className="step">02</span>
                <h2>Your score</h2>
              </div>
              <span className="preview-label">
                {transcription
                  ? "YOUR TRANSCRIPTION"
                  : result
                    ? "FIXED SAMPLE RESULT"
                    : "Demo preview"}
              </span>
            </div>
            <div className="provenance">
              <span className="tiny-dot" />
              <p>
                {transcription
                  ? "Transcribed from your recording by the correction model."
                  : "This preview uses a fixed sample, not a transcription of your recording."}
              </p>
            </div>
            {transcription ? (
              <ResultViewer key={transcription.id} result={transcription} />
            ) : (
              <ScoreViewer key={resetKey} />
            )}
          </section>
        </div>
        <footer>
          <span>Made for the melodies worth keeping.</span>
          <span>
            {live ? "LIVE DEMO" : "FRONTEND PREVIEW"}{" "}
            <span className="footer-dot">·</span>{" "}
            {live ? "MODEL CONNECTED" : "NO MODEL CONNECTED"}
          </span>
        </footer>
      </main>
      <dialog
        ref={dialog}
        onCancel={() => setHelp(false)}
        onClose={() => setHelp(false)}
        aria-labelledby="help-title"
      >
        <button
          className="icon-button modal-close"
          aria-label="Close how it works"
          onClick={() => setHelp(false)}
        >
          <X />
        </button>
        <p className="eyebrow">A NOTE ABOUT THIS STUDIO</p>
        {live ? (
          <>
            <h2 id="help-title">
              Your voice.
              <br />
              <em>Your score.</em>
            </h2>
            <ol>
              <li>
                <strong>Bring a melody.</strong> Record up to 30 seconds of solo
                singing, or choose a local audio file (20 MB max).
              </li>
              <li>
                <strong>Generate.</strong> Basic Pitch finds the notes, five
                trained correction steps fix their timing, split, remove and
                snap them, and the result is written as a score. A 10-second
                clip takes about 6–8 seconds.
              </li>
              <li>
                <strong>Listen & explore.</strong> Play the transcribed notes on
                their own or mixed with your voice, follow the cursor on the
                score, and download it as PDF, MusicXML or MIDI.
              </li>
            </ol>
            <p className="modal-note">
              Tempo is estimated from your audio and the metre is always 4/4.
              Lyrics are not transcribed. Microphone access requires localhost
              or HTTPS and browser permission.
            </p>
          </>
        ) : (
          <>
            <h2 id="help-title">
              A real workspace.
              <br />
              <em>A simulated score.</em>
            </h2>
            <ol>
              <li>
                <strong>Bring a melody.</strong> Record up to 30 seconds, choose
                a local audio file (20 MB max), or try our synthetic sample.
              </li>
              <li>
                <strong>Explore the flow.</strong> “Generate demo score” walks
                through labelled simulated stages. No model runs, no app-level
                noise suppression is performed, and no audio leaves this device.
              </li>
              <li>
                <strong>Listen & explore.</strong> The fixed Evening melody
                fixture always plays at 120 BPM in 4/4. Listen with note
                highlighting, zoom the notation, or download its sample files.
              </li>
            </ol>
            <p className="modal-note">
              This is not transcription. Lyrics are not included. Microphone
              access requires localhost or HTTPS and browser permission.
            </p>
          </>
        )}
        <button className="primary" onClick={() => setHelp(false)}>
          Back to the studio
        </button>
      </dialog>
    </>
  );
}

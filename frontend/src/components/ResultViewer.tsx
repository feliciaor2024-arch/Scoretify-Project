import { useEffect, useRef, useState } from "react";
import type { OpenSheetMusicDisplay } from "opensheetmusicdisplay";
import {
  Play,
  Pause,
  RotateCcw,
  Minus,
  Plus,
  Download,
  Music2,
} from "lucide-react";
import type { TranscriptionResult } from "../lib/demo";
import { scorePdf } from "../lib/pdf";
type Mode = "synth" | "mix";
/** A transcribed score from the back end: MusicXML drawn by OpenSheetMusicDisplay, a cursor that follows playback,
 * and a switch between the transcribed notes alone and the notes mixed with the singer. */
export function ResultViewer({ result }: { result: TranscriptionResult }) {
  const [status, setStatus] = useState<"loading" | "ready" | "error">(
      "loading",
    ),
    [zoom, setZoom] = useState(100),
    [time, setTime] = useState(0),
    [playing, setPlaying] = useState(false),
    [mode, setMode] = useState<Mode>("synth"),
    [error, setError] = useState(""),
    [attempt, setAttempt] = useState(0),
    [makingPdf, setMakingPdf] = useState(false);
  const host = useRef<HTMLDivElement>(null),
    osmd = useRef<OpenSheetMusicDisplay | null>(null),
    steps = useRef<number[]>([]),
    step = useRef(0),
    audio = useRef<HTMLAudioElement>(null),
    resume = useRef<{ time: number; play: boolean } | null>(null);
  // The sheet's beat 0 is at 0 s of the recording (le_xin_work/inference/pipeline.py, sheet_notes).
  const secondsPerWhole = (4 * 60) / result.bpm;
  useEffect(() => {
    const element = audio.current;
    return () => element?.pause();
  }, []);
  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    Promise.all([
      import("opensheetmusicdisplay"),
      fetch(result.files.musicxml).then((r) => {
        if (!r.ok) throw Error();
        return r.text();
      }),
    ])
      .then(async ([{ OpenSheetMusicDisplay }, xml]) => {
        if (cancelled || !host.current) return;
        const sheet = new OpenSheetMusicDisplay(host.current, {
          autoResize: true,
          backend: "svg",
          drawTitle: false,
          drawComposer: false,
          drawPartNames: false,
          followCursor: true,
          cursorsOptions: [
            { type: 0, color: "#b85040", alpha: 0.3, follow: true },
          ],
        });
        await sheet.load(xml);
        if (cancelled) return;
        sheet.zoom = zoom / 100;
        sheet.render();
        const cursor = sheet.cursor;
        cursor.show();
        cursor.reset();
        const times: number[] = [];
        while (!cursor.Iterator.EndReached) {
          times.push(
            cursor.Iterator.currentTimeStamp.RealValue * secondsPerWhole,
          );
          cursor.next();
        }
        cursor.reset();
        steps.current = times;
        step.current = 0;
        osmd.current = sheet;
        setStatus("ready");
      })
      .catch(() => {
        if (!cancelled) setStatus("error");
      });
    return () => {
      cancelled = true;
      osmd.current?.clear();
      osmd.current = null;
    };
    // Not on zoom: the effect below re-renders in place, so the cursor keeps its position.
  }, [result.files.musicxml, secondsPerWhole, attempt]);
  useEffect(() => {
    const sheet = osmd.current;
    if (!sheet || status !== "ready") return;
    sheet.zoom = zoom / 100;
    sheet.render();
    sheet.cursor.update();
  }, [zoom, status]);
  useEffect(() => {
    const sheet = osmd.current;
    if (!sheet || status !== "ready") return;
    const times = steps.current;
    let target = 0;
    while (target + 1 < times.length && times[target + 1] <= time) target++;
    if (target < step.current) {
      sheet.cursor.reset();
      step.current = 0;
    }
    for (; step.current < target; step.current++) sheet.cursor.next();
  }, [time, status]);
  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    const tick = () => {
      if (audio.current) setTime(audio.current.currentTime);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing]);
  const toggle = async () => {
    if (!audio.current) return;
    if (playing) {
      audio.current.pause();
    } else {
      try {
        await audio.current.play();
      } catch {
        setError("Audio playback was blocked. Please press play again.");
      }
    }
  };
  const replay = () => {
    if (audio.current) {
      audio.current.pause();
      audio.current.currentTime = 0;
    }
    setTime(0);
    setPlaying(false);
  };
  const downloadPdf = async () => {
    setMakingPdf(true);
    setError("");
    try {
      const xml = await fetch(result.files.musicxml).then((r) => {
        if (!r.ok) throw Error();
        return r.text();
      });
      const url = URL.createObjectURL(await scorePdf(xml, "Your melody"));
      const link = document.createElement("a");
      link.href = url;
      link.download = `scoretify-${result.id}.pdf`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch {
      setError("The PDF could not be made. Please try again.");
    } finally {
      setMakingPdf(false);
    }
  };
  const switchMode = (next: Mode) => {
    if (next === mode) return;
    const element = audio.current;
    resume.current = {
      time: element?.currentTime ?? 0,
      play: !!element && !element.paused,
    };
    setMode(next);
  };
  return (
    <>
      <div className="score-toolbar">
        <span>
          <Music2 size={15} /> VOICE · TREBLE
        </span>
        <div className="zoom">
          <button
            className="icon-button"
            aria-label="Zoom out"
            disabled={zoom === 75}
            onClick={() => setZoom((z) => Math.max(75, z - 25))}
          >
            <Minus size={16} />
          </button>
          <output aria-label="Score zoom">{zoom}%</output>
          <button
            className="icon-button"
            aria-label="Zoom in"
            disabled={zoom === 200}
            onClick={() => setZoom((z) => Math.min(200, z + 25))}
          >
            <Plus size={16} />
          </button>
        </div>
      </div>
      <div className="paper-wrap">
        <article className="score-paper">
          <div className="paper-header">
            <span className="eyebrow">SCORETIFY / YOUR TRANSCRIPTION</span>
            <span className="paper-ornament" aria-hidden="true">
              ♮
            </span>
            <h3>Your melody</h3>
            <p>Transcribed from your recording</p>
            <div className="score-meta">
              <span>
                Estimated tempo <span className="tempo">♩ = {result.bpm}</span>
              </span>
              <span>
                {result.notes.length} notes · {result.seconds.toFixed(1)} s to
                transcribe
              </span>
            </div>
          </div>
          {status === "error" ? (
            <div className="score-error" role="alert">
              <p>The score could not be drawn. Please try again.</p>
              <button onClick={() => setAttempt((n) => n + 1)}>
                Retry score
              </button>
            </div>
          ) : (
            status === "loading" && (
              <p className="score-loading" role="status">
                Engraving your score…
              </p>
            )
          )}
          <div
            ref={host}
            className="osmd-host"
            aria-label="Transcribed notation"
            hidden={status !== "ready"}
          />
          <div className="paper-bottom">
            <span>4/4 · {result.bpm} BPM</span>
            <span>01</span>
          </div>
        </article>
      </div>
      <div className="tabs mode-switch" role="radiogroup" aria-label="Play">
        {(
          [
            ["synth", "Transcribed notes"],
            ["mix", "Notes + your voice"],
          ] as const
        ).map(([value, label]) => (
          <button
            key={value}
            role="radio"
            aria-checked={mode === value}
            onClick={() => switchMode(value)}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="transport">
        <button
          className="play-button"
          disabled={status !== "ready"}
          aria-label={playing ? "Pause transcription" : "Play transcription"}
          onClick={() => void toggle()}
        >
          {playing ? (
            <Pause size={18} fill="currentColor" />
          ) : (
            <Play size={18} fill="currentColor" />
          )}
        </button>
        <button
          className="icon-button"
          aria-label="Restart playback"
          onClick={replay}
        >
          <RotateCcw size={17} />
        </button>
        <div className="timeline">
          <label htmlFor="result-position">
            Playback{" "}
            <span>
              {time.toFixed(1)} / {result.duration.toFixed(1)} s
            </span>
          </label>
          <input
            id="result-position"
            type="range"
            min="0"
            max={result.duration}
            step="0.01"
            value={time}
            disabled={status !== "ready"}
            onChange={(e) => {
              const t = Number(e.target.value);
              if (audio.current) audio.current.currentTime = t;
              setTime(t);
            }}
          />
        </div>
        <span className="audio-tag">
          {mode === "mix" ? "NOTES + VOICE" : "SYNTHESIZED"}
        </span>
        <audio
          ref={audio}
          src={mode === "mix" ? result.files.mix : result.files.synth}
          preload="auto"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={replay}
          onLoadedMetadata={() => {
            const r = resume.current;
            if (!r || !audio.current) return;
            resume.current = null;
            audio.current.currentTime = r.time;
            if (r.play) void audio.current.play().catch(() => {});
          }}
        />
      </div>
      {error && (
        <p role="alert" className="error score-footnote">
          {error}
        </p>
      )}
      <div className="downloads">
        <div>
          <Download size={17} />
          <span>Downloads</span>
        </div>
        <div>
          <button
            disabled={makingPdf || status !== "ready"}
            onClick={() => void downloadPdf()}
          >
            {makingPdf ? "Making PDF…" : "PDF"}
            <Download size={13} />
          </button>
          {(
            [
              ["MusicXML", result.files.musicxml, "musicxml"],
              ["MIDI", result.files.midi, "mid"],
            ] as const
          ).map(([label, href, extension]) => (
            <a
              key={label}
              href={href}
              download={`scoretify-${result.id}.${extension}`}
            >
              {label}
              <Download size={13} />
            </a>
          ))}
        </div>
      </div>
      <p className="score-footnote">
        Notes found by Basic Pitch and corrected by our models, then rounded to
        a 32nd-note grid for the sheet. The tempo is estimated from your audio
        and the metre is always 4/4.
      </p>
    </>
  );
}

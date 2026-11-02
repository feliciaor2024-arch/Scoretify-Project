import { useEffect, useRef, useState } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  Minus,
  Plus,
  Download,
  Music2,
} from "lucide-react";
interface Fixture {
  title: string;
  duration: number;
  bpm: number;
  metre: string;
  notes: { id: string; start: number; end: number; pitch: number }[];
}
export function ScoreViewer() {
  const [svg, setSvg] = useState(""),
    [fixture, setFixture] = useState<Fixture | null>(null),
    [error, setError] = useState(""),
    [zoom, setZoom] = useState(100),
    [time, setTime] = useState(0),
    [playing, setPlaying] = useState(false),
    [attempt, setAttempt] = useState(0);
  const audio = useRef<HTMLAudioElement>(null),
    score = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const element = audio.current;
    return () => element?.pause();
  }, []);
  useEffect(() => {
    let cancelled = false;
    setError("");
    Promise.all([
      fetch("/demo/sample.svg").then((r) => {
        if (!r.ok) throw Error();
        return r.text();
      }),
      fetch("/demo/sample.json").then((r) => {
        if (!r.ok) throw Error();
        return r.json() as Promise<Fixture>;
      }),
    ])
      .then(([raw, data]) => {
        if (cancelled) return;
        const doc = new DOMParser().parseFromString(raw, "image/svg+xml");
        const root = doc.documentElement;
        root.setAttribute("viewBox", "0 38 480 92");
        root.setAttribute("height", "92");
        root.setAttribute("width", "480");
        const inner = root.querySelector("svg");
        inner?.setAttribute("width", "840");
        inner?.setAttribute("height", "1080");
        root
          .querySelectorAll(".pgHead,.pgFoot,.dir,.tempo")
          .forEach((n) => n.remove());
        root.setAttribute("role", "img");
        root.setAttribute(
          "aria-label",
          "Evening melody, synthetic sample notation",
        );
        setSvg(new XMLSerializer().serializeToString(root));
        setFixture(data);
      })
      .catch(() => {
        if (!cancelled)
          setError("The sample score could not be loaded. Please try again.");
      });
    return () => {
      cancelled = true;
    };
  }, [attempt]);
  useEffect(() => {
    const nodes = score.current?.querySelectorAll<SVGGElement>(".note");
    nodes?.forEach((n) => n.classList.remove("playing-note"));
    if (!fixture) return;
    const active = fixture.notes.find((n) => time >= n.start && time < n.end);
    if (active)
      score.current
        ?.querySelector(`[id="${active.id}"]`)
        ?.classList.add("playing-note");
  }, [time, svg, fixture]);
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
  return (
    <>
      <div className="score-toolbar">
        <span>
          <Music2 size={15} /> PIANO · TREBLE
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
            <span className="eyebrow">SCORETIFY / SAMPLE NO. 01</span>
            <span className="paper-ornament" aria-hidden="true">
              ♮
            </span>
            <h3>Evening melody</h3>
            <p>A small tune for a quiet moment</p>
            <div className="score-meta">
              <span>
                Moderato <span className="tempo">♩ = 120</span>
              </span>
              <span>Synthetic demonstration</span>
            </div>
          </div>
          {error ? (
            <div className="score-error" role="alert">
              <p>{error}</p>
              <button
                onClick={() => {
                  setError("");
                  setAttempt((n) => n + 1);
                }}
              >
                Retry sample
              </button>
            </div>
          ) : !svg ? (
            <p className="score-loading" role="status">
              Opening the sample score…
            </p>
          ) : (
            <div
              className="notation-scroll"
              tabIndex={0}
              aria-label="Scrollable sample notation"
            >
              <div
                ref={score}
                className="notation"
                style={{ width: `${zoom}%` }}
                dangerouslySetInnerHTML={{ __html: svg }}
              />
            </div>
          )}
          <div className="paper-bottom">
            <span>4/4 · C major · 120 BPM</span>
            <span>01</span>
          </div>
        </article>
      </div>
      <div className="transport">
        <button
          className="play-button"
          disabled={!fixture}
          aria-label={playing ? "Pause sample score" : "Play sample score"}
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
          aria-label="Restart sample playback"
          onClick={replay}
        >
          <RotateCcw size={17} />
        </button>
        <div className="timeline">
          <label htmlFor="score-position">
            Sample playback <span>{time.toFixed(1)} / 5.0 s</span>
          </label>
          <input
            id="score-position"
            type="range"
            min="0"
            max={fixture?.duration ?? 5}
            step="0.01"
            value={time}
            disabled={!fixture}
            onChange={(e) => {
              const t = Number(e.target.value);
              if (audio.current) audio.current.currentTime = t;
              setTime(t);
            }}
          />
        </div>
        <span className="audio-tag">SYNTHETIC AUDIO</span>
        <audio
          ref={audio}
          src="/demo/sample.wav"
          preload="metadata"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={replay}
        />
      </div>
      <div className="downloads">
        <div>
          <Download size={17} />
          <span>Sample downloads</span>
        </div>
        <div>
          {[
            ["PDF", "pdf"],
            ["MusicXML", "musicxml"],
            ["MIDI", "mid"],
          ].map(([label, extension]) => (
            <a
              key={label}
              href={`/demo/sample.${extension}`}
              download={`scoretify-sample.${extension}`}
            >
              {label}
              <Download size={13} />
            </a>
          ))}
        </div>
      </div>
      <p className="score-footnote">
        A fixed notation fixture, engraved with Verovio. Not a model-generated
        transcription.
      </p>
    </>
  );
}

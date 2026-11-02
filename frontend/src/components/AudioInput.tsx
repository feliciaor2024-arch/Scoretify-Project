import { useEffect, useRef, useState } from "react";
import { Mic, Square, Upload, Music2, RefreshCw } from "lucide-react";
import { inspectAudio } from "../lib/audio";
import type { AudioClip } from "../lib/audio";
export type { AudioClip } from "../lib/audio";
export function AudioInput({
  disabled,
  onChange,
}: {
  disabled: boolean;
  onChange: (clip: AudioClip | null) => void;
}) {
  const [tab, setTab] = useState<"record" | "upload">("record"),
    [clip, setClip] = useState<AudioClip | null>(null),
    [status, setStatus] = useState<
      "idle" | "requesting" | "recording" | "validating"
    >("idle"),
    [seconds, setSeconds] = useState(0),
    [error, setError] = useState(""),
    [level, setLevel] = useState(0),
    [drag, setDrag] = useState(false);
  const generation = useRef(0),
    recorder = useRef<MediaRecorder | null>(null),
    stream = useRef<MediaStream | null>(null),
    context = useRef<AudioContext | null>(null),
    url = useRef(""),
    ticker = useRef<ReturnType<typeof setInterval> | null>(null),
    animation = useRef(0),
    fileInput = useRef<HTMLInputElement>(null),
    preview = useRef<HTMLAudioElement>(null),
    started = useRef(0);
  useEffect(() => {
    const element = preview.current;
    return () => element?.pause();
  }, [clip]);
  const release = () => {
    if (ticker.current) clearInterval(ticker.current);
    ticker.current = null;
    cancelAnimationFrame(animation.current);
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    if (context.current) {
      void context.current.close().catch(() => {});
      context.current = null;
    }
    setLevel(0);
  };
  const discard = () => {
    preview.current?.pause();
    generation.current++;
    if (recorder.current && recorder.current.state !== "inactive") {
      recorder.current.onstop = null;
      recorder.current.stop();
    }
    recorder.current = null;
    release();
    if (url.current) URL.revokeObjectURL(url.current);
    url.current = "";
  };
  useEffect(
    () => () => {
      generation.current++;
      if (recorder.current && recorder.current.state !== "inactive") {
        recorder.current.onstop = null;
        recorder.current.stop();
      }
      if (ticker.current) clearInterval(ticker.current);
      cancelAnimationFrame(animation.current);
      stream.current?.getTracks().forEach((t) => t.stop());
      void context.current?.close().catch(() => {});
      if (url.current) URL.revokeObjectURL(url.current);
    },
    [],
  );
  const clear = () => {
    discard();
    setClip(null);
    setStatus("idle");
    setSeconds(0);
    setError("");
    onChange(null);
  };
  const commit = (value: AudioClip) => {
    setClip(value);
    setStatus("idle");
    onChange(value);
  };
  const stop = () => {
    if (recorder.current?.state === "recording") recorder.current.stop();
    release();
  };
  const record = async () => {
    clear();
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setError(
        "Recording is unavailable here. Use localhost or HTTPS in a supported browser, or upload audio.",
      );
      return;
    }
    const id = generation.current;
    setStatus("requesting");
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      if (id !== generation.current) {
        media.getTracks().forEach((t) => t.stop());
        return;
      }
      stream.current = media;
      const mime = [
        "audio/webm;codecs=opus",
        "audio/ogg;codecs=opus",
        "audio/mp4",
        "audio/webm",
      ].find((m) => MediaRecorder.isTypeSupported(m));
      const rec = new MediaRecorder(
        media,
        mime ? { mimeType: mime } : undefined,
      );
      recorder.current = rec;
      const chunks: Blob[] = [];
      rec.ondataavailable = (e) => {
        if (e.data.size) chunks.push(e.data);
      };
      rec.onstop = () => {
        if (id !== generation.current) return;
        release();
        const duration = Math.min(
          30,
          (performance.now() - started.current) / 1000,
        );
        const blob = new Blob(chunks, { type: rec.mimeType });
        if (!blob.size || duration < 0.1) {
          setError("No audio was captured. Please record again.");
          setStatus("idle");
          return;
        }
        url.current = URL.createObjectURL(blob);
        commit({
          name: "Your recording",
          url: url.current,
          duration,
          source: "recording",
        });
      };
      rec.onerror = () => {
        clear();
        setError("Recording stopped unexpectedly. Please try again.");
      };
      rec.start();
      started.current = performance.now();
      setStatus("recording");
      ticker.current = setInterval(() => {
        const elapsed = (performance.now() - started.current) / 1000;
        setSeconds(Math.min(30, elapsed));
        if (elapsed >= 30) stop();
      }, 100);
      try {
        const ctx = new AudioContext();
        context.current = ctx;
        await ctx.resume();
        if (
          id !== generation.current ||
          rec.state !== "recording" ||
          context.current !== ctx
        )
          return;
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        ctx.createMediaStreamSource(media).connect(analyser);
        const data = new Uint8Array(analyser.fftSize);
        const draw = () => {
          analyser.getByteTimeDomainData(data);
          let sum = 0;
          for (const n of data) sum += ((n - 128) / 128) ** 2;
          setLevel(Math.min(1, Math.sqrt(sum / data.length) * 5));
          animation.current = requestAnimationFrame(draw);
        };
        draw();
      } catch {
        /* Recording still works when live metering is unavailable. */
      }
    } catch (e) {
      if (id !== generation.current) return;
      release();
      setStatus("idle");
      setError(
        e instanceof DOMException && e.name === "NotAllowedError"
          ? "Microphone permission was denied. Allow access in your browser or upload an audio file."
          : "We could not access your microphone. Check your device, or upload audio instead.",
      );
    }
  };
  const upload = async (file?: File) => {
    if (!file || disabled) return;
    clear();
    const id = generation.current;
    setStatus("validating");
    try {
      const duration = await inspectAudio(file);
      if (id !== generation.current) return;
      url.current = URL.createObjectURL(file);
      commit({ name: file.name, url: url.current, duration, source: "upload" });
    } catch (e) {
      if (id !== generation.current) return;
      setStatus("idle");
      setError(e instanceof Error ? e.message : "Could not open this audio.");
    }
  };
  const sample = () => {
    clear();
    commit({
      name: "Evening melody · synthetic sample",
      url: "/demo/sample.wav",
      duration: 5,
      source: "sample",
    });
  };
  const busy = disabled || status === "validating";
  return (
    <>
      <div
        className="tabs"
        role="tablist"
        aria-label="Audio source"
        onKeyDown={(event) => {
          if (
            disabled ||
            !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)
          )
            return;
          event.preventDefault();
          const next =
            event.key === "Home"
              ? "record"
              : event.key === "End"
                ? "upload"
                : tab === "record"
                  ? "upload"
                  : "record";
          clear();
          setTab(next);
          document.getElementById(`${next}-tab`)?.focus();
        }}
      >
        <button
          id="record-tab"
          role="tab"
          aria-controls="audio-tab-panel"
          aria-selected={tab === "record"}
          tabIndex={tab === "record" ? 0 : -1}
          disabled={disabled}
          onClick={() => {
            clear();
            setTab("record");
          }}
        >
          <Mic size={16} />
          Record
        </button>
        <button
          id="upload-tab"
          role="tab"
          aria-controls="audio-tab-panel"
          aria-selected={tab === "upload"}
          tabIndex={tab === "upload" ? 0 : -1}
          disabled={disabled}
          onClick={() => {
            clear();
            setTab("upload");
          }}
        >
          <Upload size={16} />
          Upload
        </button>
      </div>
      <div
        id="audio-tab-panel"
        role="tabpanel"
        aria-labelledby={tab === "record" ? "record-tab" : "upload-tab"}
      >
        {clip ? (
          <div className="clip-preview">
            <div className="clip-heading">
              <Music2 size={19} />
              <strong>{clip.name}</strong>
            </div>
            <p>
              {clip.duration.toFixed(1)} sec ·{" "}
              {clip.source === "sample" ? "Synthetic sample" : "Original audio"}
            </p>
            <audio
              ref={preview}
              key={clip.url}
              controls
              src={clip.url}
              aria-label="Preview original input audio"
            />
            <button disabled={disabled} className="text-button" onClick={clear}>
              <RefreshCw size={14} />
              {tab === "record" ? "Record another take" : "Choose another file"}
            </button>
          </div>
        ) : tab === "record" ? (
          <div
            className={
              "record-space " + (status === "recording" ? "is-recording" : "")
            }
          >
            <div
              className="waveform"
              aria-label={
                status === "recording"
                  ? "Live microphone input level"
                  : "Microphone waiting"
              }
            >
              {Array.from({ length: 35 }, (_, i) => (
                <i
                  key={i}
                  style={{
                    height:
                      status === "recording"
                        ? `${4 + level * (12 + Math.sin(i * 1.7) ** 2 * 58)}px`
                        : `${3 + Math.sin(i * 1.2) ** 2 * 5}px`,
                  }}
                />
              ))}
            </div>
            <span className="record-timer">
              00:{Math.floor(seconds).toString().padStart(2, "0")}
              <small> / 00:30</small>
            </span>
            <p>
              {status === "recording"
                ? "Listening. Let your melody unfold."
                : status === "requesting"
                  ? "Allow microphone access in your browser."
                  : "A quiet moment. A single melody."}
            </p>
            {status === "recording" ? (
              <button className="record-button recording" onClick={stop}>
                <Square size={19} fill="currentColor" />
                Stop recording
              </button>
            ) : status === "requesting" ? (
              <button className="record-button" onClick={clear}>
                Cancel microphone request
              </button>
            ) : (
              <button
                className="record-button"
                disabled={busy}
                onClick={record}
              >
                <Mic size={19} />
                Start recording
              </button>
            )}
            <span className="record-hint">Up to 30 seconds · solo voice</span>
          </div>
        ) : (
          <div
            className={"upload-space " + (drag ? "dragging" : "")}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              void upload(e.dataTransfer.files[0]);
            }}
          >
            <Upload size={28} />
            <h3>Bring your melody</h3>
            <p>
              Drop an audio file here,
              <br />
              or choose one from your device.
            </p>
            <button
              className="record-button"
              disabled={busy}
              onClick={() => fileInput.current?.click()}
            >
              {status === "validating"
                ? "Checking audio…"
                : "Choose audio file"}
            </button>
            <span className="record-hint">
              30 seconds max · 20 MB max
              <br />
              WAV, MP3 & browser-playable audio
            </span>
          </div>
        )}
        <input
          ref={fileInput}
          className="sr-only"
          tabIndex={-1}
          type="file"
          accept="audio/*,.wav,.mp3,.m4a,.ogg,.webm,.flac"
          aria-label="Choose local audio file"
          onChange={(e) => {
            void upload(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
        {status === "validating" && clip === null && tab === "record" && (
          <p role="status">Checking audio…</p>
        )}
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </div>
      <div className="sample-row">
        <span>Just exploring?</span>
        <button
          className="text-button"
          disabled={busy || status === "recording" || status === "requesting"}
          onClick={sample}
        >
          Try sample <span aria-hidden="true">↗</span>
        </button>
      </div>
    </>
  );
}

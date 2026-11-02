import type { AudioClip } from "./audio";
export interface DemoResult {
  kind: "fixed-sample";
  audioUrl: string;
}
export interface Note {
  start: number;
  end: number;
  pitch: number;
}
/** What the back end returns for one clip (backend/app.py). Times are seconds, pitch is a MIDI number. */
export interface TranscriptionResult {
  kind: "transcription";
  id: string;
  bpm: number;
  duration: number;
  seconds: number;
  basic_pitch_notes: number;
  notes: Note[];
  sheet_notes: Note[];
  files: {
    musicxml: string;
    midi: string;
    synth: string;
    mix: string;
    input: string;
  };
}
export type ScoreResult = DemoResult | TranscriptionResult;
export interface ScoreProvider {
  /** True when a real model transcribes the input (changes the labels in the studio). */
  live?: boolean;
  generate(options: {
    input: AudioClip;
    signal: AbortSignal;
    onStage: (stage: string) => void;
  }): Promise<ScoreResult>;
}
export const mockScoreProvider: ScoreProvider = {
  async generate({ signal, onStage }) {
    for (const stage of [
      "Preparing demo",
      "Simulating pitch detection",
      "Correction skipped in preview",
      "Formatting sample score",
    ]) {
      signal.throwIfAborted();
      onStage(stage);
      await new Promise<void>((resolve, reject) => {
        const cancel = () => {
          clearTimeout(timer);
          reject(new DOMException("Cancelled", "AbortError"));
        };
        const timer = setTimeout(() => {
          signal.removeEventListener("abort", cancel);
          resolve();
        }, 650);
        signal.addEventListener("abort", cancel, { once: true });
      });
    }
    signal.throwIfAborted();
    return { kind: "fixed-sample", audioUrl: "/demo/sample.wav" };
  },
};

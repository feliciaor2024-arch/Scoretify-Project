import type { AudioClip } from "./audio";
export interface DemoResult {
  kind: "fixed-sample";
  audioUrl: string;
}
export interface ScoreProvider {
  generate(options: {
    input: AudioClip;
    signal: AbortSignal;
    onStage: (stage: string) => void;
  }): Promise<DemoResult>;
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

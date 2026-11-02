import { mockScoreProvider, type ScoreProvider } from "./demo";
import { toWav } from "./wav";
/** Base URL of the back end. Empty = same origin; `npm run dev` proxies /api to port 8000 (vite.config.ts). */
const API = import.meta.env.VITE_API_URL ?? "";
export const apiScoreProvider: ScoreProvider = {
  live: true,
  async generate({ input, signal, onStage }) {
    onStage("Preparing audio");
    const body = new FormData();
    body.append("audio", await toWav(input.url, signal), "input.wav");
    signal.throwIfAborted();
    onStage("Transcribing and correcting notes");
    let response: Response;
    try {
      response = await fetch(`${API}/api/transcribe`, {
        method: "POST",
        body,
        signal,
      });
    } catch (e) {
      if (signal.aborted) throw e;
      throw new Error(
        "The transcription server is not reachable. Start the back end and try again.",
      );
    }
    if (!response.ok) {
      const detail = await response
        .json()
        .then((d: { detail?: unknown }) => d.detail)
        .catch(() => "");
      throw new Error(
        typeof detail === "string" && detail
          ? detail
          : `The server could not transcribe this clip (error ${response.status}).`,
      );
    }
    const result = await response.json();
    const prefix = (url: string) => API + url;
    return {
      ...result,
      kind: "transcription",
      files: Object.fromEntries(
        Object.entries(result.files as Record<string, string>).map(([k, v]) => [
          k,
          prefix(v),
        ]),
      ),
    };
  },
};
/** The live model, unless VITE_SCORE_PROVIDER=mock (used by the Playwright suite) or under Vitest. */
export const defaultProvider =
  import.meta.env.VITE_SCORE_PROVIDER === "mock" ||
  import.meta.env.MODE === "test"
    ? mockScoreProvider
    : apiScoreProvider;

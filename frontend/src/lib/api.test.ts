import { afterEach, describe, expect, it, vi } from "vitest";
import { apiScoreProvider } from "./api";
import { encodeWav } from "./wav";
vi.mock("./wav", async (original) => ({
  ...(await original<typeof import("./wav")>()),
  toWav: vi.fn(async () => new Blob(["wav"], { type: "audio/wav" })),
}));
const input = {
  name: "Take",
  url: "blob:take",
  duration: 4,
  source: "recording" as const,
};
const run = () =>
  apiScoreProvider.generate({
    input,
    signal: new AbortController().signal,
    onStage: () => {},
  });
afterEach(() => vi.unstubAllGlobals());
describe("live score provider", () => {
  it("posts the clip as WAV and returns a transcription", async () => {
    const fetch = vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            id: "abc",
            bpm: 120,
            notes: [],
            files: { musicxml: "/api/runs/abc/score.musicxml" },
          }),
        ),
    );
    vi.stubGlobal("fetch", fetch);
    const result = await run();
    expect(result).toMatchObject({
      kind: "transcription",
      id: "abc",
      files: { musicxml: "/api/runs/abc/score.musicxml" },
    });
    const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/transcribe");
    expect((init.body as FormData).get("audio")).toBeInstanceOf(Blob);
  });
  it("shows the server's message when it rejects a clip", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({ detail: "No sung notes were found." }),
            {
              status: 422,
            },
          ),
      ),
    );
    await expect(run()).rejects.toThrow("No sung notes were found.");
  });
  it("explains when the server is not running", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );
    await expect(run()).rejects.toThrow(/server is not reachable/);
  });
});
describe("WAV encoding", () => {
  it("writes a mono 16-bit PCM header and clamps samples", () => {
    const view = new DataView(encodeWav(new Float32Array([0, 1, -2]), 48000));
    const text = (o: number) =>
      String.fromCharCode(...new Uint8Array(view.buffer, o, 4));
    expect([text(0), text(8), text(36)]).toEqual(["RIFF", "WAVE", "data"]);
    expect(view.getUint16(22, true)).toBe(1);
    expect(view.getUint32(24, true)).toBe(48000);
    expect(view.getUint32(40, true)).toBe(6);
    expect(view.getInt16(46, true)).toBe(0x7fff);
    expect(view.getInt16(48, true)).toBe(-0x7fff);
  });
});

import { describe, it, expect, vi } from "vitest";
import { mockScoreProvider } from "./demo";
describe("fixed demo provider", () => {
  it("declares fixed synthetic provenance and simulated stages", async () => {
    vi.useFakeTimers();
    const stages: string[] = [];
    const task = mockScoreProvider.generate({
      input: {
        name: "Demo input",
        url: "/demo/sample.wav",
        duration: 5,
        source: "sample",
      },
      signal: new AbortController().signal,
      onStage: (s) => stages.push(s),
    });
    await vi.runAllTimersAsync();
    const result = await task;
    expect(result.kind).toBe("fixed-sample");
    expect(result.audioUrl).toBe("/demo/sample.wav");
    expect(stages).toEqual([
      "Preparing demo",
      "Simulating pitch detection",
      "Correction skipped in preview",
      "Formatting sample score",
    ]);
    vi.useRealTimers();
  });
});

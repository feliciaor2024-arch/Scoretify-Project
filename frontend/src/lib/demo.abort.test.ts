import { it, expect, vi } from "vitest";
import { mockScoreProvider } from "./demo";
it("aborts a stage without emitting a later result", async () => {
  vi.useFakeTimers();
  const controller = new AbortController();
  const stages: string[] = [];
  const task = mockScoreProvider.generate({
    input: {
      name: "Demo input",
      url: "/demo/sample.wav",
      duration: 5,
      source: "sample",
    },
    signal: controller.signal,
    onStage: (s) => stages.push(s),
  });
  const rejection = expect(task).rejects.toMatchObject({ name: "AbortError" });
  controller.abort();
  await rejection;
  await vi.runAllTimersAsync();
  expect(stages).toEqual(["Preparing demo"]);
  vi.useRealTimers();
});

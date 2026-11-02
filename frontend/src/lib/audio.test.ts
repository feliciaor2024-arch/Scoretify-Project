import { expect, it } from "vitest";
import { validateAudioFile, validateDuration } from "./audio";
it("rejects empty, oversized and non-audio inputs before decoding", () => {
  expect(() =>
    validateAudioFile(new File([], "empty.wav", { type: "audio/wav" })),
  ).toThrow(/empty/i);
  expect(() =>
    validateAudioFile({
      size: 21 * 1024 * 1024,
      type: "audio/wav",
      name: "large.wav",
    } as File),
  ).toThrow(/20 MB/);
  expect(() =>
    validateAudioFile(new File(["text"], "notes.txt", { type: "text/plain" })),
  ).toThrow(/audio/i);
});
it("accepts only finite playable clips up to 30 seconds", () => {
  for (const n of [0, NaN, Infinity, 30.1])
    expect(() => validateDuration(n)).toThrow();
  expect(validateDuration(30)).toBe(30);
});

import { render, screen, fireEvent } from "@testing-library/react";
import { it, expect, vi } from "vitest";
import App from "./App";
import type { ScoreProvider, TranscriptionResult } from "./lib/demo";
const transcription: TranscriptionResult = {
  kind: "transcription",
  id: "abc",
  bpm: 120,
  duration: 5,
  seconds: 1,
  basic_pitch_notes: 1,
  notes: [{ start: 0, end: 1, pitch: 60 }],
  sheet_notes: [{ start: 0, end: 1, pitch: 60 }],
  files: {
    musicxml: "/api/runs/abc/score.musicxml",
    midi: "/api/runs/abc/score.mid",
    synth: "/api/runs/abc/synth.wav",
    mix: "/api/runs/abc/mix.wav",
    input: "/api/runs/abc/input.wav",
  },
};
const live: ScoreProvider = {
  live: true,
  generate: vi.fn(async () => transcription),
};
it("puts the song name on the score, follows edits and clears on reset", async () => {
  // The score itself is drawn by OpenSheetMusicDisplay, which jsdom cannot run; the title is plain HTML.
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response("", { status: 404 })),
  );
  render(<App provider={live} />);
  const box = screen.getByRole("textbox", { name: /Song name/ });
  expect(box).toHaveAttribute("placeholder", "Your melody");
  fireEvent.change(box, { target: { value: "Moon River" } });
  fireEvent.click(screen.getByRole("button", { name: /Try sample/ }));
  fireEvent.click(screen.getByRole("button", { name: /Generate score/ }));
  expect(
    await screen.findByRole("heading", { name: "Moon River" }),
  ).toBeInTheDocument();
  fireEvent.change(box, { target: { value: "" } });
  expect(
    screen.getByRole("heading", { name: "Your melody" }),
  ).toBeInTheDocument();
  fireEvent.change(box, { target: { value: "Renamed" } });
  expect(screen.getByRole("heading", { name: "Renamed" })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Reset studio" }));
  expect(screen.getByRole("textbox", { name: /Song name/ })).toHaveValue("");
  vi.unstubAllGlobals();
});
it("has no song name box in the mock demo, whose sample never changes", () => {
  render(<App />);
  expect(
    screen.queryByRole("textbox", { name: /Song name/ }),
  ).not.toBeInTheDocument();
});

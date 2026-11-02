import { render, screen, fireEvent, act } from "@testing-library/react";
import { it, expect, vi, afterEach } from "vitest";
import { AudioInput } from "./AudioInput";
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});
it("supports arrow-key tab navigation", () => {
  render(<AudioInput disabled={false} onChange={() => {}} />);
  const record = screen.getByRole("tab", { name: "Record" });
  record.focus();
  fireEvent.keyDown(record, { key: "ArrowRight" });
  expect(screen.getByRole("tab", { name: "Upload" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  expect(screen.getByRole("tab", { name: "Upload" })).toHaveFocus();
});
it("pauses original input playback when cleared", () => {
  const pause = vi
    .spyOn(HTMLMediaElement.prototype, "pause")
    .mockImplementation(() => {});
  render(<AudioInput disabled={false} onChange={() => {}} />);
  fireEvent.click(screen.getByRole("button", { name: /Try sample/ }));
  fireEvent.click(screen.getByRole("button", { name: "Record another take" }));
  expect(pause).toHaveBeenCalled();
});
it("explains microphone denial and permits a sample fallback", async () => {
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: {
      getUserMedia: vi
        .fn()
        .mockRejectedValue(new DOMException("denied", "NotAllowedError")),
    },
  });
  vi.stubGlobal("MediaRecorder", class {});
  const changed = vi.fn();
  render(<AudioInput disabled={false} onChange={changed} />);
  fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "permission was denied",
  );
  fireEvent.click(screen.getByRole("button", { name: /Try sample/ }));
  expect(changed).toHaveBeenLastCalledWith(
    expect.objectContaining({ source: "sample", duration: 5 }),
  );
});
it("stops a late microphone stream when the request is cancelled", async () => {
  let grant!: (stream: MediaStream) => void;
  const stop = vi.fn();
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: {
      getUserMedia: () =>
        new Promise((resolve) => {
          grant = resolve;
        }),
    },
  });
  vi.stubGlobal("MediaRecorder", class {});
  render(<AudioInput disabled={false} onChange={() => {}} />);
  fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
  fireEvent.click(
    screen.getByRole("button", { name: "Cancel microphone request" }),
  );
  await act(async () =>
    grant({ getTracks: () => [{ stop }] } as unknown as MediaStream),
  );
  expect(stop).toHaveBeenCalledOnce();
  expect(
    screen.getByRole("button", { name: "Start recording" }),
  ).toBeInTheDocument();
});
it("rejects a non-audio document locally", async () => {
  render(<AudioInput disabled={false} onChange={() => {}} />);
  fireEvent.click(screen.getByRole("tab", { name: "Upload" }));
  fireEvent.change(screen.getByLabelText("Choose local audio file"), {
    target: {
      files: [new File(["hello"], "notes.txt", { type: "text/plain" })],
    },
  });
  expect(await screen.findByRole("alert")).toHaveTextContent("audio file");
});

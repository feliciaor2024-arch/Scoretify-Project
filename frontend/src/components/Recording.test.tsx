import { render, screen, fireEvent, act } from "@testing-library/react";
import { it, expect, vi, afterEach } from "vitest";
import { AudioInput } from "./AudioInput";
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});
it("caps recording at 30 seconds, stops tracks and revokes preview on reset", async () => {
  vi.useFakeTimers();
  const trackStop = vi.fn();
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: {
      getUserMedia: vi.fn(async () => ({
        getTracks: () => [{ stop: trackStop }],
      })),
    },
  });
  class Recorder {
    static isTypeSupported = () => true;
    state = "inactive";
    mimeType = "audio/webm";
    onstop: (() => void) | null = null;
    onerror = null;
    ondataavailable: ((e: { data: Blob }) => void) | null = null;
    start() {
      this.state = "recording";
    }
    stop() {
      this.state = "inactive";
      this.ondataavailable?.({
        data: new Blob(["fixture recorded data"], { type: this.mimeType }),
      });
      this.onstop?.();
    }
  }
  vi.stubGlobal("MediaRecorder", Recorder);
  vi.stubGlobal(
    "AudioContext",
    class {
      constructor() {
        throw Error("meter unavailable");
      }
    },
  );
  const revoke = vi.fn();
  vi.stubGlobal(
    "URL",
    Object.assign(URL, {
      createObjectURL: () => "blob:recording-test",
      revokeObjectURL: revoke,
    }),
  );
  const changed = vi.fn();
  const { unmount } = render(
    <AudioInput disabled={false} onChange={changed} />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Start recording" }));
  await act(async () => {});
  expect(
    screen.getByRole("button", { name: "Stop recording" }),
  ).toBeInTheDocument();
  await act(() => vi.advanceTimersByTimeAsync(30000));
  expect(changed).toHaveBeenLastCalledWith(
    expect.objectContaining({ duration: 30, source: "recording" }),
  );
  expect(trackStop).toHaveBeenCalled();
  unmount();
  expect(revoke).toHaveBeenCalledWith("blob:recording-test");
});

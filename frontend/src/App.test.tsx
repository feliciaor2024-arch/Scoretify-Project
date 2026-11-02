import { render, screen, fireEvent, act } from "@testing-library/react";
import { it, expect, vi } from "vitest";
import App from "./App";
import type { DemoResult, ScoreProvider } from "./lib/demo";
it("ignores late provider results after reset even when a provider ignores abort", async () => {
  let finish!: (result: DemoResult) => void;
  const generate = vi.fn(({ onStage }) => {
    onStage("Preparing demo");
    return new Promise<DemoResult>((resolve) => {
      finish = resolve;
    });
  });
  const provider: ScoreProvider = { generate };
  render(<App provider={provider} />);
  fireEvent.click(screen.getByRole("button", { name: /Try sample/ }));
  const submit = screen.getByRole("button", { name: /Generate demo score/ });
  fireEvent.click(submit);
  fireEvent.click(submit);
  expect(generate).toHaveBeenCalledOnce();
  expect(generate).toHaveBeenCalledWith(
    expect.objectContaining({
      input: expect.objectContaining({
        source: "sample",
        url: "/demo/sample.wav",
      }),
    }),
  );
  fireEvent.click(screen.getByRole("button", { name: "Reset studio" }));
  await act(async () =>
    finish({ kind: "fixed-sample", audioUrl: "/demo/sample.wav" }),
  );
  expect(screen.queryByText("Sample score ready")).not.toBeInTheDocument();
  expect(screen.getByText("Demo preview")).toBeInTheDocument();
});
it("keeps a labelled preview before input and exposes a fixed sample result", async () => {
  vi.useFakeTimers();
  render(<App />);
  expect(screen.getByText("Demo preview")).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: /Generate demo score/ }),
  ).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: /Try sample/ }));
  fireEvent.click(screen.getByRole("button", { name: /Generate demo score/ }));
  expect(screen.getByText("Preparing demo")).toBeInTheDocument();
  await act(() => vi.advanceTimersByTimeAsync(3000));
  expect(screen.getByText("Sample score ready")).toBeInTheDocument();
  expect(
    screen.getByText(
      "This preview uses a fixed sample, not a transcription of your recording.",
    ),
  ).toBeInTheDocument();
  vi.useRealTimers();
});
it("reset cancels work and prevents a stale result from appearing", async () => {
  vi.useFakeTimers();
  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: /Try sample/ }));
  fireEvent.click(screen.getByRole("button", { name: /Generate demo score/ }));
  fireEvent.click(screen.getByRole("button", { name: "Reset studio" }));
  await act(() => vi.advanceTimersByTimeAsync(5000));
  expect(screen.queryByText("Sample score ready")).not.toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: /Generate demo score/ }),
  ).toBeDisabled();
  vi.useRealTimers();
});

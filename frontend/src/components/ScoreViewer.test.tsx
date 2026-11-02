import { readFileSync } from "node:fs";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { it, expect, vi, afterEach } from "vitest";
import { ScoreViewer } from "./ScoreViewer";
const raw = readFileSync("public/demo/sample.svg", "utf8");
const fixture = JSON.parse(readFileSync("public/demo/sample.json", "utf8"));
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});
it("pauses sample audio when reset removes the viewer", () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(Error("offline")));
  const pause = vi
    .spyOn(HTMLMediaElement.prototype, "pause")
    .mockImplementation(() => {});
  const { unmount } = render(<ScoreViewer />);
  unmount();
  expect(pause).toHaveBeenCalledOnce();
});
it("loads genuine fixture IDs and scrubs current note highlighting without altering assets", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      text: async () => raw,
      json: async () => fixture,
    })),
  );
  const { container } = render(<ScoreViewer />);
  await screen.findByRole("img", {
    name: "Evening melody, synthetic sample notation",
  });
  expect(container.querySelectorAll(".note")).toHaveLength(8);
  fireEvent.change(screen.getByRole("slider"), { target: { value: "1.25" } });
  expect(container.querySelector(".playing-note")).toHaveAttribute(
    "id",
    "source-1-fragment-0",
  );
  fireEvent.change(screen.getByRole("slider"), { target: { value: "0" } });
  expect(container.querySelector(".playing-note")).toBeNull();
  expect(container.querySelector(".pgHead")).toBeNull();
  expect(container.querySelectorAll("defs")).toHaveLength(1);
  expect(screen.getByRole("link", { name: "MusicXML" })).toHaveAttribute(
    "href",
    "/demo/sample.musicxml",
  );
});
it("offers retry when local score assets fail", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(Error("missing")));
  render(<ScoreViewer />);
  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent("could not be loaded"),
  );
  expect(
    screen.getByRole("button", { name: "Retry sample" }),
  ).toBeInTheDocument();
});

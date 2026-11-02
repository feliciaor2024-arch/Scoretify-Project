import { test, expect } from "@playwright/test";
import path from "node:path";
test("rejects long and oversized audio before a demo can run", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("tab", { name: "Upload" }).click();
  const frames = 8000 * 31;
  const wav = Buffer.alloc(44 + frames * 2);
  wav.write("RIFF", 0);
  wav.writeUInt32LE(wav.length - 8, 4);
  wav.write("WAVEfmt ", 8);
  wav.writeUInt32LE(16, 16);
  wav.writeUInt16LE(1, 20);
  wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(8000, 24);
  wav.writeUInt32LE(16000, 28);
  wav.writeUInt16LE(2, 32);
  wav.writeUInt16LE(16, 34);
  wav.write("data", 36);
  wav.writeUInt32LE(frames * 2, 40);
  await page.getByLabel("Choose local audio file").setInputFiles({
    name: "too-long.wav",
    mimeType: "audio/wav",
    buffer: wav,
  });
  await expect(page.getByRole("alert")).toContainText("30 seconds or less");
  await expect(
    page.getByRole("button", { name: "Generate demo score" }),
  ).toBeDisabled();
  await page.getByLabel("Choose local audio file").setInputFiles({
    name: "too-large.wav",
    mimeType: "audio/wav",
    buffer: Buffer.alloc(21 * 1024 * 1024),
  });
  await expect(page.getByRole("alert")).toContainText("20 MB");
});
test("sample workflow, real fixture playback, highlights and reset", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const remote: string[] = [];
  page.on("request", (r) => {
    if (!r.url().startsWith("http://127.0.0.1:5181")) remote.push(r.url());
  });
  await page.goto("/");
  await expect(
    page.getByRole("img", {
      name: "Evening melody, synthetic sample notation",
    }),
  ).toBeVisible();
  await expect(page.locator(".note")).toHaveCount(8);
  await page.screenshot({
    path: "test-results/studio-desktop.png",
    fullPage: true,
  });
  await expect(
    page.getByRole("button", { name: "Generate demo score" }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "Try sample" }).click();
  await page.getByRole("button", { name: "Generate demo score" }).click();
  await expect(page.getByText("Preparing demo")).toBeVisible();
  await expect(page.getByText("Sample score ready")).toBeVisible();
  await page.getByRole("button", { name: "Play sample score" }).click();
  await expect(page.locator(".playing-note")).toHaveCount(1);
  await page.getByRole("button", { name: "Pause sample score" }).click();
  const held = await page.locator(".playing-note").getAttribute("id");
  expect(held).toMatch(/^source-/);
  await page.getByRole("button", { name: "Restart sample playback" }).click();
  await expect(page.locator(".playing-note")).toHaveCount(0);
  await page.getByRole("button", { name: "Zoom in" }).click();
  await expect(page.getByLabel("Score zoom")).toHaveText("125%");
  await page.getByRole("button", { name: "Reset studio" }).click();
  await expect(page.getByText("Demo preview")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Generate demo score" }),
  ).toBeDisabled();
  for (const label of ["PDF", "MusicXML", "MIDI"]) {
    const response = await page.request.get(
      (await page
        .getByRole("link", { name: label, exact: true })
        .getAttribute("href")) ?? "",
    );
    expect(response.ok()).toBeTruthy();
    expect((await response.body()).length).toBeGreaterThan(100);
  }
  expect(errors).toEqual([]);
  expect(remote).toEqual([]);
});
test("local upload validates genuine sample, rejects non-audio and shows help on mobile", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByText("Mock demo", { exact: true })).toBeVisible();
  await page.getByRole("tab", { name: "Upload" }).click();
  await page
    .getByLabel("Choose local audio file")
    .setInputFiles(path.resolve("public/demo/sample.wav"));
  await expect(page.getByText("sample.wav", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Generate demo score" }),
  ).toBeEnabled();
  await page.screenshot({
    path: "test-results/studio-mobile.png",
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  await page.getByRole("button", { name: "How it works" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await page.getByRole("button", { name: "Choose another file" }).click();
  await page.getByLabel("Choose local audio file").setInputFiles({
    name: "document.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("not audio"),
  });
  await expect(page.getByRole("alert")).toContainText("audio file");
});
test("records browser fake microphone, stops tracks and previews captured audio", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const request = navigator.mediaDevices.getUserMedia.bind(
      navigator.mediaDevices,
    );
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      const stream = await request(constraints);
      Object.defineProperty(window, "testCapturedTracks", {
        value: stream.getTracks(),
        configurable: true,
      });
      return stream;
    };
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Start recording" }).click();
  await expect(
    page.getByRole("button", { name: "Stop recording" }),
  ).toBeVisible();
  await expect(page.locator(".record-timer")).toContainText("00:01");
  await page.getByRole("button", { name: "Stop recording" }).click();
  await expect(page.getByText("Your recording", { exact: true })).toBeVisible();
  expect(
    await page.evaluate(() =>
      (
        window as unknown as { testCapturedTracks: MediaStreamTrack[] }
      ).testCapturedTracks.every((track) => track.readyState === "ended"),
    ),
  ).toBeTruthy();
  await expect(page.getByLabel("Preview original input audio")).toHaveAttribute(
    "src",
    /^blob:/,
  );
  const duration = await page
    .getByLabel("Preview original input audio")
    .evaluate(async (audio: HTMLAudioElement) => {
      if (audio.readyState < 1)
        await new Promise<void>((resolve) =>
          audio.addEventListener("loadedmetadata", () => resolve(), {
            once: true,
          }),
        );
      return audio.duration;
    });
  expect(duration).toBeGreaterThan(0);
  await page.getByRole("button", { name: "Reset studio" }).click();
  await expect(
    page.getByRole("button", { name: "Start recording" }),
  ).toBeVisible();
});

/** Decode any browser-playable clip (WebM/Opus recordings, MP3, M4A, ...) and re-encode it as mono 16-bit WAV,
 * which the back end can read without ffmpeg. */
export async function toWav(url: string, signal?: AbortSignal) {
  const data = await (await fetch(url, { signal })).arrayBuffer();
  const context = new AudioContext();
  try {
    const buffer = await context.decodeAudioData(data);
    return new Blob([encodeWav(mixDown(buffer), buffer.sampleRate)], {
      type: "audio/wav",
    });
  } finally {
    await context.close();
  }
}
function mixDown(buffer: AudioBuffer) {
  const mono = new Float32Array(buffer.length);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const channel = buffer.getChannelData(c);
    for (let i = 0; i < mono.length; i++)
      mono[i] += channel[i] / buffer.numberOfChannels;
  }
  return mono;
}
export function encodeWav(samples: Float32Array, rate: number) {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const text = (offset: number, value: string) =>
    [...value].forEach((ch, i) => view.setUint8(offset + i, ch.charCodeAt(0)));
  text(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((s, i) =>
    view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 0x7fff, true),
  );
  return view.buffer;
}

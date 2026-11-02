export interface AudioClip {
  name: string;
  url: string;
  duration: number;
  source: "sample" | "recording" | "upload";
}
export const MAX_SECONDS = 30;
export function validateAudioFile(file: File) {
  if (!file.size)
    throw new Error("This file is empty. Choose an audio recording.");
  if (file.size > 20 * 1024 * 1024)
    throw new Error("Choose an audio file of 20 MB or less.");
  if (
    !file.type.startsWith("audio/") &&
    !(file.type === "" && /\.(wav|mp3|m4a|ogg|webm|aac|flac)$/i.test(file.name))
  )
    throw new Error(
      "Choose a supported audio file, not another document type.",
    );
}
export function validateDuration(seconds: number) {
  if (!Number.isFinite(seconds) || seconds <= 0 || seconds > MAX_SECONDS)
    throw new Error("Use a playable audio clip of 30 seconds or less.");
  return seconds;
}
export async function inspectAudio(file: File) {
  validateAudioFile(file);
  const context = new AudioContext();
  try {
    const buffer = await context.decodeAudioData(await file.arrayBuffer());
    return validateDuration(buffer.duration);
  } catch (error) {
    if (error instanceof Error && error.message.includes("30 seconds"))
      throw error;
    throw new Error(
      "This audio could not be decoded. Try WAV, MP3 or a browser-playable audio file.",
    );
  } finally {
    await context.close();
  }
}

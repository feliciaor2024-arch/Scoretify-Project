/** The song name the user types: shown on the score, in the PDF and MusicXML, and used for download file names. */
export const DEFAULT_TITLE = "Your melody";
export const MAX_TITLE = 60;
export const songTitle = (name: string) => name.trim() || DEFAULT_TITLE;
/** "My Song!" -> "my-song"; non-Latin names are kept. Falls back to scoretify-<run id>. */
export function fileStem(name: string, id: string) {
  const slug = name
    .normalize("NFKD")
    .replace(/\p{M}/gu, "") // drop accents: "Café" -> "Cafe"
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 50);
  return slug || `scoretify-${id}`;
}
/** Put the song name into MusicXML. music21 writes the run id as the work and movement title. */
export function retitle(xml: string, title: string) {
  const doc = new DOMParser().parseFromString(xml, "application/xml");
  if (doc.querySelector("parsererror")) return xml;
  doc
    .querySelectorAll("work-title, movement-title")
    .forEach((n) => (n.textContent = title));
  return (
    '<?xml version="1.0" encoding="UTF-8"?>\n' +
    new XMLSerializer().serializeToString(doc)
  );
}

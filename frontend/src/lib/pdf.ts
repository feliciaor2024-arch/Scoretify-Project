/** Engrave MusicXML on A4 pages with OpenSheetMusicDisplay, off screen, and save the pages as a PDF.
 * Everything runs in the browser (jsPDF + svg2pdf.js), so the back end needs no MuseScore or LilyPond. */
export async function scorePdf(xml: string, title: string) {
  const [{ OpenSheetMusicDisplay }, { jsPDF }] = await Promise.all([
    import("opensheetmusicdisplay"),
    import("jspdf"),
  ]);
  await import("svg2pdf.js"); // adds pdf.svg()
  const host = document.createElement("div");
  host.style.cssText = "position:fixed;left:-10000px;top:0;width:794px"; // A4 width at 96 dpi
  document.body.append(host);
  try {
    const sheet = new OpenSheetMusicDisplay(host, {
      autoResize: false,
      backend: "svg",
      pageFormat: "A4_P",
      pageBackgroundColor: "#FFFFFF",
      drawSubtitle: false, // music21 writes the run id there
      drawComposer: false,
      drawPartNames: false,
    });
    await sheet.load(xml);
    sheet.Sheet.TitleString = title; // the MusicXML title is the run id
    sheet.render();
    const pdf = new jsPDF({ unit: "pt", format: "a4" });
    const width = pdf.internal.pageSize.getWidth();
    const pages = [...host.querySelectorAll("svg")];
    for (const [i, svg] of pages.entries()) {
      if (i) pdf.addPage();
      // svg2pdf drops text sized in pt (OSMD uses it for the tempo mark's "= 136"), so give it in px.
      svg.querySelectorAll("text[font-size$='pt']").forEach((t) => {
        const size = parseFloat(t.getAttribute("font-size")!);
        t.setAttribute("font-size", `${(size * 4) / 3}px`);
      });
      const box = svg.getBoundingClientRect();
      const scale = width / box.width;
      const label = i === 0 ? titleImage(svg, title) : null;
      await pdf.svg(svg, { x: 0, y: 0, width, height: box.height * scale });
      if (label) {
        const r = label.rect;
        pdf.addImage(
          label.png,
          "PNG",
          (r.left - box.left) * scale,
          (r.top - box.top) * scale,
          r.width * scale,
          r.height * scale,
        );
      }
    }
    return pdf.output("blob");
  } finally {
    host.remove();
  }
}
/** jsPDF's built-in fonts only cover Latin-1, so a title such as "说谎" would come out as gibberish. For those
 * titles, take the title text out of the page and return it drawn by the browser (its own fonts) as an image,
 * with the place it occupied. Latin-1 titles stay as real PDF text. */
function titleImage(svg: SVGSVGElement, title: string) {
  if (/^[\x20-\xff]*$/.test(title)) return null;
  const text = [...svg.querySelectorAll("text")].find(
    (t) => t.textContent === title,
  );
  if (!text) return null;
  const rect = text.getBoundingClientRect();
  const ink = 4; // pixels per screen pixel, for print sharpness
  const canvas = document.createElement("canvas");
  canvas.width = Math.ceil(rect.width * ink);
  canvas.height = Math.ceil(rect.height * ink);
  const ctx = canvas.getContext("2d")!;
  ctx.scale(ink, ink);
  ctx.font = `${text.getAttribute("font-size")} ${text.getAttribute("font-family")}, serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(title, rect.width / 2, rect.height / 2);
  text.remove();
  return { png: canvas.toDataURL("image/png"), rect };
}

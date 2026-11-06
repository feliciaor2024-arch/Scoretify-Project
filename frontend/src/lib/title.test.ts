import { describe, expect, it } from "vitest";
import { fileStem, retitle, songTitle } from "./title";
describe("song title", () => {
  it("falls back to the default name when empty or blank", () => {
    expect(songTitle("")).toBe("Your melody");
    expect(songTitle("   ")).toBe("Your melody");
    expect(songTitle("  Moon River ")).toBe("Moon River");
  });
  it("turns the name into a safe download file name", () => {
    expect(fileStem("Moon River!", "abc")).toBe("moon-river");
    expect(fileStem("Café / Été", "abc")).toBe("cafe-ete");
    expect(fileStem("说谎", "abc")).toBe("说谎");
    expect(fileStem("  ?!  ", "abc")).toBe("scoretify-abc");
  });
  it("writes the name into MusicXML and keeps the rest", () => {
    const xml =
      '<?xml version="1.0" encoding="utf-8"?>\n<score-partwise version="4.0"><work><work-title>score</work-title></work><movement-title>score</movement-title><part id="P1"/></score-partwise>';
    const out = retitle(xml, "Tom & Jerry <3");
    expect(out).toContain("<work-title>Tom &amp; Jerry &lt;3</work-title>");
    expect(out).toContain(
      "<movement-title>Tom &amp; Jerry &lt;3</movement-title>",
    );
    expect(out).toContain('<part id="P1"/>');
    expect(out.startsWith("<?xml")).toBe(true);
    expect(retitle("not xml <", "x")).toBe("not xml <");
  });
});

// Convert the report Markdown (a small, known subset) into a styled DOCX.
// Usage: node md2docx.js <input.md> <output.docx>
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, TableOfContents, Footer, Header, PageNumber,
  LevelFormat, PageBreak, TabStopType,
} = require("docx");

const [, , inFile, outFile] = process.argv;
const baseDir = path.dirname(path.resolve(inFile));
const md = fs.readFileSync(inFile, "utf8").replace(/\r\n/g, "\n");
const [titleBlock, body] = md.split("<!-- BODY -->");

const FONT = "Calibri";
const ACCENT = "1F3864";
const CONTENT_W = 9026; // A4 width 11906 minus 2 x 1440 margins

// ---------- inline formatting: **bold**, *italic*, `code` ----------
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), ...base, bold: true }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), ...base, font: "Consolas", size: 20 }));
    else out.push(new TextRun({ text: t.slice(1, -1), ...base, italics: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}

const caption = (text) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 60, after: 240 },
  children: runs(text, { italics: true, size: 19, color: "404040" }),
});

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
function makeTable(rows) {
  const cells = rows.map((r) => r.replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
  const header = cells[0];
  const data = cells.slice(2); // skip |---| line
  const n = header.length;
  // column widths proportional to longest text (bounded), summing to CONTENT_W
  const lens = header.map((_, i) => Math.min(60, Math.max(6, ...[header, ...data].map((r) => (r[i] || "").replace(/\*\*/g, "").length))));
  const total = lens.reduce((a, b) => a + b, 0);
  const widths = lens.map((l) => Math.floor((l / total) * CONTENT_W));
  widths[n - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const emptyHeader = header.every((h) => h === "");
  const mkRow = (r, isHeader, keep = true) => new TableRow({
    tableHeader: isHeader, cantSplit: true,
    children: r.map((c, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: isHeader ? { type: ShadingType.CLEAR, fill: "D9E2F3", color: "auto" } : undefined,
      margins: { top: 50, bottom: 50, left: 90, right: 90 },
      borders: { top: border, bottom: border, left: border, right: border },
      children: [new Paragraph({ keepNext: keep, spacing: { before: 0, after: 0 }, children: runs(c, { size: 19, bold: isHeader || undefined }) })],
    })),
  });
  const tableRows = (emptyHeader ? [] : [mkRow(header, true)]).concat(data.map((r, k) => mkRow(r, false, k < data.length - 1)));
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, rows: tableRows });
}

// ---------- body parser ----------
function parseBody(text) {
  const lines = text.split("\n");
  const out = [];
  let i = 0, listInstance = 0, inList = null;
  while (i < lines.length) {
    const line = lines[i];
    const t = line.trim();
    if (!/^(- |\d+\. )/.test(t)) inList = null;
    if (t === "") { i++; continue; }
    if (t === "\\pagebreak") { out.push(new Paragraph({ children: [new PageBreak()] })); i++; continue; }
    if (t === "[[TOC]]") {
      out.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Table of Contents")] }));
      out.push(new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }));
      i++; continue;
    }
    if (t.startsWith("```")) {
      const code = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) code.push(lines[i++]);
      i++;
      code.forEach((c, k) => out.push(new Paragraph({
        shading: { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" },
        spacing: { before: k === 0 ? 120 : 0, after: k === code.length - 1 ? 200 : 0, line: 240 },
        children: [new TextRun({ text: c.replace(/ /g, "Â ") || "Â ", font: "Consolas", size: 17 })],
      })));
      continue;
    }
    const h = /^(#{1,3}) (.*)$/.exec(t);
    if (h) {
      const level = [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3][h[1].length - 1];
      out.push(new Paragraph({ heading: level, keepNext: true, children: [new TextRun(h[2])] }));
      i++; continue;
    }
    const img = /^!\[(.*)\]\((.*)\)$/.exec(t);
    if (img) {
      const file = path.resolve(baseDir, img[2]);
      const { w, h: hh, data } = pngSize(file);
      const isShot = /screenshots/.test(file);
      const maxW = isShot ? 480 : 600, maxH = isShot ? 800 : 560; // points-ish pixels at 96 dpi
      const scale = Math.min(maxW / w, maxH / hh);
      out.push(new Paragraph({
        alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 0 },
        children: [new ImageRun({ type: "png", data, transformation: { width: Math.round(w * scale), height: Math.round(hh * scale) },
          altText: { title: img[1], description: img[1], name: path.basename(file) } })],
      }));
      out.push(caption(img[1]));
      i++; continue;
    }
    if (t.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(lines[i++].trim());
      out.push(makeTable(rows));
      // caption line (Table N: ...) directly after the table
      let j = i; while (j < lines.length && lines[j].trim() === "") j++;
      if (j < lines.length && /^Table \d+:/.test(lines[j].trim())) { out.push(caption(lines[j].trim())); i = j + 1; }
      else out.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
      continue;
    }
    if (/^- /.test(t)) {
      out.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: runs(t.slice(2)) }));
      i++; continue;
    }
    const num = /^(\d+)\. (.*)$/.exec(t);
    if (num) {
      if (inList !== "num") { listInstance++; inList = "num"; }
      out.push(new Paragraph({ numbering: { reference: "numbers", level: 0, instance: listInstance }, spacing: { after: 60 }, children: runs(num[2]) }));
      i++; continue;
    }
    // ordinary paragraph (join soft-wrapped lines)
    const para = [t];
    i++;
    while (i < lines.length && lines[i].trim() !== "" && !/^(#|\||!\[|```|- |\d+\. |\\pagebreak|\[\[TOC)/.test(lines[i].trim())) para.push(lines[i++].trim());
    out.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 140 }, children: runs(para.join(" ")) }));
  }
  return out;
}

// ---------- title page ----------
const meta = {};
titleBlock.split("\n").forEach((l) => {
  const m = /^\| (.+?) \| (.+?) \|$/.exec(l.trim());
  if (m && m[1].trim()) meta[m[1].trim()] = m[2].trim();
});
const title = /^# (.*)$/m.exec(titleBlock)[1];
const centered = (children, spacing = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing, children });
// Title-page fields come from the table above <!-- BODY --> in the Markdown file.
const FIELD_ORDER = ["Submitted by", "Roll Number", "Course", "Institute", "Assignment", "Maximum Marks",
  "Submission Date", "Code Repository"];
const titlePage = [
  centered([new TextRun({ text: (meta["Institute"] || "").toUpperCase(), bold: true, size: 28, color: ACCENT })], { before: 600, after: 60 }),
  centered([new TextRun({ text: meta["Department"] || "", size: 24 })], { after: 1100 }),
  centered([new TextRun({ text: meta["Document"] || "Project Report", size: 26, color: "595959" })], { after: 200 }),
  centered([new TextRun({ text: title, bold: true, size: 40, color: ACCENT })], { after: 200 }),
  centered([new TextRun({ text: meta["Subtitle"] || "", italics: true, size: 24, color: "404040" })], { after: 1200 }),
  ...FIELD_ORDER.filter((k) => meta[k]).map((k) =>
    new Paragraph({
      spacing: { after: 110 }, indent: { left: 500 },
      tabStops: [{ type: TabStopType.LEFT, position: 2700 }],
      children: [new TextRun({ text: k, bold: true, size: 22 }), new TextRun({ text: `	${meta[k]}`, size: 22 })],
    })),
  new Paragraph({ children: [new PageBreak()] }),
];

const doc = new Document({
  creator: meta["Submitted by"] || "Student",
  title,
  description: "Internship project report",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { size: 32, bold: true, color: ACCENT, font: FONT }, paragraph: { spacing: { after: 240 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: ACCENT, font: FONT }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: "2E5597", font: FONT }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, color: "2E5597", font: FONT }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "â€¢", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    ],
  },
  sections: [
    { properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } }, children: titlePage },
    {
      properties: { page: { margin: { top: 1440, bottom: 1300, left: 1440, right: 1440 }, pageNumbers: { start: 1 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: meta["Running Header"] || title, size: 17, color: "7F7F7F" })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 18, color: "7F7F7F" })] })] }) },
      children: parseBody(body),
    },
  ],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(outFile, buf); console.log("wrote", outFile); });


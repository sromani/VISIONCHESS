#!/usr/bin/env node
/**
 * Unpack debug_failures_*.json (from app Export failures) into debug_failures/<id>/
 * Usage: node scripts/unpack-debug-failures.cjs path/to/export.json
 */
const fs = require("fs");
const path = require("path");

const input = process.argv[2];
if (!input) {
  console.error("Usage: node scripts/unpack-debug-failures.cjs <export.json>");
  process.exit(1);
}

const records = JSON.parse(fs.readFileSync(input, "utf8"));
const outRoot = path.join(process.cwd(), "debug_failures");

function writeDataUrl(dir, name, dataUrl) {
  if (!dataUrl || !dataUrl.startsWith("data:")) return;
  const m = dataUrl.match(/^data:([^;]+);base64,(.+)$/);
  if (!m) return;
  const ext = m[1].includes("png") ? "png" : "jpg";
  fs.writeFileSync(path.join(dir, `${name}.${ext}`), Buffer.from(m[2], "base64"));
}

for (const r of records) {
  const dir = path.join(outRoot, r.id);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(
    path.join(dir, "meta.json"),
    JSON.stringify(
      {
        id: r.id,
        timestamp: r.timestamp,
        mode: r.mode,
        reason: r.reason,
        fileName: r.fileName,
        localizationScore: r.localizationScore,
        boardFound: r.boardFound,
        pieceCount: r.pieceCount,
        bestSource: r.bestSource,
        rejectionReason: r.rejectionReason,
        candidates: r.candidates,
      },
      null,
      2,
    ),
  );
  writeDataUrl(dir, "original", r.originalDataUrl);
  writeDataUrl(dir, "overlay", r.overlayDataUrl);
  console.log("Wrote", dir);
}

console.log(`Done — ${records.length} case(s) in ${outRoot}`);

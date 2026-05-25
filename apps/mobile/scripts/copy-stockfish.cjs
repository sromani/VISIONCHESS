const fs = require("fs");
const path = require("path");

const pkgDir = path.join(__dirname, "..", "node_modules", "stockfish.js");
const publicDir = path.join(__dirname, "..", "public");

const files = ["stockfish.js", "stockfish.wasm.js", "stockfish.wasm"];

if (!fs.existsSync(pkgDir)) {
  console.warn("stockfish.js package not found — run npm install first");
  process.exit(0);
}

fs.mkdirSync(publicDir, { recursive: true });

for (const file of files) {
  const src = path.join(pkgDir, file);
  const dest = path.join(publicDir, file);
  if (fs.existsSync(src)) {
    fs.copyFileSync(src, dest);
    console.log(`Copied ${file} to public/`);
  } else {
    console.warn(`Missing ${file} in stockfish.js package`);
  }
}

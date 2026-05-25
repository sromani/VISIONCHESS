#!/usr/bin/env node
/**
 * VisionChess branding — tight app icons (no text, ~88% fill).
 * Source: branding/source-app-icon.png (may include legacy text; stripped automatically).
 */
const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

const root = path.join(__dirname, "..");
const sourcePath = path.join(root, "branding", "source-app-icon.png");
const archivePath = path.join(root, "branding", "archive", "source-app-icon-with-text.jpg");
const masterPath = path.join(root, "branding", "app-icon-master.png");
const previewsDir = path.join(root, "branding", "previews");

const publicDir = path.join(root, "public");
const iosAppIcon = path.join(root, "ios/App/App/Assets.xcassets/AppIcon.appiconset");
const iosSplash = path.join(root, "ios/App/App/Assets.xcassets/Splash.imageset");
const androidRes = path.join(root, "android/app/src/main/res");

const LAUNCHER_BG = { r: 0, g: 0, b: 0, alpha: 255 };
const TRANSPARENT = { r: 0, g: 0, b: 0, alpha: 0 };
const SPLASH_BG = "#09090b";

/** Square launcher / iOS / PWA — subject fills this fraction of canvas (80–90% target). */
const ICON_FILL = 0.96;
/** Android adaptive foreground — fills safe zone on circular launchers. */
const ADAPTIVE_FOREGROUND_FILL = 0.88;
/** PWA maskable — extra inset so circle/squircle masks do not clip the knight. */
const MASKABLE_FILL = 0.82;
const SPLASH_ICON_SCALE = 0.52;

const MIPMAP_LAUNCHER = {
  "mipmap-mdpi": 48,
  "mipmap-hdpi": 72,
  "mipmap-xhdpi": 96,
  "mipmap-xxhdpi": 144,
  "mipmap-xxxhdpi": 192,
};

const MIPMAP_FOREGROUND = {
  "mipmap-mdpi": 108,
  "mipmap-hdpi": 162,
  "mipmap-xhdpi": 216,
  "mipmap-xxhdpi": 324,
  "mipmap-xxxhdpi": 432,
};

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

async function writePng(pipeline, dest) {
  ensureDir(path.dirname(dest));
  await pipeline.png({ compressionLevel: 9, adaptiveFiltering: true }).toFile(dest);
}

/** Row luminance score for text-band detection. */
function rowScores(data, width, height) {
  const scores = new Array(height).fill(0);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      if (r + g + b > 90) scores[y]++;
    }
  }
  return scores;
}

/**
 * Y limit for graphic crop — finds the low-density row between artwork and wordmark.
 * (Row luminance alone is unreliable because the gradient is bright mid-frame.)
 */
function detectTextBandTop(scores, height) {
  const lo = Math.floor(height * 0.44);
  const hi = Math.floor(height * 0.58);
  let cutY = height - 1;
  let minScore = Infinity;

  for (let y = lo; y <= hi; y++) {
    if (scores[y] < minScore) {
      minScore = scores[y];
      cutY = y;
    }
  }

  if (minScore > 420) return height - 1;
  return Math.max(0, cutY - 10);
}

/** Bounding box of visible artwork (gradient + knight + lens), excluding bottom text. */
function contentBbox(data, width, height, maxY) {
  let minX = width;
  let minY = height;
  let maxX = 0;
  let maxY2 = 0;

  for (let y = 0; y <= maxY; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      if (r + g + b > 55 && !(r < 14 && g < 14 && b < 14)) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY2) maxY2 = y;
      }
    }
  }

  if (maxX < minX || maxY2 < minY) {
    return { left: 0, top: 0, width, height: Math.min(height, maxY + 1) };
  }

  const pad = 4;
  const left = Math.max(0, minX - pad);
  const top = Math.max(0, minY - pad);
  const right = Math.min(width - 1, maxX + pad);
  const bottom = Math.min(maxY, maxY2 + pad);
  return {
    left,
    top,
    width: right - left + 1,
    height: bottom - top + 1,
  };
}

/** Crop mark (knight + lens + reticle), no wordmark. */
async function extractMarkBuffer(source) {
  const { data, info } = await sharp(source).ensureAlpha().raw().toBuffer({
    resolveWithObject: true,
  });
  const scores = rowScores(data, info.width, info.height);
  const textTop = detectTextBandTop(scores, info.height);
  const bbox = contentBbox(data, info.width, info.height, textTop);

  const cropped = await sharp(source).extract(bbox).png().toBuffer();
  try {
    return await sharp(cropped).trim({ threshold: 14 }).png().toBuffer();
  } catch {
    return cropped;
  }
}

/** Center mark on square canvas; cover-fill so the knight dominates at small sizes. */
async function composeIcon(markBuffer, size, fill, background) {
  const inner = Math.round(size * fill);
  const resized = await sharp(markBuffer)
    .resize(inner, inner, { fit: "cover", position: "centre", kernel: sharp.kernel.lanczos3 })
    .png()
    .toBuffer();

  return sharp({
    create: {
      width: size,
      height: size,
      channels: 4,
      background,
    },
  }).composite([{ input: resized, gravity: "center" }]);
}

async function buildMaster(markBuffer) {
  return composeIcon(markBuffer, 1024, ICON_FILL, LAUNCHER_BG);
}

async function adaptiveForeground(markBuffer, size) {
  return composeIcon(markBuffer, size, ADAPTIVE_FOREGROUND_FILL, TRANSPARENT);
}

/** Native launch splash: solid dark only (branding lives in React loading screen). */
function nativeLaunchSplash(width, height) {
  return sharp({
    create: { width, height, channels: 3, background: SPLASH_BG },
  });
}

async function splashCanvas(markBuffer, width, height) {
  const logoSize = Math.round(Math.min(width, height) * SPLASH_ICON_SCALE);
  const logo = await sharp(markBuffer)
    .resize(logoSize, logoSize, { fit: "cover", position: "centre", kernel: sharp.kernel.lanczos3 })
    .png()
    .toBuffer();

  return sharp({
    create: {
      width,
      height,
      channels: 3,
      background: SPLASH_BG,
    },
  }).composite([{ input: logo, gravity: "center" }]);
}

const PREVIEW_WALL = { r: 30, g: 30, b: 35, alpha: 255 };

/** Simulated iOS home screen (icon on dark wallpaper). */
async function previewIosHome(markBuffer, size) {
  const pad = Math.max(8, Math.round(size * 0.12));
  const canvas = size + pad * 2;
  const icon = await (await composeIcon(markBuffer, size, ICON_FILL, LAUNCHER_BG))
    .png()
    .toBuffer();
  return sharp({
    create: { width: canvas, height: canvas, channels: 3, background: PREVIEW_WALL },
  }).composite([{ input: icon, top: pad, left: pad }]);
}

/** Simulated Android launcher (foreground + black bg, circle crop via SVG). */
async function previewAndroidAdaptive(markBuffer, size) {
  const pad = 10;
  const fgBuf = await (await adaptiveForeground(markBuffer, size)).png().toBuffer();
  const flat = await sharp({
    create: { width: size, height: size, channels: 4, background: LAUNCHER_BG },
  })
    .composite([{ input: fgBuf, gravity: "center" }])
    .png()
    .toBuffer();

  const masked = await sharp(flat)
    .resize(size, size)
    .ensureAlpha()
    .composite([
      {
        input: await sharp(
          Buffer.from(
            `<svg width="${size}" height="${size}"><circle cx="${size / 2}" cy="${size / 2}" r="${size / 2 - 1}" fill="white"/></svg>`,
          ),
        )
          .png()
          .resize(size, size)
          .toBuffer(),
        blend: "dest-in",
      },
    ])
    .png()
    .toBuffer();

  const canvas = size + pad * 2;
  return sharp({
    create: { width: canvas, height: canvas, channels: 3, background: PREVIEW_WALL },
  }).composite([{ input: masked, top: pad, left: pad }]);
}

async function generatePreviews(markBuffer, masterPipeline) {
  ensureDir(previewsDir);
  console.log("Preview sheets (open branding/previews/)…");

  await writePng(
    await previewIosHome(markBuffer, 60),
    path.join(previewsDir, "ios-home-60.png"),
  );
  await writePng(
    await previewIosHome(markBuffer, 120),
    path.join(previewsDir, "ios-home-120.png"),
  );
  await writePng(
    await previewIosHome(markBuffer, 180),
    path.join(previewsDir, "ios-home-180.png"),
  );
  await writePng(
    await previewAndroidAdaptive(markBuffer, 192),
    path.join(previewsDir, "android-launcher-192.png"),
  );
  await writePng(
    await composeIcon(markBuffer, 48, ICON_FILL, LAUNCHER_BG),
    path.join(previewsDir, "pwa-favicon-48.png"),
  );
  await writePng(
    await composeIcon(markBuffer, 32, ICON_FILL, LAUNCHER_BG),
    path.join(previewsDir, "pwa-favicon-32.png"),
  );
  await writePng(
    await adaptiveForeground(markBuffer, 512),
    path.join(previewsDir, "android-foreground-512.png"),
  );
  await writePng(masterPipeline.clone(), path.join(previewsDir, "master-1024.png"));
}

async function main() {
  if (!fs.existsSync(sourcePath)) {
    console.error(`Missing source: ${sourcePath}`);
    process.exit(1);
  }

  if (!fs.existsSync(archivePath)) {
    ensureDir(path.dirname(archivePath));
    fs.copyFileSync(sourcePath, archivePath);
    console.log(`Archived original → ${path.relative(root, archivePath)}`);
  }

  const designSource = fs.existsSync(archivePath) ? archivePath : sourcePath;
  console.log(`Extracting mark from ${path.relative(root, designSource)} (no text)…`);
  const markBuffer = await extractMarkBuffer(designSource);
  const master = await buildMaster(markBuffer);
  await writePng(master.clone(), masterPath);
  // Keep source-app-icon.png as design original; ship tight master separately

  const icon = sharp(markBuffer);

  console.log("Generating PWA / web assets…");
  const brandingPublic = path.join(publicDir, "branding");
  ensureDir(brandingPublic);
  ensureDir(path.join(publicDir, "icons"));

  const square1024 = await buildMaster(markBuffer);
  await writePng(square1024.clone(), path.join(brandingPublic, "app-icon-1024.png"));
  await writePng(
    (await composeIcon(markBuffer, 32, ICON_FILL, LAUNCHER_BG)).clone(),
    path.join(publicDir, "favicon.png"),
  );
  await writePng(
    (await composeIcon(markBuffer, 16, ICON_FILL, LAUNCHER_BG)).clone(),
    path.join(publicDir, "favicon-16.png"),
  );
  await writePng(
    (await composeIcon(markBuffer, 180, ICON_FILL, LAUNCHER_BG)).clone(),
    path.join(publicDir, "apple-touch-icon.png"),
  );
  await writePng(
    (await composeIcon(markBuffer, 192, ICON_FILL, LAUNCHER_BG)).clone(),
    path.join(publicDir, "icons/icon-192.png"),
  );
  await writePng(
    (await composeIcon(markBuffer, 512, ICON_FILL, LAUNCHER_BG)).clone(),
    path.join(publicDir, "icons/icon-512.png"),
  );
  await writePng(
    (await composeIcon(markBuffer, 512, MASKABLE_FILL, LAUNCHER_BG)).clone(),
    path.join(publicDir, "icons/icon-512-maskable.png"),
  );

  console.log("iOS App Icon + Splash…");
  await writePng(square1024.clone(), path.join(iosAppIcon, "AppIcon-512@2x.png"));

  const splash2732 = nativeLaunchSplash(2732, 2732);
  for (const name of ["splash-2732x2732.png", "splash-2732x2732-1.png", "splash-2732x2732-2.png"]) {
    await writePng(splash2732.clone(), path.join(iosSplash, name));
  }

  console.log("Android mipmaps + splash…");
  for (const [folder, size] of Object.entries(MIPMAP_LAUNCHER)) {
    const dir = path.join(androidRes, folder);
    await writePng(
      (await composeIcon(markBuffer, size, ICON_FILL, LAUNCHER_BG)).clone(),
      path.join(dir, "ic_launcher.png"),
    );
    await writePng(
      (await composeIcon(markBuffer, size, ICON_FILL, LAUNCHER_BG)).clone(),
      path.join(dir, "ic_launcher_round.png"),
    );
  }

  for (const [folder, size] of Object.entries(MIPMAP_FOREGROUND)) {
    const dir = path.join(androidRes, folder);
    await writePng(
      (await adaptiveForeground(markBuffer, size)).clone(),
      path.join(dir, "ic_launcher_foreground.png"),
    );
  }

  ensureDir(path.join(androidRes, "drawable"));
  await writePng(nativeLaunchSplash(1280, 1280).clone(), path.join(androidRes, "drawable", "splash.png"));

  await generatePreviews(markBuffer, square1024);

  console.log("Done.");
  console.log(`  Master (no text, ${ICON_FILL * 100}% fill): ${path.relative(root, masterPath)}`);
  console.log(`  Previews: ${path.relative(root, previewsDir)}/`);
  console.log("  Run: npm run cap:sync");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

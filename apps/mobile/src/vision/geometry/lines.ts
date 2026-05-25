/** Line helpers — simplified port of LC2FEN SLID + intersections. */

export type Segment = [[number, number], [number, number]];

export function segmentIntersection(
  a: Segment,
  b: Segment,
): [number, number] | null {
  const [[x1, y1], [x2, y2]] = a;
  const [[x3, y3], [x4, y4]] = b;
  const denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (Math.abs(denom) < 1e-6) return null;
  const px =
    ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom;
  const py =
    ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom;
  const within = (x: number, lo: number, hi: number) => x >= lo - 2 && x <= hi + 2;
  if (
    !within(px, Math.min(x1, x2), Math.max(x1, x2)) ||
    !within(py, Math.min(y1, y2), Math.max(y1, y2)) ||
    !within(px, Math.min(x3, x4), Math.max(x3, x4)) ||
    !within(py, Math.min(y3, y4), Math.max(y3, y4))
  ) {
    return null;
  }
  return [px, py];
}

export function clusterPoints(points: [number, number][], maxDist = 10): [number, number][] {
  const clusters: [number, number][][] = [];
  for (const p of points) {
    let found = false;
    for (const c of clusters) {
      const [cx, cy] = c[0];
      if (Math.hypot(cx - p[0], cy - p[1]) < maxDist) {
        c.push(p);
        found = true;
        break;
      }
    }
    if (!found) clusters.push([p]);
  }
  return clusters.map((c) => {
    const x = c.reduce((s, p) => s + p[0], 0) / c.length;
    const y = c.reduce((s, p) => s + p[1], 0) / c.length;
    return [x, y] as [number, number];
  });
}

export function findLineIntersections(segments: Segment[], maxPoints = 400): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i < segments.length; i += 1) {
    for (let j = i + 1; j < segments.length; j += 1) {
      const hit = segmentIntersection(segments[i], segments[j]);
      if (hit) pts.push(hit);
      if (pts.length >= maxPoints) return pts;
    }
  }
  return pts;
}

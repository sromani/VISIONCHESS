import type { Point2 } from "./types";

/** Port of vision/board/corners.order_points_y_sort — TL, TR, BR, BL. */
export function orderPointsYSort(points: Point2[]): Point2[] {
  if (points.length !== 4) {
    throw new Error(`orderPointsYSort expects 4 points, got ${points.length}`);
  }
  const sorted = [...points].sort((a, b) => a.y - b.y);
  const top = sorted.slice(0, 2).sort((a, b) => a.x - b.x);
  const bottom = sorted.slice(2, 4).sort((a, b) => a.x - b.x);
  return [top[0], top[1], bottom[1], bottom[0]];
}

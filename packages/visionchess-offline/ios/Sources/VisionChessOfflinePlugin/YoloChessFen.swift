import Foundation

/// Port of `yolo_pieces._fen_from_assignments` + square assignment.
enum YoloChessFen {

    private static let labelToFen: [String: String] = [
        "white_pawn": "P", "white_knight": "N", "white_bishop": "B", "white_rook": "R",
        "white_queen": "Q", "white_king": "K",
        "black_pawn": "p", "black_knight": "n", "black_bishop": "b", "black_rook": "r",
        "black_queen": "q", "black_king": "k",
    ]

    static func assignSquares(_ detections: [YoloPieceDetection], boardSize: Int) -> [YoloPieceDetection] {
        let cell = Float(boardSize) / 8.0
        return detections.map { det in
            let cx = Float(det.bbox.x) + Float(det.bbox.w) / 2
            let cy = Float(det.bbox.y) + Float(det.bbox.h) / 2
            let col = min(7, max(0, Int(cx / cell)))
            let row = min(7, max(0, Int(cy / cell)))
            let rank = 8 - row
            let file = Character(UnicodeScalar(97 + col)!)
            var d = det
            d.squareName = "\(file)\(rank)"
            return d
        }
    }

    static func onePerSquare(_ detections: [YoloPieceDetection]) -> [String: YoloPieceDetection] {
        var best = [String: YoloPieceDetection]()
        for det in detections {
            guard let sq = det.squareName else { continue }
            if let prev = best[sq] {
                if det.confidence > prev.confidence { best[sq] = det }
            } else {
                best[sq] = det
            }
        }
        return best
    }

    static func placementFen(assigned: [String: YoloPieceDetection]) -> String {
        var ranks = [String]()
        for rank in (1...8).reversed() {
            var row = ""
            var empty = 0
            for fileIdx in 0..<8 {
                let name = "\(Character(UnicodeScalar(97 + fileIdx)!))\(rank)"
                let symbol = assigned[name].flatMap { labelToFen[$0.label] }
                if symbol == nil {
                    empty += 1
                } else {
                    if empty > 0 {
                        row += String(empty)
                        empty = 0
                    }
                    row += symbol!
                }
            }
            if empty > 0 { row += String(empty) }
            ranks.append(row)
        }
        return ranks.joined(separator: "/")
    }

    static func buildSquaresPayload(assigned: [String: YoloPieceDetection], boardSize: Int) -> [[String: Any]] {
        var out = [[String: Any]]()
        for rank in (1...8).reversed() {
            for fileIdx in 0..<8 {
                let name = "\(Character(UnicodeScalar(97 + fileIdx)!))\(rank)"
                let det = assigned[name]
                let label = det?.label ?? "empty"
                let conf = det?.confidence ?? 0
                let occupied = det != nil && label != "empty" && label != "piece"
                let bbox: [Int] = det.map { [$0.bbox.x, $0.bbox.y, $0.bbox.w, $0.bbox.h] } ?? [0, 0, 0, 0]
                out.append([
                    "name": name,
                    "label": label,
                    "confidence": conf,
                    "occupied": occupied,
                    "bbox": bbox,
                ])
            }
        }
        return out
    }
}

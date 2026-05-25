import Foundation
import UIKit

/// Mirrors `apps/api/vision/inference/yolo_detector.py` + `yolov8_chess_pieces.json`.
struct YoloChessConfig {
    let inputSize: Int = 640
    let inputName: String = "images"
    let outputName: String = "output0"
    let confThreshold: Float
    let iouThreshold: Float = 0.45
    let skipClasses: Set<String> = ["board"]
    let localizationOnly: Bool = false
    let boardClassIndex: Int = 0
    let coordsNormalized: Bool = true
    let maxBoxRatio: Float = 0.32
    let minBoxPx: Int = 18
    let classNames: [String] = [
        "board",
        "white_king", "white_queen", "white_rook", "white_bishop", "white_knight", "white_pawn",
        "black_king", "black_queen", "black_rook", "black_bishop", "black_knight", "black_pawn",
    ]

    static func pieceClassifier(confThreshold: Float = 0.30) -> YoloChessConfig {
        YoloChessConfig(confThreshold: confThreshold)
    }
}

struct YoloPieceDetection {
    let label: String
    let confidence: Float
    let bbox: (x: Int, y: Int, w: Int, h: Int)
    var squareName: String?
}

struct YoloRecognizeTimings {
    var preprocessMs: Int = 0
    var inferenceMs: Int = 0
    var postprocessMs: Int = 0
    var totalMs: Int = 0
}

struct YoloRecognizeResult {
    let placementFen: String
    let squares: [[String: Any]]
    let detections: [[String: Any]]
    let timings: YoloRecognizeTimings
    let overlayJpegBase64: String
    let logLines: [String]
}

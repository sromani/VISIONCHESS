import Foundation

/// Port of `yolo_detector._postprocess`, `_reshape_predictions`, `_nms`.
enum YoloChessPostprocess {

    static func decode(
        output: [Float],
        shape: [Int],
        origW: Int,
        origH: Int,
        scale: Float,
        config: YoloChessConfig
    ) -> [YoloPieceDetection] {
        guard let pred = reshapePredictions(output: output, shape: shape),
              !pred.isEmpty,
              pred[0].count >= 5 else {
            return []
        }

        let numRows = pred.count
        var boxes = pred.map { Array($0[0..<4]) }
        let numClasses = pred[0].count - 4

        var classIds = [Int]()
        var confidences = [Float]()
        var labels = [String]()

        let coordsNorm = config.coordsNormalized || (boxes.flatMap { $0 }.max() ?? 0) <= 1.5
        if coordsNorm {
            for i in 0..<numRows {
                for j in 0..<4 { boxes[i][j] *= Float(config.inputSize) }
            }
        }

        if config.localizationOnly && numClasses > 1 && config.boardClassIndex >= 0 {
            for i in 0..<numRows {
                var bestConf: Float = 0
                var bestId = 0
                for c in 0..<numClasses {
                    if c == config.boardClassIndex { continue }
                    let sc = pred[i][4 + c]
                    if sc > bestConf {
                        bestConf = sc
                        bestId = c
                    }
                }
                var mapped = bestId
                if mapped >= config.boardClassIndex { mapped += 1 }
                classIds.append(mapped)
                confidences.append(bestConf)
            }
        } else {
            for i in 0..<numRows {
                var bestConf: Float = -1
                var bestId = 0
                for c in 0..<numClasses {
                    let sc = pred[i][4 + c]
                    if sc > bestConf {
                        bestConf = sc
                        bestId = c
                    }
                }
                classIds.append(bestId)
                confidences.append(bestConf)
            }
        }

        for i in 0..<numRows {
            let cid = classIds[i]
            if cid < config.classNames.count {
                labels.append(config.classNames[cid])
            } else {
                labels.append("class_\(cid)")
            }
        }

        var keptIndices = [Int]()
        for i in 0..<numRows {
            if labels[i] == "board" || config.skipClasses.contains(labels[i]) { continue }
            if confidences[i] < config.confThreshold { continue }
            keptIndices.append(i)
        }

        if keptIndices.isEmpty { return [] }

        var boxesXYXY = [[Float]]()
        var keptConf = [Float]()
        var keptLabels = [String]()
        for i in keptIndices {
            let b = boxes[i]
            let x1 = b[0] - b[2] / 2
            let y1 = b[1] - b[3] / 2
            let x2 = b[0] + b[2] / 2
            let y2 = b[1] + b[3] / 2
            boxesXYXY.append([
                max(0, min(Float(origW), x1 / scale)),
                max(0, min(Float(origH), y1 / scale)),
                max(0, min(Float(origW), x2 / scale)),
                max(0, min(Float(origH), y2 / scale)),
            ])
            keptConf.append(confidences[i])
            keptLabels.append(labels[i])
        }

        let maxW = Float(origW) * config.maxBoxRatio
        let maxH = Float(origH) * config.maxBoxRatio
        var sizeFiltered = [Int]()
        for (idx, box) in boxesXYXY.enumerated() {
            let bw = box[2] - box[0]
            let bh = box[3] - box[1]
            if bw >= Float(config.minBoxPx) && bh >= Float(config.minBoxPx) && bw <= maxW && bh <= maxH {
                sizeFiltered.append(idx)
            }
        }

        if sizeFiltered.isEmpty { return [] }

        var sfBoxes = [[Float]]()
        var sfConf = [Float]()
        var sfLabels = [String]()
        for idx in sizeFiltered {
            sfBoxes.append(boxesXYXY[idx])
            sfConf.append(keptConf[idx])
            sfLabels.append(keptLabels[idx])
        }

        let nmsKeep = nms(boxes: sfBoxes, scores: sfConf, iouThreshold: config.iouThreshold)
        var detections = [YoloPieceDetection]()
        for idx in nmsKeep {
            let box = sfBoxes[idx]
            let x1 = Int(box[0])
            let y1 = Int(box[1])
            let w = max(1, Int(box[2] - box[0]))
            let h = max(1, Int(box[3] - box[1]))
            let label = config.localizationOnly ? "piece" : sfLabels[idx]
            detections.append(YoloPieceDetection(
                label: label,
                confidence: sfConf[idx],
                bbox: (x1, y1, w, h)
            ))
        }

        detections.sort { $0.confidence > $1.confidence }
        return detections
    }

    private static func reshapePredictions(output: [Float], shape: [Int]) -> [[Float]]? {
        guard shape.count >= 2 else { return nil }
        var dims = shape
        if dims.first == 1 { dims.removeFirst() }
        let rows = dims.count >= 2 ? dims[0] : 0
        let cols = dims.count >= 2 ? dims[1] : 0
        guard rows > 0, cols > 0 else { return nil }

        var matrix = [[Float]](repeating: [Float](repeating: 0, count: cols), count: rows)
        var idx = 0
        for r in 0..<rows {
            for c in 0..<cols {
                if idx < output.count { matrix[r][c] = output[idx] }
                idx += 1
            }
        }
        if rows <= 64 && cols > rows {
            return transpose(matrix)
        }
        if cols <= 64 && rows > cols {
            return transpose(matrix)
        }
        return transpose(matrix)
    }

    private static func transpose(_ m: [[Float]]) -> [[Float]] {
        guard let cols = m.first?.count, cols > 0 else { return [] }
        var out = Array(repeating: Array(repeating: Float(0), count: m.count), count: cols)
        for (r, row) in m.enumerated() {
            for (c, v) in row.enumerated() {
                out[c][r] = v
            }
        }
        return out
    }

    private static func nms(boxes: [[Float]], scores: [Float], iouThreshold: Float) -> [Int] {
        guard !boxes.isEmpty else { return [] }
        let order = (0..<scores.count).sorted { scores[$1] > scores[$0] }
        var keep = [Int]()
        var remaining = order
        while !remaining.isEmpty {
            let i = remaining[0]
            keep.append(i)
            if remaining.count == 1 { break }
            let rest = Array(remaining.dropFirst())
            var next = [Int]()
            let bi = boxes[i]
            let areaI = (bi[2] - bi[0]) * (bi[3] - bi[1])
            for j in rest {
                let bj = boxes[j]
                let xx1 = max(bi[0], bj[0])
                let yy1 = max(bi[1], bj[1])
                let xx2 = min(bi[2], bj[2])
                let yy2 = min(bi[3], bj[3])
                let w = max(0, xx2 - xx1)
                let h = max(0, yy2 - yy1)
                let inter = w * h
                let areaJ = (bj[2] - bj[0]) * (bj[3] - bj[1])
                let iou = inter / (areaI + areaJ - inter + 1e-6)
                if iou <= iouThreshold {
                    next.append(j)
                }
            }
            remaining = next
        }
        return keep
    }
}

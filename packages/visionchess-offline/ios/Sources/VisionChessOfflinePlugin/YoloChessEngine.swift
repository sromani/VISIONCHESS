import Foundation
import onnxruntime_objc

/// ONNX Runtime session — mirrors `YoloPieceDetector` (piece classifier mode).
final class YoloChessEngine {
    static let shared = YoloChessEngine()

    private var env: ORTEnv?
    private var session: ORTSession?
    private var inputName: String = "images"
    private var outputName: String = "output0"
    private let lock = NSLock()

    private init() {}

    func ensureSession(log: inout [String]) throws {
        lock.lock()
        defer { lock.unlock() }
        if session != nil { return }

        let t0 = CFAbsoluteTimeGetCurrent()
        env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)

        let modelPath = try Self.resolveModelPath()
        log.append("model_path=\(modelPath)")

        let options = try ORTSessionOptions()
        do {
            try options.appendCoreMLExecutionProvider(with: ORTCoreMLExecutionProviderOptions())
            log.append("execution_provider=CoreML")
        } catch {
            log.append("execution_provider=CPU (CoreML append failed: \(error.localizedDescription))")
        }

        guard let env else { throw YoloEngineError.message("ORTEnv nil") }
        session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)

        let names = try session!.inputNames()
        if let first = names.first { inputName = first }
        let outs = try session!.outputNames()
        if let first = outs.first { outputName = first }

        let ms = Int((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        log.append("session_load_ms=\(ms) inputs=\(names) outputs=\(outs)")
    }

    func detect(image: UIImage, config: YoloChessConfig, log: inout [String]) throws -> YoloRecognizeResult {
        try ensureSession(log: &log)

        guard let session else { throw YoloEngineError.message("ORT session not loaded") }

        let totalStart = CFAbsoluteTimeGetCurrent()
        var timings = YoloRecognizeTimings()

        let preStart = CFAbsoluteTimeGetCurrent()
        let (blob, scale, origW, origH) = YoloChessImage.letterboxRGB(image: image, inputSize: config.inputSize)
        timings.preprocessMs = Int((CFAbsoluteTimeGetCurrent() - preStart) * 1000)
        log.append("preprocess_ms=\(timings.preprocessMs) orig=\(origW)x\(origH) scale=\(scale)")

        let shape: [NSNumber] = [1, 3, NSNumber(value: config.inputSize), NSNumber(value: config.inputSize)]
        let inputData = NSMutableData(bytes: blob, length: blob.count * MemoryLayout<Float>.size)
        let inputValue = try ORTValue(
            tensorData: inputData,
            elementType: ORTTensorElementDataType.float,
            shape: shape
        )

        let inferStart = CFAbsoluteTimeGetCurrent()
        let outputs = try session.run(
            withInputs: [inputName: inputValue],
            outputNames: Set([outputName]),
            runOptions: nil
        )
        timings.inferenceMs = Int((CFAbsoluteTimeGetCurrent() - inferStart) * 1000)
        log.append("inference_ms=\(timings.inferenceMs)")

        guard let outTensor = outputs[outputName] else {
            throw YoloEngineError.message("Missing output \(outputName)")
        }

        let postStart = CFAbsoluteTimeGetCurrent()
        let typeShape = try outTensor.tensorTypeAndShapeInfo()
        var outShape = typeShape.shape.map { $0.intValue }
        if outShape.isEmpty { outShape = [1, 17, 8400] }

        let rawData = try outTensor.tensorData() as Data
        let floatCount = rawData.count / MemoryLayout<Float>.size
        let floats: [Float] = rawData.withUnsafeBytes { ptr in
            let bind = ptr.bindMemory(to: Float.self)
            return Array(bind.prefix(floatCount))
        }
        log.append("output_shape=\(outShape) floats=\(floats.count)")

        var detections = YoloChessPostprocess.decode(
            output: floats,
            shape: outShape,
            origW: origW,
            origH: origH,
            scale: scale,
            config: config
        )
        log.append("raw_detections=\(detections.count)")

        let boardSize = min(origW, origH)
        detections = YoloChessFen.assignSquares(detections, boardSize: boardSize)
        let assigned = YoloChessFen.onePerSquare(detections)
        let placement = YoloChessFen.placementFen(assigned: assigned)
        timings.postprocessMs = Int((CFAbsoluteTimeGetCurrent() - postStart) * 1000)

        let overlay = YoloChessOverlay.draw(image: image, detections: Array(assigned.values), boardSize: boardSize)
        let overlayB64 = YoloChessOverlay.jpegBase64(overlay)

        timings.totalMs = Int((CFAbsoluteTimeGetCurrent() - totalStart) * 1000)
        log.append("postprocess_ms=\(timings.postprocessMs) assigned=\(assigned.count) fen=\(placement)")

        let detPayload: [[String: Any]] = detections.map { d in
            [
                "label": d.label,
                "confidence": d.confidence,
                "square": d.squareName ?? "?",
                "bbox": [d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h],
            ]
        }

        return YoloRecognizeResult(
            placementFen: placement,
            squares: YoloChessFen.buildSquaresPayload(assigned: assigned, boardSize: boardSize),
            detections: detPayload,
            timings: timings,
            overlayJpegBase64: overlayB64,
            logLines: log
        )
    }

    private static func resolveModelPath() throws -> String {
        let bundle = Bundle(for: YoloChessEngine.self)
        if let url = bundle.url(forResource: "VisionchessOfflineResources", withExtension: "bundle"),
           let resBundle = Bundle(url: url),
           let model = resBundle.path(forResource: "yolov8_chess_pieces", ofType: "onnx") {
            return model
        }
        if let model = bundle.path(forResource: "yolov8_chess_pieces", ofType: "onnx") {
            return model
        }
        throw YoloEngineError.message(
            "yolov8_chess_pieces.onnx not in bundle. Run: node apps/mobile/scripts/copy-offline-models.cjs"
        )
    }
}

enum YoloEngineError: LocalizedError {
    case message(String)
    var errorDescription: String? {
        switch self {
        case .message(let m): return m
        }
    }
}

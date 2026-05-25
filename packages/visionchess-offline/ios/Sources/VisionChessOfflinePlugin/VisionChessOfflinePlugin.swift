import Foundation
import Capacitor
import UIKit

@objc(VisionChessOfflinePlugin)
public class VisionChessOfflinePlugin: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "VisionChessOfflinePlugin"
    public let jsName = "VisionChessOffline"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "recognizeFromWarpedJpeg", returnType: CAPPluginReturnPromise),
    ]

    @objc func recognizeFromWarpedJpeg(_ call: CAPPluginCall) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                guard let b64 = call.getString("jpegBase64"), !b64.isEmpty else {
                    call.reject("jpegBase64 required")
                    return
                }
                let width = call.getInt("width") ?? 512
                let height = call.getInt("height") ?? 512
                let conf = call.getFloat("confThreshold") ?? 0.30

                guard let data = Data(base64Encoded: b64),
                      let image = UIImage(data: data) else {
                    call.reject("Invalid warped JPEG base64")
                    return
                }

                var logs = [String]()
                logs.append("M1 VisionChessOffline recognizeFromWarpedJpeg")
                logs.append("warped_reported=\(width)x\(height) image_pixels=\(Int(image.size.width))x\(Int(image.size.height))")
                logs.append("conf_threshold=\(conf)")

                let config = YoloChessConfig.pieceClassifier(confThreshold: conf)
                let result = try YoloChessEngine.shared.detect(image: image, config: config, log: &logs)

                var payload = JSObject()
                payload["placementFen"] = result.placementFen
                payload["squares"] = result.squares
                payload["detections"] = result.detections
                payload["timings"] = [
                    "preprocessMs": result.timings.preprocessMs,
                    "inferenceMs": result.timings.inferenceMs,
                    "postprocessMs": result.timings.postprocessMs,
                    "totalMs": result.timings.totalMs,
                ]
                payload["debug"] = [
                    "overlayJpegBase64": result.overlayJpegBase64,
                    "logLines": logs + result.logLines,
                ]

                call.resolve(payload)
            } catch {
                call.reject("Native YOLO failed: \(error.localizedDescription)", nil, error)
            }
        }
    }
}

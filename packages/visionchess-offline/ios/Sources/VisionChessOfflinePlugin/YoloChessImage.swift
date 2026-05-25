import UIKit

/// RGB letterbox preprocess — port of `yolo_detector._preprocess`.
enum YoloChessImage {

    static func letterboxRGB(
        image: UIImage,
        inputSize: Int
    ) -> (data: [Float], scale: Float, width: Int, height: Int) {
        let w = max(1, Int(image.size.width.rounded()))
        let h = max(1, Int(image.size.height.rounded()))
        let scale = Float(inputSize) / Float(max(w, h))
        let newW = max(1, Int((Float(w) * scale).rounded()))
        let newH = max(1, Int((Float(h) * scale).rounded()))

        UIGraphicsBeginImageContextWithOptions(
            CGSize(width: inputSize, height: inputSize),
            true,
            1
        )
        UIColor(red: 114 / 255, green: 114 / 255, blue: 114 / 255, alpha: 1).setFill()
        UIRectFill(CGRect(x: 0, y: 0, width: inputSize, height: inputSize))
        image.draw(in: CGRect(x: 0, y: 0, width: newW, height: newH))
        let padded = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        guard let padded, let cg = padded.cgImage else {
            return ([], scale, w, h)
        }

        let bytesPerRow = inputSize * 4
        var rgba = [UInt8](repeating: 0, count: inputSize * inputSize * 4)
        guard let ctx = CGContext(
            data: &rgba,
            width: inputSize,
            height: inputSize,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return ([], scale, w, h)
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: inputSize, height: inputSize))

        var floats = [Float](repeating: 0, count: 3 * inputSize * inputSize)
        var fi = 0
        for c in 0..<3 {
            for y in 0..<inputSize {
                for x in 0..<inputSize {
                    let px = (y * inputSize + x) * 4
                    floats[fi] = Float(rgba[px + c]) / 255.0
                    fi += 1
                }
            }
        }
        return (floats, scale, w, h)
    }
}

import UIKit

/// Port of `render_yolo_class_overlay` (simplified).
enum YoloChessOverlay {

    static func draw(image: UIImage, detections: [YoloPieceDetection], boardSize: Int) -> UIImage {
        let size = image.size
        let scale = image.scale
        UIGraphicsBeginImageContextWithOptions(size, true, scale)
        image.draw(at: .zero)
        guard let ctx = UIGraphicsGetCurrentContext() else {
            UIGraphicsEndImageContext()
            return image
        }

        let cell = CGFloat(boardSize) / 8.0
        ctx.setLineWidth(max(1, size.width / 256))

        for det in detections {
            let rect = CGRect(
                x: CGFloat(det.bbox.x),
                y: CGFloat(det.bbox.y),
                width: CGFloat(det.bbox.w),
                height: CGFloat(det.bbox.h)
            )
            let isWhite = det.label.hasPrefix("white_")
            ctx.setStrokeColor(isWhite ? UIColor.cyan.cgColor : UIColor.systemPurple.cgColor)
            ctx.stroke(rect)

            if let sq = det.squareName {
                let short = det.label.replacingOccurrences(of: "_", with: " ")
                let text = String(format: "%@ %@ %.2f", sq, String(short.prefix(12)), det.confidence)
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: UIFont.systemFont(ofSize: max(8, cell / 5)),
                    .foregroundColor: UIColor.white,
                ]
                (text as NSString).draw(at: CGPoint(x: rect.minX + 2, y: rect.minY + 2), withAttributes: attrs)
            }
        }

        // 8x8 grid
        ctx.setStrokeColor(UIColor.white.withAlphaComponent(0.35).cgColor)
        ctx.setLineWidth(0.5)
        for i in 1..<8 {
            let p = CGFloat(i) * cell
            ctx.move(to: CGPoint(x: p, y: 0))
            ctx.addLine(to: CGPoint(x: p, y: size.height))
            ctx.move(to: CGPoint(x: 0, y: p))
            ctx.addLine(to: CGPoint(x: size.width, y: p))
        }
        ctx.strokePath()

        let composed = UIGraphicsGetImageFromCurrentImageContext() ?? image
        UIGraphicsEndImageContext()
        return composed
    }

    static func jpegBase64(_ image: UIImage, quality: CGFloat = 0.88) -> String {
        guard let data = image.jpegData(compressionQuality: quality) else { return "" }
        return data.base64EncodedString()
    }
}

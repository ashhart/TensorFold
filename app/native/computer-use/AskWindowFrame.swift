import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import ScreenCaptureKit

func askWindowFrameImage(_ buffer: CVPixelBuffer) -> CGImage? {
    let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA,
        width > 0, height > 0, width <= 1800, height <= 1800,
        stride >= width * 4, stride * height <= 32 * 1024 * 1024 else { return nil }
    guard CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess else { return nil }
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
    let copy = Data(bytes: base, count: stride * height)
    guard let provider = CGDataProvider(data: copy as CFData) else { return nil }
    return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: stride, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
}

@available(macOS 15.2, *)
final class AskWindowFrameCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    private let completion: (CGImage?, Error?) -> Void
    private var stream: SCStream?
    private var timer: Timer?
    private var settled = false
    private var acceptedFrame = false
    private let frames = DispatchQueue(label: "dev.tensorfold.ask-frame")

    init(completion: @escaping (CGImage?, Error?) -> Void) { self.completion = completion }

    func start(filter: SCContentFilter, configuration: SCStreamConfiguration) {
        configuration.pixelFormat = kCVPixelFormatType_32BGRA
        configuration.minimumFrameInterval = CMTime(value: 1, timescale: 2)
        configuration.queueDepth = 2
        let capture = SCStream(filter: filter, configuration: configuration, delegate: self)
        stream = capture
        do { try capture.addStreamOutput(self, type: .screen, sampleHandlerQueue: frames) }
        catch { finish(nil, error, stopped: true); return }
        let timeout = Timer(timeInterval: 15, repeats: false) { [weak self] _ in
            self?.finish(nil, HelperError("window_timeout", "The selected window did not produce a frame."))
        }
        timer = timeout; RunLoop.main.add(timeout, forMode: .common)
        capture.startCapture { error in
            if let error { DispatchQueue.main.async { self.finish(nil, error, stopped: true) } }
        }
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, !acceptedFrame, CMSampleBufferIsValid(sampleBuffer),
            let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
            attachments.first?[.status] as? Int == SCFrameStatus.complete.rawValue,
            let buffer = CMSampleBufferGetImageBuffer(sampleBuffer), let image = askWindowFrameImage(buffer) else { return }
        acceptedFrame = true
        DispatchQueue.main.async { self.finish(image, nil) }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        DispatchQueue.main.async { self.finish(nil, error, stopped: true) }
    }

    private func finish(_ image: CGImage?, _ error: Error?, stopped: Bool = false) {
        guard !settled else { return }
        settled = true; timer?.invalidate()
        guard let stream, !stopped else { completion(image, error); return }
        stream.stopCapture { _ in DispatchQueue.main.async { self.completion(image, error) } }
    }
}

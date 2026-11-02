import AppKit
import Foundation
import ScreenCaptureKit

let askWindowDeniedBundles = [
    "com.apple.SecurityAgent", "com.apple.loginwindow", "com.apple.keychainaccess", "com.apple.Passwords",
    "com.agilebits.onepassword7", "com.1password.1password", "com.bitwarden.desktop",
]

func askWindowKind(app: String, title: String, text: String) -> String {
    let name = app.lowercased(), window = title.lowercased()
    if ["safari", "chrome", "firefox", "edge", "brave", "browser", "arc.app"].contains(where: name.contains) { return "web" }
    if [".png", ".jpg", ".jpeg", ".webp", ".heic", ".tiff"].contains(where: window.hasSuffix)
        || ["photos", "pixelmator", "photoshop", "affinity photo"].contains(where: name.contains) { return "image" }
    if ["textedit", "pages", "notes", "word", "libreoffice", "code", "xcode", "preview"].contains(where: name.contains)
        || text.count > 80 { return "document" }
    return "window"
}

func askWindowPng(_ image: CGImage) throws -> Data {
    var width = min(image.width, 1600), height = max(1, Int(Double(image.height) * Double(width) / Double(image.width)))
    if height > 1100 { width = max(1, width * 1100 / height); height = 1100 }
    for _ in 0..<6 {
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw HelperError("window_image", "The selected window image could not be prepared.")
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        if let scaled = context.makeImage(), let data = NSBitmapImageRep(cgImage: scaled).representation(using: .png, properties: [:]),
            data.count <= 1_700_000 { return data }
        width = max(1, width * 3 / 4); height = max(1, height * 3 / 4)
    }
    throw HelperError("window_image", "The selected window image is too large. Choose a smaller window.")
}

@available(macOS 15.2, *)
final class AskWindowPicker: NSObject, SCContentSharingPickerObserver {
    let ownBundle: String
    private var settled = false
    private var selected = false
    private var timer: Timer?
    private var frameCapture: AskWindowFrameCapture?

    init(ownBundle: String) { self.ownBundle = ownBundle }

    func start() {
        var configuration = SCContentSharingPickerConfiguration()
        configuration.allowedPickerModes = [.singleWindow]
        configuration.excludedBundleIDs = askWindowDeniedBundles + [ownBundle]
        configuration.allowsChangingSelectedContent = false
        let picker = SCContentSharingPicker.shared
        picker.defaultConfiguration = configuration
        picker.maximumStreamCount = 1
        picker.add(self)
        picker.isActive = true
        picker.present(using: .window)
        writeLine(["state": "picking"])
    }

    func finish(_ result: [String: Any]) {
        guard !settled else { return }
        settled = true
        timer?.invalidate()
        let picker = SCContentSharingPicker.shared
        picker.remove(self)
        picker.isActive = false
        writeLine(result)
        exit(0)
    }

    func contentSharingPicker(_ picker: SCContentSharingPicker, didCancelFor stream: SCStream?) {
        DispatchQueue.main.async { self.finish(["cancelled": true]) }
    }

    func contentSharingPickerStartDidFailWithError(_ error: Error) {
        DispatchQueue.main.async { self.finish(["error": "macOS could not open its window picker. Try again."]) }
    }

    func contentSharingPicker(_ picker: SCContentSharingPicker, didUpdateWith filter: SCContentFilter, for stream: SCStream?) {
        DispatchQueue.main.async { self.capture(filter) }
    }

    private func capture(_ filter: SCContentFilter) {
        guard !selected, !settled else { return }
        selected = true
        writeLine(["state": "capturing"])
        let timeout = Timer(timeInterval: 30, repeats: false) { [weak self] _ in
            self?.finish(["error": "The selected window took too long to read. Try again."])
        }
        timer = timeout
        RunLoop.main.add(timeout, forMode: .common)
        guard filter.includedWindows.count == 1, let window = filter.includedWindows.first,
            let owner = window.owningApplication, owner.bundleIdentifier != ownBundle,
            !askWindowDeniedBundles.contains(owner.bundleIdentifier), owner.processID != getpid() else {
            finish(["error": "Select one ordinary app window. TensorFold and protected system windows cannot be attached."])
            return
        }
        let rect = filter.contentRect
        guard rect.width > 0, rect.height > 0, rect.width <= 16384, rect.height <= 16384 else {
            finish(["error": "The selected window has unsupported dimensions."]); return
        }
        let scale = min(1, 1800 / max(rect.width, rect.height))
        let configuration = SCStreamConfiguration()
        configuration.width = max(1, Int(rect.width * scale))
        configuration.height = max(1, Int(rect.height * scale))
        configuration.showsCursor = false
        configuration.capturesAudio = false
        configuration.ignoreShadowsSingleWindow = true
        let capture = AskWindowFrameCapture { image, _ in
            DispatchQueue.main.async {
                guard !self.settled else { return }
                guard let image else { self.finish(["error": "The selected window could not be read. It may have closed or declined capture."]); return }
                do {
                    let text = String((try? recognizeAttachmentImage(image).text) ?? "").prefix(16_000)
                    let png = try askWindowPng(image)
                    self.finish(["app": String(owner.applicationName.prefix(100)), "title": String((window.title ?? "Selected window").prefix(260)),
                        "kind": askWindowKind(app: owner.applicationName, title: window.title ?? "", text: String(text)),
                        "text": String(text), "png": png.base64EncodedString(), "method": "visible-window-ocr"])
                } catch { self.finish(["error": "The selected window could not be prepared. Try a smaller window."]) }
            }
        }
        frameCapture = capture
        capture.start(filter: filter, configuration: configuration)
    }
}

func runAskWindowPicker(_ ownBundle: String) -> Never {
    if #available(macOS 15.2, *) {
        let application = NSApplication.shared
        application.setActivationPolicy(.accessory)
        let picker = AskWindowPicker(ownBundle: ownBundle)
        DispatchQueue.main.async { picker.start() }
        withExtendedLifetime(picker) { application.run() }
    }
    writeLine(["error": "The native window picker needs macOS 15.2 or later."])
    exit(0)
}

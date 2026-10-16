import AppKit
import ApplicationServices
import CoreGraphics
import ScreenCaptureKit
import Vision

// This protocol has no app, PID, window, path, key-combination or desktop
// target argument. Only Apple's system iPhone Mirroring app can be pinned.
let phoneBundleId = "com.apple.ScreenContinuity"
let phoneAppURL = URL(fileURLWithPath: "/System/Applications/iPhone Mirroring.app")
let phoneMaxImageBytes = 8 * 1024 * 1024

final class PhoneSession {
    let token = UUID().uuidString
    let pid: pid_t
    let windowId: CGWindowID
    var observedFrame: CGRect?
    var observedSize: CGSize?
    var observedAt: Date?
    init(pid: pid_t, windowId: CGWindowID) { self.pid = pid; self.windowId = windowId }
}
var phoneSession: PhoneSession?

func phoneLocalPoint(_ point: CGPoint, size: CGSize) -> Bool {
    point.x.isFinite && point.y.isFinite && size.width.isFinite && size.height.isFinite && size.width >= 1 && size.height >= 1 && point.x >= 0 && point.y >= 0 && point.x < size.width && point.y < size.height
}
func phoneTypeAllowed(_ text: String) -> Bool {
    text.unicodeScalars.allSatisfy { scalar in
        scalar == "\n" || scalar == "\t" || scalar.value >= 32 && scalar.value != 127 && !(0xE000...0xF8FF).contains(scalar.value)
    }
}
func phoneContentRect(_ frame: CGRect) -> CGRect? {
    // Keep native window controls and the hidden Mac toolbar outside the image
    // and pointer area. These conservative insets also exclude the bezel edge.
    guard frame.width >= 200, frame.height >= 200, frame.width <= 4096, frame.height <= 4096 else { return nil }
    return CGRect(x: frame.minX + 10, y: frame.minY + 48, width: frame.width - 20, height: frame.height - 58)
}
func phoneProtectedText(_ text: String) -> Bool {
    let value = text.lowercased()
    return ["enter passcode", "enter your passcode", "iphone is locked", "iphone unavailable", "security lockout", "unlock your iphone", "unlock iphone", "enter your password", "apple account password", "mac login password", "touch id to", "get started with iphone mirroring", "unable to connect to iphone", "iphone in use", "mirroring has ended", "mirroring paused", "connect your iphone", "iphone not found"].contains { value.contains($0) }
}

func phoneStatus() -> [String: Any] {
    let supported: Bool
    if #available(macOS 15.0, *) { supported = FileManager.default.fileExists(atPath: phoneAppURL.path) } else { supported = false }
    return ["available": supported, "accessibility": AXIsProcessTrusted(), "screenRecording": CGPreflightScreenCaptureAccess()]
}
func phoneApp(_ pid: pid_t? = nil) -> NSRunningApplication? {
    let apps = pid.flatMap { NSRunningApplication(processIdentifier: $0) }.map { [$0] } ?? NSRunningApplication.runningApplications(withBundleIdentifier: phoneBundleId)
    return apps.first { !$0.isTerminated && $0.bundleIdentifier == phoneBundleId && $0.bundleURL?.standardizedFileURL == phoneAppURL.standardizedFileURL }
}
func phoneWindowInfo(_ session: PhoneSession? = nil) -> (pid_t, CGWindowID, CGRect)? {
    guard let app = phoneApp(session?.pid), let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] else { return nil }
    let matches = list.compactMap { info -> (pid_t, CGWindowID, CGRect)? in
        guard (info[kCGWindowOwnerPID as String] as? NSNumber)?.int32Value == app.processIdentifier,
              (info[kCGWindowLayer as String] as? NSNumber)?.intValue == 0,
              let number = info[kCGWindowNumber as String] as? NSNumber,
              let bounds = info[kCGWindowBounds as String] as? NSDictionary,
              let frame = CGRect(dictionaryRepresentation: bounds as CFDictionary), phoneContentRect(frame) != nil,
              session == nil || number.uint32Value == session?.windowId else { return nil }
        return (app.processIdentifier, number.uint32Value, frame)
    }
    return matches.max { $0.2.width * $0.2.height < $1.2.width * $1.2.height }
}
func requirePhone(_ token: String, fresh: Bool = false) throws -> (PhoneSession, CGRect) {
    guard let session = phoneSession, session.token == token else { throw HelperError("phone_disconnected", "The iPhone connection has ended; connect again.") }
    guard let (_, _, frame) = phoneWindowInfo(session) else {
        phoneSession = nil
        throw HelperError("phone_disconnected", "The pinned iPhone Mirroring window closed or disappeared; reconnect it.")
    }
    if fresh {
        guard let observed = session.observedFrame, let at = session.observedAt, Date().timeIntervalSince(at) < 30, observed == frame else {
            throw HelperError("phone_stale", "The iPhone window moved, resized, or needs a fresh screen; refresh before acting.")
        }
    }
    return (session, frame)
}
func phoneConnect() throws -> [String: Any] {
    guard #available(macOS 15.0, *), FileManager.default.fileExists(atPath: phoneAppURL.path) else { throw HelperError("phone_unsupported", "iPhone Mirroring requires macOS 15 or later and Apple's supported hardware and region.") }
    try requireAccessibility()
    guard CGPreflightScreenCaptureAccess() else { throw HelperError("no_screen_recording", "Allow Screen Recording in macOS Settings before connecting iPhone Mirroring.") }
    guard phoneSession == nil else { throw HelperError("phone_busy", "Disconnect the current iPhone Mirroring session before reconnecting.") }
    guard phoneApp() != nil else { throw HelperError("phone_setup", "Open Apple's iPhone Mirroring app and complete its pairing and authentication, then connect again.") }
    guard let (pid, id, _) = phoneWindowInfo() else { throw HelperError("phone_setup", "Show the iPhone screen in Apple's iPhone Mirroring window, then connect again.") }
    let session = PhoneSession(pid: pid, windowId: id)
    phoneSession = session
    return ["token": session.token, "label": "iPhone", "transport": "wifi"]
}
func phoneDisconnect(_ token: String) -> [String: Any] {
    if phoneSession?.token == token { phoneSession = nil }
    return [:]
}

final class PhoneWait<T> {
    private let lock = NSLock()
    private var result: Result<T, Error>?
    func finish(_ value: T?, _ error: Error?) {
        lock.lock(); defer { lock.unlock() }
        if let value { result = .success(value) } else { result = .failure(error ?? HelperError("phone_capture", "iPhone Mirroring could not provide its window image.")) }
    }
    func wait(_ seconds: Double) throws -> T {
        let deadline = Date().addingTimeInterval(seconds)
        while Date() < deadline {
            lock.lock(); let current = result; lock.unlock()
            if let current {
                switch current {
                case .success(let value): return value
                case .failure: throw HelperError("phone_capture", "The pinned iPhone window could not be captured; check Screen Recording permission and the Mirroring connection.")
                }
            }
            pump(0.02)
        }
        throw HelperError("phone_timeout", "The pinned iPhone window did not respond in time.")
    }
}
@available(macOS 15.0, *)
func capturePhoneWindow(_ session: PhoneSession, frame: CGRect) throws -> CGImage {
    guard CGPreflightScreenCaptureAccess() else { throw HelperError("no_screen_recording", "Allow Screen Recording in macOS Settings to view the iPhone window.") }
    let contentWait = PhoneWait<SCShareableContent>()
    SCShareableContent.getExcludingDesktopWindows(true, onScreenWindowsOnly: true) { content, error in contentWait.finish(content, error) }
    let content = try contentWait.wait(4)
    guard let window = content.windows.first(where: { $0.windowID == session.windowId && $0.owningApplication?.processID == session.pid && $0.owningApplication?.bundleIdentifier == phoneBundleId }), window.frame == frame else { throw HelperError("phone_stale", "The pinned iPhone window changed during capture; refresh it.") }
    let filter = SCContentFilter(desktopIndependentWindow: window)
    let options = SCStreamConfiguration()
    options.width = Int(frame.width.rounded())
    options.height = Int(frame.height.rounded())
    options.ignoreShadowsSingleWindow = true
    options.showsCursor = false
    options.capturesAudio = false
    let imageWait = PhoneWait<CGImage>()
    SCScreenshotManager.captureImage(contentFilter: filter, configuration: options) { image, error in imageWait.finish(image, error) }
    let image = try imageWait.wait(4)
    guard image.width > 0, image.height > 0, image.width <= 4096, image.height <= 4096,
          let clip = phoneContentRect(frame) else { throw HelperError("phone_capture", "The iPhone window image has unsupported dimensions.") }
    let scaleX = CGFloat(image.width) / frame.width, scaleY = CGFloat(image.height) / frame.height
    let crop = CGRect(x: (clip.minX - frame.minX) * scaleX, y: (clip.minY - frame.minY) * scaleY, width: clip.width * scaleX, height: clip.height * scaleY).integral
    guard let cropped = image.cropping(to: crop) else { throw HelperError("phone_capture", "The iPhone window image could not be cropped.") }
    return cropped
}

func phoneObserve(_ token: String) throws -> [String: Any] {
    guard #available(macOS 15.0, *) else { throw HelperError("phone_unsupported", "iPhone Mirroring requires macOS 15 or later.") }
    try requireAccessibility()
    let (session, frame) = try requirePhone(token)
    // No app-wide screenshot, main-display fallback, or desktop crop is used.
    let image = try capturePhoneWindow(session, frame: frame)
    _ = try requirePhone(token)
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    let handler = VNImageRequestHandler(cgImage: image)
    do { try handler.perform([request]) } catch { throw HelperError("phone_ocr", "The iPhone window text could not be read; refresh it.") }
    let recognized = request.results ?? []
    let text = recognized.compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
    guard !phoneProtectedText(text) else {
        session.observedAt = nil
        throw HelperError("phone_secure", "Complete the iPhone lock, authentication or Mirroring setup directly, then refresh its screen.")
    }
    let current = try requirePhone(token).1
    guard current == frame else { throw HelperError("phone_stale", "The iPhone window changed during capture; refresh it.") }
    let bitmap = NSBitmapImageRep(cgImage: image)
    guard let png = bitmap.representation(using: .png, properties: [:]), png.count <= phoneMaxImageBytes else { throw HelperError("phone_capture", "The iPhone window image exceeds the supported size.") }
    let nodes: [[String: Any]] = recognized.prefix(240).compactMap { item in
        guard let candidate = item.topCandidates(1).first else { return nil }
        let box = item.boundingBox
        return ["role": "text", "text": String(candidate.string.prefix(240)), "bounds": ["x": box.minX * CGFloat(image.width), "y": (1 - box.maxY) * CGFloat(image.height), "width": box.width * CGFloat(image.width), "height": box.height * CGFloat(image.height)], "enabled": true, "secure": false]
    }
    session.observedFrame = frame
    session.observedSize = CGSize(width: image.width, height: image.height)
    session.observedAt = Date()
    return ["width": image.width, "height": image.height, "base64": png.base64EncodedString(), "nodes": nodes, "truncated": recognized.count > 240, "warnings": ["Mirroring uses image text recognition; controls without labels may require a visual model."]]
}

func phoneAXWindow(_ session: PhoneSession, frame: CGRect) -> AXUIElement? {
    let app = AXUIElementCreateApplication(session.pid)
    let windows = axAttribute(app, kAXWindowsAttribute) as? [AnyObject] ?? []
    return windows.compactMap(axElement).first { elementFrame($0) == frame }
}
func phoneTopWindow(at point: CGPoint) -> CGWindowID? {
    guard let windows = CGWindowListCopyWindowInfo([.optionOnScreenOnly], kCGNullWindowID) as? [[String: Any]] else { return nil }
    return phoneFirstWindow(at: point, windows: windows)
}
func phoneFirstWindow(at point: CGPoint, windows: [[String: Any]]) -> CGWindowID? {
    for info in windows {
        guard let bounds = info[kCGWindowBounds as String] as? NSDictionary,
              let frame = CGRect(dictionaryRepresentation: bounds as CFDictionary), frame.contains(point),
              let id = info[kCGWindowNumber as String] as? NSNumber else { continue }
        if let alpha = info[kCGWindowAlpha as String] as? NSNumber, alpha.doubleValue < 0.01 { continue }
        return id.uint32Value
    }
    return nil
}
func phoneActivate(_ token: String) throws -> (PhoneSession, CGRect) {
    let (session, frame) = try requirePhone(token, fresh: true)
    guard let app = phoneApp(session.pid), let window = phoneAXWindow(session, frame: frame) else { throw HelperError("phone_stale", "The exact iPhone window is no longer available; refresh it.") }
    // Explicit phone input may raise only the exact pinned system app window.
    _ = AXUIElementSetAttributeValue(window, kAXMainAttribute as CFString, kCFBooleanTrue)
    _ = AXUIElementPerformAction(window, kAXRaiseAction as CFString)
    app.activate(options: [.activateIgnoringOtherApps])
    pump(0.15)
    try phoneInputTarget(session, frame: frame)
    return (session, frame)
}
func phoneInputTarget(_ session: PhoneSession, frame: CGRect, point: CGPoint? = nil, typing: Bool = false) throws {
    pump(0.002)
    let (_, liveFrame) = try requirePhone(session.token, fresh: true)
    guard liveFrame == frame, NSWorkspace.shared.frontmostApplication?.processIdentifier == session.pid,
          let focused = axElement(axAttribute(AXUIElementCreateApplication(session.pid), kAXFocusedWindowAttribute)), elementFrame(focused) == frame else { throw HelperError("phone_focus", "Phone input stopped because you changed the active window; refresh before continuing.") }
    guard let content = phoneContentRect(frame), phoneTopWindow(at: point ?? CGPoint(x: content.midX, y: content.midY)) == session.windowId else { throw HelperError("phone_focus", "Phone input stopped because the exact mirrored phone window is covered or no longer focused.") }
    if let point {
        guard let clip = phoneContentRect(frame), clip.contains(point), windowOwner(at: point) == session.pid else { throw HelperError("phone_focus", "Phone input stopped because another window covers that point.") }
    }
    if typing {
        guard let focusedElement = axElement(axAttribute(systemWide, kAXFocusedUIElementAttribute)), elementPid(focusedElement) == session.pid else { throw HelperError("phone_focus", "Focus an editable field on the mirrored phone, then try typing again.") }
        let role = axAttribute(focusedElement, kAXRoleAttribute) as? String ?? ""
        guard !isSecureRole(role, axAttribute(focusedElement, kAXSubroleAttribute) as? String) else { throw HelperError("phone_secure", "Enter secure fields directly on your phone.") }
    }
}
func phoneMouse(_ type: CGEventType, _ point: CGPoint, _ session: PhoneSession) {
    guard let event = CGEvent(mouseEventSource: eventSource, mouseType: type, mouseCursorPosition: point, mouseButton: .left) else { return }
    event.flags = []
    // Process-targeted delivery prevents a race from redirecting an event into
    // another desktop app after the checks above.
    event.postToPid(session.pid)
}
func phoneKey(_ code: CGKeyCode, flags: CGEventFlags, session: PhoneSession) {
    for down in [true, false] {
        guard let event = CGEvent(keyboardEventSource: eventSource, virtualKey: code, keyDown: down) else { continue }
        event.flags = flags
        event.postToPid(session.pid)
    }
}
func phoneGlobalPoint(_ local: CGPoint, size: CGSize, frame: CGRect) throws -> CGPoint {
    guard phoneLocalPoint(local, size: size), let clip = phoneContentRect(frame) else { throw badRequest("Phone coordinates must be inside the latest phone screen.") }
    return CGPoint(x: clip.minX + local.x / size.width * clip.width, y: clip.minY + local.y / size.height * clip.height)
}
func phoneInput(_ token: String, _ input: PhoneInput) throws -> [String: Any] {
    try requireAccessibility()
    let (session, frame) = try phoneActivate(token)
    switch input {
    case .tap(let local, let size):
        guard size == session.observedSize else { throw HelperError("phone_stale", "Use the dimensions of the latest phone screen.") }
        let point = try phoneGlobalPoint(local, size: size, frame: frame)
        try phoneInputTarget(session, frame: frame, point: point)
        phoneMouse(.mouseMoved, point, session)
        phoneMouse(.leftMouseDown, point, session)
        hold(0.03)
        phoneMouse(.leftMouseUp, point, session)
    case .swipe(let from, let to, let size, let duration):
        guard size == session.observedSize else { throw HelperError("phone_stale", "Use the dimensions of the latest phone screen.") }
        let first = try phoneGlobalPoint(from, size: size, frame: frame), last = try phoneGlobalPoint(to, size: size, frame: frame)
        let points = dragPath(from: first, to: last)
        try phoneInputTarget(session, frame: frame, point: first)
        phoneMouse(.mouseMoved, first, session)
        phoneMouse(.leftMouseDown, first, session)
        defer { phoneMouse(.leftMouseUp, last, session) }
        for point in points {
            try phoneInputTarget(session, frame: frame, point: point)
            phoneMouse(.leftMouseDragged, point, session)
            hold(duration / Double(points.count))
        }
    case .type(let text):
        for segment in typeSegments(text) {
            try phoneInputTarget(session, frame: frame, typing: true)
            switch segment {
            case .key(let code): phoneKey(code, flags: [], session: session)
            case .text(let value):
                for units in unicodeChunks(value) {
                    try phoneInputTarget(session, frame: frame, typing: true)
                    for down in [true, false] {
                        guard let event = CGEvent(keyboardEventSource: eventSource, virtualKey: 0, keyDown: down) else { continue }
                        event.flags = []
                        units.withUnsafeBufferPointer { event.keyboardSetUnicodeString(stringLength: $0.count, unicodeString: $0.baseAddress) }
                        event.postToPid(session.pid)
                    }
                    hold(0.008)
                }
            }
        }
    case .home:
        try phoneInputTarget(session, frame: frame)
        phoneKey(18, flags: .maskCommand, session: session) // Apple's documented Command-1 Home shortcut.
    case .enter:
        try phoneInputTarget(session, frame: frame, typing: true)
        phoneKey(36, flags: [], session: session)
    }
    session.observedAt = nil
    return [:]
}

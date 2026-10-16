import AppKit
import ApplicationServices
import CoreGraphics

func permissionsJSON() -> [String: Any] {
    ["accessibility": AXIsProcessTrusted(), "screenRecording": CGPreflightScreenCaptureAccess()]
}

// Each prompt appears only while the permission is missing; macOS never changes a decision from here.
func requestPermissions(accessibility: Bool, screenRecording: Bool) -> [String: Any] {
    if accessibility && !AXIsProcessTrusted() {
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(options)
    }
    if screenRecording && !CGPreflightScreenCaptureAccess() { _ = CGRequestScreenCaptureAccess() }
    return permissionsJSON()
}

func appsJSON() -> [String: Any] {
    pump(0.02)
    let apps = NSWorkspace.shared.runningApplications
        .filter { $0.activationPolicy == .regular && !$0.isTerminated }
        .map { app -> [String: Any] in
            var out = appJSON(app)
            out["active"] = app.isActive
            return out
        }
    return ["apps": apps]
}

func frontmostNow() throws -> [String: Any] {
    pump(0.02)
    guard NSWorkspace.shared.frontmostApplication != nil else { throw HelperError("no_app", "No app is frontmost.") }
    return frontmostJSON()
}

// A running app by its shown name first, then an .app bundle of that name in the usual folders.
func appURL(named name: String) -> URL? {
    let lowered = name.lowercased()
    let wanted = lowered.hasSuffix(".app") ? String(lowered.dropLast(4)) : lowered
    if let app = NSWorkspace.shared.runningApplications.first(where: { $0.activationPolicy == .regular && $0.localizedName?.lowercased() == wanted }),
       let url = app.bundleURL {
        return url
    }
    let folders = ["/Applications", "/System/Applications", "/System/Applications/Utilities", "/Applications/Utilities", NSHomeDirectory() + "/Applications"]
    for folder in folders {
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: folder),
              let match = names.first(where: { $0.lowercased() == wanted + ".app" }) else { continue }
        return URL(fileURLWithPath: folder).appendingPathComponent(match)
    }
    return nil
}

final class LaunchWait {
    private let lock = NSLock()
    private var done = false
    private var failure: Error?

    func finish(_ error: Error?) {
        lock.lock()
        done = true
        failure = error
        lock.unlock()
    }

    var state: (done: Bool, error: Error?) {
        lock.lock()
        defer { lock.unlock() }
        return (done, failure)
    }
}

func openApp(name: String?, bundleId: String?) throws -> [String: Any] {
    let url = bundleId.flatMap { NSWorkspace.shared.urlForApplication(withBundleIdentifier: $0) } ?? name.flatMap { appURL(named: $0) }
    guard let url else { throw HelperError("not_found", "No app matching \(String((name ?? bundleId ?? "").prefix(128))) was found.") }
    let target = Bundle(url: url)?.bundleIdentifier
    let parentURL = NSRunningApplication(processIdentifier: config.parentPid)?.bundleURL?.standardizedFileURL
    if isOwnBundle(target, own: config.ownBundleId) || parentURL == url.standardizedFileURL { throw ownAppError() }
    let wait = LaunchWait()
    let configuration = NSWorkspace.OpenConfiguration()
    configuration.activates = true
    NSWorkspace.shared.openApplication(at: url, configuration: configuration) { _, error in wait.finish(error) }
    let deadline = Date().addingTimeInterval(10)
    while !wait.state.done && Date() < deadline { pump(0.05) }
    let state = wait.state
    guard state.done else { throw HelperError("launch_failed", "\(url.lastPathComponent) did not open within 10 seconds.") }
    if let error = state.error { throw HelperError("launch_failed", "\(url.lastPathComponent) did not open: \(error.localizedDescription)") }
    let settle = Date().addingTimeInterval(2)
    while Date() < settle && NSWorkspace.shared.frontmostApplication?.bundleIdentifier != target { pump(0.05) }
    pump(0.2)
    return frontmostJSON()
}

import AppKit
import ApplicationServices
import Foundation

// tensorfold-computer: newline-delimited JSON requests on stdin, one JSON reply line each on stdout.
// Requests run one at a time on the main thread; logs go to stderr only.

func writeLine(_ object: [String: Any]) {
    do {
        try FileHandle.standardOutput.write(contentsOf: encodeLine(object))
    } catch {
        exit(0)
    }
}

func logLine(_ message: String) {
    FileHandle.standardError.write(Data("tensorfold-computer: \(message)\n".utf8))
}

func execute(_ command: Command) throws -> [String: Any] {
    switch command {
    case .permissions: return permissionsJSON()
    case .requestPermissions(let accessibility, let screenRecording): return requestPermissions(accessibility: accessibility, screenRecording: screenRecording)
    case .apps: return appsJSON()
    case .frontmost: return try frontmostNow()
    case .observe(let maxElements, let pid): return try observe(maxElements: maxElements, pid: pid)
    case .click(let target, let button, let count): return try click(target, button, count)
    case .move(let point): return try move(to: point)
    case .drag(let from, let to): return try drag(from: from, to: to)
    case .scroll(let target, let dx, let dy): return try scroll(at: target, dx: dx, dy: dy)
    case .type(let text): return try typeText(text)
    case .key(let combo): return try pressKeys(combo)
    case .openApp(let name, let bundleId): return try openApp(name: name, bundleId: bundleId)
    case .phoneStatus: return phoneStatus()
    case .phoneConnect: return try phoneConnect()
    case .phoneObserve(let token): return try phoneObserve(token)
    case .phoneInput(let token, let input): return try phoneInput(token, input)
    case .phoneDisconnect(let token): return phoneDisconnect(token)
    case .stop: return [:]
    }
}

func handle(_ line: String) {
    let parsed = parseRequest(line)
    switch parsed.command {
    case .failure(let error):
        writeLine(errorReply(parsed.id, error))
    case .success(let command):
        let id = parsed.id ?? 0
        do {
            writeLine(okReply(id, try execute(command)))
        } catch let error as HelperError {
            writeLine(errorReply(id, error))
        } catch {
            writeLine(errorReply(id, HelperError("internal", String(describing: error))))
        }
        if case .stop = command { exit(0) }
    }
}

// The reader waits for each request to finish, so a nested pump on the main thread never starts a second one.
func startReader() {
    let done = DispatchSemaphore(value: 0)
    let reader = Thread {
        while let raw = readLine(strippingNewline: true) {
            let line = raw.hasSuffix("\r") ? String(raw.dropLast()) : raw
            if line.trimmingCharacters(in: .whitespaces).isEmpty { continue }
            RunLoop.main.perform(inModes: [.default]) {
                handle(line)
                done.signal()
            }
            CFRunLoopWakeUp(CFRunLoopGetMain())
            done.wait()
        }
        RunLoop.main.perform(inModes: [.default]) { exit(0) }
        CFRunLoopWakeUp(CFRunLoopGetMain())
    }
    reader.start()
}

func argumentValue(_ flag: String) -> String? {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: flag), index + 1 < arguments.count else { return nil }
    return arguments[index + 1]
}

// Attachment extraction is an isolated byte-only mode, independent of computer control.
if CommandLine.arguments.contains("--attachment-text-capability") {
    writeLine(attachmentOcrCapability())
    exit(0)
}
if CommandLine.arguments.contains("--extract-attachment-text") {
    exit(runAttachmentTextMode())
}

if CommandLine.arguments.contains("--self-test") {
    exit(runSelfTest())
}
guard let ownBundleId = argumentValue("--own-bundle-id"), isBundleIdentifier(ownBundleId) else {
    logLine("--own-bundle-id <TensorFold bundle id> is required.")
    exit(2)
}
config.ownBundleId = ownBundleId
config.parentPid = getppid()
signal(SIGPIPE, SIG_IGN)
AXUIElementSetMessagingTimeout(systemWide, 2.0)
let keepAlive = Timer(fire: .distantFuture, interval: 0, repeats: false) { _ in }
RunLoop.main.add(keepAlive, forMode: .default)
startReader()
CFRunLoopRun()

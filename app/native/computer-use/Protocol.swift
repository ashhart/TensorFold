import CoreGraphics
import Foundation

struct HelperError: Error {
    let code: String
    let message: String
    init(_ code: String, _ message: String) {
        self.code = code
        self.message = message
    }
}

enum MouseButton: String {
    case left, right
}

// Where a pointer action lands: a global point, or the live frame centre of an element from the latest observe.
enum Target: Equatable {
    case point(CGPoint)
    case element(Int)
}

enum PhoneInput: Equatable {
    case tap(CGPoint, CGSize)
    case swipe(CGPoint, CGPoint, CGSize, Double)
    case type(String)
    case home
    case enter
}

enum Command: Equatable {
    case permissions
    case requestPermissions(accessibility: Bool, screenRecording: Bool)
    case apps
    case frontmost
    case observe(maxElements: Int, pid: pid_t?)
    case click(Target, MouseButton, Int)
    case move(CGPoint)
    case drag(CGPoint, CGPoint)
    case scroll(Target?, dx: Int, dy: Int)
    case type(String)
    case key(KeyCombo)
    case openApp(name: String?, bundleId: String?)
    case phoneStatus
    case phoneConnect
    case phoneObserve(String)
    case phoneInput(String, PhoneInput)
    case phoneDisconnect(String)
    case stop
}

struct ParsedLine {
    let id: Int?
    let command: Result<Command, HelperError>
}

let maxLineBytes = 1 << 20
let maxTypeCharacters = 10_000
let defaultObserveElements = 150
let maxObserveElements = 400
let maxScrollLines = 200

func badRequest(_ message: String) -> HelperError {
    HelperError("bad_request", message)
}

func isJSONBool(_ value: Any?) -> Bool {
    guard let number = value as? NSNumber else { return false }
    return CFGetTypeID(number) == CFBooleanGetTypeID()
}

func jsonNumber(_ value: Any?) -> Double? {
    guard let number = value as? NSNumber, !isJSONBool(number) else { return nil }
    let double = number.doubleValue
    return double.isFinite ? double : nil
}

func jsonInteger(_ value: Any?) -> Int? {
    guard let double = jsonNumber(value), double == double.rounded(), abs(double) <= 9_007_199_254_740_991 else { return nil }
    return Int(double)
}

func isBundleIdentifier(_ text: String) -> Bool {
    guard !text.isEmpty, text.count <= 255 else { return false }
    return text.unicodeScalars.allSatisfy { ($0.isASCII && CharacterSet.alphanumerics.contains($0)) || $0 == "." || $0 == "-" }
}

struct Params {
    let object: [String: Any]

    func present(_ key: String) -> Bool {
        guard let value = object[key] else { return false }
        return !(value is NSNull)
    }

    func number(_ key: String) throws -> Double? {
        guard present(key) else { return nil }
        guard let value = jsonNumber(object[key]) else { throw badRequest("\(key) must be a finite number.") }
        return value
    }

    func integer(_ key: String) throws -> Int? {
        guard present(key) else { return nil }
        guard let value = jsonInteger(object[key]) else { throw badRequest("\(key) must be an integer.") }
        return value
    }

    func bool(_ key: String) throws -> Bool? {
        guard present(key) else { return nil }
        guard isJSONBool(object[key]), let value = object[key] as? Bool else { throw badRequest("\(key) must be true or false.") }
        return value
    }

    func string(_ key: String) throws -> String? {
        guard present(key) else { return nil }
        guard let value = object[key] as? String else { throw badRequest("\(key) must be a string.") }
        return value
    }

    // Top-level x and y: both or neither.
    func optionalXY() throws -> CGPoint? {
        let x = try number("x")
        let y = try number("y")
        if x == nil && y == nil { return nil }
        guard let x, let y else { throw badRequest("x and y must be given together.") }
        return CGPoint(x: x, y: y)
    }

    func xy() throws -> CGPoint {
        guard let point = try optionalXY() else { throw badRequest("x and y are required.") }
        return point
    }

    // x and y, or element, or neither.
    func target() throws -> Target? {
        let point = try optionalXY()
        let element = try integer("element")
        switch (point, element) {
        case (nil, nil): return nil
        case (let point?, nil): return .point(point)
        case (nil, let element?):
            guard element >= 1 else { throw badRequest("element must be an id from the latest observe.") }
            return .element(element)
        default: throw badRequest("Give either x and y or element, not both.")
        }
    }

    func point(_ key: String) throws -> CGPoint {
        guard present(key), let nested = object[key] as? [String: Any] else { throw badRequest("\(key) must be an object with x and y.") }
        guard let point = try Params(object: nested).optionalXY() else { throw badRequest("\(key) needs x and y.") }
        return point
    }
}

func parseRequest(_ line: String) -> ParsedLine {
    guard line.utf8.count <= maxLineBytes else { return ParsedLine(id: nil, command: .failure(badRequest("Request line is longer than 1 MiB."))) }
    guard let data = line.data(using: .utf8), let any = try? JSONSerialization.jsonObject(with: data, options: []) else {
        return ParsedLine(id: nil, command: .failure(HelperError("bad_json", "Request is not valid JSON.")))
    }
    guard let object = any as? [String: Any] else { return ParsedLine(id: nil, command: .failure(badRequest("Request must be a JSON object."))) }
    guard let id = jsonInteger(object["id"]), id >= 0 else {
        return ParsedLine(id: nil, command: .failure(badRequest("Request needs a non-negative integer id.")))
    }
    guard let cmd = object["cmd"] as? String else { return ParsedLine(id: id, command: .failure(badRequest("Request needs a cmd string."))) }
    do {
        return ParsedLine(id: id, command: .success(try parseCommand(cmd, Params(object: object))))
    } catch let error as HelperError {
        return ParsedLine(id: id, command: .failure(error))
    } catch {
        return ParsedLine(id: id, command: .failure(HelperError("internal", String(describing: error))))
    }
}

func parseCommand(_ cmd: String, _ p: Params) throws -> Command {
    switch cmd {
    case "permissions":
        return .permissions
    case "request_permissions":
        let accessibility = try p.bool("accessibility") ?? false
        let screenRecording = try p.bool("screenRecording") ?? false
        guard accessibility || screenRecording else { throw badRequest("Set accessibility or screenRecording to true.") }
        return .requestPermissions(accessibility: accessibility, screenRecording: screenRecording)
    case "apps":
        return .apps
    case "frontmost":
        return .frontmost
    case "observe":
        let limit = try p.integer("maxElements") ?? defaultObserveElements
        guard limit >= 1 else { throw badRequest("maxElements must be at least 1.") }
        let pid = try p.integer("pid")
        if let pid, pid <= 0 || pid > Int(Int32.max) { throw badRequest("pid must be a positive process id.") }
        return .observe(maxElements: min(limit, maxObserveElements), pid: pid.map { pid_t($0) })
    case "click":
        guard let target = try p.target() else { throw badRequest("click needs either x and y or element.") }
        let buttonName = try p.string("button") ?? "left"
        guard let button = MouseButton(rawValue: buttonName) else { throw badRequest("button must be left or right.") }
        let count = try p.integer("count") ?? 1
        guard count == 1 || count == 2 else { throw badRequest("count must be 1 or 2.") }
        return .click(target, button, count)
    case "move":
        return .move(try p.xy())
    case "drag":
        return .drag(try p.point("from"), try p.point("to"))
    case "scroll":
        let at = try p.target()
        let dx = Int((try p.number("dx") ?? 0).rounded())
        let dy = Int((try p.number("dy") ?? 0).rounded())
        guard dx != 0 || dy != 0 else { throw badRequest("scroll needs a non-zero dx or dy.") }
        guard abs(dx) <= maxScrollLines, abs(dy) <= maxScrollLines else { throw badRequest("scroll is limited to \(maxScrollLines) lines per call.") }
        return .scroll(at, dx: dx, dy: dy)
    case "type":
        guard let text = try p.string("text"), !text.isEmpty else { throw badRequest("type needs non-empty text.") }
        guard text.count <= maxTypeCharacters else { throw badRequest("type is limited to \(maxTypeCharacters) characters per call.") }
        return .type(text)
    case "key":
        guard let keys = try p.string("keys") else { throw badRequest("key needs keys, like cmd+shift+t.") }
        return .key(try parseKeyCombo(keys))
    case "open_app":
        let name = try p.string("name")
        let bundleId = try p.string("bundleId")
        switch (name, bundleId) {
        case (let name?, nil):
            guard !name.isEmpty, name.count <= 128, !name.contains("/"), !name.contains(":"), !name.hasPrefix(".") else {
                throw badRequest("name must be an app name, not a path.")
            }
        case (nil, let bundleId?):
            guard isBundleIdentifier(bundleId) else { throw badRequest("bundleId is not a bundle identifier.") }
        default:
            throw badRequest("open_app needs exactly one of name or bundleId.")
        }
        return .openApp(name: name, bundleId: bundleId)
    case "phone_status": return .phoneStatus
    case "phone_connect": return .phoneConnect
    case "phone_observe": return .phoneObserve(try phoneToken(p))
    case "phone_disconnect": return .phoneDisconnect(try phoneToken(p))
    case "phone_input":
        let token = try phoneToken(p)
        guard let action = try p.string("action") else { throw badRequest("phone_input needs an action.") }
        switch action {
        case "tap", "swipe":
            guard let width = try p.number("width"), let height = try p.number("height"), width >= 1, height >= 1, width <= 4096, height <= 4096 else { throw badRequest("Phone input needs the latest bounded screen dimensions.") }
            let size = CGSize(width: width, height: height)
            let from = try p.xy()
            guard phoneLocalPoint(from, size: size) else { throw badRequest("Phone coordinates must be inside the latest phone screen.") }
            if action == "tap" { return .phoneInput(token, .tap(from, size)) }
            guard let toX = try p.number("toX"), let toY = try p.number("toY") else { throw badRequest("Phone swipe needs toX and toY.") }
            let to = CGPoint(x: toX, y: toY)
            guard phoneLocalPoint(to, size: size) else { throw badRequest("Phone swipe must end inside the phone screen.") }
            let duration = try p.number("durationMs") ?? 400
            guard duration >= 100, duration <= 2000 else { throw badRequest("Phone swipe duration must be between 100 and 2000 milliseconds.") }
            return .phoneInput(token, .swipe(from, to, size, duration / 1000))
        case "type":
            guard let text = try p.string("text"), !text.isEmpty, text.utf16.count <= 4000, phoneTypeAllowed(text) else { throw badRequest("Phone typing needs 1 to 4000 characters without control or private key codes.") }
            return .phoneInput(token, .type(text))
        case "home": return .phoneInput(token, .home)
        case "enter": return .phoneInput(token, .enter)
        default: throw badRequest("Mirroring supports tap, swipe, type, home and enter only.")
        }
    case "stop":
        return .stop
    default:
        throw HelperError("unknown_cmd", "Unknown command \(String(cmd.prefix(40))).")
    }
}

func phoneToken(_ p: Params) throws -> String {
    guard let token = try p.string("token"), token.count == 36, UUID(uuidString: token) != nil else { throw badRequest("Phone requests need the current connection token.") }
    return token
}

// One line of JSON plus a newline; anything JSONSerialization would reject becomes an internal error line.
func encodeLine(_ object: [String: Any]) -> Data {
    var data: Data
    if JSONSerialization.isValidJSONObject(object), let encoded = try? JSONSerialization.data(withJSONObject: object, options: [.withoutEscapingSlashes]) {
        data = encoded
    } else {
        let id: Any = object["id"] ?? NSNull()
        let fallback: [String: Any] = ["id": id, "ok": false, "code": "internal", "error": "The reply could not be encoded as JSON."]
        data = (try? JSONSerialization.data(withJSONObject: fallback, options: [])) ?? Data(#"{"id":null,"ok":false,"code":"internal"}"#.utf8)
    }
    data.append(0x0A)
    return data
}

func okReply(_ id: Int, _ body: [String: Any]) -> [String: Any] {
    var out = body
    out["id"] = id
    out["ok"] = true
    return out
}

func errorReply(_ id: Int?, _ error: HelperError) -> [String: Any] {
    ["id": id.map { $0 as Any } ?? NSNull(), "ok": false, "code": error.code, "error": error.message]
}

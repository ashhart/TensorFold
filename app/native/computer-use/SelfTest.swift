import CoreGraphics
import Foundation

// Pure checks only: nothing here reads the accessibility tree, the window list or posts an event.

final class Checks {
    var count = 0
    var failures: [String] = []

    func expect(_ condition: Bool, _ label: String) {
        count += 1
        if !condition { failures.append(label) }
    }
}

func failureCode(_ line: String) -> String? {
    if case .failure(let error) = parseRequest(line).command { return error.code }
    return nil
}

func parsedCommand(_ line: String) -> Command? {
    if case .success(let command) = parseRequest(line).command { return command }
    return nil
}

func keyChecks(_ c: Checks) {
    let good: [(String, CGKeyCode, CGEventFlags)] = [
        ("cmd+shift+t", 17, [.maskCommand, .maskShift]), ("Return", 36, []), ("enter", 36, []),
        ("ctrl+opt+delete", 51, [.maskControl, .maskAlternate]), ("command+alt+esc", 53, [.maskCommand, .maskAlternate]),
        ("f12", 111, []), ("F1", 122, []), ("cmd+,", 43, [.maskCommand]), ("cmd+=", 24, [.maskCommand]),
        ("shift+`", 50, [.maskShift]), ("pagedown", 121, []), ("pageup", 116, []), ("fn+f1", 122, [.maskSecondaryFn]),
        (" cmd + a ", 0, [.maskCommand]), ("cmd+cmd+a", 0, [.maskCommand]), ("space", 49, []), ("up", 126, []),
        ("down", 125, []), ("left", 123, []), ("right", 124, []), ("home", 115, []), ("end", 119, []),
        ("forwarddelete", 117, []), ("backspace", 51, []), ("tab", 48, []), ("control+[", 33, [.maskControl]),
        ("cmd+]", 30, [.maskCommand]), ("ctrl+;", 41, [.maskControl]), ("'", 39, []), ("/", 44, []), (".", 47, []),
        ("-", 27, []), ("0", 29, []), ("9", 25, []), ("z", 6, []), ("option+shift+k", 40, [.maskAlternate, .maskShift]),
    ]
    for (text, code, flags) in good {
        let combo = try? parseKeyCombo(text)
        c.expect(combo?.keyCode == code, "key \(text) code")
        c.expect(combo?.flags == flags, "key \(text) flags")
    }
    let order = try? parseKeyCombo("shift+cmd+t")
    c.expect(order?.modifiers.map { $0.keyCode } == [56, 55], "modifiers keep their order")
    c.expect((try? parseKeyCombo("cmd+cmd+a"))?.modifiers.count == 1, "a repeated modifier is pressed once")
    c.expect((try? parseKeyCombo("fn+f1"))?.modifiers.first?.keyCode == nil, "fn only sets a flag")
    let bad = ["", "   ", "cmd", "cmd+shift", "cmd+foo", "a+b", "f13", "hyper+a", "cmd++", "+", "control+option+", String(repeating: "a", count: 100)]
    for text in bad {
        do {
            _ = try parseKeyCombo(text)
            c.expect(false, "key \(text.prefix(20)) should fail")
        } catch let error as HelperError {
            c.expect(error.code == "bad_key", "key \(text.prefix(20)) fails with bad_key")
        } catch {
            c.expect(false, "key \(text.prefix(20)) threw something else")
        }
    }
    let printable: [(String, Bool)] = [("a", true), ("shift+a", true), ("space", true), ("cmd+a", false), ("ctrl+c", false), ("return", false), ("left", false), ("tab", false)]
    for (text, expected) in printable {
        c.expect((try? parseKeyCombo(text)).map(comboTypesCharacter) == expected, "\(text) types a character: \(expected)")
    }
}

func requestChecks(_ c: Checks) {
    let permissions = parseRequest(#"{"id":1,"cmd":"permissions"}"#)
    c.expect(permissions.id == 1, "id is read")
    c.expect(parsedCommand(#"{"id":1,"cmd":"permissions"}"#) == .permissions, "permissions parses")
    let codes: [(String, String)] = [
        ("not json", "bad_json"), ("[1,2]", "bad_request"), (#"{"cmd":"apps"}"#, "bad_request"),
        (#"{"id":"1","cmd":"apps"}"#, "bad_request"), (#"{"id":1.5,"cmd":"apps"}"#, "bad_request"),
        (#"{"id":true,"cmd":"apps"}"#, "bad_request"), (#"{"id":-1,"cmd":"apps"}"#, "bad_request"),
        (#"{"id":2}"#, "bad_request"), (#"{"id":3,"cmd":"dance"}"#, "unknown_cmd"),
        (#"{"id":4,"cmd":"click"}"#, "bad_request"), (#"{"id":4,"cmd":"click","x":1}"#, "bad_request"),
        (#"{"id":4,"cmd":"click","x":1,"y":2,"element":3}"#, "bad_request"),
        (#"{"id":4,"cmd":"click","x":1,"y":2,"button":"middle"}"#, "bad_request"),
        (#"{"id":4,"cmd":"click","x":1,"y":2,"count":3}"#, "bad_request"), (#"{"id":4,"cmd":"click","x":"1","y":2}"#, "bad_request"),
        (#"{"id":4,"cmd":"click","x":true,"y":2}"#, "bad_request"), (#"{"id":4,"cmd":"click","element":0}"#, "bad_request"),
        (#"{"id":5,"cmd":"observe","maxElements":0}"#, "bad_request"), (#"{"id":5,"cmd":"observe","pid":-5}"#, "bad_request"),
        (#"{"id":6,"cmd":"scroll"}"#, "bad_request"), (#"{"id":6,"cmd":"scroll","dy":500}"#, "bad_request"),
        (#"{"id":6,"cmd":"scroll","x":1,"y":2,"element":3,"dy":1}"#, "bad_request"), (#"{"id":6,"cmd":"scroll","element":0,"dy":1}"#, "bad_request"),
        (#"{"id":7,"cmd":"type","text":""}"#, "bad_request"), (#"{"id":7,"cmd":"type","text":5}"#, "bad_request"),
        (#"{"id":8,"cmd":"key","keys":"cmd+nope"}"#, "bad_key"), (#"{"id":8,"cmd":"key"}"#, "bad_request"),
        (#"{"id":9,"cmd":"drag","from":{"x":1},"to":{"x":3,"y":4}}"#, "bad_request"), (#"{"id":9,"cmd":"drag","from":{"x":1,"y":2}}"#, "bad_request"),
        (#"{"id":10,"cmd":"open_app","name":"../Evil"}"#, "bad_request"), (#"{"id":10,"cmd":"open_app","name":"/Applications/X.app"}"#, "bad_request"),
        (#"{"id":10,"cmd":"open_app","bundleId":"com apple"}"#, "bad_request"), (#"{"id":10,"cmd":"open_app"}"#, "bad_request"),
        (#"{"id":10,"cmd":"open_app","name":"Notes","bundleId":"com.apple.Notes"}"#, "bad_request"),
        (#"{"id":11,"cmd":"request_permissions"}"#, "bad_request"), (#"{"id":11,"cmd":"request_permissions","accessibility":1}"#, "bad_request"),
    ]
    for (line, code) in codes {
        c.expect(failureCode(line) == code, "\(line.prefix(60)) -> \(code)")
    }
    c.expect(parseRequest(#"{"id":2}"#).id == 2, "a bad request keeps its id")
    c.expect(parseRequest("not json").id == nil, "unreadable JSON has no id")
    let tooLong = #"{"id":12,"cmd":"type","text":""# + String(repeating: "a", count: maxTypeCharacters + 1) + #""}"#
    c.expect(failureCode(tooLong) == "bad_request", "type text is capped")
    let huge = #"{"id":13,"cmd":"apps","pad":""# + String(repeating: "a", count: maxLineBytes) + #""}"#
    c.expect(failureCode(huge) == "bad_request", "request lines are capped")
    let good: [(String, Command)] = [
        (#"{"id":1,"cmd":"click","x":10,"y":20}"#, .click(.point(CGPoint(x: 10, y: 20)), .left, 1)),
        (#"{"id":1,"cmd":"click","element":4,"button":"right","count":2}"#, .click(.element(4), .right, 2)),
        (#"{"id":1,"cmd":"observe"}"#, .observe(maxElements: 150, pid: nil)),
        (#"{"id":1,"cmd":"observe","maxElements":1000,"pid":123}"#, .observe(maxElements: 400, pid: 123)),
        (#"{"id":1,"cmd":"scroll","dy":3}"#, .scroll(nil, dx: 0, dy: 3)),
        (#"{"id":1,"cmd":"scroll","x":1,"y":2,"dx":-2}"#, .scroll(.point(CGPoint(x: 1, y: 2)), dx: -2, dy: 0)),
        (#"{"id":1,"cmd":"scroll","element":5,"dy":-1}"#, .scroll(.element(5), dx: 0, dy: -1)),
        (#"{"id":1,"cmd":"type","text":"hi\n"}"#, .type("hi\n")),
        (#"{"id":1,"cmd":"key","keys":"cmd+q"}"#, .key(KeyCombo(keyCode: 12, modifiers: [Modifier(flag: .maskCommand, keyCode: 55)]))),
        (#"{"id":1,"cmd":"drag","from":{"x":1,"y":2},"to":{"x":3,"y":4}}"#, .drag(CGPoint(x: 1, y: 2), CGPoint(x: 3, y: 4))),
        (#"{"id":1,"cmd":"move","x":5.5,"y":6}"#, .move(CGPoint(x: 5.5, y: 6))),
        (#"{"id":1,"cmd":"open_app","name":"Notes"}"#, .openApp(name: "Notes", bundleId: nil)),
        (#"{"id":1,"cmd":"open_app","bundleId":"com.apple.Notes"}"#, .openApp(name: nil, bundleId: "com.apple.Notes")),
        (#"{"id":1,"cmd":"request_permissions","accessibility":true}"#, .requestPermissions(accessibility: true, screenRecording: false)),
        (#"{"id":1,"cmd":"apps","extra":null}"#, .apps), (#"{"id":1,"cmd":"frontmost"}"#, .frontmost), (#"{"id":1,"cmd":"stop"}"#, .stop),
    ]
    for (line, command) in good {
        c.expect(parsedCommand(line) == command, "\(line.prefix(60)) parses")
    }
}

func pointChecks(_ c: Checks) {
    let displays = [CGRect(x: 0, y: 0, width: 1512, height: 982), CGRect(x: 1512, y: -200, width: 2560, height: 1440)]
    let cases: [(CGPoint, Bool)] = [
        (CGPoint(x: 0, y: 0), true), (CGPoint(x: 1511.5, y: 981.5), true), (CGPoint(x: 1512, y: 982), true),
        (CGPoint(x: 2000, y: -150), true), (CGPoint(x: 1512, y: 1300), false), (CGPoint(x: -1, y: 10), false),
        (CGPoint(x: 100, y: 990), false), (CGPoint(x: 4072, y: 0), false), (CGPoint(x: Double.nan, y: 1), false),
        (CGPoint(x: Double.infinity, y: 1), false),
    ]
    for (point, expected) in cases {
        c.expect(pointOnAnyDisplay(point, displays) == expected, "point \(point) on a display: \(expected)")
    }
    c.expect(!pointOnAnyDisplay(CGPoint(x: 1, y: 1), []), "no displays, no valid point")
}

func filterChecks(_ c: Checks) {
    let displays = [CGRect(x: 0, y: 0, width: 1000, height: 800)]
    let clip = CGRect(x: 100, y: 100, width: 600, height: 400)
    func node(_ role: String, _ frame: CGRect = CGRect(x: 150, y: 150, width: 80, height: 24), title: String? = nil, value: String? = nil, subrole: String? = nil) -> NodeInfo {
        var out = NodeInfo(role: role)
        out.frame = frame
        out.title = title
        out.value = value
        out.subrole = subrole
        return out
    }
    c.expect(keptElement(node("AXButton", title: "OK"), clip: clip, displays: displays)?.title == "OK", "a visible button is kept")
    c.expect(keptElement(node("AXButton", CGRect(x: 150, y: 150, width: 0, height: 24)), clip: clip, displays: displays) == nil, "zero size is dropped")
    c.expect(keptElement(node("AXButton", CGRect(x: 150, y: 600, width: 80, height: 24)), clip: clip, displays: displays) == nil, "outside the window is dropped")
    c.expect(keptElement(node("AXButton", CGRect(x: 2000, y: 10, width: 80, height: 24)), clip: nil, displays: displays) == nil, "off every display is dropped")
    c.expect(keptElement(node("AXStaticText", value: ""), clip: clip, displays: displays) == nil, "empty static text is dropped")
    c.expect(keptElement(node("AXStaticText", value: "  \n"), clip: clip, displays: displays) == nil, "blank static text is dropped")
    c.expect(keptElement(node("AXStaticText", value: "Hello"), clip: clip, displays: displays)?.value == "Hello", "static text with a value is kept")
    c.expect(keptElement(node("AXGroup", title: "Sidebar"), clip: clip, displays: displays) == nil, "groups are dropped")
    let secure = keptElement(node("AXTextField", value: "hunter2", subrole: "AXSecureTextField"), clip: clip, displays: displays)
    c.expect(secure != nil && secure?.value == nil, "a password field is kept without its value")
    c.expect((secure.map { elementJSON(1, $0) }?["secure"] as? Bool) == true, "a password field is marked secure")
    c.expect(secure.map { elementJSON(1, $0) }?["value"] == nil, "a password field reply has no value")
    c.expect(keptElement(node("AXRow"), clip: clip, displays: displays) == nil, "a row without text is dropped")
    c.expect(keptElement(node("AXCell", title: "Row 1"), clip: clip, displays: displays) != nil, "a cell with text is kept")
    c.expect(keptElement(node("AXHeading", title: "Inbox"), clip: clip, displays: displays) != nil, "a heading with text is kept")
    c.expect(keptElement(node("AXTextField"), clip: clip, displays: displays) != nil, "an empty text field is kept")
    let long = keptElement(node("AXLink", title: String(repeating: "x", count: 500)), clip: clip, displays: displays)
    c.expect(long?.title?.count == maxTextLength, "long strings are cut to 200 characters")
    let json = elementJSON(7, node("AXButton", CGRect(x: 10.04, y: 20.06, width: 30, height: 40), title: "Go"))
    let frame = json["frame"] as? [String: Any]
    c.expect(json["id"] as? Int == 7 && frame?["x"] as? Double == 10.0 && frame?["y"] as? Double == 20.1, "frames round to a tenth of a point")
    c.expect(json["subrole"] == nil && json["enabled"] as? Bool == true, "absent fields are left out")
}

func textChecks(_ c: Checks) {
    c.expect(typeSegments("ab\ncd\te") == [.text("ab"), .key(36), .text("cd"), .key(48), .text("e")], "line breaks and tabs become keys")
    c.expect(typeSegments("x\r\ny") == [.text("x"), .key(36), .text("y")], "CRLF is one Return")
    c.expect(typeSegments("\n\n") == [.key(36), .key(36)], "blank lines are Returns")
    let plain = unicodeChunks(String(repeating: "a", count: 40))
    c.expect(plain.map { $0.count } == [16, 16, 8], "ASCII splits into 16-unit chunks")
    let samples = [String(repeating: "😀", count: 9), "héllo 👨‍👩‍👧‍👦 🇬🇧 wörld", "e" + String(repeating: "\u{301}", count: 20), String(repeating: "中文", count: 12)]
    for sample in samples {
        let chunks = unicodeChunks(sample)
        c.expect(chunks.allSatisfy { $0.count <= 16 && !$0.isEmpty }, "chunks fit one keyboard event")
        c.expect(String(utf16CodeUnits: chunks.flatMap { $0 }, count: chunks.reduce(0) { $0 + $1.count }) == sample, "chunks rebuild the text")
        c.expect(chunks.allSatisfy { chunk in !(chunk.last.map { UTF16.isLeadSurrogate($0) } ?? false) }, "no chunk ends inside a surrogate pair")
    }
}

func miscChecks(_ c: Checks) {
    let own = "dev.tensorfold.studio"
    let bundles: [(String?, Bool)] = [(own, true), ("DEV.TensorFold.Studio", true), ("dev.tensorfold.studio.helper", true), ("dev.tensorfold.studiox", false), (nil, false), ("com.apple.Notes", false)]
    for (bundle, expected) in bundles {
        c.expect(isOwnBundle(bundle, own: own) == expected, "own bundle \(bundle ?? "nil"): \(expected)")
    }
    c.expect(!isOwnBundle("com.apple.Notes", own: ""), "an empty own id matches nothing")
    c.expect(isBundleIdentifier(own) && !isBundleIdentifier("") && !isBundleIdentifier("a b") && !isBundleIdentifier("é.app"), "bundle id syntax")
    c.expect(scrollSteps(dx: 0, dy: 3) == [WheelStep(vertical: -3, horizontal: 0)], "dy down is a negative wheel")
    c.expect(scrollSteps(dx: 0, dy: 25).map { $0.vertical } == [-10, -10, -5], "long scrolls split into steps")
    c.expect(scrollSteps(dx: -4, dy: 0) == [WheelStep(vertical: 0, horizontal: 4)], "dx left is a positive wheel")
    c.expect(scrollSteps(dx: 12, dy: -3) == [WheelStep(vertical: 3, horizontal: -10), WheelStep(vertical: 0, horizontal: -2)], "mixed scrolls")
    let from = CGPoint(x: 10, y: 10)
    let to = CGPoint(x: 410, y: 310)
    let path = dragPath(from: from, to: to)
    c.expect(path.last == to && path.first != from && (8...60).contains(path.count), "drag path ends on the target")
    c.expect(dragPath(from: from, to: from).count == 8, "a zero drag still moves in steps")
    let line = encodeLine(okReply(3, ["title": "a\nb", "path": "/x"]))
    let text = String(decoding: line, as: UTF8.self)
    c.expect(text.hasSuffix("\n") && text.filter { $0 == "\n" }.count == 1, "replies are exactly one line")
    c.expect(text.contains(#""path":"/x""#) && text.contains(#"\n"#), "replies escape newlines and keep slashes")
    let broken = String(decoding: encodeLine(["id": 4, "ok": true, "x": Double.nan]), as: UTF8.self)
    c.expect(broken.contains(#""code":"internal""#) && broken.contains(#""id":4"#), "an unencodable reply becomes an internal error")
    let error = errorReply(nil, HelperError("bad_json", "nope"))
    c.expect(error["id"] is NSNull && error["ok"] as? Bool == false && error["code"] as? String == "bad_json", "error replies carry code and null id")
}

func runSelfTest() -> Int32 {
    let checks = Checks()
    keyChecks(checks)
    requestChecks(checks)
    pointChecks(checks)
    filterChecks(checks)
    textChecks(checks)
    miscChecks(checks)
    if checks.failures.isEmpty {
        writeLine(["ok": true, "checks": checks.count])
        return 0
    }
    writeLine(["ok": false, "checks": checks.count, "failures": checks.failures])
    return 1
}

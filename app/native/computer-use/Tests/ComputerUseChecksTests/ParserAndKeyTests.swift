import Testing
@testable import ComputerUseChecks

private func failureCode(_ line: String) -> String? {
    if case .failure(let error) = parseRequest(line).command { return error.code }
    return nil
}

private func parsedCommand(_ line: String) -> Command? {
    if case .success(let command) = parseRequest(line).command { return command }
    return nil
}

@Test func keyCombosKeepTheirCodesAndFlags() throws {
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
        let combo = try parseKeyCombo(text)
        #expect(combo.keyCode == code, "key \(text) code")
        #expect(combo.flags == flags, "key \(text) flags")
    }
    let order = try parseKeyCombo("shift+cmd+t")
    #expect(order.modifiers.map(\.keyCode) == [56, 55])
    #expect(try parseKeyCombo("cmd+cmd+a").modifiers.count == 1)
    #expect(try parseKeyCombo("fn+f1").modifiers.first?.keyCode == nil)
}

@Test func badKeyCombosFail() {
    let bad = ["", "   ", "cmd", "cmd+shift", "cmd+foo", "a+b", "f13", "hyper+a", "cmd++", "+", "control+option+", String(repeating: "a", count: 100)]
    for text in bad {
        #expect(throws: HelperError.self) { try parseKeyCombo(text) }
        do {
            _ = try parseKeyCombo(text)
            Issue.record("key \(text.prefix(20)) should fail")
        } catch let error as HelperError {
            #expect(error.code == "bad_key")
        } catch {
            Issue.record("key \(text.prefix(20)) threw something else")
        }
    }
}

@Test func printableKeysStayDistinctFromShortcuts() throws {
    let printable: [(String, Bool)] = [("a", true), ("shift+a", true), ("space", true), ("cmd+a", false), ("ctrl+c", false), ("return", false), ("left", false), ("tab", false)]
    for (text, expected) in printable {
        #expect(comboTypesCharacter(try parseKeyCombo(text)) == expected)
    }
}

@Test func requestsParseOrFailWithTheirCodes() {
    let permissions = parseRequest(#"{"id":1,"cmd":"permissions"}"#)
    #expect(permissions.id == 1)
    #expect(parsedCommand(#"{"id":1,"cmd":"permissions"}"#) == .permissions)
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
        #expect(failureCode(line) == code, "\(line.prefix(60)) -> \(code)")
    }
    #expect(parseRequest(#"{"id":2}"#).id == 2)
    #expect(parseRequest("not json").id == nil)
    let tooLong = #"{"id":12,"cmd":"type","text":""# + String(repeating: "a", count: maxTypeCharacters + 1) + #""}"#
    #expect(failureCode(tooLong) == "bad_request")
    let huge = #"{"id":13,"cmd":"apps","pad":""# + String(repeating: "a", count: maxLineBytes) + #""}"#
    #expect(failureCode(huge) == "bad_request")
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
        #expect(parsedCommand(line) == command, "\(line.prefix(60)) parses")
    }
}

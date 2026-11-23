import Foundation
import Testing
@testable import ComputerUseChecks

func failureCode(_ line: String) -> String? {
    if case .failure(let error) = parseRequest(line).command { return error.code }
    return nil
}

func parsedCommand(_ line: String) -> Command? {
    if case .success(let command) = parseRequest(line).command { return command }
    return nil
}

struct ProtocolTests {
    @Test func requestsParse() {
        let permissions = parseRequest(#"{"id":1,"cmd":"permissions"}"#)
        #expect(permissions.id == 1, "id is read")
        #expect(parsedCommand(#"{"id":1,"cmd":"permissions"}"#) == .permissions, "permissions parses")
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

    @Test func invalidRequestsAreRejected() {
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
        #expect(parseRequest(#"{"id":2}"#).id == 2, "a bad request keeps its id")
        #expect(parseRequest("not json").id == nil, "unreadable JSON has no id")
        let tooLong = #"{"id":12,"cmd":"type","text":""# + String(repeating: "a", count: maxTypeCharacters + 1) + #""}"#
        #expect(failureCode(tooLong) == "bad_request", "type text is capped")
        let huge = #"{"id":13,"cmd":"apps","pad":""# + String(repeating: "a", count: maxLineBytes) + #""}"#
        #expect(failureCode(huge) == "bad_request", "request lines are capped")
        #expect(isBundleIdentifier("dev.tensorfold.studio"))
        for name in ["", "a b", "é.app"] { #expect(isBundleIdentifier(name) == false) }
    }

    @Test func phoneRequestsStayBounded() {
        let token = "12345678-1234-1234-1234-123456789ABC"
        #expect(parsedCommand(#"{"id":1,"cmd":"phone_status"}"#) == .phoneStatus, "phone status parses without permission prompts")
        #expect(parsedCommand(#"{"id":1,"cmd":"phone_connect","bundleId":"com.example.other"}"#) == .phoneConnect, "phone connect has a fixed native target")
        let observe = "{\"id\":1,\"cmd\":\"phone_observe\",\"token\":\"\(token)\"}"
        #expect(parsedCommand(observe) == .phoneObserve(token), "phone observe requires connection token")
        let tap = "{\"id\":1,\"cmd\":\"phone_input\",\"token\":\"\(token)\",\"action\":\"tap\",\"x\":10,\"y\":20,\"width\":300,\"height\":600}"
        #expect(parsedCommand(tap) == .phoneInput(token, .tap(CGPoint(x: 10, y: 20), CGSize(width: 300, height: 600))), "phone tap uses bounded relative coordinates")
        for line in [tap.replacingOccurrences(of: "\"x\":10", with: "\"x\":300"), tap.replacingOccurrences(of: "\"x\":10", with: "\"x\":-1"), tap.replacingOccurrences(of: "\"width\":300", with: "\"width\":0"), tap.replacingOccurrences(of: "\"tap\"", with: "\"key\""), tap.replacingOccurrences(of: token, with: "wrong"), #"{"id":1,"cmd":"phone_observe"}"#] {
            #expect(failureCode(line) == "bad_request", "phone invalid coordinates, token and arbitrary keys refused")
        }
        let size = CGSize(width: 300, height: 600)
        #expect(phoneLocalPoint(CGPoint(x: 0, y: 0), size: size), "phone origin is inside")
        #expect(phoneLocalPoint(CGPoint(x: 300, y: 0), size: size) == false, "phone right boundary is excluded")
        #expect(phoneLocalPoint(CGPoint(x: 1, y: CGFloat.nan), size: size) == false, "phone NaN point refused")
        #expect(phoneTypeAllowed("Hello\n世界\t😀"), "phone unicode and explicit line break typing supported")
        #expect(phoneTypeAllowed("\u{E03D}") == false && phoneTypeAllowed("\u{1B}") == false, "phone private key codes and escape refused")
        let swipe = tap.replacingOccurrences(of: #""action":"tap""#, with: #""action":"swipe","toX":15,"toY":25"#)
        #expect(parsedCommand(swipe) == .phoneInput(token, .swipe(CGPoint(x: 10, y: 20), CGPoint(x: 15, y: 25), size, 0.4)))
        #expect(failureCode(swipe.replacingOccurrences(of: #""toY":25"#, with: #""toY":600"#)) == "bad_request")
        let typing = #"{"id":1,"cmd":"phone_input","token":"TOKEN","action":"type","text":"TEXT"}"#
            .replacingOccurrences(of: "TOKEN", with: token)
        #expect(parsedCommand(typing.replacingOccurrences(of: "TEXT", with: #"Hello\n世界\t😀"#)) == .phoneInput(token, .type("Hello\n世界\t😀")))
        for text in ["", #"\u001b"#, "\u{E03D}", String(repeating: "😀", count: 2001)] {
            #expect(failureCode(typing.replacingOccurrences(of: "TEXT", with: text)) == "bad_request")
        }
        let limit = String(repeating: "😀", count: 2000)
        #expect(parsedCommand(typing.replacingOccurrences(of: "TEXT", with: limit)) == .phoneInput(token, .type(limit)))
    }

    @Test func repliesStayOneJSONLine() throws {
        let line = encodeLine(okReply(3, ["title": "a\nb", "path": "/x"]))
        let text = String(decoding: line, as: UTF8.self)
        #expect(text.hasSuffix("\n") && text.filter { $0 == "\n" }.count == 1, "replies are exactly one line")
        #expect(text.contains(#""path":"/x""#) && text.contains(#"\n"#), "replies escape newlines and keep slashes")
        let decoded = try #require(JSONSerialization.jsonObject(with: line) as? [String: Any])
        #expect(decoded["title"] as? String == "a\nb")
        #expect(decoded["path"] as? String == "/x")
        #expect(decoded["id"] as? Int == 3)
        #expect(decoded["ok"] as? Bool == true)
        let broken = String(decoding: encodeLine(["id": 4, "ok": true, "x": Double.nan]), as: UTF8.self)
        #expect(broken.contains(#""code":"internal""#) && broken.contains(#""id":4"#), "an unencodable reply becomes an internal error")
        let error = errorReply(nil, HelperError("bad_json", "nope"))
        #expect(error["id"] is NSNull && error["ok"] as? Bool == false && error["code"] as? String == "bad_json", "error replies carry code and null id")
    }
}

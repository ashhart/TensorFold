import class Foundation.NSNull
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

@Test func phoneInputRejectsATapOutsideTheScreen() {
    let token = "12345678-1234-1234-1234-123456789ABC"
    #expect(parsedCommand(#"{"id":1,"cmd":"phone_status"}"#) == .phoneStatus)
    #expect(parsedCommand(#"{"id":1,"cmd":"phone_connect","bundleId":"com.example.other"}"#) == .phoneConnect)
    #expect(parsedCommand("{\"id\":1,\"cmd\":\"phone_observe\",\"token\":\"\(token)\"}") == .phoneObserve(token))
    let tap = "{\"id\":1,\"cmd\":\"phone_input\",\"token\":\"\(token)\",\"action\":\"tap\",\"x\":10,\"y\":20,\"width\":300,\"height\":600}"
    let expected: Command = .phoneInput(token, .tap(CGPoint(x: 10, y: 20), CGSize(width: 300, height: 600)))
    #expect(parsedCommand(tap) == expected)
    let refused = [
        tap.replacingOccurrences(of: "\"x\":10", with: "\"x\":300"),
        tap.replacingOccurrences(of: "\"x\":10", with: "\"x\":-1"),
        tap.replacingOccurrences(of: "\"width\":300", with: "\"width\":0"),
        tap.replacingOccurrences(of: "\"tap\"", with: "\"key\""),
        tap.replacingOccurrences(of: token, with: "wrong"),
        #"{"id":1,"cmd":"phone_observe"}"#,
    ]
    for line in refused {
        #expect(failureCode(line) == "bad_request")
    }
    let size = CGSize(width: 300, height: 600)
    #expect(phoneLocalPoint(CGPoint(x: 0, y: 0), size: size))
    #expect(!phoneLocalPoint(CGPoint(x: 300, y: 0), size: size))
    #expect(!phoneLocalPoint(CGPoint(x: 1, y: CGFloat.nan), size: size))
    #expect(phoneTypeAllowed("Hello\n世界\t😀"))
    #expect(!phoneTypeAllowed("\u{E03D}"))
    #expect(!phoneTypeAllowed("\u{1B}"))
}

@Test func repliesStayOneJsonLine() {
    let line = encodeLine(okReply(3, ["title": "a\nb", "path": "/x"]))
    let text = String(decoding: line, as: UTF8.self)
    #expect(text.hasSuffix("\n"))
    #expect(text.filter { $0 == "\n" }.count == 1)
    #expect(text.contains(#""path":"/x""#))
    #expect(text.contains(#"\n"#))
    let broken = String(decoding: encodeLine(["id": 4, "ok": true, "x": Double.nan]), as: UTF8.self)
    #expect(broken.contains(#""code":"internal""#))
    #expect(broken.contains(#""id":4"#))
    let error = errorReply(nil, HelperError("bad_json", "nope"))
    #expect(error["id"] is NSNull)
    #expect(error["ok"] as? Bool == false)
    #expect(error["code"] as? String == "bad_json")
}

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
}

func phoneChecks(_ c: Checks) {
    let token = "12345678-1234-1234-1234-123456789ABC"
    let size = CGSize(width: 300, height: 600)
    c.expect(phoneContentRect(CGRect(x: 100, y: 200, width: 320, height: 700)) == CGRect(x: 110, y: 248, width: 300, height: 642), "phone image and input exclude window controls")
    c.expect(phoneContentRect(CGRect(x: 0, y: 0, width: 50, height: 50)) == nil, "phone cannot target tiny toolbar windows")
    let global = try? phoneGlobalPoint(CGPoint(x: 150, y: 300), size: size, frame: CGRect(x: 100, y: 200, width: 320, height: 700))
    c.expect(global == CGPoint(x: 260, y: 569), "phone coordinates map only into clipped window content")
    c.expect(phoneProtectedText("Enter your passcode") && phoneProtectedText("iPhone in use") && phoneProtectedText("Apple Account password"), "phone lock, authentication and disconnected states are refused")
    c.expect(!phoneProtectedText("Synthetic sample home screen"), "ordinary phone screen text accepted")
    let windowBounds: [String: Any] = ["X": 0, "Y": 0, "Width": 300, "Height": 600]
    let backWindow: [String: Any] = [kCGWindowBounds as String: windowBounds, kCGWindowNumber as String: 11, kCGWindowAlpha as String: 1]
    let frontWindow: [String: Any] = [kCGWindowBounds as String: windowBounds, kCGWindowNumber as String: 22, kCGWindowAlpha as String: 1]
    c.expect(phoneFirstWindow(at: CGPoint(x: 40, y: 60), windows: [frontWindow, backWindow]) == 22, "phone input checks exact top window id, not just its process")
    var invisible = frontWindow
    invisible[kCGWindowAlpha as String] = 0
    c.expect(phoneFirstWindow(at: CGPoint(x: 40, y: 60), windows: [invisible, backWindow]) == 11, "phone ignores an invisible window overlay")
    c.expect(phoneFirstWindow(at: CGPoint(x: 400, y: 60), windows: [frontWindow, backWindow]) == nil, "phone input cannot target outside pinned windows")
    do { _ = try requirePhone(token); c.expect(false, "phone token absent should fail") }
    catch let error as HelperError { c.expect(error.code == "phone_disconnected", "revoked phone token fails before any desktop access") }
    catch { c.expect(false, "phone absent token threw unexpected error") }
}

func attachmentTextChecks(_ c: Checks) {
    let capability = attachmentOcrCapability()
    c.expect(capability["available"] as? Bool == true, "byte-only OCR capability is available on the supported Mac target")
    c.expect(capability["method"] as? String == "apple-vision", "OCR capability names its local engine")
    c.expect(capability["maxBytes"] as? Int == 20 * 1024 * 1024, "OCR bytes are bounded independently")
    c.expect(capability["maxPages"] as? Int == 8, "OCR pages are bounded independently")
    c.expect(attachmentTextReviewWarning.contains("even when its confidence score is high"), "high OCR confidence never suppresses review uncertainty")
    do { _ = try extractAttachmentText(Data(), mime: "image/png"); c.expect(false, "empty OCR bytes must fail") }
    catch let error as HelperError { c.expect(error.code == "attachment_size", "empty OCR bytes fail before decode") }
    catch { c.expect(false, "empty OCR bytes return a supported error") }
    do { _ = try extractAttachmentText(Data("not PDF".utf8), mime: "application/pdf"); c.expect(false, "bad PDF must fail") }
    catch let error as HelperError { c.expect(error.code == "attachment_pdf", "invalid PDF bytes fail visibly") }
    catch { c.expect(false, "invalid PDF returns a supported error") }
}

func runSelfTest() -> Int32 {
    let checks = Checks()
    pointChecks(checks)
    filterChecks(checks)
    textChecks(checks)
    miscChecks(checks)
    phoneChecks(checks)
    attachmentTextChecks(checks)
    if checks.failures.isEmpty {
        writeLine(["ok": true, "checks": checks.count])
        return 0
    }
    writeLine(["ok": false, "checks": checks.count, "failures": checks.failures])
    return 1
}

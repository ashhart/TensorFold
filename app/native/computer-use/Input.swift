import AppKit
import ApplicationServices
import CoreGraphics

final class HelperConfig {
    var ownBundleId = ""
    var parentPid: pid_t = 0
}

let config = HelperConfig()
let eventSource = CGEventSource(stateID: .hidSystemState)

// Runs the main run loop so NSWorkspace sees app switches; it keeps a timer, so this never returns early.
func pump(_ seconds: Double) {
    let deadline = Date().addingTimeInterval(seconds)
    while true {
        let remaining = deadline.timeIntervalSinceNow
        if remaining <= 0 { return }
        if CFRunLoopRunInMode(.defaultMode, remaining, false) == .finished { usleep(10_000) }
    }
}

func hold(_ seconds: Double) {
    usleep(useconds_t(seconds * 1_000_000))
}

func isOwnBundle(_ bundleId: String?, own: String) -> Bool {
    guard let bundleId, !own.isEmpty else { return false }
    let id = bundleId.lowercased()
    let ownId = own.lowercased()
    return id == ownId || id.hasPrefix(ownId + ".")
}

// The parent is TensorFold's main process; in a dev build its bundle id is Electron's, so the pid counts too.
func isOwnPid(_ pid: pid_t) -> Bool {
    if pid == getpid() || (config.parentPid > 1 && pid == config.parentPid) { return true }
    return isOwnBundle(NSRunningApplication(processIdentifier: pid)?.bundleIdentifier, own: config.ownBundleId)
}

func ownAppError() -> HelperError {
    HelperError("own_app", "That target is TensorFold itself, which computer use never controls.")
}

func requireAccessibility() throws {
    guard AXIsProcessTrusted() else { throw HelperError("no_accessibility", "TensorFold needs Accessibility permission to control other apps.") }
}

func windowOwner(at point: CGPoint) -> pid_t? {
    guard let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly], kCGNullWindowID) as? [[String: Any]] else { return nil }
    for info in list {
        guard let bounds = info[kCGWindowBounds as String] as? NSDictionary,
              let frame = CGRect(dictionaryRepresentation: bounds as CFDictionary), frame.contains(point) else { continue }
        if let alpha = (info[kCGWindowAlpha as String] as? NSNumber)?.doubleValue, alpha < 0.01 { continue }
        if let pid = (info[kCGWindowOwnerPID as String] as? NSNumber)?.int32Value { return pid }
    }
    return nil
}

// Refuses when either the accessibility hit test or the topmost window at the point belongs to TensorFold.
func checkPoint(_ point: CGPoint) throws {
    guard pointOnAnyDisplay(point, displayBounds()) else {
        throw HelperError("bad_point", "(\(rounded(point.x)), \(rounded(point.y))) is not on any display.")
    }
    var owners: [pid_t] = []
    var hit: AXUIElement?
    if AXUIElementCopyElementAtPosition(systemWide, Float(point.x), Float(point.y), &hit) == .success, let hit, let pid = elementPid(hit) {
        owners.append(pid)
    }
    if let pid = windowOwner(at: point) { owners.append(pid) }
    if owners.contains(where: isOwnPid) { throw ownAppError() }
}

func checkKeyboardTarget(typesCharacters: Bool) throws {
    pump(0.02)
    guard let front = NSWorkspace.shared.frontmostApplication else { throw HelperError("no_app", "No app is frontmost to receive keys.") }
    if isOwnPid(front.processIdentifier) { throw ownAppError() }
    if let app = axElement(axAttribute(systemWide, kAXFocusedApplicationAttribute)), let pid = elementPid(app), isOwnPid(pid) { throw ownAppError() }
    guard let focused = axElement(axAttribute(systemWide, kAXFocusedUIElementAttribute)) else { return }
    if let pid = elementPid(focused), isOwnPid(pid) { throw ownAppError() }
    let role = axAttribute(focused, kAXRoleAttribute) as? String ?? ""
    if typesCharacters && isSecureRole(role, axAttribute(focused, kAXSubroleAttribute) as? String) {
        throw HelperError("secure_field", "The focused field is a password field, and computer use never types into one.")
    }
}

func actionResult(via: String? = nil, settle: Double = 0.15) -> [String: Any] {
    pump(settle)
    var out = frontmostJSON()
    if let via { out["via"] = via }
    return out
}

func postMouse(_ type: CGEventType, _ point: CGPoint, _ button: CGMouseButton, clickState: Int64 = 1) {
    guard let event = CGEvent(mouseEventSource: eventSource, mouseType: type, mouseCursorPosition: point, mouseButton: button) else { return }
    event.setIntegerValueField(.mouseEventClickState, value: clickState)
    event.flags = []
    event.post(tap: .cghidEventTap)
}

func mouseClick(at point: CGPoint, button: MouseButton, count: Int) {
    let cgButton: CGMouseButton = button == .left ? .left : .right
    let down: CGEventType = button == .left ? .leftMouseDown : .rightMouseDown
    let up: CGEventType = button == .left ? .leftMouseUp : .rightMouseUp
    postMouse(.mouseMoved, point, cgButton)
    hold(0.03)
    for press in 1...count {
        postMouse(down, point, cgButton, clickState: Int64(press))
        hold(0.03)
        postMouse(up, point, cgButton, clickState: Int64(press))
        if press < count { hold(0.06) }
    }
}

func dragPath(from: CGPoint, to: CGPoint) -> [CGPoint] {
    let steps = min(60, max(8, Int(hypot(to.x - from.x, to.y - from.y) / 12)))
    return (1...steps).map { step in
        let t = CGFloat(step) / CGFloat(steps)
        return CGPoint(x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t)
    }
}

struct WheelStep: Equatable {
    let vertical: Int32
    let horizontal: Int32
}

// Positive dy scrolls toward later content and positive dx to the right; wheel deltas carry the opposite sign.
func scrollSteps(dx: Int, dy: Int, perEvent: Int = 10) -> [WheelStep] {
    var steps: [WheelStep] = []
    var restX = dx
    var restY = dy
    while restX != 0 || restY != 0 {
        let stepX = max(-perEvent, min(perEvent, restX))
        let stepY = max(-perEvent, min(perEvent, restY))
        steps.append(WheelStep(vertical: Int32(-stepY), horizontal: Int32(-stepX)))
        restX -= stepX
        restY -= stepY
    }
    return steps
}

enum TypeSegment: Equatable {
    case text(String)
    case key(CGKeyCode)
}

// Line breaks and tabs go out as Return and Tab presses; everything else as Unicode text.
func typeSegments(_ text: String) -> [TypeSegment] {
    var segments: [TypeSegment] = []
    var current = ""
    for character in text {
        let code: CGKeyCode? = character == "\n" || character == "\r" || character == "\r\n" ? 36 : character == "\t" ? 48 : nil
        guard let code else {
            current.append(character)
            continue
        }
        if !current.isEmpty { segments.append(.text(current)) }
        current = ""
        segments.append(.key(code))
    }
    if !current.isEmpty { segments.append(.text(current)) }
    return segments
}

// Chunks stay under the 20 UTF-16 units a keyboard event carries and never split a character.
func unicodeChunks(_ text: String, maxUnits: Int = 16) -> [[UniChar]] {
    var chunks: [[UniChar]] = []
    var current: [UniChar] = []
    for character in text {
        let units = Array(String(character).utf16)
        if units.count > maxUnits {
            if !current.isEmpty { chunks.append(current) }
            current = []
            chunks.append(contentsOf: character.unicodeScalars.map { Array(String($0).utf16) })
            continue
        }
        if current.count + units.count > maxUnits {
            chunks.append(current)
            current = []
        }
        current.append(contentsOf: units)
    }
    if !current.isEmpty { chunks.append(current) }
    return chunks
}

func postText(_ units: [UniChar]) {
    for down in [true, false] {
        guard let event = CGEvent(keyboardEventSource: eventSource, virtualKey: 0, keyDown: down) else { continue }
        event.flags = []
        units.withUnsafeBufferPointer { event.keyboardSetUnicodeString(stringLength: $0.count, unicodeString: $0.baseAddress) }
        event.post(tap: .cghidEventTap)
    }
}

func postKey(_ code: CGKeyCode, down: Bool, flags: CGEventFlags) {
    guard let event = CGEvent(keyboardEventSource: eventSource, virtualKey: code, keyDown: down) else { return }
    event.flags = flags
    event.post(tap: .cghidEventTap)
}

func pressCombo(_ combo: KeyCombo) {
    var held: CGEventFlags = []
    for modifier in combo.modifiers {
        held.insert(modifier.flag)
        if let code = modifier.keyCode {
            postKey(code, down: true, flags: held)
            hold(0.01)
        }
    }
    postKey(combo.keyCode, down: true, flags: held)
    hold(0.02)
    postKey(combo.keyCode, down: false, flags: held)
    for modifier in combo.modifiers.reversed() {
        held.remove(modifier.flag)
        if let code = modifier.keyCode {
            hold(0.01)
            postKey(code, down: false, flags: held)
        }
    }
}

func supportsPress(_ element: AXUIElement) -> Bool {
    var names: CFArray?
    guard AXUIElementCopyActionNames(element, &names) == .success, let list = names as? [String] else { return false }
    return list.contains(kAXPressAction)
}

func staleElement(_ id: Int) -> HelperError {
    HelperError("stale_element", "Element \(id) is not in the latest observe snapshot or has gone away; observe again.")
}

// The element's current frame centre, refused when the element is TensorFold's or no longer on screen.
func elementCentre(_ id: Int) throws -> CGPoint {
    guard let element = snapshot.elements[id], let pid = elementPid(element) else { throw staleElement(id) }
    if isOwnPid(pid) { throw ownAppError() }
    guard let frame = elementFrame(element), frame.width > 0, frame.height > 0 else { throw staleElement(id) }
    return CGPoint(x: frame.midX, y: frame.midY)
}

func click(_ target: Target, _ button: MouseButton, _ count: Int) throws -> [String: Any] {
    try requireAccessibility()
    switch target {
    case .point(let point):
        try checkPoint(point)
        mouseClick(at: point, button: button, count: count)
        return actionResult(via: "mouse")
    case .element(let id):
        guard let element = snapshot.elements[id], let pid = elementPid(element) else { throw staleElement(id) }
        if isOwnPid(pid) { throw ownAppError() }
        if button == .left && count == 1 && supportsPress(element) {
            let result = AXUIElementPerformAction(element, kAXPressAction as CFString)
            // cannotComplete is the messaging timeout, which a press that opens a modal often hits after it worked.
            if result == .success || result == .cannotComplete { return actionResult(via: "press") }
            if result == .invalidUIElement { throw staleElement(id) }
        }
        let centre = try elementCentre(id)
        try checkPoint(centre)
        mouseClick(at: centre, button: button, count: count)
        return actionResult(via: "mouse")
    }
}

func move(to point: CGPoint) throws -> [String: Any] {
    try requireAccessibility()
    try checkPoint(point)
    postMouse(.mouseMoved, point, .left)
    return actionResult(settle: 0.05)
}

func drag(from: CGPoint, to: CGPoint) throws -> [String: Any] {
    try requireAccessibility()
    try checkPoint(from)
    try checkPoint(to)
    postMouse(.mouseMoved, from, .left)
    hold(0.03)
    postMouse(.leftMouseDown, from, .left)
    hold(0.08)
    for point in dragPath(from: from, to: to) {
        postMouse(.leftMouseDragged, point, .left)
        hold(0.012)
    }
    hold(0.05)
    postMouse(.leftMouseUp, to, .left)
    return actionResult()
}

func scroll(at target: Target?, dx: Int, dy: Int) throws -> [String: Any] {
    try requireAccessibility()
    var point: CGPoint?
    switch target {
    case .point(let at): point = at
    case .element(let id): point = try elementCentre(id)
    case nil: point = nil
    }
    try checkPoint(point ?? CGEvent(source: nil)?.location ?? .zero)
    if let point {
        postMouse(.mouseMoved, point, .left)
        hold(0.03)
    }
    for step in scrollSteps(dx: dx, dy: dy) {
        guard let event = CGEvent(scrollWheelEvent2Source: eventSource, units: .line, wheelCount: 2, wheel1: step.vertical, wheel2: step.horizontal, wheel3: 0) else { continue }
        event.flags = []
        event.post(tap: .cghidEventTap)
        hold(0.02)
    }
    return actionResult()
}

// Focus is checked again before every segment, because a Return or Tab can move it to another app or a password field.
func typeText(_ text: String) throws -> [String: Any] {
    try requireAccessibility()
    let total = text.count
    var typed = 0
    for segment in typeSegments(text) {
        let isText: Bool
        if case .text = segment { isText = true } else { isText = false }
        do {
            try checkKeyboardTarget(typesCharacters: isText)
        } catch let error as HelperError where typed > 0 {
            throw HelperError(error.code, "Stopped after \(typed) of \(total) characters. \(error.message)")
        }
        switch segment {
        case .key(let code):
            pressCombo(KeyCombo(keyCode: code, modifiers: []))
            typed += 1
            hold(0.03)
        case .text(let chunkText):
            for chunk in unicodeChunks(chunkText) {
                postText(chunk)
                hold(0.008)
            }
            typed += chunkText.count
        }
    }
    var out = actionResult(settle: 0.1)
    out["typed"] = typed
    return out
}

func pressKeys(_ combo: KeyCombo) throws -> [String: Any] {
    try requireAccessibility()
    try checkKeyboardTarget(typesCharacters: comboTypesCharacter(combo))
    pressCombo(combo)
    return actionResult(settle: 0.1)
}

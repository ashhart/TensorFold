import AppKit
import ApplicationServices
import CoreGraphics

struct NodeInfo: Equatable {
    var role: String
    var subrole: String?
    var title: String?
    var value: String?
    var description: String?
    var placeholder: String?
    var frame: CGRect?
    var enabled = true
    var focused = false
}

let actionRoles: Set<String> = [
    "AXButton", "AXLink", "AXTextField", "AXTextArea", "AXSearchField", "AXComboBox", "AXCheckBox",
    "AXRadioButton", "AXPopUpButton", "AXMenuButton", "AXMenuItem", "AXMenuBarItem", "AXSlider",
    "AXIncrementor", "AXDisclosureTriangle", "AXColorWell", "AXDateField", "AXTimeField", "AXTab",
]
let textRoles: Set<String> = ["AXStaticText", "AXHeading", "AXCell", "AXRow"]
let maxTextLength = 200
let maxWalkVisits = 5000
let maxWalkDepth = 60
let walkSeconds = 4.0

func isSecureRole(_ role: String, _ subrole: String?) -> Bool {
    role == "AXSecureTextField" || subrole == "AXSecureTextField"
}

func shortText(_ text: String?) -> String? {
    guard let text, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
    return text.count > maxTextLength ? String(text.prefix(maxTextLength)) : text
}

func pointOnAnyDisplay(_ point: CGPoint, _ displays: [CGRect]) -> Bool {
    guard point.x.isFinite, point.y.isFinite else { return false }
    return displays.contains { point.x >= $0.minX && point.x < $0.maxX && point.y >= $0.minY && point.y < $0.maxY }
}

// What observe reports: controls a person acts on, and text-bearing elements a person reads, when visible.
func keptElement(_ node: NodeInfo, clip: CGRect?, displays: [CGRect]) -> NodeInfo? {
    guard let frame = node.frame, frame.width > 0, frame.height > 0 else { return nil }
    if let clip, !frame.intersects(clip) { return nil }
    guard displays.contains(where: { $0.intersects(frame) }) else { return nil }
    let secure = isSecureRole(node.role, node.subrole)
    var out = node
    out.title = shortText(node.title)
    out.description = shortText(node.description)
    out.placeholder = shortText(node.placeholder)
    out.value = secure ? nil : shortText(node.value)
    if secure || actionRoles.contains(node.role) { return out }
    let hasText = out.title != nil || out.value != nil || out.description != nil
    return textRoles.contains(node.role) && hasText ? out : nil
}

func rounded(_ value: CGFloat) -> Double {
    (Double(value) * 10).rounded() / 10
}

func frameJSON(_ rect: CGRect) -> [String: Any] {
    ["x": rounded(rect.minX), "y": rounded(rect.minY), "width": rounded(rect.width), "height": rounded(rect.height)]
}

func elementJSON(_ id: Int, _ node: NodeInfo) -> [String: Any] {
    var out: [String: Any] = [
        "id": id, "role": node.role, "frame": frameJSON(node.frame ?? .zero),
        "enabled": node.enabled, "focused": node.focused, "secure": isSecureRole(node.role, node.subrole),
    ]
    if let subrole = node.subrole { out["subrole"] = subrole }
    if let title = node.title { out["title"] = title }
    if let value = node.value { out["value"] = value }
    if let description = node.description { out["description"] = description }
    if let placeholder = node.placeholder { out["placeholder"] = placeholder }
    return out
}

let systemWide = AXUIElementCreateSystemWide()

func axAttribute(_ element: AXUIElement, _ name: String) -> AnyObject? {
    var value: CFTypeRef?
    guard AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success, let value else { return nil }
    return value as AnyObject
}

func axElement(_ value: AnyObject?) -> AXUIElement? {
    guard let value, CFGetTypeID(value) == AXUIElementGetTypeID() else { return nil }
    return (value as! AXUIElement)
}

func axValue(_ value: AnyObject?, _ type: AXValueType) -> AXValue? {
    guard let value, CFGetTypeID(value) == AXValueGetTypeID() else { return nil }
    let axValue = value as! AXValue
    return AXValueGetType(axValue) == type ? axValue : nil
}

func axFrame(position: AnyObject?, size: AnyObject?) -> CGRect? {
    guard let positionValue = axValue(position, .cgPoint), let sizeValue = axValue(size, .cgSize) else { return nil }
    var origin = CGPoint.zero
    var extent = CGSize.zero
    guard AXValueGetValue(positionValue, .cgPoint, &origin), AXValueGetValue(sizeValue, .cgSize, &extent) else { return nil }
    return CGRect(origin: origin, size: extent)
}

func axString(_ value: AnyObject?) -> String? {
    if let string = value as? String { return string }
    if let number = value as? NSNumber { return number.stringValue }
    return nil
}

func elementFrame(_ element: AXUIElement) -> CGRect? {
    axFrame(position: axAttribute(element, kAXPositionAttribute), size: axAttribute(element, kAXSizeAttribute))
}

func elementPid(_ element: AXUIElement) -> pid_t? {
    var pid: pid_t = 0
    return AXUIElementGetPid(element, &pid) == .success ? pid : nil
}

let walkAttributes: [String] = [
    kAXRoleAttribute, kAXSubroleAttribute, kAXTitleAttribute, kAXValueAttribute, kAXDescriptionAttribute,
    "AXPlaceholderValue", kAXPositionAttribute, kAXSizeAttribute, kAXEnabledAttribute, kAXFocusedAttribute,
    kAXChildrenAttribute,
]

// One round trip per element; missing attributes come back as AXValue error placeholders.
func readNode(_ element: AXUIElement) -> (NodeInfo, [AXUIElement])? {
    var raw: CFArray?
    guard AXUIElementCopyMultipleAttributeValues(element, walkAttributes as CFArray, AXCopyMultipleAttributeOptions(rawValue: 0), &raw) == .success,
          let values = raw as? [AnyObject], values.count == walkAttributes.count else { return nil }
    func at(_ index: Int) -> AnyObject? {
        let value = values[index]
        if CFGetTypeID(value) == AXValueGetTypeID(), AXValueGetType(value as! AXValue) == .axError { return nil }
        return value
    }
    guard let role = at(0) as? String else { return nil }
    var node = NodeInfo(role: role)
    node.subrole = at(1) as? String
    node.title = at(2) as? String
    node.value = axString(at(3))
    node.description = at(4) as? String
    node.placeholder = at(5) as? String
    node.frame = axFrame(position: at(6), size: at(7))
    node.enabled = (at(8) as? Bool) ?? true
    node.focused = (at(9) as? Bool) ?? false
    let children = (at(10) as? [AnyObject] ?? []).compactMap { axElement($0) }
    return (node, children)
}

struct WalkResult {
    var kept: [(AXUIElement, NodeInfo)] = []
    var truncated = false
}

// Depth-first in reading order. Scroll areas narrow the clip, and a subtree whose frame misses the clip is skipped.
func walkWindow(_ window: AXUIElement, windowFrame: CGRect?, displays: [CGRect], limit: Int) -> WalkResult {
    var result = WalkResult()
    var stack: [(AXUIElement, Int, CGRect?)] = [(window, 0, windowFrame)]
    var visits = 0
    let deadline = Date().addingTimeInterval(walkSeconds)
    while let entry = stack.popLast() {
        let (element, depth, clip) = entry
        visits += 1
        if visits > maxWalkVisits || Date() > deadline {
            result.truncated = true
            break
        }
        guard let read = readNode(element) else { continue }
        let (node, children) = read
        var childClip = clip
        if let frame = node.frame, frame.width > 0, frame.height > 0 {
            if let clip, !frame.intersects(clip) { continue }
            if node.role == "AXScrollArea" { childClip = clip.map { $0.intersection(frame) } ?? frame }
        }
        if depth > 0, let kept = keptElement(node, clip: clip, displays: displays) {
            if result.kept.count >= limit {
                result.truncated = true
                break
            }
            result.kept.append((element, kept))
        }
        if depth < maxWalkDepth {
            for child in children.reversed() { stack.append((child, depth + 1, childClip)) }
        }
    }
    return result
}

final class Snapshot {
    var pid: pid_t = 0
    var elements: [Int: AXUIElement] = [:]
}

let snapshot = Snapshot()
var webAccessibilityRequested = Set<pid_t>()

// Electron apps build their AX tree only after AXManualAccessibility is set; other apps ignore it.
func requestWebAccessibility(_ app: AXUIElement, pid: pid_t) -> Bool {
    guard !webAccessibilityRequested.contains(pid) else { return false }
    webAccessibilityRequested.insert(pid)
    return AXUIElementSetAttributeValue(app, "AXManualAccessibility" as CFString, kCFBooleanTrue) == .success
}

func activeDisplays() -> [(CGDirectDisplayID, CGRect)] {
    var count: UInt32 = 0
    guard CGGetActiveDisplayList(0, nil, &count) == .success, count > 0 else { return [] }
    var ids = [CGDirectDisplayID](repeating: 0, count: Int(count))
    guard CGGetActiveDisplayList(count, &ids, &count) == .success else { return [] }
    return ids.prefix(Int(count)).map { ($0, CGDisplayBounds($0)) }
}

func displayBounds() -> [CGRect] {
    activeDisplays().map { $0.1 }
}

func screenJSON(near frame: CGRect?) -> [String: Any] {
    let displays = activeDisplays()
    let centre = frame.map { CGPoint(x: $0.midX, y: $0.midY) }
    let id = displays.first(where: { centre != nil && $0.1.contains(centre!) })?.0 ?? CGMainDisplayID()
    let bounds = CGDisplayBounds(id)
    var scale = 1.0
    if let mode = CGDisplayCopyDisplayMode(id), mode.width > 0 { scale = Double(mode.pixelWidth) / Double(mode.width) }
    return ["x": rounded(bounds.minX), "y": rounded(bounds.minY), "width": rounded(bounds.width), "height": rounded(bounds.height), "scale": scale]
}

func appJSON(_ app: NSRunningApplication) -> [String: Any] {
    ["name": app.localizedName ?? "", "bundleId": app.bundleIdentifier ?? NSNull(), "pid": Int(app.processIdentifier)]
}

func targetWindow(_ app: AXUIElement) -> AXUIElement? {
    if let window = axElement(axAttribute(app, kAXFocusedWindowAttribute)) { return window }
    if let window = axElement(axAttribute(app, kAXMainWindowAttribute)) { return window }
    return (axAttribute(app, kAXWindowsAttribute) as? [AnyObject])?.lazy.compactMap { axElement($0) }.first
}

func axWindowJSON(_ window: AXUIElement) -> [String: Any]? {
    guard let frame = elementFrame(window) else { return nil }
    return ["title": shortText(axAttribute(window, kAXTitleAttribute) as? String) ?? "", "frame": frameJSON(frame)]
}

// Without Accessibility, the window server still knows the frontmost window's bounds (titles need Screen Recording).
func windowServerWindowJSON(pid: pid_t) -> [String: Any]? {
    guard let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] else { return nil }
    for info in list {
        guard (info[kCGWindowOwnerPID as String] as? NSNumber)?.int32Value == pid,
              (info[kCGWindowLayer as String] as? NSNumber)?.intValue == 0,
              let bounds = info[kCGWindowBounds as String] as? NSDictionary,
              let frame = CGRect(dictionaryRepresentation: bounds as CFDictionary) else { continue }
        return ["title": shortText(info[kCGWindowName as String] as? String) ?? "", "frame": frameJSON(frame)]
    }
    return nil
}

func windowJSON(pid: pid_t) -> [String: Any]? {
    if AXIsProcessTrusted() {
        let app = AXUIElementCreateApplication(pid)
        if let window = targetWindow(app), let json = axWindowJSON(window) { return json }
    }
    return windowServerWindowJSON(pid: pid)
}

func frontmostJSON() -> [String: Any] {
    guard let app = NSWorkspace.shared.frontmostApplication else { return ["app": NSNull(), "window": NSNull()] }
    return ["app": appJSON(app), "window": windowJSON(pid: app.processIdentifier) ?? NSNull()]
}

// Walks one window only; the menu bar is left out, so menus are reached with key or a click by point.
func observe(maxElements: Int, pid requested: pid_t?) throws -> [String: Any] {
    guard AXIsProcessTrusted() else { throw HelperError("no_accessibility", "TensorFold needs Accessibility permission to read other apps.") }
    let running: NSRunningApplication?
    if let requested { running = NSRunningApplication(processIdentifier: requested) } else { running = NSWorkspace.shared.frontmostApplication }
    guard let app = running, !app.isTerminated else { throw HelperError("no_app", requested == nil ? "No app is frontmost." : "No running app has pid \(requested!).") }
    let pid = app.processIdentifier
    let appElement = AXUIElementCreateApplication(pid)
    if requestWebAccessibility(appElement, pid: pid) { pump(0.3) }
    snapshot.pid = pid
    snapshot.elements = [:]
    var body: [String: Any] = ["app": appJSON(app), "window": NSNull(), "elements": [Any](), "truncated": false]
    guard let window = targetWindow(appElement) else {
        body["screen"] = screenJSON(near: nil)
        return body
    }
    let frame = elementFrame(window)
    let walk = walkWindow(window, windowFrame: frame, displays: displayBounds(), limit: maxElements)
    var elements: [Any] = []
    for (index, entry) in walk.kept.enumerated() {
        snapshot.elements[index + 1] = entry.0
        elements.append(elementJSON(index + 1, entry.1))
    }
    body["window"] = axWindowJSON(window) ?? NSNull()
    body["screen"] = screenJSON(near: frame)
    body["elements"] = elements
    body["truncated"] = walk.truncated
    return body
}

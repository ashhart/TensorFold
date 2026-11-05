import CoreGraphics

struct Modifier: Equatable {
    let flag: CGEventFlags
    // nil for fn: a lone fn press can open the emoji picker or dictation, so it only sets the flag.
    let keyCode: CGKeyCode?
}

struct KeyCombo: Equatable {
    let keyCode: CGKeyCode
    let modifiers: [Modifier]

    var flags: CGEventFlags {
        modifiers.reduce(into: CGEventFlags()) { $0.insert($1.flag) }
    }
}

// US ANSI virtual key codes (Carbon kVK_*); they name key positions, not characters.
let namedKeyCodes: [String: CGKeyCode] = [
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25, "7": 26,
    "-": 27, "8": 28, "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35,
    "return": 36, "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43,
    "/": 44, "n": 45, "m": 46, ".": 47, "tab": 48, "space": 49, "`": 50, "delete": 51,
    "escape": 53, "f5": 96, "f6": 97, "f7": 98, "f3": 99, "f8": 100, "f9": 101,
    "f11": 103, "f10": 109, "f12": 111, "home": 115, "pageup": 116, "forwarddelete": 117,
    "f4": 118, "end": 119, "f2": 120, "pagedown": 121, "f1": 122,
    "left": 123, "right": 124, "down": 125, "up": 126,
]

let keyAliases: [String: String] = [
    "enter": "return", "esc": "escape", "backspace": "delete", "minus": "-", "equal": "=", "equals": "=",
    "comma": ",", "period": ".", "slash": "/", "backslash": "\\", "semicolon": ";", "quote": "'",
    "backtick": "`", "grave": "`", "leftbracket": "[", "rightbracket": "]",
]

let modifierNames: [String: Modifier] = [
    "cmd": Modifier(flag: .maskCommand, keyCode: 55), "command": Modifier(flag: .maskCommand, keyCode: 55),
    "shift": Modifier(flag: .maskShift, keyCode: 56),
    "opt": Modifier(flag: .maskAlternate, keyCode: 58), "option": Modifier(flag: .maskAlternate, keyCode: 58),
    "alt": Modifier(flag: .maskAlternate, keyCode: 58),
    "ctrl": Modifier(flag: .maskControl, keyCode: 59), "control": Modifier(flag: .maskControl, keyCode: 59),
    "fn": Modifier(flag: .maskSecondaryFn, keyCode: nil),
]

// Keys that put a character into a text field when pressed without cmd or ctrl.
func comboTypesCharacter(_ combo: KeyCombo) -> Bool {
    if combo.flags.contains(.maskCommand) || combo.flags.contains(.maskControl) { return false }
    let nonPrinting: Set<CGKeyCode> = [36, 48, 51, 53, 96, 97, 98, 99, 100, 101, 103, 109, 111, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126]
    return !nonPrinting.contains(combo.keyCode)
}

func parseKeyCombo(_ text: String) throws -> KeyCombo {
    let shown = String(text.prefix(64))
    let lowered = text.trimmingCharacters(in: .whitespaces).lowercased()
    guard !lowered.isEmpty, lowered.count <= 64 else { throw HelperError("bad_key", "Keys must be a short combination like cmd+shift+t.") }
    var modifiers: [Modifier] = []
    var key: CGKeyCode?
    for raw in lowered.split(separator: "+", omittingEmptySubsequences: false) {
        let part = raw.trimmingCharacters(in: .whitespaces)
        if let modifier = modifierNames[part] {
            if !modifiers.contains(modifier) { modifiers.append(modifier) }
            continue
        }
        guard key == nil else { throw HelperError("bad_key", "\(shown) names more than one key.") }
        guard let code = namedKeyCodes[keyAliases[part] ?? part] else { throw HelperError("bad_key", "Unknown key \"\(part)\" in \(shown).") }
        key = code
    }
    guard let key else { throw HelperError("bad_key", "\(shown) has modifiers but no key.") }
    return KeyCombo(keyCode: key, modifiers: modifiers)
}
